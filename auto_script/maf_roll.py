"""import lsst maf packages."""
import lsst.sims.maf
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.metricBundles as metricBundles

import numpy as np
import pandas as pd
import sys

# setup database connection
outDir = 'maf_out'
RollV2 = '/home/mount/Opsim DB/enigma_1260_sqlite.db'
# specify output directory for metrics result
resultsDb = db.ResultsDb(outDir=outDir)


class expDateStacker(stackers.BaseStacker):
    """Convert MJD to seconds with respect to beginning of survey."""
    
    colsAdded = ['expDate']

    def __init__(self, mjdcol='observationStartMJD', startMJD=59853):
        """Init method."""

        self.units = ['seconds']
        self.colsReq = [mjdcol]
        self.mjdcol = mjdcol
        self.startMJD = startMJD

    def _run(self, simData, cols_present=False):
        """Calculate new column for normalized airmass."""
        # Run method is required to calculate column.
        # Driver runs getColInfo to know what columns are needed from db & which are calculated,
        #  then gets data from db and then calculates additional columns (via run methods here).
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return simData
        
        simData['expDate'] = np.round((simData[self.mjdcol] - self.startMJD)*86400)
        return simData


def run_maf(dbFile, ra, dec):
    """Retrive min inter_night gap, and observation history with the input of database file name and arrays of RA and DEC."""

    # establish connection to sqllite database file.
    opsimdb = db.OpsimDatabase(dbFile)
    
    # While we're in transition between opsim v3 and v4, this may be helpful: print("{dbFile} is an opsim version {version} database".format(dbFile=dbFile, version=opsimdb.opsimVersion))
    if opsimdb.opsimVersion == "V3":
        # For v3 databases:
        mjdcol = 'expMJD'
        degrees = False
        cols = ['filter', 'fiveSigmaDepth', mjdcol, 'expDate', 'fieldID']
        stackerList = []
    else:
        # For v4 and alternate scheduler databases.
        mjdcol = 'observationStartMJD'
        degrees = True
        cols = ['filter', 'fiveSigmaDepth', mjdcol, 'fieldId']
        stackerList = [expDateStacker()]
    
    # IntraNightGapsMetric returns the gap (in days) between observations within the same night custom reduceFunc to find min gaps 
    metric = metrics.cadenceMetrics.IntraNightGapsMetric(reduceFunc=np.amin, mjdCol=mjdcol)
    # PassMetric just pass all values
    metric_pass = metrics.simpleMetrics.PassMetric(cols=cols)
    # slicer for slicing pointing history
    slicer = slicers.UserPointsSlicer(ra, dec, lonCol='fieldRA', latCol='fieldDec', latLonDeg=degrees)
    # sql constrains, here I put none
    # sql = 'propID IN (540, 541, 542, 543, 544, 545, 546, 547, 548, 549)'
    sql = ''
    
    # bundles to combine metric, slicer and sql constrain together
    bundle = metricBundles.MetricBundle(metric, slicer, sql)
    date_bundle = metricBundles.MetricBundle(metric_pass, slicer, sql, stackerList)
    
    # create metric bundle group and returns
    bg = metricBundles.MetricBundleGroup({'sep': bundle, 'cadence': date_bundle}, opsimdb, outDir=outDir, resultsDb=resultsDb)
    bg.runAll()
    opsimdb.close()
    return bg

if __name__ == "__main__":

    # read coordinate from csv
    loc_path = sys.argv[1]
    loc_df = pd.read_csv(loc_path)
    ra = list(loc_df['ra'])
    dec = list(loc_df['dec'])

    # run maf
    result = run_maf(RollV2, ra, dec)

    # get data
    cad = result.bundleDict['cadence'].metricValues.data
    sep = result.bundleDict['sep'].metricValues.data

    # determine location returned no result
    cad_del = [i for i in range(len(cad)) if cad[i] is None]

    # save sep to input csv
    loc_df['sep'] = sep
    loc_df.drop(cad_del)
    # loc_df.to_csv(loc_path, index=False)

    # save cadence to npy file
    cad = np.delete(cad, cad_del)
    np.savez(sys.argv[2], ra=loc_df['ra'], dec=loc_df['dec'], sep=loc_df['sep'], cad=cad)
