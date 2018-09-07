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
Baseline = '/home/data/Opsim DB/v4/baseline2018a.db'
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


def run_maf(dbFile):
    """Retrive min inter_night gap, and observation history with the input of database file name and arrays of RA and DEC."""

    # establish connection to sqllite database file.
    opsimdb = db.OpsimDatabase(dbFile)
    
    # While we're in transition between opsim v3 and v4, this may be helpful: print("{dbFile} is an opsim version {version} database".format(dbFile=dbFile, version=opsimdb.opsimVersion))
    if opsimdb.opsimVersion == "V3":
        # For v3 databases:
        mjdcol = 'expMJD'
        cols = ['filter', 'fiveSigmaDepth', mjdcol, 'expDate', 'fieldID']
        stackerList = []
    else:
        # For v4 and alternate scheduler databases.
        mjdcol = 'observationStartMJD'
        cols = ['filter', 'fiveSigmaDepth', mjdcol, 'fieldId']
        stackerList = [expDateStacker()]
    
    # PassMetric just pass all values
    metric_pass = metrics.simpleMetrics.PassMetric(cols=cols)
    slicer = slicers.UniSlicer()
    # sql constrains, here I put none
    sql = 'fieldId IN (744, 1427, 2412, 2786)'
    
    # bundles to combine metric, slicer and sql constrain together
    ddf = metricBundles.MetricBundle(metric_pass, slicer, sql, stackerList=stackerList)
    
    # create metric bundle group and returns
    bg = metricBundles.MetricBundleGroup({'ddf': ddf}, opsimdb, outDir=outDir, resultsDb=resultsDb)
    bg.runAll()
    opsimdb.close()
    return bg

if __name__ == "__main__":

    # run maf
    result = run_maf(Baseline)

    # get data and save to file
    cad = result.bundleDict['ddf'].metricValues.data[0]
    np.savez(sys.argv[1], Elias=cad[cad['fieldID'] == 744], Chandra=cad[cad['fieldID'] == 1427], XMM=cad[cad['fieldID'] == 2412], Cosmos=cad[cad['fieldID'] == 2786])
