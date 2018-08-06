"""import lsst maf packages."""
import lsst.sims.maf
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles

import numpy as np
import pandas as pd
import sys

# setup database connection
outDir = 'maf_out'
Baseline = '/home/mount/Opsim DB/minion_1016_sqlite.db'
# specify output directory for metrics result
resultsDb = db.ResultsDb(outDir=outDir)


def run_maf(dbFile, ra, dec):
    """Retrive min inter_night gap, and observation history with the input of database file name and arrays of RA and DEC."""

    # establish connection to sqllite database file.
    opsimdb = db.OpsimDatabase(dbFile)
    
    # While we're in transition between opsim v3 and v4, this may be helpful: print("{dbFile} is an opsim version {version} database".format(dbFile=dbFile, version=opsimdb.opsimVersion))
    if opsimdb.opsimVersion == "V3":
        # For v3 databases:
        mjdcol = 'expMJD'
        degrees = False
    else:
        # For v4 and alternate scheduler databases.
        mjdcol = 'observationStartMJD'
        degrees = True
    
    # IntraNightGapsMetric returns the gap (in days) between observations within the same night custom reduceFunc to find min gaps 
    metric = metrics.cadenceMetrics.IntraNightGapsMetric(reduceFunc=np.amin, mjdCol=mjdcol)
    # PassMetric just pass all values
    metric_pass = metrics.simpleMetrics.PassMetric(cols=['filter', 'fiveSigmaDepth', mjdcol, 'expDate', 'fieldID'])
    # slicer for slicing pointing history
    slicer = slicers.UserPointsSlicer(ra, dec, lonCol='fieldRA', latCol='fieldDec', latLonDeg=degrees)
    # sql constrains, here I put none
    sql = 'propID = 54'
    
    # bundles to combine metric, slicer and sql constrain together
    bundle = metricBundles.MetricBundle(metric, slicer, sql)
    date_bundle = metricBundles.MetricBundle(metric_pass, slicer, sql)
    
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
    result = run_maf(Baseline, ra, dec)

    # get data
    cad = result.bundleDict['cadence'].metricValues.data
    sep = result.bundleDict['sep'].metricValues.data

    # determine location returned no result
    cad_del = [i for i in range(len(cad)) if cad[i] is None]

    # save sep to input csv
    loc_df['sep'] = sep
    loc_df.drop(cad_del)
    loc_df.to_csv(loc_path, index=False)

    # save cadence to npy file
    cad = np.delete(cad, cad_del)
    np.save(sys.argv[2], cad)

