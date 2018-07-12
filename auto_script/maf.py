# import lsst maf packages
import lsst.sims.maf
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as metricBundles

import numpy as np
import pandas as pd

# setup database connection
outDir ='maf_out'
Baseline = '/home/mount/Opsim DB/minion_1016_sqlite.db'
# specify output directory for metrics result
resultsDb = db.ResultsDb(outDir=outDir)

#retrive min inter_night gap, and observation history with the input of database file name and arrays of RA and DEC
def run_maf(dbFile, ra, dec):
    # establish connection to sqllite database file
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
    metric_pass = metrics.simpleMetrics.PassMetric(cols=['filter','fiveSigmaDepth', mjdcol, 'expDate', 'fieldID'])
    # slicer for slicing pointing history
    slicer = slicers.UserPointsSlicer(ra, dec, lonCol='fieldRA', latCol='fieldDec', latLonDeg=degrees)
    # sql constrains, here I put none
    sql = '' #'night < 365'
    
    # bundles to combine metric, slicer and sql constrain together
    bundle = metricBundles.MetricBundle(metric,slicer,sql)
    date_bundle = metricBundles.MetricBundle(metric_pass, slicer, sql)
    # In case you are using a dither stacker, we can check what columns are actually being pulled from the database.
    print(bundle.dbCols)
    
    # create metric bundle group and returns
    bg = metricBundles.MetricBundleGroup({'sep': bundle, 'cadence':date_bundle}, opsimdb, outDir=outDir, resultsDb=resultsDb)
    bg.runAll()
    opsimdb.close()
    return bg

if __name__ == "__main__":
    
    # read coordinate from csv
    loc_df = pd.read_csv('/home/mount/lsst_cadence/auto_script/loc.csv')
    ra = list(loc_df['ra'])
    dec = list(loc_df['dec'])

    print(ra, dec)
    # run maf
    result = run_maf(Baseline, ra, dec)

    for i in loc_df.index:
        odf = loc_df.copy()
        odf['sep'] = result.bundleDict['sep'].metricValues.data
        odf['db'] = 'minion_1016'
        odf.to_csv('./maf_meta.csv')

        ofile = './maf_cadence'
        np.save(ofile, result.bundleDict['cadence'].metricValues.data)























