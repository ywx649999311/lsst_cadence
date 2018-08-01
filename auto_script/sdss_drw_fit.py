"""Fit LC to DRW model."""
import numpy as np
import pandas as pd
import kali.carma
import kali
from joblib import Parallel, delayed
import os, sys, glob
sys.path.insert(0, '/home/mount/')
from lsstlc import *  # derived LSST lightcurve sub-class


names = ['mjd_u', 'mjd_g', 'mjd_r', 'mjd_i', 'mjd_z', 'ra', 'dec']
cols = [0, 3, 6, 9, 12, 15, 16] 


def sdss_fit(lc, mjd_path):
    """Take full mock LC and SDSS cadence to find best_fit params.

    Args:
        lc: Kali LC object, full mock LC.
        mjd_path(str): The path to sdss cadence source provided by C. MacLeod.
    """
    mjd = pd.read_csv(mjd_path, sep=' ', usecols=cols, names=names)
    ra = mjd.loc[0, 'ra']
    dec = mjd.loc[0, 'dec']
    bands = ['mjd_r']  # will expand to all bands once enough cpu
    best_param = []  # list to hold result from different bands
    task = kali.carma.CARMATask(1, 0, nsteps=1000, nwalkers=200, nthreads=1)
    
    for band in bands:
        task.clear()
        cad = (np.array(mjd[band])-mjd[band][0])*86400
        lc_down = lsstlc(ra, dec, cad, lc, fix_dt=True, band=band.split("_")[1])
        task.fit(lc_down)
        fit = list(task.bestTau)
        fit.append(band.split("_")[1])
        best_param.append(fit)
    
    df_p = pd.DataFrame(best_param, columns=['tau', 'sigma', 'filter'])
    df_p['ra'] = ra
    df_p['dec'] = dec
    df_p['file_name'] = os.path.split(mjd_path)[1]
    
    return df_p

if __name__ == '__main__':
    
    lc_dir = sys.argv[1]
    cad_dir = sys.argv[2]
    input_param = sys.argv[3]
    final_path = sys.argv[4]

    cad_files = glob.glob(os.path.join(cad_dir, '*'))
    df_param = pd.read_csv(input_param)

    array_ls = []  # master list to hold result for all simulated objects
    for i in range(len(df_param)):
        lc_path = os.path.join(lc_dir, df_param.loc[i, 'file'])
        lc = extLC(lc_path)

        # randomly select 100 cads to fit (not including n_obs < 50)
        rd_idx = np.random.randint(len(cad_files), size=100)
        s_100 = [cad_files[i] for i in rd_idx]
        
        # start parallel code
        best = Parallel(n_jobs=-1)(delayed(sdss_fit)(lc, path) for path in s_100)
        
        # put into one master frame and then to rec.array
        dff = pd.concat(best, ignore_index=True)
        dff['std_x'] = np.std(lc.x)
        dff['std_y'] = np.std(lc.y)
        dff['tauIn'] = df_param.loc[i, 'tau']
        dff['sigmaIn'] = df_param.loc[i, 'sigma']
        rec_arr = dff.to_records(index=False)

        # append to master list
        array_ls.append(rec_arr)

    # finally save all result to a file
    np.save(final_path, array_ls)
