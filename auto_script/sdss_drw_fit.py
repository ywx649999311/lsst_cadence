"""Fit LC to DRW model."""
import numpy as np
import pandas as pd
import kali.carma
import kali
from joblib import Parallel, delayed, load, dump
from concurrent.futures import ProcessPoolExecutor
import os, sys, glob, gc, random
sys.path.insert(0, '/home/mount/lsst_cadence')
from lsstlc import *  # derived LSST lightcurve sub-class


# pre-defined for loading C. M. light curves
names = ['mjd_u', 'mjd_g', 'mjd_r', 'mjd_i', 'mjd_z', 'ra', 'dec']
cols = [0, 3, 6, 9, 12, 15, 16]

# carma task arguments
nwalkers = 200
nsteps = 1000

bands = ['mjd_r']  # will loop over for each band in list
shape = (nwalkers, nsteps)
dtype = np.dtype([('LnPosterior', np.float64, shape), ('Chain[0]', np.float64, shape), ('Chain[1]', np.float64, shape), ('rootChain[0]', np.complex128, shape), ('rootChain[1]', np.complex128, shape)])
n_cpu = os.cpu_count()
cwd = os.getcwd()


def sdss_fit(lc, mjd_path):
    """Take full mock LC and SDSS cadence to find best_fit params.

    Args:
        lc: Kali LC object, full mock LC.
        mjd_path(str): The path to sdss cadence source provided by C. MacLeod.
    """
    mjd = pd.read_csv(mjd_path, sep=' ', usecols=cols, names=names)
    ra = mjd.loc[0, 'ra']
    dec = mjd.loc[0, 'dec']
    best_param = []  # store best-fit params
    mcmc = []  # store mcmc chain
    task = kali.carma.CARMATask(1, 0, nsteps=nsteps, nwalkers=nwalkers, nthreads=1)
    
    for band in bands:

        cad = (np.array(mjd[band])-mjd[band][0])*86400  # subtract first date
        
        # start fitting
        task.clear()
        lc_down = lsstlc(ra, dec, cad, lc, fix_dt=True, band=band.split("_")[1])
        task.fit(lc_down)
        
        # fitted params and chains to array and pass back
        fit = list(task.bestTau)
        fit.append(band.split("_")[1])
        best_param.append(fit)
        mcmc.append(np.rec.array([task.LnPosterior, task.Chain[0], task.Chain[1], task.rootChain[0], task.rootChain[1]], dtype=dtype))

    df_p = pd.DataFrame(best_param, columns=['tau', 'sigma', 'filter'])
    df_p['file_name'] = os.path.split(mjd_path)[1]
    
    return (df_p, mcmc) 


def read_lc(path_combo):
    """Given path combo, return lc object."""

    idx = path_combo[1]
    path = os.path.join(path_combo[0], df_param.loc[idx, 'file'])
    lc = extLC(path)
    return lc


def save(data):
    """Parallelized task for saving fit data to file."""
    
    sd = data[1]
    np.savez_compressed(data[0], fit=sd[0], chain=sd[1])


if __name__ == '__main__':
    
    lc_dir = sys.argv[1]  # Directory containning all simulated LCs
    cad_dir = sys.argv[2]  # C. MacLeod LC path
    input_param = sys.argv[3]  # csv containing the select mock LCs to fit among all
    final_dir = sys.argv[4]  # The directory to save all output, one file per object

    if not os.path.exists(final_dir):
        os.mkdir(final_dir)
        print("Creating diretory: {}".format(final_dir))

    cad_files = glob.glob(os.path.join(cad_dir, '*'))  # grab all C.M. lc files
    df_param = pd.read_csv(input_param)  # load shortened object list
    rt_id = 'c10_fit_{:.2f}_{:.2f}_{:d}'  # specify the saved record file standard
    save_ls = []  # Hold results for every 4 objects in the first loop

    # random index to select from all lc
    rd_idx = random.sample(range(len(cad_files)), k=8)
    s_100 = [cad_files[i] for i in rd_idx]

    for i in df_param.index[::n_cpu]:

        # pack arguments into tuple
        path_combos = [(lc_dir, i) for i in df_param.index[i:i+n_cpu]]

        # using process pool to parallize file IO
        with ProcessPoolExecutor(n_cpu) as executor:
            LCs = list(executor.map(read_lc, path_combos))

        for j in range(len(LCs)):
            
            lc = LCs[j]  # take lc object out
            rt_path = os.path.join(final_dir, rt_id.format(df_param.loc[i+j, 'tau'], df_param.loc[i+j, 'sigma'], i+j))  # file path for saved result
        
            # start parallel code
            try:
                best = Parallel(n_jobs=-1)(delayed(sdss_fit)(lc, path) for path in s_100)
            except Exception as inst:
                print(inst)
                print('Failed at object #{:d}'.format(i))
                continue
            
            # put into one master frame and then to rec.array
            df_ls = [best[i][0] for i in range(len(best))]
            mcmc_ls = [best[i][1] for i in range(len(best))]
            dff = pd.concat(df_ls, ignore_index=True)
            dff['std_x'] = np.std(lc.x)
            dff['std_y'] = np.std(lc.y)
            dff['tauIn'] = df_param.loc[i, 'tau']
            dff['sigmaIn'] = df_param.loc[i, 'sigma']
            rec_dff = dff.to_records(index=False)
            mcmc_arr = [mcmc_band for mcmc_path in mcmc_ls for mcmc_band in mcmc_path]

            # put data need to be saved to a list
            save_ls.append((rt_path, [rec_dff, mcmc_arr]))

        with ProcessPoolExecutor(n_cpu) as executor:
            executor.map(save, save_ls)

        # force garbage collection every 4 objects
        gc.collect()
