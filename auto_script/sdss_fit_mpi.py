"""SDSS fit using MPI."""
from mpi4py import MPI
import glob, sys, os, random, gc
import pandas as pd
import kali.carma
import kali
import h5py
sys.path.insert(0, '/home/mount/lsst_cadence')
from lsstlc import *  # derived LSST lightcurve sub-class
import time

start = time.time()
# pre-defined for loading C. M. light curves
names = ['mjd_u', 'mjd_g', 'mjd_r', 'mjd_i', 'mjd_z', 'ra', 'dec']
cols = [0, 3, 6, 9, 12, 15, 16]

# carma task arguments
nwalkers = 200
nsteps = 1000

mjd_bands = ['mjd_r', 'mjd_u']  # will loop over for each band in list
shape = (nwalkers, nsteps)
dtype = np.dtype([('LnPosterior', np.float64, shape), ('Chain[0]', np.float64, shape), ('Chain[1]', np.float64, shape), ('rootChain[0]', np.complex128, shape), ('rootChain[1]', np.complex128, shape)])

obj_dir = sys.argv[1]
cad_dir = sys.argv[2]  # C. MacLeod LC path
input_param = sys.argv[3]  # csv containing the select mock LCs to fit among all
final_dir = sys.argv[4]  # The directory to save all output, one file per object

if not os.path.exists(final_dir):
        os.mkdir(final_dir)
        print("Creating diretory: {}".format(final_dir))

# hdf5 reference special data type
ref_dtype = h5py.special_dtype(ref=h5py.Reference)
dt = h5py.special_dtype(vlen=str)


def sdss_fit(lc, mjd_path, grp):
    """Take full mock LC and SDSS cadence to find best_fit params.

    Args:
        lc: Kali LC object, full mock LC.
        mjd_path(str): The path to sdss cadence source provided by C. MacLeod.
    """
    mjd_df = pd.read_csv(mjd_path, sep=' ', usecols=cols, names=names)  # df storing C.M. lc mjd
    ra = mjd_df.loc[0, 'ra']
    dec = mjd_df.loc[0, 'dec']
    best_param = []  # store best-fit params
    ref_ls = []
    mjd_fname = os.path.split(mjd_path)[1]
    task = kali.carma.CARMATask(1, 0, nsteps=nsteps, nwalkers=nwalkers, nthreads=1)
    
    for mjd_band in mjd_bands:

        cad = (np.array(mjd_df[mjd_band])-mjd_df[mjd_band][0])*86400  # subtract first date
        band = mjd_band.split("_")[1]  # retrive the band
        # start fitting
        task.clear()
        lc_down = lsstlc(ra, dec, cad, lc, fix_dt=True, band=band)
        task.fit(lc_down)
        
        # fitted params and chains to array and pass back
        fit = list(task.bestTau)
        fit.append(band)
        best_param.append(fit)
        mcmc_rec = np.rec.array([task.LnPosterior, task.Chain[0], task.Chain[1], task.rootChain[0], task.rootChain[1]], dtype=dtype)

        # create hdf5 dataset given id as combination of mjd_fname and band
        dset = grp.create_dataset('{}_{}'.format(mjd_fname, band), dtype=dtype, data=mcmc_rec, shape=())

        # create reference to this dataset and store in para_fit dataframe
        ref_ls.append(dset.ref)

    df_p = pd.DataFrame(best_param, columns=['tau', 'sigma', 'band'])
    df_p['file_name'] = mjd_fname
    df_p['file_name'] = df_p['file_name'].astype(np.uint32)
    df_p['ref2chain'] = ref_ls

    # flush data into file
    grp.file.flush()
    gc.collect()
    
    return df_p

# MPI code
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if __name__ == '__main__':

    # ls = glob.glob(os.path.join(obj_dir, '*'))
    cad_files = glob.glob(os.path.join(cad_dir, '*'))  # grab all C.M. lc files
    df_param = pd.read_csv(input_param)  # load shortened object list
    rt_id = 'c10_fit_{:.2f}_{:.2f}_{:d}.hdf5'  # specify the saved record file standard

    # random index to select from all lc
    rd_idx = random.sample(range(len(cad_files)), k=2)
    s_100 = [cad_files[i] for i in rd_idx]
    
    # list to store all returned param df
    df_ls = []
    
    # loop over all sdss lc cadence
    for i in range(rank, len(df_param), size):
        print(rank, i, df_param.loc[i, 'file'])
        
        # create hdf5 file and chain group
        f = h5py.File(os.path.join(final_dir, rt_id.format(df_param.loc[i, 'tau'], df_param.loc[i, 'sigma'], i)), 'w')
        grp = f.create_group('chain')
        # read lc 
        lc = extLC(os.path.join(obj_dir, df_param.loc[i, 'file']))

        for cad in s_100:

            df_rt = sdss_fit(lc, cad, grp)
            df_ls.append(df_rt)

        dff = pd.concat(df_ls, ignore_index=True)
        dff.insert(0, 'std_x', np.std(lc.x))
        dff.insert(0, 'std_y', np.std(lc.y))
        dff.insert(0, 'tauIn', df_param.loc[i, 'tau'])
        dff.insert(0, 'sigmaIn', df_param.loc[i, 'sigma'])
        rec_dff = dff.to_records(index=False)
        # print(rec_dff.dtype)

        # now redefine dtype for hdf5
        descr = rec_dff.dtype.descr
        descr[-3] = ('band', dt)
        descr[-1] = ('ref2chain', ref_dtype)
        newdp = np.dtype(descr)
        
        # add fit dataset to hdf5, flush() and close()
        
        f.create_dataset('fit', data=rec_dff, dtype=newdp)
        f.flush()
        f.close()
        gc.collect()

    end = time.time()
    print('Total seconds spent {} for process {}'.format(end-start, rank))

