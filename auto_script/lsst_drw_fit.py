"""LSST DRW fit using MPI/HDF5."""
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

# carma task arguments
nwalkers = 200
nsteps = 1000

bands = ['r']  # will loop over for each band in list
shape = (nwalkers, nsteps)
dtype = np.dtype([('LnPosterior', np.float64, shape), ('Chain[0]', np.float64, shape), ('Chain[1]', np.float64, shape), ('rootChain[0]', np.complex128, shape), ('rootChain[1]', np.complex128, shape)])

obj_dir = sys.argv[1]  # simulated lc path
lsst_maf = sys.argv[2]  # LSST cadence output
loc_csv = sys.argv[3]
input_param = sys.argv[4]  # csv containing the select mock LCs to fit among all
final_dir = sys.argv[5]  # The directory to save all output, one file per object

if not os.path.exists(final_dir):
    os.mkdir(final_dir)
    print("Creating diretory: {}".format(final_dir))

# hdf5 reference special data type
ref_dtype = h5py.special_dtype(ref=h5py.Reference)
dt = h5py.special_dtype(vlen=str)

# load maf output
maf = np.load(lsst_maf)
df_loc = pd.read_csv(loc_csv)


def lsst_fit(lc, rd_cad, grp):
    """Take full mock LC and SDSS cadence to find best_fit params.

    Args:
        lc: Kali LC object, full mock LC.
        mjd_path(str): The path to sdss cadence source provided by C. MacLeod.
    """
    
    best_param = []  # store best-fit params
    ref_ls = []
    task = kali.carma.CARMATask(1, 0, nsteps=nsteps, nwalkers=nwalkers, nthreads=1)
    
    for cad_idx in rd_cad:
        
        cad = maf[cad_idx]
        ra = df_loc.loc[cad_idx, 'ra']
        dec = df_loc.loc[cad_idx, 'dec']
        
        # loop over required bands
        for band in bands:

            # start fitting
            task.clear()
            lc_down = lsstlc(ra, dec, cad[cad['filter'] == band]['expDate'], lc, fix_dt=True, band=band)
            task.fit(lc_down)
            
            # fitted params and chains to array and pass back
            fit = list(task.bestTau)
            fit.append(band)
            fit.append(ra)
            fit.append(dec)
            best_param.append(fit)
            mcmc_rec = np.rec.array([task.LnPosterior, task.Chain[0], task.Chain[1], task.rootChain[0], task.rootChain[1]], dtype=dtype)

            # create hdf5 dataset given id as combination of ra, dec and band
            dset = grp.create_dataset('{}_{}_{}'.format(ra, dec, band), dtype=dtype, data=mcmc_rec, shape=())

            # create reference to this dataset and store in para_fit dataframe
            ref_ls.append(dset.ref)

        df_p = pd.DataFrame(best_param, columns=['tau', 'sigma', 'band', 'ra', 'dec'])
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

    # sim_lcs = glob.glob(os.path.join(cad_dir, '*'))  # grab all C.M. lc files
    df_param = pd.read_csv(input_param)  # load shortened object list
    rt_id = 'c10_fit_{:.2f}_{:.2f}_{:d}_m2016.hdf5'  # specify the saved record file standard

    # random index to select from all simulated objects
    rd_obj = random.sample(range(len(df_param)), k=4)
    rd_obj.sort()
    
    # random index to select from all cadence
    rd_cad = random.sample(range(len(maf)), k=2)

    # list to store all returned param df
    df_ls = []
    
    # loop over all sdss lc cadence
    for i in range(rank, len(rd_obj), size):
        print(rank, i, df_param.loc[i, 'file'])
        
        j = rd_obj[i]
        # create hdf5 file and chain group
        f = h5py.File(os.path.join(final_dir, rt_id.format(df_param.loc[j, 'tau'], df_param.loc[j, 'sigma'], j)), 'w')
        print(rt_id.format(df_param.loc[j, 'tau'], df_param.loc[j, 'sigma'], j))
        grp = f.create_group('chain')
        # read lc 
        lc = extLC(os.path.join(obj_dir, df_param.loc[j, 'file']))

        try:
            df_rt = lsst_fit(lc, rd_cad, grp)
            df_ls.append(df_rt)
        except Exception as exter:
            print(exter)
            print('Failed at object {:d}'.format(j))
            continue

        dff = pd.concat(df_ls, ignore_index=True)
        dff.insert(0, 'std_x', np.std(lc.x))
        dff.insert(0, 'std_y', np.std(lc.y))
        dff.insert(0, 'sigmaIn', df_param.loc[i, 'sigma'])
        dff.insert(0, 'tauIn', df_param.loc[i, 'tau'])
        rec_dff = dff.to_records(index=False)

        # now redefine dtype for hdf5
        descr = rec_dff.dtype.descr
        descr[-4] = ('band', dt)
        descr[-1] = ('ref2chain', ref_dtype)
        newdp = np.dtype(descr)
        
        # add fit dataset to hdf5, flush() and close()
        
        f.create_dataset('fit', data=rec_dff, dtype=newdp)
        f.flush()
        f.close()
        gc.collect()

    end = time.time()
    print('Total seconds spent {} for process {}'.format(end-start, rank))
