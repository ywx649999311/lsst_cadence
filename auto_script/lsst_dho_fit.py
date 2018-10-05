"""LSST DRW fit using MPI/HDF5."""
from mpi4py import MPI
import glob, sys, os, random, gc
import pandas as pd
import kali
import kali.carma
import h5py
sys.path.insert(0, '/home/mount/lsst_cadence')
from lsstlc import *  # derived LSST lightcurve sub-class
import time

start = time.time()

# MPI code
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# command-line args
obj_dir = sys.argv[1]  # simulated lc path
lsst_maf = sys.argv[2]  # LSST cadence output
input_param = sys.argv[3]  # csv containing simulated lc params
final_dir = sys.argv[4]  # The directory to save all output, one file per object

# load maf output
maf = np.load(lsst_maf)
run_postfix = 'e1260'
df_param = pd.read_csv(input_param)  # load shortened object list
rt_id = 'c21_fit_{:.2f}_{:.2f}_{:d}_{}.hdf5'  # specify the saved record file standard

# Max/Min LC index and Cadence index
obj_max = 12
obj_min = 10
cad_max = 1
cad_min = 0

if rank == 0 and not os.path.exists(final_dir):
    os.mkdir(final_dir)
    print("Creating diretory: {}".format(final_dir))

# if rank == 0:
#     # random index to select from all simulated objects
#     rd_obj = random.sample(range(len(df_param)), k=1)
    
#     # random index to select from all cadence
#     rd_cad = random.sample(range(len(maf['cad'])), k=300)
# else:
#     rd_obj = None
#     rd_cad = None

# rd_obj = comm.bcast(rd_obj, root=0)
# rd_cad = comm.bcast(rd_cad, root=0)

# carma task arguments
nwalkers = 200
nsteps = 1000

bands = ['r']  # will loop over for each band in list
shape = (nwalkers, nsteps)
dtype = np.dtype([('LnPosterior', np.float64, shape), ('Chain[0]', np.float64, shape), ('Chain[1]', np.float64, shape), ('Chain[2]', np.float64, shape), ('Chain[3]', np.float64, shape), ('rootChain[0]', np.complex128, shape), ('rootChain[1]', np.complex128, shape), ('rootChain[2]', np.complex128, shape), ('rootChain[3]', np.complex128, shape)])

# hdf5 reference special data type
ref_dtype = h5py.special_dtype(ref=h5py.Reference)
dt = h5py.special_dtype(vlen=str)


def lsst_fit(lc, grp):
    """Take full mock LC and SDSS cadence to find best_fit params.

    Args:
        lc: Kali LC object, full mock LC.
        grp: HDF5 group storing the MCMC chains.
    """
    
    best_param = []  # store best-fit params
    ref_ls = []
    task = kali.carma.CARMATask(2, 1, nsteps=nsteps, nwalkers=nwalkers, nthreads=1)
    
    for cad_idx in range(cad_min, cad_max):
        
        # for new maf output
        cad = maf['cad'][cad_idx]
        ra = maf['ra'][cad_idx]
        dec = maf['dec'][cad_idx]
                                
        # loop over required bands
        for band in bands:

            # start fitting
            task.clear()
            lc_down = lsstlc(ra, dec, cad[cad['filter'] == band]['expDate'], lc, fix_dt=True, band=band)
            task.fit(lc_down)
            
            # fitted params and chains to array and pass back
            fit = list(task.bestTheta)
            fit.append(band)
            fit.append(ra)
            fit.append(dec)
            best_param.append(fit)
            mcmc_rec = np.rec.array([task.LnPosterior, task.Chain[0], task.Chain[1], task.Chain[2], task.Chain[3], task.rootChain[0], task.rootChain[1], task.rootChain[2], task.rootChain[3]], dtype=dtype)

            # create hdf5 dataset given id as combination of ra, dec and band
            dset = grp.create_dataset('{}_{}_{}'.format(ra, dec, band), dtype=dtype, data=mcmc_rec, shape=())

            # create reference to this dataset and store in para_fit dataframe
            ref_ls.append(dset.ref)

        df_p = pd.DataFrame(best_param, columns=['a1', 'a2', 'b0', 'b1', 'band', 'ra', 'dec'])
        df_p['ref2chain'] = ref_ls

    # flush data into file
    grp.file.flush()
    gc.collect()
    
    return df_p


if __name__ == '__main__':
    
    # loop over all sdss lc cadence
    for i in range(rank+obj_min, obj_max, size):
        # print(rank+obj_min, i, df_param.loc[i, 'fname'])
        
        # create hdf5 file and chain group
        f = h5py.File(os.path.join(final_dir, rt_id.format(df_param.loc[i, 'a1'], df_param.loc[i, 'b1']/df_param.loc[i, 'b0'], i, run_postfix)), 'w')
        grp = f.create_group('chain')
        # read lc 
        lc = extLC(os.path.join(obj_dir, df_param.loc[i, 'fname']+'.npz'))

        try:
            dff = lsst_fit(lc, grp)
        except Exception as exter:
            print(exter)
            print('Failed at object {:d}'.format(i))
            continue

        # dff.insert(0, 'std_x', np.std(lc.x))
        # dff.insert(0, 'std_y', np.std(lc.y))
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
