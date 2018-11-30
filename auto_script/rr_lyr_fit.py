"""LSST DRW fit for RR_Lyr using joblib.

!!Need to config a few settings:!!
1. Path of lsstlc.py script, set in sys.path.insert()

"""
import sys, os, gc
import pandas as pd
import numpy as np
from joblib import delayed, Parallel
import kali
import kali.carma
from kali import lc
sys.path.insert(0, '/home/mount/lsst_cadence')
from lsstlc import *  # derived LSST lightcurve sub-class
import time

start = time.time()

# command-line args
rr_lyr_temp = pd.read_csv(sys.argv[1])
rr_lyr_param = pd.read_csv(sys.argv[2])
maf_fname = sys.argv[3]
opt_fname = sys.argv[4]

# load maf
maf = np.load(maf_fname)
cad_all = maf['cad']
ra_all = maf['ra']
dec_all = maf['dec']

# rr_lyr template statistics
temp_std = rr_lyr_temp['y_inv'].std()
temp_mean = rr_lyr_temp['y_inv'].mean()

# kali task params
nwalkers = 100
nsteps = 1000


# kali class to read external LC
class extlc(lc.lc):
    """Fix Kali's original externalLC class."""

    def read(self, name, band, path=None, **kwargs):
        """Read function for LC object."""

        self.name = name
        self.band = band
        t = kwargs.get('tIn')
        self._ancillary = None
        if t is not None:
            self.t = np.require(t, requirements=['F', 'A', 'W', 'O', 'E'])
        else:
            raise KeyError('Must supply key-word argument t!')
        self.startT = self.t[0]
        self.t = self.t - self.startT
        self._numCadences = self.t.shape[0]
        self.x = np.require(
            kwargs.get('x', np.zeros(self.numCadences)), requirements=['F', 'A', 'W', 'O', 'E'])
        y = kwargs.get('yIn')
        if y is not None:
            self.y = np.require(y, requirements=['F', 'A', 'W', 'O', 'E'])
        else:
            raise KeyError('Must supply key-word argument y!')
        yerr = kwargs.get('yerrIn')
        if yerr is not None:
            self.yerr = np.require(yerr, requirements=['F', 'A', 'W', 'O', 'E'])
        else:
            raise KeyError('Must supply key-word argument yerr!')
        mask = kwargs.get('maskIn')
        if mask is not None:
            self.mask = np.require(mask, requirements=['F', 'A', 'W', 'O', 'E'])
        else:
            raise KeyError('Must supply key-word argument mask!')
        self.xunit = kwargs.get('xunit', r'$t$')
        self.yunit = kwargs.get('yunit', r'$F$')
    
    def write(self, name=None, band=None, pwd=None, **kwargs):
        """Neverd used write function for LC subclass."""

        pass


# function to simulate rr_lyr light curves
def rr_lyr_lc(param):
    """Function to simulate RR_Lyrae light curve using template."""
    
    # get lc param
    std = param['sigma']
    amp = std/temp_std
    T = param['period']
    
    # generate lc components
    t = np.linspace(0, 3654, int(500*3654/T))
    y_idx = range(len(t))
    y = (rr_lyr_temp['y_inv'].iloc[np.mod(y_idx, 500)].values - temp_mean)*amp + std/0.15
    
    # add noise
    y_err = np.zeros_like(y)
    noise_level = y[0:500]*param['noise']
    for i in range(500):
        y_err[i::500] = np.random.normal(0, noise_level[i], int(3654/T))
    
    # put components into kali lc object
    lc = extlc('lc', 'r', tIn=t, yIn=y, yerrIn=y_err, maskIn=np.ones_like(t))
    
    return lc


def fit(idx, lc):
    """MCMC fit function return best-fits."""

    # get exact cadence using index
    cad = cad_all[idx['cad_idx']]
    cad = cad[cad['filter'] == 'r']

    # lc period
    T = rr_lyr_param.loc[idx['obj_idx']]['period']
    
    # create task, downsample lc and fit
    task = kali.carma.CARMATask(1, 0, nsteps=1000, nwalkers=100, nthreads=1)

    try:
        lc_down = lsstlc(0, 0, cad['expDate'], lc, min_sep=T*24/500, band='r')
    except:
        print (idx)

    task.clear()
    task.fit(lc_down)
    
    ra_dec = np.array([ra_all[idx['cad_idx']], dec_all[idx['cad_idx']]])
    lc_param = rr_lyr_param.loc[idx['obj_idx']]

    return np.concatenate((lc_param, ra_dec, task.bestTau))


if __name__ == '__main__':

    # iMac Test
    obj_size = 4
    cad_size = 20

    # init index grid
    # obj_size = len(rr_lyr_param)
    # cad_size = len(cad_all)
    x = range(obj_size)
    y = range(cad_size)
    xx, yy = np.meshgrid(x, y)
    idx_grid = np.vstack([xx.ravel(), yy.ravel()])
    idx_df = pd.DataFrame({'obj_idx': idx_grid[0], 'cad_idx': idx_grid[1]})
    
    # iMac Test
    lc_list = Parallel(n_jobs=-1)(delayed(rr_lyr_lc)(rr_lyr_param.loc[i]) for i in range(4))
    # simulate lc in parallel
    # lc_list = Parallel(n_jobs=-1)(delayed(rr_lyr_lc)(rr_lyr_param.loc[i]) for i in range(len(rr_lyr_param)))

    # parallel fitting
    result = Parallel(n_jobs=-1)(delayed(fit)(idx_df.iloc[i], lc_list[idx_df.loc[i]['obj_idx']]) for i in range(len(idx_df)))

    # now save meta result into a master csv file
    clms = np.concatenate((rr_lyr_param.columns.values, ['ra', 'dec', 'tau', 'sigma']))
    df_master = pd.DataFrame(result, columns=clms)
    df_master.to_csv(opt_fname, index=False)

    
end = time.time()
print('Total seconds spent {} mins'.format((end-start)/(60)))
