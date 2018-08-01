"""Code to generate full mock LCs."""
import numpy as np
import kali
import kali.carma
import pandas as pd
import sys
from joblib import Parallel, delayed
sys.path.insert(0, '/home/mount/lsst_cadence')
from lsstlc import *  # derived LSST lightcurve sub-class


def genLC(params, i, save_dir):
    """Generating simulated light curves with input.

    Args:
        params: A pandas series containing tau, sigma and noise level
        i(int): The integer index of this object in the input csv
        save_dir(str): Where to store the simulated LCs

    """
    Task = kali.carma.CARMATask(1, 0, nthreads=1)
    r_1 = (-1.0/float(params['tau'])) + 0j
    sigma = float(params['sigma'])
    noise = float(params['noise'])

    Rho = np.array([r_1, sigma])
    Theta = kali.carma.coeffs(1, 0, Rho)
    
    # check whether the coefficients are valid before simulation
    if (Task.check(Theta) is False):
        return 0

    dt = 30.0/86400
    Task.set(dt, Theta)
    lc = Task.simulate(duration=3650)
    lc.fracNoiseToSignal = noise
    Task.observe(lc)

    # now save to file
    lc2file(save_dir, lc, full=True, timescales=[params['tau'], sigma, i])
    return 1

if __name__ == '__main__':

    idf = pd.read_csv(sys.argv[1])
    lc_dir = sys.argv[2]

    result = Parallel(n_jobs=-1)(delayed(genLC)(idf.loc[i], i, lc_dir) for i in idf.index)
    np.save('lc_log', result)
