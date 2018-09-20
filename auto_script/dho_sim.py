"""Code to generate full mock LCs."""
import numpy as np
import kali
import kali.carma
import pandas as pd
import sys
from joblib import Parallel, delayed
sys.path.insert(0, '/home/mount/lsst_cadence')
from lsstlc import *  # derived LSST lightcurve sub-class


def genLC(params, save_dir):
    """Generating simulated light curves with input.

    Args:
        params: A pandas series containing tau, sigma and noise level
        save_dir(str): Where to store the simulated LCs

    """
    Task = kali.carma.CARMATask(2, 1)
    noise = float(params['noise'])
    Theta = np.array([params['a1'], params['a2'], params['b0'], params['b1']])
    # print(Theta)
    # check whether the coefficients are valid before simulation
    if (Task.check(Theta) is False):
        return 0

    dt = 30.0/86400
    Task.set(dt, Theta)
    lc = Task.simulate(duration=3653)
    lc.fracNoiseToSignal = noise
    Task.observe(lc)

    # now save to file
    fname = lc2file(save_dir, lc, full=True, timescales=[params['a1'], params['b1']/params['b0']])
    return fname

if __name__ == '__main__':

    idf = pd.read_csv(sys.argv[1])
    lc_dir = sys.argv[2]

    result = Parallel(n_jobs=-1)(delayed(genLC)(idf.loc[i], lc_dir) for i in idf.index)
    idf['fname'] = result

    # save lc fname back to input csv
    idf.to_csv(sys.argv[1], index=False)
    # np.save('lc_log', result)
