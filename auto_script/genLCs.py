## Code to generate full mock LCs
import numpy as np
import kali.carma
import kali
import pandas as pd

idf = pd.read_csv('drw_input.csv')

if __name__ == '__main__':
    
    for i in idf.index:
        # initial carma task object
        task = kali.carma.CARMATask(1,0)
        r_1 = (-1.0)/float(idf.loc[i,'tau1']) + 0j
        amp = float(idf.loc[i,'sigma'])
        Rho = np.array([r_1, amp])    
        Theta = kali.carma.coeffs(1,0,Rho)
        
        # set basic parameters for task
        dt = 30.0/86400 # float number is required
        task.set(dt, Theta)
        
        #simulate LCs
        lc = task.simulate(duration=3650)
        
        #simulate noise, VNR = (1+fracintrisicVar)*fracNoiseToSignal
        lc.fracNoiseToSignal = float(idf.loc[i, 'noise'])
        lc.fracIntrinsicVar = 0.15
        task.observe(lc)
        
        lc2file('/home/mount/Full LCs/', lc, full=True, timescales = list(idf.loc[i]))
        del task
        del lc
