import numpy as np
import pandas as pd
import kali.carma
import kali
import os, sys, glob
sys.path.insert(0,'/home/mount/lsst_cadence')
from lsstlc import * # derived LSST lightcurve sub-class

# dir for earlier input files
lc_dir = '/home/mount/Full LCs/'
maf_dir = '/home/mount/lsst_cadence/auto_script/'
chain_dir = '/home/mount/Chain/'
maf_file = 'minion_1016_cadence.csv'
lc_param_file ='drw_input.csv' 
pos_file = 'loc.csv'
chain_id = 'c10_{}_{}_{}'

# read csv files
param_df = pd.read_csv('./drw_input.csv')
loc_df = pd.read_csv('./loc.csv')
cad_df = pd.read_csv('./minion_1016_cadence.csv')


# create a master dataframe to hold all info
para_col = list(param_df.columns)
loc_col = list(loc_df.columns)
all_col = para_col + loc_col + ['b_tau_1', 'b_sigma', 'chain_path', ]
df_all = pd.DataFrame(columns=all_col)

# glob lc file list
lc_ls = glob.glob('/home/mount/Full LCs/c10*.npz')

if __name__ == '__main__':

	for i in range(len(lc_ls)):
		# import lc based on order in folder (must match order in input.csv)
		lc = extLC(lc_ls[i])
		para_ls = lc.name.split('_')

		# loop over all observing locations
		for j in loc_df.index:
			row = loc_df.loc[j].to_dict()
			row['tau1'] = para_ls[1]
			row['sigma'] = para_ls[2]
			row['noise'] = para_ls[3]

			# now down sample lc
			obs_df = cad_df[cad_df['loc'] == j]
			lc_d = lsstlc(row['ra'], row['dec'], np.array(obs_df['expDate']), lc, fix_dt=True)
			task = kali.carma.CARMATask(1,0, nsteps=100, maxEvals=1)
			task.fit(lc_d)
			row['b_tau_1'] = task.bestTau[0]
			row['b_sigma'] = task.bestTau[1]
			row['chain_path'] = os.path.join(chain_dir, chain_id.format(row['ra'], row['dec'], row['noise']))
			df_all = df_all.append(row, ignore_index=True)

			del row, task
		del lc
	# save master df to csv
	df_all.to_csv('./final.csv', index=False)


















