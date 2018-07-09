import numpy as np
import kali
import math
from astropy import stats
import os

def lc2file(file_dir, lc, full = False, timescales = None):
    '''Save light curve to a .npz file
    Args:
        file_path(str): path + file name
        lc: Kali light curve object
        full(bool): Full mock LC or not
        timescales(list): CARMA coefficients in timescale format
    '''
    
    # Check out if full LC is saved, if so require more input!
    if full:
        if timescales is None:
            raise Exception('Full LC need CARMA timescales!')
        elif not isinstance(timescales, list):
            raise Exception('Timescales must contained in a list!')

        lc_id = 'c{}{}'.format(lc.pSim, lc.qSim)
        for i in range(len(timescales)):
            lc_id += '_{}'.format(timescales[i])

    meta = [lc.pSim, lc.qSim, lc.fracNoiseToSignal, lc.fracIntrinsicVar]
    if 'mock_t' in lc.__dict__:
        np.savez(os.path.join(file_dir, lc.name), t=lc.t, x=lc.x, y=lc.y, yerr=lc.yerr, mask=lc.mask, meta=meta, 
            mock_t= lc.mock_t)
    else: 
        np.savez(os.path.join(file_dir, lc_id), t=lc.t, x=lc.x, y=lc.y, yerr=lc.yerr, mask=lc.mask, meta=meta)

class lsstlc(kali.lc.lc):
    """A subclass of Kali's lc class. 
    This class down sample the mock lc with given dates. More flexible plotting is also available.
    """

    def __init__(self, ra, dec, obsTimes, mockLC, min_sep, band = 'a', fix_dt = False, **kwargs):
        """Initiation method

        Args:
            ra(float): Right ascension
            dec(float): Declination
            obsTimes(ndarray): A numpy array of the observing dates in seconds
            mockLC: Mock lightcuve simulated using Kali
            min_sep(float): Min intra-night seperation (in hours) for LSST observations of the particular
                point on the sky
            band(str): Observing band, defaults to 'a' stands for all bands.
            fixed_dt:(bool): Whether the full LC is sampled every 30 sec. 
        """
       	self._ra, self._dec = ra, dec
        
        if fix_dt:
            self.min_sep = 30 
        else:
            self.min_sep = np.floor(min_sep*3600-1) # hours to seconds (floor to avoid dim error)
        
        self.obsTimes = obsTimes
        self.mockLC = mockLC
        name = '{}_{}_{}_c{}{}'.format(ra, dec, band, mockLC.pSim, mockLC.qSim)
        kali.lc.lc.__init__(self, name=name, band=band, pSim = self.mockLC.pSim, 
            qSim = self.mockLC.qSim)
        
    def read(self, name=None, band=None, path=None, **kwargs):
        """Method will be called first during object initiation
        
        Args:
            name(str, optional): Name of the light curve
            band(str, optional): Observing band
            path(str, optional): Not used in the context, just to match the orgianl function format

        """
        self.name = name
        self.band = band
        self.xunit = kwargs.get('xunit', r'$t$')
        self.yunit = kwargs.get('yunit', r'$F$')

        opSim_index = np.around(self.obsTimes/self.min_sep).astype('int')
        mark = np.full((self.mockLC.t.shape[0],), False, dtype='bool')
        mark[opSim_index] = True
        
        self.mock_t = self.mockLC.t[np.where(mark)] # mock_t is time from mock LC
        self.x = self.mockLC.x[np.where(mark)]
        self.y = self.mockLC.y[np.where(mark)]
        self.yerr = self.mockLC.yerr[np.where(mark)]
        self.mask = self.mockLC.mask[np.where(mark)]
        self._numCadences = self.mock_t.shape[0]
        self.startT = self.mock_t[0]
        self.t = self.mock_t - self.startT # get t for new LC
    
    def write(self, name=None, band=None, pwd=None, **kwargs):
        """Not implemented, but required to complet the class"""
        pass
    
    
    def plot_x_y_err(self):
        """"Return the entries for plotting

        Returns:
            x(ndarray): An array storing the observation timestamps in days
            y(ndarray): An array storing the intrinsic flux of the AGN
            err_x(ndarray): same as x
            err_y(ndarray): An array storing the observed flux (with noise) of the AGN
            err_yerr(ndarray): An array storing the error bar magnitude 
        """
        x = self.mock_t
        y = self.x - np.mean(self.x) + np.mean(self.y[np.where(self.mask == 1.0)[0]])
        err_x = self.mock_t[np.where(self.mask == 1.0)[0]]
        err_y = self.y[np.where(self.mask == 1.0)[0]]
        err_yerr = self.yerr[np.where(self.mask == 1.0)[0]]
        
        return x, y, err_x, err_y, err_yerr
      
    def periodogram_sb(self, nterms=1):
        
        if (hasattr(self, '_periodogramfreqs_sb') and
                hasattr(self, '_periodogram_sb') and
                hasattr(self, '_periodogramerr_sb')):
            return self._periodogramfreqs_sb, self._periodogram_sb, self._periodogramerr_sb
        else:
            # if self.numCadences > 50:
            #     model = gatspy.periodic.LombScargleFast()
            # else:
            #     model = gatspy.periodic.LombScargle()

	        ls = stats.LombScargle(self.t, self.y, self.yerr, nterms=nterms)
	        f, psd = ls.autopower(method='fast', normalization='psd', maximum_frequency=1/self.mindt)
	        self._periodogramfreqs_sb = np.require(np.array(f), requirements=['F', 'A', 'W', 'O', 'E'])
	        self._periodogram_sb = np.require(np.array(psd), requirements=['F', 'A', 'W', 'O', 'E'])
	        self._periodogramerr_sb = np.require(np.array(self._periodogram_sb.shape[0]*[0.0]),
                                          requirements=['F', 'A', 'W', 'O', 'E'])
        for i in xrange(self._periodogram_sb.shape[0]):
            if self._periodogram_sb[i] <= 0.0:
                self._periodogram_sb[i] = np.nan
        return self._periodogramfreqs_sb, self._periodogram_sb, self._periodogramerr_sb    

class extLC(kali.lc.lc):

    # Class variable are used to update LC model parameters
    pSim = 0
    qSim = 0
    fracIntrinsicVar = 0.15
    fracNoiseToSignal = 0.001

    def __init__(self, file_path):

        path = file_path
        base = os.path.basename(file_path)
        name = os.path.splitext(base)[0]
        band = ''

        kali.lc.lc.__init__(self, path=path, name=name, band=band)
        self._pSim = extLC.pSim
        self._qSim = extLC.qSim
        self._fracIntrinsicVar = extLC.fracIntrinsicVar
        self._fracNoiseToSignal = extLC.fracNoiseToSignal

    def read(self, path, name=None, band=None, **kwargs):

        lc_data = np.load(path)
        if 'mock_t' in lc_data.files:
            self.mock_t = lc_data['mock_t']

        self.t = lc_data['t']
        self.x = lc_data['x']
        self.y = lc_data['y']
        self.yerr = lc_data['yerr']
        self.mask = lc_data['mask']
        
        meta = lc_data['meta']
        self.startT = self.t[0]
        self._numCadences = self.t.shape[0]
        self.name = name
        self.band = band
        self.xunit = kwargs.get('xunit', r'$t$')
        self.yunit = kwargs.get('yunit', r'$F$')
        
        extLC.pSim = int(meta[0])
        extLC.qSim = int(meta[1])
        extLC.fracNoiseToSignal = float(meta[2])
        extLC.fracIntrinsicVar = float(meta[3])

    def write(self, name=None, band=None, pwd=None, **kwargs):
        """Not implemented, but required to complet the class"""
        pass



    
        
            
            
            