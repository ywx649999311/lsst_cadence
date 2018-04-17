import numpy as np
import kali
import math
from astropy import stats

class lsstlc(kali.lc.lc):
    """A subclass of Kali's lc class. 
    This class down sample the mock lc with given dates. More flexible plotting is also available.
    """

    def __init__(self, ra, dec, cadence, mockLC, min_gap, **kwargs):
        """Initiation method

        Args:
            ra(float): Right ascension
            dec(float): Declination
            cadence(ndarray): A numpy array of the observing dates in seconds
            mockLC: Mock lightcuve simulated using Kali
            min_gap(float): Min intra-night gap for LSST observations of particular ra and dec
        """
        self._ra, self._dec = ra, dec
        self.min_gap = np.around(min_gap*3600)
        self.cadence = cadence
        self.mockLC = mockLC
        name = 'lsst_{}_{}_CARMA_{}_{}'.format(ra, dec, mockLC.pSim, mockLC.qSim)
        band = ''
        kali.lc.lc.__init__(self, name=name, band=band, pSim = self.mockLC.pSim, qSim = self.mockLC.qSim)
        
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
        mock_s = np.around(self.mockLC.t*86400).astype('int')
        opSim_index = np.around(self.cadence/self.min_gap)
        mark = [round(x/self.min_gap) in opSim_index for x in mock_s]
        
        self.t = self.mockLC.t[np.where(mark)]
        self.x = self.mockLC.x[np.where(mark)]
        self.y = self.mockLC.y[np.where(mark)]
        self.yerr = self.mockLC.yerr[np.where(mark)]
        self.mask = self.mockLC.mask[np.where(mark)]
        self._numCadences = self.t.shape[0]
        self.startT = self.t[0]
    
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
        x = self.t
        y = self.x - np.mean(self.x) + np.mean(self.y[np.where(self.mask == 1.0)[0]])
        err_x = self.t[np.where(self.mask == 1.0)[0]]
        err_y = self.y[np.where(self.mask == 1.0)[0]]
        err_yerr = self.yerr[np.where(self.mask == 1.0)[0]]
        
        return x, y, err_x, err_y, err_yerr
      

    def regularize(self, newdt=None):
        """!
        \brief Re-sample the light curve on a grid of spacing newdt
        Creates a new LC on gridding newdt and copies in the required points.
        """
        if not self.isRegular:
            if not newdt:
                if hasattr(self, 'terr'):
                    newdt = np.mean(self.terr)
                else:
                    newdt = self.mindt/10.0
            if newdt > self.mindt:
                raise ValueError('newdt cannot be greater than mindt')
            newLC = self.copy()
            newLC.dt = newdt
            newLC.meandt = float(np.nanmean(self.t[1:] - self.t[:-1]))
            newLC.mindt = float(np.nanmin(self.t[1:] - self.t[:-1]))
            newLC.maxdt = float(np.nanmax(self.t[1:] - self.t[:-1]))
            newLC.meandt = float(self.t[-1] - self.t[0])
            newLC.numCadences = int(math.floor(self.T/newLC.dt))+1 # avoid index out bounds error
            del newLC.t
            del newLC.x
            del newLC.y
            del newLC.yerr
            del newLC.mask
            newLC.t = np.require(np.zeros(newLC.numCadences), requirements=[
                                 'F', 'A', 'W', 'O', 'E'])  # Numpy array of timestamps.
            newLC.x = np.require(np.zeros(newLC.numCadences), requirements=[
                                 'F', 'A', 'W', 'O', 'E'])  # Numpy array of intrinsic fluxes.
            newLC.y = np.require(np.zeros(newLC.numCadences), requirements=[
                                 'F', 'A', 'W', 'O', 'E'])  # Numpy array of observed fluxes.
            newLC.yerr = np.require(np.zeros(newLC.numCadences), requirements=[
                                    'F', 'A', 'W', 'O', 'E'])  # Numpy array of observed flux errors.
            newLC.mask = np.require(np.zeros(newLC.numCadences), requirements=[
                                    'F', 'A', 'W', 'O', 'E'])  # Numpy array of mask values.
            for i in xrange(newLC.numCadences):
                newLC.t[i] = i*newLC.dt + self.t[0]
            for i in xrange(self.numCadences):
                tOff = (self.t[i] - self.t[0])
                index = int(math.floor(tOff/newLC.dt))
                newLC.x[index] = self.x[i]
                newLC.y[index] = self.y[i]
                newLC.yerr[index] = self.yerr[i]
                newLC.mask[index] = 1.0
            newLC._statistics()
            return newLC
        else:
            return self
    
    def periodogram_sb(self, nterms=1):
        
        # if (hasattr(self, '_periodogramfreqs') and
        #         hasattr(self, '_periodogram') and
        #         hasattr(self, '_periodogramerr')):
        #     return self._periodogramfreqs, self._periodogram, self._periodogramerr
        # else:
        #     if self.numCadences > 50:
        #         model = gatspy.periodic.LombScargleFast()
        #     else:
        #         model = gatspy.periodic.LombScargle()

        ls = stats.LombScargle(self.t, self.y, self.yerr, nterms=nterms)
        f, psd = ls.autopower(method='fast', normalization='psd', maximum_frequency=1/self.mindt, minimum_frequency=1/self.T)
        self._periodogramfreqs_sb = np.require(np.array(f), requirements=['F', 'A', 'W', 'O', 'E'])
        self._periodogram_sb = np.require(np.array(psd), requirements=['F', 'A', 'W', 'O', 'E'])
        self._periodogramerr_sb = np.require(np.array(self._periodogram_sb.shape[0]*[0.0]),
                                          requirements=['F', 'A', 'W', 'O', 'E'])
        for i in xrange(self._periodogram_sb.shape[0]):
            if self._periodogram_sb[i] <= 0.0:
                self._periodogram_sb[i] = np.nan
        return self._periodogramfreqs_sb, self._periodogram_sb, self._periodogramerr_sb    
        
            
            
            