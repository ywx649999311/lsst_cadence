import numpy as np
import kali

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
        kali.lc.lc.__init__(self, name=name, pSim = self.mockLC.pSim, qSim = self.mockLC.qSim)
        
    def read(self, name=None, band=None, path=None, **kwargs):
        """Method will be called first during object initiation
        
        Args:
            name(str, optional): Name of the light curve
            band(str, optional): Observing band
            path(str, optional): Not used in the context, just to match the orgianl function format

        """
        self.name = name
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
      
            
          
            
            
            
            
            
            