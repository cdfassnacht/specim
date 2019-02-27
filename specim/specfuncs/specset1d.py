"""

specset1d - Defines a SpecSet1d class that contains code to handle operations
            on several 1d spectra, e.g., coaddition

"""

import numpy as np
from matplotlib import pyplot as plt
from .spec1d import Spec1d


class SpecSet1d(list):
    """

    The SpecSet1d class is effectively just a list of Spec1d instances that
    includes operations that act on the whole list (e.g., coadds)

    """

    # ------------------------------------------------------------------------

    def __init__(self, inlist=None, informat=None, spec1dlist=None,
                 wavlist=None, fluxlist=None, varlist=None, skylist=None,
                 verbose=True):
        """

        There are three ways to create a SpecSet1d instance, all of which
        use lists to create the instance.

        Option 1: List of filenames
         For this option, two of the parameters must be set:
           inlist   - a list of filenames
           informat - format of the files (see help file for Spec1d class
                      for available formats)

        Option 2: List of Spec1d instances
         For this option, one of the parameters must be set:
           spec1dlist - list of Spec1d instances

        Option 3: Lists of (at minimum) wavelength and flux arrays
         For this option, a minimum of two of the parameters must be set:
           wavlist  - a list of wavelength arrays (either astropy.Table
                       or numpy.ndarray instances) containing the wavelength
                       information
           fluxlist - a list of arrays containing the flux information
          * Optional parameters for option 2:
           varlist  - a list of arrays containing the variance on the fluxes
           skylist  - a list of arrays containing the sky-level information
                      (used if the flux arrays have been sky-subtracted)

        """

        if inlist is not None:
            if informat is not None:
                for i in inlist:
                    try:
                        spec = Spec1d(infile=i, informat=informat,
                                      verbose=verbose)
                    except IOError:
                        raise IOError
                    self.append(spec)
            else:
                msg = 'ERROR: inlist given but no informat has been given'
                raise IOError(msg)

        elif spec1dlist is not None:
            for spec in spec1dlist:
                if isinstance(spec, Spec1d):
                    self.append(spec)
                else:
                    msg = 'ERROR: spec1dlist is set but does not contain '
                    msg += 'only Spec1d instances'
                    raise IOError(msg)

        elif (wavlist is not None) and (fluxlist is not None):
            nspec = wavlist.size
            if varlist is None:
                varlist = []
                for i in range(nspec):
                    varlist.append(None)
            if skylist is None:
                skylist = []
                for i in range(nspec):
                    skylist.append(None)

            for w, f, v, s in zip(wavlist, fluxlist, varlist, skylist):
                spec = Spec1d(wav=w, flux=f, var=v, sky=s)
                self.append(spec)

    # ------------------------------------------------------------------------

    def coadd(self, doplot=True, outfile=None, verbose=True, **kwargs):
        """

        Do a variance-weighted sum of the spectra

        """

        """ Initialize """
        nx = self[0]['wav'].size
        wtflux = np.zeros(nx)
        skysum = np.zeros(nx)
        wtsum = np.zeros(nx)

        """ Create the weighted sum """
        for spec in self:
            wt = np.zeros(nx)
            mask = spec['var'] != 0
            wt[mask] = 1.0 / spec['var'][mask]
            wtflux += wt * spec['flux']
            if spec.sky:
                skysum += spec['sky']
            wtsum += wt
            del wt

        """
        Normalize the flux, and calculate the variance of the coadded
         spectrum.
        Note that the equation below for the variance only works for the case
         of inverse variance weighting.
        """
        wtflux[wtsum == 0] = 0
        wtsum[wtsum == 0] = 1
        outflux = wtflux / wtsum
        outvar = 1.0 / wtsum
        if self[0].sky is None:
            outsky = None
        else:
            outsky = skysum / len(self)

        """ Create a Spec1d structure for the output spectrum """
        outspec = Spec1d(wav=self[0]['wav'], flux=outflux, var=outvar,
                         sky=outsky, verbose=verbose)

        """ Plot the combined spectrum """
        if doplot:
            outspec.plot(title='Combined spectrum', **kwargs)

        """ Save the combined spectrum """
        if outfile is not None:
            outspec.save(outfile, outformat=informat)

        """ Return the coadded spectrum """
        return outspec
