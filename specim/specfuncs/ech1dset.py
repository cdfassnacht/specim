"""

ech2dset.py - Code to perform actions on multiple 1d echelle data sets

"""

from os import path
from matplotlib import pyplot as plt
from .specset1d import SpecSet1d
from .echelle1d import Ech1d

class Ech1dSet(list):
    """

    The Ech1dSet class is effectively just a list of Ech1d objects that
    includes operations that act on the whole list (e.g., coadds)

    """

    # ------------------------------------------------------------------------

    def __init__(self, inlist, ordinfo=None, indir='.', informat='text',
                 summary=True, verbose=True):

        """

        Reads the input data and possibly its associated variance.
        The input files can be designated in one of two ways:

          1. A list of filenames

          2. A list of Esi1d objects

        """

        """ Read in the data """
        for i in inlist:
            if isinstance(i, str):
                espec = Esi1d(i, ordinfo=ordinfo, informat=informat,
                              summary=summary, verbose=verbose)
            else:
                espec = i
            self.append(espec)

    # ------------------------------------------------------------------------

    def coadd(self, doplot=True, outroot=None, debug=False, **kwargs):
        """

        For the set of Ech1d objects in this Ech1dSet class, creates an
         output Ech1d object, where each spectral order in the output
         echelle container is the coadd of the corresponding spectral
         orders in the input Ech1d objects

        """

        """ Set up a list container for the coadded spectral orders"""
        coaddlist = []

        """ Put in some debugging info """
        if debug:
            print(type(self))
            print(len(self))
            print(type(self[0]))
            print(len(self[0]))

        """ Loop through the each of spectral orders in this class """
        for i in range(len(self[0])):

            """ Create a list of Spec1d objects """
            speclist = []
            for espec in self:
                speclist.append(espec[i])

            """ Coadd the spectra in the list """
            specall = SpecSet1d(spec1dlist=speclist)
            coaddlist.append(specall.coadd(doplot=False))
            plt.show()
            del(speclist)

        """
        Convert the list of coadded spectra to an output Ech1d object, and
        plot it if requested
        """
        outspec = Ech1d(coaddlist, ordinfo=self[0].ordinfo)
        if doplot:
            outspec.plot_all(**kwargs)
            plt.show()

        """
        Return the coadded Ech1d object, possibly saving it to a file as well
        """
        if outroot is not None:
            outspec.save_multi(outroot, outformat='multitab')
        return outspec
