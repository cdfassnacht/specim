"""

echelle1d.py

A generic set of code to process echelle 1d spectra that are stored as a
collection of 1d spectra.  This code is used by some of the ESI and NIRES
code that are in the keckcode repository

"""

from numpy import arange
from matplotlib import pyplot as plt
from astropy.io import fits as pf
from astropy.table import Table
from specim import specfuncs as ss


# ---------------------------------------------------------------------------

class Ech1d(list):
    """
    A generic set of code to process echelle 1d spectra that are stored as a
    collection of 1d spectra.  This code is used by some of the ESI and NIRES
    code that are in the keckcode repository
    """

    def __init__(self, inspec, informat='text', ordinfo=None, summary=True,
                 verbose=True):
        """

        Load the data.  This can be accomplished in one of three ways:
          1. by providing a list of Spec1d instances
          2. by providing a single filename, where the file contains the
             full set of single 1d spectra that are associated with a
             single echelle exposure for which the extraction has been done
          3. by providing a filelist, where each file in the list contains
             a single 1d spectrum in one of the approved formats
             (see Spec1d for possible informat values)

        """

        """ Initialize some variables """
        self.ordinfo = ordinfo

        """
        Load the spectra in a manner reflecting how the input spectra
         are provided
        """
        if isinstance(inspec, list):
            """
            Input spectra are a list of either Spec1d objects or filenames
            """
            for i, spec in enumerate(inspec):
                if isinstance(spec, ss.Spec1d):
                    self.append(spec)
                elif isinstance(spec, str):
                    if self.ordinfo is not None:
                        info = self.ordinfo[i]
                        tsec = [info['pixmin'], info['pixmax']]
                    else:
                        tsec = None
                    spec1d = ss.Spec1d(spec, informat=informat, trimsec=tsec,
                                       verbose=False)
                    self.append(spec1d)
                else:
                    errstr = '\n\nERROR: Input list must contain either '
                    errstr+ 'filenames or Spec1d instances\n\n'
                    raise TypeError(errstr)

        elif isinstance(inspec, str):
            """ Input spectra are contained in a single input file """
            if verbose:
                print('Opening echelle spectrum file: %s' % inspec)
                print('')
            try:
                hdu = pf.open(inspec)
            except IOError:
                raise IOError
            norder = len(hdu) - 1
            for i in range(1, len(hdu)):
                spec = ss.Spec1d(hdu[i], verbose=False)
                self.append(spec)

        else:
            print('')
            print('ERROR: First input must be either a filename or a list')
            print('  of either filenames or Spec1d instances')
            print('')
            raise TypeError

        """ Summarize the input if requested """
        if summary:
            self.input_summary()

    # -----------------------------------------------------------------------

    def input_summary(self):
        """

        Prints a summary of the data in the input file

        """

        print('Ord N_pix lam_start lam_end  dispave')
        print('--- ----- --------- -------- -------')


        """ Get information on the order labels, if it exists """
        if self.ordinfo is not None:
            orders = self.ordinfo['order']
        else:
            orders = arange(len(self))

        """ Print out the information for each order """
        for spec, order in zip(self, orders):
            print(' %2d %5d  %8.2f %8.2f  %5.2f' %
                  (order, len(spec), spec['wav'][0], spec['wav'][-1],
                   spec.dispave))

    # -----------------------------------------------------------------------

    def plot_all(self, plotmode='single', mode='input', title=None, z=None,
                 showz=True, linetype='strongem', smo=None, pltylim=None,
                 verbose=False, **kwargs):
        """

        Plots all the spectra in the echelle data set in a single plot.

        """

        """ Set starting values """
        wmin = 1.e12
        wmax = 0.

        print(len(self))
        """ Plot the spectra """
        for count, spec in enumerate(self):

            """ Set plot labeling """
            if count == (len(self) - 1):
                if title is None:
                    title = 'Extracted Spectrum'
                showz = True
                print('Bang')
            else:
                tmptitle = title
                title = None
                showz = False

            if title is not None:
                print(count, title)
            else:
                print(count, 'None')

            """ Plot the spectrum """
            if smo is not None:
                spec.smooth(smo, mode=mode, title=title, verbose=verbose,
                            **kwargs)
            else:
                spec.plot(mode=mode, title=title, **kwargs)
            title = tmptitle

            """ Adjust the plot limits """
            if spec['wav'].min() < wmin:
                wmin = spec['wav'].min()
            if spec['wav'].max() > wmax:
                wmax = spec['wav'].max()

        """ Set final plot limits """
        plt.xlim(wmin, wmax)
        if pltylim is not None:
            plt.ylim(pltylim)

        """
        Mark spectral lines if desired
        Do this in a separate loop since the tickmark length depends
         on the y-axis range of the display
        """
        for spec in self:
            if z is not None:
                if smo is not None:
                    usesmooth = True
                else:
                    usesmooth = False
                try:
                    spec.mark_lines(linetype, z, showz=showz, 
                                    usesmooth=usesmooth)
                except NameError:
                    spec.mark_lines('strongem', z, showz=showz, 
                                    usesmooth=usesmooth)

    # ------------------------------------------------------------------------

    def save_multi(self, outroot, mode='input', outformat='multitab',
                   verbose=True):
        """

        Saves the echelle spectra, contained in separate Spec1d instances
         within this class, to one or more output files. 
        The output options are:
         1. 'multitab' - Writes a single output file, which is a 
                         multiextention fits file.  Each extension contains
                         a fits table ('fitstab' format in the Spec1d lingo)
         2. 'text'     - Produces several output files, one for each order.
                         The files are ascii text files, with columns for,
                         at minimum, wavelength and flux, and possibly also
                         variance and sky

        """

        if outformat == 'multitab':

            """
            Set up the required, essentially empty, primary HDU and the
            HDUList
            """
            phdu = pf.PrimaryHDU()
            hdulist = pf.HDUList(phdu)

            """
            Make a separate HDU for each echelle order and then create a
            HDUList
            """
            # lll = [pf.table_to_hdu(Table(spec)) for spec in self]
            # hdulist = pf.HDUList([phdu] + lll)
            for spec in self:
                savespec = spec.select_mode(mode)
                hdulist.append(pf.table_to_hdu(Table(savespec)))

            """ Save the output """
            outfile = '%s.fits' % outroot
            hdulist.writeto(outfile, overwrite=True)

        else:
            print('')
            print('Not yet implemented yet')
            print('')
            return

        if verbose:
            print('Spectrum saved in mode %s to file %s' % 
                  (outformat, outfile))
