"""

echelle1d.py

A generic set of code to process echelle 1d spectra that are stored as a
collection of 1d spectra.  This code is used by some of the ESI and NIRES
code that are in the keckcode repository

"""

from matplotlib import pyplot as plt

class Ech1d(list):
    """
    A generic set of code to process echelle 1d spectra that are stored as a
    collection of 1d spectra.  This code is used by some of the ESI and NIRES
    code that are in the keckcode repository
    """

    def __init__(self, speclist=None, echfile=None, filelist=None,
                 informat=None):
        """

        Load the data.  This can be accomplished in one of three ways:
          1. by providing a list of Spec1d instances
          2. by providing a single filename, where the file contains the
             full set of single 1d spectra that are associated with a
             single echelle exposure for which the extraction has been done
             [NOT YET IMPLEMENTED]
          3. by providing a filelist, where each file in the list contains
             a single 1d spectrum in one of the approved formats
             (see Spec1d for possible informat values)
             [NOT YET IMPLEMENTED]

        """

        if speclist is not None:
            for spec in speclist:
                self.append(spec)

        else:
            print('')
            print('NOTE: Right now only the speclist input option is'
                  'implemented')
            print('')

    # ------------------------------------------------------------------------

    def plot_all(self, plotmode='single', mode='input', title=None, z=None,
                 smo=None, linetype='strongem', **kwargs):
        """

        Plots all the spectra in the echelle data set in a single plot.

        """

        """ Set starting values """
        i = 0
        wmin = 1.e12
        wmax = 0.

        """ Plot the spectra """
        for count, spec in enumerate(self):

            """ Set plot labeling """
            if count == len(spec) - 1:
                if title is None:
                    title = 'Extracted Spectrum'
                showz = True
            else:
                title = False
                showz = False

            """ Plot the spectrum """
            if smo is not None:
                spec.smooth(smo, mode=mode, title=title, **kwargs)
            else:
                spec.plot(mode=mode, title=title, **kwargs)

            """ Mark spectral lines if desired """
            if z is not None:
                if smo is not None:
                    spec.mark_lines(linetype, z, showz=showz, usesmooth=True)
                else:
                    spec.mark_lines(linetype, z, showz=showz)

            """ Adjust the plot limits """
            if spec['wav'].min() < wmin:
                wmin = spec['wav'].min()
            if spec['wav'].max() > wmax:
                wmax = spec['wav'].max()
            i += 1

        """ Set final plot limits """
        plt.xlim(wmin, wmax)

    # ------------------------------------------------------------------------

    def save_multi(self, outformat='not_used_yet'):
        """

        Saves the echelle spectra in a single file but with multiple
         extensions, one for each order.  The indiv
        To save a single conjoined output spectrum use the (NOT YET WRITTEN)
         save_single method

        """
