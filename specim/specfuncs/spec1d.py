"""
spec1d.py

This file contains the Spec1d class, which is used to process and plot
1d spectroscopic data.
"""

import os
import numpy as np
from scipy import interpolate, ndimage
import matplotlib.pyplot as plt
try:
    from astropy.io import fits as pf
except ImportError:
    import pyfits as pf
from cdfutils import datafuncs as df

# ===========================================================================
#
# Start of Spec1d class
#
# NOTE: There is one stand-alone function defined at the end of this file.
#       It will eventually be incorporated into the Spec1d class.
#
# ===========================================================================


class Spec1d(df.Data1d):
    """
    A class to process and analyze 1-dimensional spectra.
    """

    def __init__(self, infile=None, informat='text',
                 wav=None, flux=None, var=None, sky=None, logwav=False,
                 debug=False):
        """

        Reads in the input 1-dimensional spectrum.
        This can be done in two mutually exclusive ways:

         1. By providing some 1-d arrays containing the wavelength array,
            the flux array, and, optionally, the variance array and the sky .

                      ---- or ----

         2. By providing the name of a file that contains the spectrum.
            There are several possible input file formats:
              fits: A multi-extension fits file
                 Extension 1 is the wavelength
                 Extension 2 is the extracted spectrum (flux)
                 Extension 3 is the variance spectrum
                 [OPTIONAL] Extension 4 is the sky spectrum
              fitstab: A binary fits table, stored as a recarray, that has
                 columns (or "fields" as they are referred to in a recarray
                 structure) corresponding to, at a mininum, wavelength and
                 flux.
                 NOTE. The table is assumed to be in Extension 1 and the
                 table is assumed to have wavelength in field 0 and flux in
                 field 1
              fitsflux:  A file with a single 1-dimensional HDU that contains
                 the flux portion of the spectrum.  The associated
                 wavelength array is described by the CRPIX1, CRVAL1, and
                 CDELT1 (or CD1_1) keywords in the fits header.  This is
                 the format, for example, for the template stellar spectra
                 in the Indo-US set.
              deimos: A multiextension fits file with the following setup:
                 HDU1 - binary table with blue-side spectral info
                 HDU2 - binary table with red-side spectral info
                 In each table there are columns for wavelength, flux,
                  variance, and sky (among many others)
              mwa:  A multi-extension fits file with wavelength info in
                 the fits header
                 Extension 1 is the extracted spectrum (flux)
                 Extension 3 is the variance spectrum
              text: An ascii text file with information in columns:
                  Column 1 is the wavelength
                  Column 2 is the extracted spectrum
                  Column 3 (optional) is the variance spectrum
                  Column 4 (optional) is the sky spectrum [NOT YET IMPLEMENTED]

                  Thus, an input text file could have one of three formats:
                      A.  wavelength flux
                      B.  wavelength flux variance
                      C.  wavelength flux variance sky


        Inputs (all inputs are optional, but at least one way of specifiying
        the input spectrum must be used):
            infile   - Name of the input file.  If infile is None, then the
                        spectrum must be provided via the wavelength and flux
                        vectors.
            informat - format of input file ('fits', 'fitstab', 'fitsflux',
                        'mwa', or 'text').
                        Default = 'text'
            wav      - 1-dimensional array containing "wavelength" information,
                        either as actual wavelength or in pixels
            flux     - 1-dimensional array containing the flux information for
                        the extracted spectrum
            var      - 1-dimensional array containing the variance
                        spectrum.  Remember: rms = sqrt(variance)
            sky      - 1-dimensional array containing the sky spectrum
            logwav   - if True then input wavelength is logarithmic, i.e., the
                        numbers in the input wavelength vector are actually
                        log10(wavelength)
                       if False (the default), then the input wavelength vector
                        is linear.
        """

        """ Initialize some variables """
        self.hasvar = False
        self.sky = False
        self.atm_trans = None
        self.infile = None
        self.dispave = None
        names0 = ['wav', 'flux', 'var']
        spec0 = [None, None, None, None]

        self.logwav = logwav

        """ Read in the spectrum """
        if infile is not None:
            self.infile = infile
            try:
                spec0, self.dispave, self.hasvar = \
                    self.read_from_file(infile, informat, debug=debug)
            except IOError:
                print('')
                print('Could not read input file %s' % infile)
                print('')
                return None
        elif wav is not None and flux is not None:
            if self.logwav:
                spec0[0] = 10.**wav
            else:
                spec0[0] = wav.copy()
            spec0[1] = flux.copy()
            if var is not None:
                spec0[2] = var.copy()
                self.hasvar = True
            if sky is not None:
                spec0[3] = sky.copy()
                self.sky = True
        else:
            print('')
            print('ERROR: Must provide either:')
            print('  1. A name of an input file containing the spectrum')
            print('  2. At minimum, both of the following:')
            print('         A. a wavelength vector (wav)')
            print('         B. a flux vector (flux)')
            print('      and optionally one or both of the following')
            print('         C. a variance vector (var)')
            print('         D. a sky spectrum vector (sky)')
            print('')
            return

        """
        Call the superclass initialization for useful Data1d attributes
        """
        if debug:
            print(names0)
            print('Wavelength vector size: %d' % spec0[0].size)
            print('Flux vector size: %d' % spec0[1].size)
        if spec0[2] is not None:
            df.Data1d.__init__(self, spec0[0], spec0[1], spec0[2],
                               names=names0)
        else:
            names = names0[:-1]
            df.Data1d.__init__(self, spec0[0], spec0[1], names=names)

        """ Add the sky vector to the Table structure if it is not none """
        if spec0[3] is not None:
            self['sky'] = spec0[3]

        """ Read in the list that may be used for marking spectral lines """
        self.load_linelist()

    # -----------------------------------------------------------------------

    def read_from_file(self, infile, informat, verbose=True, debug=False):
        """

        Reads a 1-d spectrum from a file.  The file must have one of the
        following formats, which is indicated by the informat parameter:
          fits
          fitstab
          fitsflux
          deimos
          mwa
          text

        """

        if verbose:
            print('')
            print('Reading spectrum from %s' % infile)
            print('Expected file format: %s' % informat)

        """ Set default parameters """
        wav = None
        flux = None
        var = None
        sky = None
        hasvar = False

        """ Read in the input spectrum """
        if informat == 'fits':
            hdu = pf.open(infile)
            if self.logwav:
                wav = 10.**(hdu[1].data)
            else:
                wav = hdu[1].data.copy()
            flux = hdu[2].data.copy()
            if len(hdu) > 3:
                var = hdu[3].data.copy()
            hasvar = True
            if len(hdu) > 4:
                sky = hdu[4].data.copy()
            del hdu
        elif informat == 'fitstab':
            hdu = pf.open(infile)
            tdat = hdu[1].data
            wav = tdat.field(0)
            flux = tdat.field(1)
            if len(tdat[0]) > 2:
                var = tdat.field(3)
                hasvar = True
            del hdu
        elif informat == 'fitsflux':
            hdu = pf.open(infile)
            if len(hdu[0].data.shape) == 2:
                flux = hdu[0].data[0, :]
            else:
                flux = hdu[0].data.copy()
            hasvar = False
            wav = np.arange(flux.size)
            hdr1 = hdu[0].header
            if debug:
                print('Wavelength vector size: %d' % wav.size)
                print('Flux vector size: %d' % flux.size)
                print(flux.shape)
                testkeys = ['crval1', 'cd1_1']
                for k in testkeys:
                    if k.upper() in hdr1.keys():
                        print('%s: %f' % (k.upper(), hdr1[k]))
                    else:
                        print('ERROR: could not find %s in header' % k.upper())
            if self.logwav:
                wav = 10.**(hdr1['crval1'] + wav * hdr1['cd1_1'])
            else:
                wav = hdr1['crval1'] + wav*hdr1['cd1_1']
            del hdu
        elif informat.lower() == 'deimos':
            hdu = pf.open(infile)
            tab1 = hdu[1].data
            tab2 = hdu[2].data
            bwav = tab1['lambda'][0, :]
            bflux = tab1['spec'][0, :]
            bvar = 1. / tab1['ivar'][0, :]
            bsky = tab1['skyspec'][0, :]
            rwav = tab2['lambda'][0, :]
            rflux = tab2['spec'][0, :]
            rvar = 1. / tab2['ivar'][0, :]
            rsky = tab2['skyspec'][0, :]
            flux = np.concatenate((bflux, rflux))
            wav = np.concatenate((bwav, rwav))
            var = np.concatenate((bvar, rvar))
            sky = np.concatenate((bsky, rsky))
            hasvar = True
        elif informat.lower() == 'esi':
            hdu = pf.open(infile)
            wav = 10.**(hdu[1].data)
            flux = hdu[2].data.copy()
            var = hdu[3].data.copy()
            hasvar = True
            del hdu
        elif informat == 'mwa':
            hdu = pf.open(infile)
            flux = hdu[1].data.copy()
            var = hdu[3].data.copy()
            hasvar = True
            wav = np.arange(flux.size)
            hdr1 = hdu[1].header
            if self.logwav:
                wav = hdr1['crval1'] + 10.**(wav*hdr1['cd1_1'])
            else:
                wav = hdr1['crval1'] + wav*hdr1['cd1_1']
            del hdu
        else:
            spec = np.loadtxt(infile)
            if self.logwav:
                wav = 10.**(spec[:, 0])
            else:
                wav = spec[:, 0]
            flux = spec[:, 1]
            if spec.shape[1] > 2:
                var = spec[:, 2]
                hasvar = True
            if spec.shape[1] > 3:
                sky = spec[:, 3]
                self.sky = True
            del spec

        """ Check for NaN's, which this code can't handle """
        if hasvar:
            mask = (np.isnan(flux)) | (np.isnan(var))
            varmax = var[~mask].max()
            var[mask] = varmax * 5.
        else:
            mask = (np.isnan(flux))
        flux[mask] = 0
        disp0 = wav[1] - wav[0]
        dispave = (wav[-1] - wav[0]) / (wav.size - 1)
        if verbose:
            print(' Spectrum Start: %8.2f' % wav[0])
            print(' Spectrum End:    %8.2f' % wav[-1])
            if disp0 < 0.01:
                print(' Dispersion (1st pixel): %g' % disp0)
            else:
                print(' Dispersion (1st pixel): %6.2f' % disp0)
            if dispave < 0.01:
                print(' Dispersion (average):    %g' % dispave)
            else:
                print(' Dispersion (average):    %6.2f' % dispave)
            print('')

        """ Return the data """
        # self.dispave = dispave
        return [wav, flux, var, sky], dispave, hasvar

    # -----------------------------------------------------------------------

    def make_atm_trans(self, fwhm=15., modfile='default'):
        """
        Creates an extension to the class that contains the
        transmission of the Earth's atmosphere as a function of wavelength.
        For now this is just for the near-infrared (NIR) part of the spectrum,
        which is what the default gives, but there is some functionality for
        a different transmission spectrum to be provided.
        The transmission spectrum is stored as self.atm_trans

        Inputs:
          fwhm    - smoothing parameter for the output spectrum
          modfile - the full path+name of the file containing the atmospheric
                    transmission data.  The default location is in the Data
                    subdirectory contained within the directory in which this
                    code is found.
        """

        """ Read in the atmospheric transmission data"""
        if modfile != 'default':
            infile = modfile
        else:
            moddir = '%s' % (os.path.split(__file__)[0])
            infile = '%s/Data/atm_trans_maunakea.fits' % moddir
        print('Loading atmospheric data from %s' % infile)
        try:
            atm0, da, hv = self.read_from_file(infile, informat='fitstab')
        except IOError:
            print('ERROR: Cannot read atmospheric transmission data file')
            raise IOError
        atm0[0] *= 1.0e4

        """
        Only use the relevant part of the atmospheric transmission spectrum
        """
        w0 = atm0[0]
        w = self['wav']
        mask = (w0 >= w.min()) & (w0 <= w.max())
        if mask.sum() == 0:
            print('')
            print('Warning: %s only has data outside the requested wavelength'
                  'range' % infile)
            print('   %8.2f - %8.2f' % (w.min(), w.max()))
            self.atm_trans = None
            del atm0
            raise ValueError
        else:
            watm = atm0[0][mask]
            trans = atm0[1][mask]

        """ Smooth the spectrum """
        trans = ndimage.gaussian_filter(trans, fwhm)

        """ Resample the smoothed spectrum """
        tmpspec = Spec1d(wav=watm, flux=trans)
        tmpspec.resample(w)

        """ Store result in the atm_trans holder """
        self.atm_trans = tmpspec.rsflux

        """ Clean up """
        del atm0, watm, trans, tmpspec

    # -----------------------------------------------------------------------

    def plot_atm_trans(self, scale=1., offset=0., ls='-', color='g',
                       fwhm=15., modfile='default', label='default',
                       title=None, xlabel=None, ylabel=None):
        """

        Plots the atmospheric transmission for the wavelength range
        corresponding to the spectrum contained in this Spec1d instance.
        If the transmission spectrum does not yet exist, then the
        make_atm_trans method gets called first.

        """

        """
        Make the atmospheric transmission spectrum if it doesn't already
        exist
        """
        if self.atm_trans is None:
            try:
                self.make_atm_trans(fwhm=fwhm, modfile=modfile)
            except IOError:
                return
            except ValueError:
                return

        """ Set some plotting parameters """
        if label == 'default':
            plabel = 'atmTrans'
        elif label is None:
            plabel = None
        pltls = "steps%s" % ls

        """ Now do the plotting """
        tmp = self.atm_trans.copy()
        tmp *= self['flux'].max() * scale
        tmp += offset
        if plabel is not None:
            plt.plot(self['wav'], tmp, color, linestyle=pltls, label=plabel)
        else:
            plt.plot(self['wav'], tmp, color, linestyle=pltls)

        """ Label things if requested """
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if title:
            plt.title(title)

        """ Clean up """
        del tmp

    # -----------------------------------------------------------------------

    def plot(self, xlabel='Wavelength (Angstroms)', ylabel='Relative Flux',
             title='default', docolor=True, color='b', linestyle='',
             showzero=True, model=None, modcolor='g',
             label=None, fontsize=12, rmscolor='r', rmsoffset=0, rmsls=None,
             add_atm_trans=False, atmscale=1.05, atmfwhm=15., atmoffset=0.,
             atmls='-', atmmodfile='default', usesmooth=False, verbose=True):
        """
        Plots the spectrum

        Inputs:
          model     - If not None, then plot a model on top of the spectrum.
                      NOTE: this model must be in the form of an
                      astropy.modeling model
          modcolor  - Color to use for the model plot
          atmscale
          usesmooth
        """

        """ Set the title """
        if title == 'default':
            if self.infile is None:
                title = 'Extracted Spectrum'
            else:
                title = 'Spectrum for %s' % self.infile

        """ Override color assignments if docolor is False"""
        if not docolor:
            color = 'k'
            rmscolor = 'k'

        """ Draw the flux=0 line"""
        if showzero:
            plt.axhline(color='k')

        """ Plot the spectrum """
        if usesmooth and self.ysmooth is not None:
            flux = self.ysmooth
            if self.varsmooth is not None:
                var = self.varsmooth
        else:
            flux = self['flux']
            try:
                var = self['var']
            except KeyError:
                var = None
        ls = "steps%s" % linestyle
        if label is not None:
            plabel = label
        else:
            plabel = 'Flux'
        plt.plot(self['wav'], flux, color, linestyle=ls, label=plabel)
        plt.tick_params(labelsize=fontsize)
        plt.xlabel(xlabel, fontsize=fontsize)

        """
        Plot the model, given as an astropy.modeling model, if requested
        """
        if model is not None:
            fmod = model(self['wav'])
            plt.plot(self['wav'], fmod, color=modcolor)

        """ Plot the RMS spectrum if the variance spectrum exists """
        if var is not None:
            rms = np.sqrt(var) + rmsoffset
            if rmsls is None:
                if docolor:
                    rlinestyle = 'steps'
                else:
                    rlinestyle = 'steps:'
            else:
                rlinestyle = 'steps%s' % rmsls
            if docolor:
                plt.plot(self['wav'], rms, rmscolor, linestyle=rlinestyle,
                         label='RMS')
            else:
                plt.plot(self['wav'], rms, rmscolor, linestyle=rlinestyle,
                         label='RMS', lw=2)

        """ More plot labels """
        plt.ylabel(ylabel, fontsize=fontsize)
        if(title):
            plt.title(title)
        if(self['wav'][0] > self['wav'][-1]):
            plt.xlim([self['wav'][-1], self['wav'][0]])
        else:
            plt.xlim([self['wav'][0], self['wav'][-1]])
        # print self['wav'][0], self['wav'][-1]

        """ Plot the atmospheric transmission if requested """
        if add_atm_trans:
            self.plot_atm_trans(ls=atmls, scale=atmscale, offset=atmoffset,
                                fwhm=atmfwhm, modfile=atmmodfile)

    # -----------------------------------------------------------------------

    def plot_sky(self, color='g', linestyle='-', xlabel='default',
                 title='default', label='default', verbose=True):
        """

        Plots the sky spectrum, if the appropriate information is available.
        There are two ways in which the sky spectrum may be stored:
          1. In the sky variable
          2. In the var variable if the sky variable is not available.  In this
              case the rms spectrum (i.e., the square root of the variance
              spectrum) should be a decent representation of the sky if the
              object was not super bright.
        """

        """ Check to make sure that there is a spectrum to plot """
        if self.sky:
            skyflux = self['sky']
            skylab = 'Sky spectrum'
        elif self.hasvar:
            skyflux = np.sqrt(self['var'])
            print('Using RMS spectrum as a proxy for the sky spectrum')
            skylab = 'RMS spectrum'
        else:
            if verbose:
                print('')
                print('Cannot plot sky spectrum.')
                print('')
            raise KeyError('No sky or variance information in the spectrum')

        """ Set up for plotting """
        ls = 'steps%s' % linestyle
        if xlabel == 'default':
            xlab = 'Wavelength (Angstroms)'
        else:
            xlab = xlabel
        if label == 'default':
            lab = skylab
        else:
            lab = label

        """ Plot the spectrum """
        if lab is not None:
            plt.plot(self['wav'], skyflux, ls=ls, color=color, label=lab)
        else:
            plt.plot(self['wav'], skyflux, ls=ls, color=color)
        if title == 'default':
            plttitle = 'Sky Spectrum'
        else:
            plttitle = title
        if title is not None:
            plt.title(plttitle)
        plt.xlabel(xlab)
        plt.ylabel('Relative flux')
        if(self['wav'][0] > self['wav'][-1]):
            plt.xlim([self['wav'][-1], self['wav'][0]])
        else:
            plt.xlim([self['wav'][0], self['wav'][-1]])

    # -----------------------------------------------------------------------

    def smooth(self, filtwidth, smfunc='boxcar', doplot=True, outfile=None,
               color='b', title='default', xlabel='Wavelength (Angstroms)'):
        """

        Smooths the spectrum using the requested function.  The smoothing
        function is set by the smfunc parameter.  Available functions are:
          'boxcar' - the default value and only available value for now

        Simple usage example, for a spectrum called "myspec":

           myspec.smooth(7)

         This will do a variance weighted boxcar smoothing with a 7-pixel
          smoothing width, if the variance spectrum is available.  Otherwise
          it will do a uniformly-weighted boxcar smoothing
        """

        """
        Smooth the spectrum using the requested smoothing function
         [For now, only boxcar smoothing is allowed]
        The smoothing functions are inherited from the Data1d class
        """
        if smfunc == 'boxcar':
            self.ysmooth, self.varsmooth = self.smooth_boxcar(filtwidth)
        else:
            print('')
            print('For smoothing, smfunc can only be one of the following:')
            print("  'boxcar'")
            print('')
            raise ValueError

        """ Plot the smoothed spectrum if desired """
        if doplot:
            self.plot(usesmooth=True, title=title, xlabel=xlabel, color=color)

        # """ Save the output file if desired """
        # if(outfile):
        #     print "Saving smoothed spectrum to %s" % outfile
        #     if varwt:
        #         save_spectrum(outfile, wavelength, outflux, outvar)
        #     else:
        #         save_spectrum(outfile, wavelength, outflux)
        #     print('')

    # -----------------------------------------------------------------------

    def load_linelist(self, linefile='default'):

        linefmt = [('name', 'S10'), ('wavelength', float), ('label', 'S10'),
                   ('dxlab', float), ('type', int), ('plot', bool)]
        self.lineinfo = np.array([
                ('He II',         256.32, 'HeII',    0.0, 2, True),
                ('He II',         303.78, 'HeII',    0.0, 2, True),
                ('He I',          537.03, 'HeI',     0.0, 2, True),
                ('He I',          584.33, 'HeI',     0.0, 2, True),
                ('Ly-gamma',      972.54, 'Ly',      0.0, 3, True),
                ('Ly-beta',      1025.7,  'Ly',      0.0, 3, True),
                ('O VI',         1035.07, 'OVI',     0.0, 2, True),
                ("Ly-alpha",     1216.,   r"Ly$\alpha$", 0.0, 4, True),
                ('N V',          1240.1,  'NV',      0.0, 4, True),
                ('Si II',        1263.3,  'SiII',    0.0, 4, True),
                ('O I',          1303.5,  'OI',      0.0, 4, True),
                ('C II',         1334.53, 'CII',     0.0, 4, True),
                ('Si IV',        1396.7,  'SiIV',    0.0, 2, False),
                ('Si IV/O IV]',  1400,    'SiIV/OIV]',    0.0, 4, True),
                ('O IV]',        1402.2,  'OIV]',    0.0, 2, False),
                ('N IV]',        1486.5,  'NIV]',    0.0, 4, True),
                ("C IV",         1549.1,  "C IV",    0.0, 4, True),
                ('He II ',       1640.5,  'HeII',    0.0, 2, True),
                ('O III]',       1663.0,  'OIII]',   0.0, 2, True),
                ('N III]',       1750.4,  'NIII]',   0.0, 2, True),
                ('Al III',       1858.7,  'AlIII',   0.0, 4, True),
                ('Si III]',      1892.0,  'SiIII]',  0.0, 4, True),
                ("C III]",       1908.7,  "C III]",  100.0, 4, True),
                ('Fe III',       2075,    'FeIII',   0.0, 0, True),
                ('C II] ',       2326,    'CII]',    0.0, 2, True),
                ('Fe II',        2375,    'FeII',  -10.0, 0, True),
                ('Fe II',        2383,    'FeII',   20.0, 0, True),
                ('[Ne IV]',      2423,    '[NeIV]',  0.0, 2, True),
                ('Fe II',        2587,    'FeII',  -10.0, 0, True),
                ('Fe II',        2600,    'FeII',   20.0, 0, True),
                ('Fe II',        2750.3,  'FeII',    0.0, 0, False),
                ('Mg II',        2799.8,  'MgII',    0.0, 4, True),
                ('Mg II',        2795.53, 'MgII',    0.0, 0, False),
                ('Mg II',        2802.71, 'MgII',    0.0, 0, True),
                ('Mg I',         2852,    'MgI',     0.0, 0, True),
                ('O III',        3047,    'OIII',    0.0, 2, True),
                ('O III ',       3133,    'OIII',    0.0, 2, True),
                ('[Ne V]',       3346,    '[NeV]',   0.0, 2, True),
                ('[Ne V]',       3426,    '[NeV]',   0.0, 2, True),
                ('[O II]',       3726.03, '[O II]',  0.0, 4, True),
                ('[O II]',       3728.82, '[O II]',  0.0, 4, False),
                ('H-kappa',      3750,    r'H$\kappa$', 0.0, 0, True),
                ('[Fe VII]',     3761.4,  '[FeVII]', 0.0, 0, True),
                ('H-iota',       3770,    r'H$\iota$', 0.0, 0, True),
                ('H-theta',      3797,    r'H$\theta$', 0.0, 0, True),
                ('H-eta',        3835,    r'H$\eta$', 0.0, 0, True),
                ('CN bandhd',    3883,    'CN',      0.0, 0, True),
                ('CaII K',       3933.67, 'CaII K',  0.0, 0, True),
                ('CaII H',       3968.47, 'CaII H',  0.0, 0, True),
                ('H-delta',      4101,    r'H$\delta$', 0.0, 1, True),
                ('G-band',       4305,    'G-band',  0.0, 0, True),
                ('H-gamma',      4340,    r'H$\gamma$', 0.0, 1, True),
                ('Fe4383',       4383,    'Fe4383',  0.0, 0, True),
                ('Ca4455',       4455,    'Ca4455',  0.0, 0, True),
                ('Fe4531',       4531,    'Fe4531',  0.0, 0, True),
                ('H-beta',       4861,    r'H$\beta$', 0.0, 3, True),
                ('[O III]',      4962.,   '[O III]', 0.0, 4, False),
                ('[O III]',      5007.,   '[O III]', 0.0, 4, True),
                ('Mg I (b)',     5176,    'Mg b',    0.0, 0, True),
                ('[N I]',        5199.,   '[N I]',   0.0, 2, True),
                ('HeI',          5876.,   'He I',    0.0, 2, True),
                ('Na I (D)',     5889.95, '     ',   0.0, 0, True),
                ('Na I (D)',     5895.92, 'Na D ',   0.0, 0, True),
                ('[O I]',        6300.,   '[O I]',   0.0, 2, True),
                ('[N II]',       6548.,   '[N II]',  0.0, 2, False),
                ('H-alpha',      6562.8,  r'H$\alpha$', 0.0, 3, True),
                ('[N II]',       6583.5,  '[N II]',  0.0, 2, False),
                ('[S II]',       6716.4,  '[S II]',  0.0, 2, False),
                ('[S II]',       6730.8,  '[S II]',  0.0, 2, True),
                ('Ca triplet',   8498.03, 'CaII',    0.0, 0, True),
                ('Ca triplet',   8542.09, 'CaII',    0.0, 0, True),
                ('Ca triplet',   8662.14, 'CaII',    0.0, 0, True),
                ('[S III]',      9069,    '[S III]', 0.0, 2, True),
                ('[S III]',      9532,    '[S III]', 0.0, 2, True),
                ('Pa-gamma',    10900.,   r'Pa$\gamma$', 0.0, 4, True),
                ('Pa-beta',     12800.,   r'Pa$\beta$',  0.0, 4, True),
                ('Pa-alpha',    18700.,   r'Pa$\alpha$', 0.0, 4, True)
                ], dtype=linefmt)

    # -----------------------------------------------------------------------

    def draw_tick(self, lam, linetype, ticklen, usesmooth=False, labww=20.,
                  tickfac=0.75):
        """
        This method is called by mark_lines
        It labels a spectral line by drawing a tickmark above or below the
         spectrum at the given wavelength (lam).
        """

        """ Choose whether to use the smoothed flux or not """
        if usesmooth:
            flux = self.ysmooth
        else:
            flux = self['flux']

        """ Check linetype """
        if linetype == 'abs':
            pm = -1.
            labva = 'top'
        elif linetype == 'em' or linetype == 'strongem':
            pm = 1.
            labva = 'bottom'
        else:
            print('')
            print("ERROR: linetype must be either 'abs', 'em', or 'strongem'")
            print('')
            return None, None

        """ Set up the tick parameters"""
        tmpmask = np.fabs(self['wav'] - lam) < (labww / 2.)
        if linetype == 'em' or linetype == 'strongem':
            specflux = flux[tmpmask].max()
        else:
            specflux = flux[tmpmask].min()
        tickstart = specflux + pm * tickfac * ticklen
        tickend = tickstart + pm * ticklen
        labstart = tickstart + pm * 1.5 * ticklen

        """ Draw the tick mark """
        plt.plot([lam, lam], [tickstart, tickend], 'k')

        """ Return relevant info for plotting """
        return labstart, labva

    # -----------------------------------------------------------------------

    def mark_lines(self, linetype, z, usesmooth=False, marktype='tick',
                   labww=20., labfs=12, tickfrac=0.05, tickfac=0.75,
                   showz=True, zstr='z', zfs=16, labloc='default',
                   labcolor='k', namepos='top', markatm=True):
        """
        A generic routine for marking spectral lines in the plotted spectrum.
        The required linetype parameter can be either 'abs' or 'em' and will
         determine whether absorption or emission lines are marked.

        Inputs:
          linetype - Must be either 'abs' or 'em' to mark absorption or
                     emission lines, respectively.  A third option, 'strongem'
                     only marks strong emission lines
          z        - redshift to be marked
          labww    - width in pixels of the window used to set the vertical
                      location of the tickmark (location is set from the
                      minimum or maximum value within the window).
          labfs    - font size for labels, in points
          ticklen  - override of auto-determination of tick length if > 0
        """

        """ Check linetype """
        if linetype == 'abs':
            labva = 'top'
        elif linetype == 'em' or linetype == 'strongem':
            labva = 'bottom'
        else:
            print('')
            print("ERROR: linetype must be either 'abs', 'em', or 'strongem'")
            print('')
            return

        """ Set the display limits """
        lammin, lammax = self['wav'].min(), self['wav'].max()
        x0, x1 = plt.xlim()
        y0, y1 = plt.ylim()
        if x0 > lammin:
            lammin = x0
        if x1 < lammax:
            lammax = x1
        # xdiff = x1 - x0
        ydiff = y1 - y0
        # dlocwin = labww / 2.

        """ Select lines within current display range """
        zlines = (z + 1.0) * self.lineinfo['wavelength']
        zmask = np.logical_and(zlines > lammin, zlines < lammax)
        tmptype = self.lineinfo['type']
        if linetype == 'em':
            tmask = tmptype > 0
        elif linetype == 'strongem':
            tmask = tmptype > 2
        else:
            tmask = (tmptype < 2) | (tmptype == 3)
        mask = zmask & tmask
        tmplines = self.lineinfo[mask]
        zlines = (z + 1.0) * tmplines['wavelength']
        print('')
        print('Line        lambda_rest  lambda_obs')
        print('----------  -----------  -----------')
        for i in range(len(tmplines)):
            line = tmplines[i]
            print('%-10s   %8.2f      %8.2f' %
                  (line['name'], line['wavelength'], zlines[i]))

        """ Set the length of the ticks """
        ticklen = tickfrac * ydiff

        print('')
        if (len(tmplines) == 0):
            print('')
            print('No lines of the requested type within the wavelength')
            print(' range covered by this spectrum.')
            print('')
            return

        xarr = tmplines['wavelength'] * (z + 1.)

        """
        Mark the location of the spectral lines with either tickmarks (default)
        or vertical dashed lines
        """
        for i in range(len(tmplines)):
            info = tmplines[i]
            if marktype == 'tick':
                labstart, labva = \
                    self.draw_tick(xarr[i], linetype, ticklen,
                                   usesmooth=usesmooth, labww=labww,
                                   tickfac=tickfac)
                # tmpmask = np.fabs(self['wav']-xarr[i]) < dlocwin
                # if linetype == 'em' or linetype == 'strongem':
                #     specflux = flux[tmpmask].max()
                # else:
                #     specflux = flux[tmpmask].min()
                # tickstart = specflux + pm * tickfac*ticklen
                # tickend = tickstart + pm * ticklen
                # labstart = tickstart + pm * 1.5*ticklen
                # plt.plot([xarr[i], xarr[i]], [tickstart, tickend], 'k')
                labha = 'center'
            else:
                plt.axvline(xarr[i], color='k', ls='--')
                labha = 'right'
                if namepos == 'bottom':
                    labstart = y0 + 0.05 * ydiff
                else:
                    labstart = y1 - 0.05 * ydiff
                    labva = 'top'

            """ Label the lines """
            if info['plot']:
                plt.text(xarr[i] + info['dxlab'], labstart, info['label'],
                         rotation='vertical', ha=labha, va=labva,
                         color=labcolor, fontsize=labfs)

        """ Label the plot with the redshift, if requested """
        ax = plt.gca()
        if showz:
            if labloc == 'topright':
                labx = 0.99
                laby = 0.9
                ha = 'right'
            else:
                labx = 0.01
                laby = 0.99
                ha = 'left'
            plt.text(labx, laby, '%s = %5.3f' % (zstr, z), ha=ha, va='top',
                     color=labcolor, fontsize=zfs, transform=ax.transAxes)

    # -----------------------------------------------------------------------

    def apply_wavecal_linear(self, lambda0, dlambda, outfile=None,
                             outformat='text', doplot=True):
        """

        Applies a very simple linear mapping from pixels to wavelength
        and saves the output if desired.  The required inputs provide an
        intercept (lambda0) and a slope (dlambda) that are used to define the
        linear mapping, i.e.,
           wavelength = lambda0 + pix * dlambda

        Required inputs:
          lambda0:    Intercept value in Angstrom
          dlambda:    Slope (dispersion) in Angstrom/pix
        Optional inputs:
          outfile:    Name of output file, if one is desired.  The default
                       value (None) means no output file is produced
          outformat: Format of output file (see help file for Spec1d.save for
                       the possible values).  Default value is 'text'
          doplot:     Plot the spectrum with the new wavelength calibration if
                       desired.  Default value (True) means make the plot.

        """

        """ Make the new wavelength vector """
        x = np.arange(self['wav'].size)
        self['wav'] = lambda0 + dlambda * x

        """ Plot the spectrum if desired """
        if doplot:
            self.plot()

        """ Save the wavelength-calibrated spectrum if desired """
        if outfile is not None:
            self.save(outfile, outformat=outformat)

    # -----------------------------------------------------------------------

    def check_wavecal(self, modsmooth='default', verbose=True):
        """

        Plots the observed wavelength-calibrated sky spectrum on top of a
        smoothed a priori model of the night sky emission so that
        the quality of the wavelength calibration can be evaluated.

        Inputs:
          modsmooth - Smoothing kernel for the model sky spectrum in Angstrom??
                       The default value is set under the assumption that
                       the dispersion of the spectrum gives three pixels
                       across the FWHM of the spectral resolution.  Therefore
                       the smoothing kernel should be:
                           sigma = fwhm / sqrt{2 ln 2} ~ fwhm / 1.177
                       meaning that:
                           sigma ~ 3. * dispersion / 1.177 ~ 2.55 * dispersion

        """

        """
        For the observed sky spectrum use either:
          1. The actual sky spectrum, if it exists (preferred)
          2. The square root of the variance spectrum, if it exists
        """

        """ Plot the observed sky spectrum """
        try:
            self.plot_sky()
        except KeyError:
            return

        if self.sky:
            skyflux = self['sky']
        elif self.hasvar:
            skyflux = np.sqrt(self['var'])
            mask = np.isfinite(skyflux)
            skyflux = skyflux[mask]

        """ Create the model sky spectrum, with the appropriate smoothing """
        print('')
        if modsmooth == 'default':
            modsmooth = 2.55 * self.dispave
            print('Smoothing sky spectrum with default value of %6.3f Ang'
                  % modsmooth)
        elif isinstance(modsmooth, float):
            print('Smoothing sky spectrum with passed value of %6.3f Ang'
                  % modsmooth)
        else:
            print('ERROR: modsmooth parameter must be a float')
            raise TypeError
        waveobs = self['wav'].copy()
        skymod = make_sky_model(self['wav'], smooth=modsmooth)

        """
        Scale the sky spectrum to roughly be 75% of the amplitude of the
        observed spectrum
        """

        ymin, ymax = plt.ylim()
        deltaobs = ymax - ymin
        deltamod = skymod['flux'].max() - skymod['flux'].min()
        print deltaobs, deltamod
        print skyflux.mean(), skymod['flux'].mean()
        skymod['flux'] *= 0.75 * deltaobs / deltamod
        skymod['flux'] += skyflux.mean() - skymod['flux'].mean()

        """ Make the plot """
        wrange = waveobs.max() - waveobs.min()
        xmin = waveobs.min() - 0.05*wrange
        xmax = waveobs.max() + 0.05*wrange
        skymod.plot(color='r', label='Model sky')
        plt.legend()
        plt.xlim(xmin, xmax)

        """ Clean up """
        del waveobs, skymod

    # -----------------------------------------------------------------------

    def resample(self, owave=None):
        """
        Resample the spectrum onto a linearized wavelength grid.  The grid
        can either be defined by the input wavelength range itself
        (the default) or by a wavelength vector that is passed to the function.
        """

        if owave is None:
            w0 = self['wav'][0]
            w1 = self['wav'][-1]
            owave = np.linspace(w0, w1, self['wav'].size)

        specmod = interpolate.splrep(self['wav'], self['flux'])
        outspec = interpolate.splev(owave, specmod)

        """ Store resampled spectrum """
        print('resample: replacing input spectrum with resampled version')
        print('resample: for now not resampling the variance')
        self.rswav = owave
        self.rsflux = outspec

    # -----------------------------------------------------------------------

    def save(self, outfile, outformat='text', useresamp=False, verbose=True):
        """
        Saves a spectrum into the designated output file.
        Right now, there are only the followoing options:
          1. 'text'     - produces a text file with columns for wavelength,
                          flux, variance (if available), and sky (if available)
          2. 'fits'     - produces a multiextension fits file with separate
                          HDUs for wavelength, flux, variance (if available),
                          and sky (if available).
          3. 'fitsflux' - produces a single-extension fits file with a 1-dim
                          data vector for the flux.  The wavelength information
                          is stored in the crpix1 and cd1_1 keywords.
                          NOTE: By storing the wavelength info in this way,
                           the wavelength vector must be evenly spaced in
                           wavelength (and not log(wavelength)).
        """

        """
        Set temporary variables for the vectors.  This will allow for future
        flexibility
        """
        if useresamp:
            wav = self.rswav
            flux = self.rsflux
            var = None
        else:
            wav = self['wav']
            vsize = len(wav)
            flux = self['flux']
            if self.hasvar:
                var = self['var']
            else:
                var = None
            if self.sky:
                sky = self['sky']

        """ Save the spectrum in the requested format """
        if outformat == 'fits':
            hdu = pf.HDUList()
            phdu = pf.PrimaryHDU()
            hdu.append(phdu)
            outwv = pf.ImageHDU(wav, name='wavelength')
            outflux = pf.ImageHDU(flux, name='flux')
            hdu.append(outwv)
            hdu.append(outflux)
            if self.hasvar:
                outvar = pf.ImageHDU(var, name='variance')
                hdu.append(outvar)
            if self.sky:
                outsky = pf.ImageHDU(sky, name='sky')
                hdu.append(outsky)
            hdu.writeto(outfile, overwrite=True)

        elif outformat == 'fitsflux':
            """ Only use this if the wavelength vector is evenly spaced """
            phdu = pf.PrimaryHDU(flux)
            phdu.header['crpix1'] = 1
            phdu.header['crval1'] = wav[0]
            phdu.header['ctype1'] = 'PIXEL'
            phdu.header['cd1_1'] = wav[1] - wav[0]
            phdu.writeto(outfile, overwrite=True)

        elif outformat == 'text':
            # CONSIDER JUST USING THE WRITE() METHOD FOR THE TABLE HERE!
            if self.hasvar:
                if self.sky:
                    outdata = np.zeros((vsize, 4))
                    fmtstring = '%7.2f %9.3f %10.4f %9.3f'
                    outdata[:, 3] = sky
                else:
                    outdata = np.zeros((vsize, 3))
                    fmtstring = '%7.2f %9.3f %10.4f'
                outdata[:, 2] = var
            else:
                outdata = np.zeros((vsize, 2))
                fmtstring = '%7.2f %9.3f'
            outdata[:, 0] = wav
            outdata[:, 1] = flux
            print('')
            np.savetxt(outfile, outdata, fmt=fmtstring)
            del outdata

        if verbose:
            print('Saved spectrum to file %s in format %s' %
                  (outfile, outformat))

    # -----------------------------------------------------------------------

# ===========================================================================

# -----------------------------------------------------------------------


def make_sky_model(wavelength, smooth=25., doplot=False, verbose=True):
    """
    Given an input wavelength vector, creates a smooth model of the
    night sky emission that matches the wavelength range and stepsize
    of the input vector.
    """

    """ Get info from input wavelength vector """
    wstart = wavelength.min()
    wend = wavelength.max()
    disp = wavelength[1] - wavelength[0]

    """
    Read in the appropriate skymodel:
     * If the starting wavelength is > 9000 Ang, then use the NIR sky model
     * Otherwise use the optical sky model
    These are in a B-spline format, which is apparently what
     the call to interpolate.splev needs (see code just below).
    However, they used to be stored in a numpy savefile (for the NIR spectrum)
     or a pickle save format (for the optical spectrum).
     To make things more consistent with the rest of the code here, I converted
     them to the 'fitstab' format (i.e., binary fits tables) and then will,
     in the code below, convert them back into the appropriate B-spline tuple
     format that interpolate.splev requires.
    """
    if __file__ == 'spec_simple.py':
        moddir = '.'
    else:
        moddir = __file__.split('/spec_simple')[0]
    if wstart >= 9000.:
        modfile = '%s/Data/nirspec_skymodel.fits' % moddir
    else:
        modfile = '%s/Data/uves_skymodel.fits' % moddir
    try:
        modspec = Spec1d(modfile, informat='fitstab')
    except IOError:
        raise IOError
    skymodel = (modspec['wav'], modspec['flux'], 3)

    """
    Make sure that the requested wavelength range does not exceed the range
    in the model
    """
    redo_wav = False
    if wstart < modspec['wav'][0]:
        wstart = modspec['wav'][0] + 1.
        redo_wav = True
    if wend > modspec['wav'][-1]:
        wend = modspec['wav'][-1] - 10.
        redo_wav = True
    if redo_wav:
        print('Limiting wavelength range for model sky to %8.3f - %8.3f'
              % (wstart, wend))

    if verbose:
        print('Making model sky')
        print('--------------------------------------')
        print('Model starting wavelength: %f') % wstart
        print('Model ending wavelength:    %f') % wend
        print('Model dispersion:             %f') % disp

    """
    Resample and smooth the model spectrum.
    The call to splev does a spline interpolation of the sky model onto the
     points defined by the "wave" array
    """
    wave = np.arange(wstart, wend, 0.2)
    tmpskymod = interpolate.splev(wave, skymodel)
    tmpskymod = ndimage.gaussian_filter(tmpskymod, smooth)

    """
    Create a B-spline representation of the smoothed curve for use in
    the wavecal optimization
    """
    model = interpolate.splrep(wave, tmpskymod)

    """
    Finally use the initial guess for the dispersion and evaluate the
     model sky at those points, using the B-spline model
    """
    skyflux = interpolate.splev(wave, model)

    """ Create a Spec1d instance containing the sky model """
    skymod = Spec1d(wav=wave, flux=skyflux)

    """ Plot the output model if requested """
    if doplot:
        skymod.plot(title='Model Sky Spectrum')

    """ Clean up and return """
    del skymodel, tmpskymod, wave
    return skymod
