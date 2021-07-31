"""
spec1d.py

This file contains the Spec1d class, which is used to process and plot
1d spectroscopic data.
"""

import os
import numpy as np
from scipy import interpolate, ndimage
import matplotlib.pyplot as plt

from astropy.io import ascii
from astropy.table import Table
from astropy.io import fits as pf
# from astropy.modeling.blackbody import blackbody_lambda

from cdfutils import datafuncs as df

import sys
pyversion = sys.version_info.major

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

    def __init__(self, inspec=None, informat='text',
                 wav=None, flux=None, var=None, sky=None, logwav=False,
                 colnames=['wav', 'flux', 'var', 'sky'], tabext=1,
                 trimsec=None, verbose=True, debug=False):
        """

        Reads in the input 1-dimensional spectrum.
        This can be done in two mutually exclusive ways:

         1. By providing some 1-d arrays containing the following:
               wavelength - required
               flux       - "mostly required".  For nearly all applications
                            the user will want both wavelength and flux.
                            However, as a first step in generating, e.g.,
                            a model spectrum, starting with only a
                            wavelength array should be OK.  In that
                            case a temporary flux array with all values
                            set to 1.0 will be created.
               variance   - optional
               sky        - optional

                      ---- or ----

         2. By providing a single input which can be one of the following:

            a. A filename, in which case the informat parameter is important

            b. A astropy Table that contains, at a minimum, wavelength and
               flux columns

            c. A  HDU from a previously-read fits file, where the
               HDU data are in a binary fits table that contains the
               wavelength, flux, and possibly variance and sky information.

         ------

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
           pypeit: A binary table HDU that contains wavelength,
              flux, and inverse variance among many others.
           mwa:  A multi-extension fits file with wavelength info in
              the fits header
              Extension 1 is the extracted spectrum (flux)
              Extension 3 is the variance spectrum
           text: An ascii text file with information in columns:
               Column 1 is the wavelength
               Column 2 is the extracted spectrum
               Column 3 (optional) is the variance spectrum
               Column 4 (optional) is the sky spectrum
                 [NOT YET IMPLEMENTED]

               Thus, an input text file could have one of three formats:
                   A.  wavelength flux
                   B.  wavelength flux variance
                   C.  wavelength flux variance sky


        Inputs (all inputs are optional, but at least one way of specifiying
        the input spectrum must be used):
          inspec   - A single-parameter designation of the input spectrum,
                     either:
                       a. A filename (string)
                       b. An astropy Table
                       c. A HDU containing the spectrum as a binary table
                     NOTE: If inspec is None, then the spectrum must be
                      provided via the wavelength and flux vectors.

          informat - format of input file (see above for possibilities)
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
        self.ordnum = None
        self.sky = False
        self.infile = None
        self.dispave = None
        self.names0 = ['wav', 'flux', 'var']
        spec0 = None
        self.smospec = None
        self.ysmooth = None
        self.varsmooth = None
        self.atm_trans = None
        self.atmcorr = None
        self.respcorr = None

        self.logwav = logwav

        """
        Read in the spectrum.
        First see if the spectrum is being passed by the inspec parameter
        """
        if inspec is not None:

            if isinstance(inspec, str):
                """ Input as a filename """
                self.infile = inspec
                try:
                    spec0 = self.read_file(inspec, informat, colnames=colnames,
                                           tabext=tabext, verbose=verbose,
                                           debug=debug)
                except IOError:
                    print('')
                    print('Could not read input file %s' % inspec)
                    print('')
                    raise IOError

            elif isinstance(inspec, Table):
                """ Input as a astropy Table """
                spec0 = self.read_table(inspec, colnames)

            elif isinstance(inspec, pf.BinTableHDU):
                """ Input as a previously loaded HDU """
                spec0 = self.read_hdu(inspec, colnames)

            else:
                print('')
                print('ERROR: inspec parameter must be one of the following')
                print('  1. A filename')
                print('  2. An astropy Table')
                print('  3. A previously loaded HDU (as a BinTableHDU)')
                print('')
                raise TypeError

        elif wav is not None:
            spec0 = self._read_arrays(wav, flux, var, sky)

        else:
            print('')
            print('ERROR: Must provide either:')
            print('  1. A single parameter that is one of the following:')
            print('      A. the name of a file containing the spectrum')
            print('      B. an astropy Table containing the spectrum')
            print('      C. a previously-loaded HDU containing the spectrum')
            print('         as a binary table')
            print('')
            print('             OR')
            print('')
            print('  2. At minimum, both of the following:')
            print('         A. a wavelength vector (wav)')
            print('         B. a flux vector (flux)')
            print('      and optionally one or both of the following')
            print('         C. a variance vector (var)')
            print('         D. a sky spectrum vector (sky)')
            print('')
            return

        """
        Trim the spectra if requested
        """
        if trimsec is not None:
            xmin = trimsec[0]
            xmax = trimsec[1]
            spec0 = spec0[xmin:xmax]
            # spec0[0] = spec0[0][xmin:xmax]
            # spec0[1] = spec0[1][xmin:xmax]
            # if spec0[2] is not None:
            #     spec0[2] = spec0[2][xmin:xmax]
            # if spec0[3] is not None:
            #     spec0[3] = spec0[3][xmin:xmax]

        """ Get average dispersion """
        self.dispave = self.find_dispave(spec0['wav'], verbose=verbose)

        """
        Call the superclass initialization for useful Data1d attributes
        """
        if debug:
            print(self.names0)
            print('Wavelength vector size: %d' % spec0['wav'].size)
            print('Flux vector size: %d' % spec0['flux'].size)
        if 'var' in spec0.colnames:
            var = spec0['var']
            names = self.names0
            """ Variance=0 causes problems, so fix here """
            mask = var == 0.
            var[mask] = 10. * var.max()
        else:
            var = None
            names = self.names0[:-1]
        if pyversion == 2:
            super(Spec1d, self).__init__(spec0['wav'], spec0['flux'], var,
                                         names=names)
        else:
            super().__init__(spec0['wav'], spec0['flux'], var, names=names)

        """ Add the sky vector to the Table structure if it is not none """
        if 'sky' in spec0.colnames:
            self['sky'] = spec0['sky'].copy()
            self.sky = True

        """ Read in the list that may be used for marking spectral lines """
        self.load_linelist()

    # -----------------------------------------------------------------------

    def _read_tab_gen(self, indat, keynames=None, tabext=1):
        """

        This is a generic table reader that takes advantage of the 
         Table.read() method's ability to guess the input file structure
         correctly for at least some inputs.  This generic reader can be
         used for the following formats:
           * binary fits file (in which case the tabext parameter is used)
           * ascii file in which the first line defines the column names
           * fits HDU, where the fits file has been opened by another function
           * Table structure
           * recarray structure (maybe)
        """

        """ Read in the table """
        if isinstance(indat, str):
            if indat[-4:] == 'fits':
                intab = Table.read(indat, hdu=tabext)
        else:
            intab = Table.read(indat)

        """ Rename the columns if requested """
        if keynames is not None:
            if len(keynames) == 2:
                outnames = ['wav', 'flux']
            elif len(keynames) == 3:
                outnames = ['wav', 'flux', 'var']
            elif len(keynames) == 4:
                outnames = ['wav', 'flux', 'var', 'sky']
            else:
                raise IndexError('keynames had %d elements, but must have '
                                 '2-4 elements')
        for inkey, outkey in zip(keynames, outnames):
            if inkey.lower() not in intab.colnames and \
               inkey.upper() not in intab.colnames:
                raise KeyError('Expected column %s not found in %s'
                               % (inkey, indat))
            intab.rename_column(inkey, outkey)

        """ Return the table """
        return intab

    # -----------------------------------------------------------------------

    def _read_pypeit(self, infile, tabext=1, verbose=True, debug=False):
        """

        Reads in a spectrum that has been produced by the pypeit data
        reduction pipeline.

        The input file contains a binary fits table that can be read
        by the standard Table.read() code.  The only change from a standard
        read is that the primary columns need to be renamed:
         opt_wave --> wav
         opt_counts --> flux
         opt_counts_ivar --> var
         opt_counts_sky --> sky

        """

        """ Identify the appropriate columns """
        keynames = ['opt_wave', 'opt_counts', 'opt_counts_ivar',
                    'opt_counts_sky']

        """ Read the data and return the subsequent table """
        try:
            intab = self._read_tab_gen(infile, keynames=keynames,
                                       tabext=tabext)
        except KeyError:
            for i in range(len(keynames)):
                keynames[i] = keynames[i].upper()
            intab = self._read_tab_gen(infile, keynames=keynames,
                                       tabext=tabext)
            """
            The pypeit format has inverse variance rather than variance
            """
            mask = intab['var'] > 0
            intab['var'][mask] = 1. / intab['var'][mask]
            intab['var'][~mask] = 10. * intab['var'][mask].max()
            
        return intab

        
    # -----------------------------------------------------------------------

    def read_file(self, infile, informat, tabext=1,
                  colnames=['wav', 'flux', 'var', 'sky'], verbose=True,
                  debug=False):
        """

        Reads a 1-d spectrum from a file.  The file must have one of the
        following formats, which is indicated by the informat parameter:
          fits
          fitstab
          fitsflux
          deimos
          pypeit
          mwa
          iraf
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
            thdu = hdu[tabext]
            tmp = self.read_hdu(thdu, colnames)
            wav = tmp['wav'].copy()
            flux = tmp['flux'].copy()
            if 'var' in tmp.colnames:
                var = tmp['var'].copy()
                hasvar = True
            if 'sky' in tmp.colnames:
                sky = tmp['sky'].copy()
            del hdu, tmp
        elif informat == 'iraf':
            hdu = pf.open(infile)
            hdr = hdu[0].header
            specdat = hdu[0].data
            if specdat.ndim == 3:
                flux = specdat[0, 0, :]
                flux_nowht = specdat[1, 0, :]
                bkgd = specdat[2, 0, :]
                rms = specdat[3, 0, :]
                var = rms**2
            else:
                flux = specdat[0, :]
            wav = np.arange(flux.size)
            if self.logwav:
                wav = 10.**(hdr['crval1'] + wav * hdr['cd1_1'])
            else:
                wav = hdr['crval1'] + wav*hdr['cd1_1']
            print('flux: ',flux.shape)
            print('wav: ',wav.shape)
            del hdu, specdat
        elif informat == 'fitsflux':
            hdu = pf.open(infile)
            if hdu[0].data.ndim == 2:
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
        elif informat.lower() == 'pypeit':
            tab = self._read_pypeit(infile, tabext=tabext)
            hasvar = True
            wav = tab['wav']
            flux = tab['flux']
            var = tab['var']
            sky = tab['sky']
        elif informat.lower() == 'esi':
            hdu = pf.open(infile)
            wav = 10.**(hdu[1].data)
            flux = hdu[2].data.copy()
            var = hdu[3].data.copy()
            hasvar = True
            del hdu
        elif informat.lower() == 'lpipe':
            colnames = ['wav', 'flux', 'sky', 'rms', 'xpixel', 'ypixel',
                        'response', 'flag']
            tab = ascii.read(infile, names=colnames)
            wav = tab['wav']
            flux = tab['flux']
            var = tab['rms']**2
            sky = tab['sky']
            hasvar = True
        elif informat.lower() == 'nsx':
            tab = ascii.read(infile)
            wav = tab['angstrom'].copy()
            flux = tab['object'].copy()
            var = (tab['error'].copy())**2
            hasvar = True
            del tab
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
            spectab = ascii.read(infile)
            if self.logwav:
                wav = 10.**spectab.columns[0].data
            else:
                wav = spectab.columns[0].data
            flux = spectab.columns[1].data
            if len(spectab.columns) > 2:
                var = spectab.columns[2].data
                hasvar = True
            if len(spectab.columns) > 3:
                sky = spectab.columns[3].data
                self.sky = True

        """ Fix var=0 values, since they will cause things to crash """
        if hasvar:
            mask = var <= 0.
            var[mask] = var.max() * 3.

        """ Check for NaN's, which this code can't handle """
        if hasvar:
            mask = (np.isnan(flux)) | (np.isnan(var))
            varmax = var[~mask].max()
            var[mask] = varmax * 5.
        else:
            mask = (np.isnan(flux))
        flux[mask] = 0

        """ Return the data """
        spec0 = Table([wav, flux], names=(['wav', 'flux']))
        if var is not None:
            spec0['var'] = var
        if sky is not None:
            spec0['sky'] = sky
        return spec0

    # -----------------------------------------------------------------------

    def read_table(self, inspec, colnames=['wav', 'flux', 'var', 'sky']):
        """

        Read the spectrum from an astropy Table

        """

        """ Set some default values """
        var = None
        sky = None

        """ Read the relevant columns of the input table """
        wav = inspec[colnames[0]]
        if colnames[1] not in inspec.colnames:
            flux = np.ones(len(inspec))
        else:
            flux = inspec[colnames[1]]
        if colnames[2] in inspec.colnames:
            var = inspec[colnames[2]]
        if colnames[3] in inspec.colnames:
            sky = inspec[colnames[3]]

        """ Save the columns in a new table and return """
        spec0 = Table([wav, flux], names=(['wav', 'flux']))
        if var is not None:
            spec0['var'] = var
        if sky is not None:
            spec0['sky'] = sky
        return spec0

    # -----------------------------------------------------------------------

    def read_hdu(self, hdu, colnames=['wav', 'flux', 'var', 'sky']):
        """

        Read the spectrum from an previously loaded HDU, which is in a
        binary table format

        """

        """ Set up defaults """
        wav = None
        flux = None
        var = None
        sky = None

        """ Read the data """
        tdat = hdu.data
        try:
            wav = tdat[colnames[0]]
        except KeyError:
            wav = tdat.field(0)
        try:
            flux = tdat[colnames[1]]
        except KeyError:
            flux = tdat.field(1)
        if len(tdat.columns) > 2:
            try:
                var = tdat[colnames[2]]
            except KeyError:
                var = tdat.field(2)

        """
        For the sky, only read in a column if it is actually called 'sky'
        """
        if len(tdat.columns) > 3:
            try:
                sky = tdat[colnames[3]]
            except KeyError:
                pass

        """ Save the arrays in a table and return """
        spec0 = Table([wav, flux], names=(['wav', 'flux']))
        if var is not None:
            spec0['var'] = var
        if sky is not None:
            spec0['sky'] = sky
        return spec0

    # -----------------------------------------------------------------------

    def _read_arrays(self, wav, flux, var, sky):
        """

        Reads the spectrum from individual arrays

        """

        """ Get the wavelength vector into linear units """
        if self.logwav:
            w = 10.**wav
        else:
            w = wav.copy()

        """ Make the flux identically 1.0 if it has not been given """
        if flux is None:
            flux = np.ones(wav.size)

        """ Save the arrays in a table and return """
        spec0 = Table([w, flux], names=(['wav', 'flux']))
        if var is not None:
            spec0['var'] = var
        if sky is not None:
            spec0['sky'] = sky
        return spec0

    # -----------------------------------------------------------------------

    def find_dispave(self, wav, verbose=True):
        """

        Finds the average dispersion from the wavelength array

        """

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

        return dispave

    # -----------------------------------------------------------------------

    def select_mode(self, mode):
        """

        Selects the "mode" of the spectrum that is used for subsequent
        actions such as plotting or saving.

        These modes can be produced by various processing steps within the
         class and include the following

          'input'    - just the spectrum as it was read in
          'smooth'   - the smoothed spectrum
          'atmcorr'  - the spectrum after applying an atmospheric absorption
                        correction
          'respcorr' - the spectrum after applying a response correction

        Inputs:
          mode - one of the values listed above ('input', 'smooth', etc.)
        """

        if mode == 'input':
            spec = self
        elif mode == 'smooth':
            spec = self.smospec
        elif mode == 'atmcorr':
            spec = self.atmcorr
        elif mode == 'respcorr':
            spec = self.respcorr
        else:
            print('')
            errstr = 'Invalid mode (%s) for select_mode. See help for ' % mode
            errstr += 'allowed values\n\n'
            raise ValueError(errstr)

        return spec

    # -----------------------------------------------------------------------

    def __add__(self, other):
        """

        Do a variance-weighted sum of this spectrum with another

        """

        """ Initialize """
        nx = self['wav'].size
        mask = self['var'] != 0
        wtsum = np.zeros(nx)
        wtsum[mask] = 1.0 / self['var'][mask]
        wtflux = wtsum * self['flux']
        if self.sky:
            skysum = self['sky']
        else:
            skysum = np.zeros(nx)

        """ Create the weighted sum """
        wt = np.zeros(nx)
        mask = other['var'] != 0
        wt[mask] = 1.0 / other['var'][mask]
        wtflux += wt * other['flux']
        if other.sky:
            skysum += other['sky']
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
        if self.sky is None:
            outsky = None
        else:
            outsky = skysum / 2.

        """ Return the coadded spectrum as a Spec1d object """
        return Spec1d(wav=self['wav'], flux=outflux, var=outvar,
                      sky=outsky)

    # -----------------------------------------------------------------------

    def __radd__(self, other):
        """

        This is the "reverse add" method that is needed in order to sum
        spectra with the sum function

        NOTE: This doesn't seem to be working yet

        """
        if isinstance(other, (int, float)):
            return self
        else:
            return self.__add__(other)

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
            atm0 = self.read_file(infile, informat='fitstab')
        except IOError:
            print('ERROR: Cannot read atmospheric transmission data file')
            raise IOError
        atm0['wav'] *= 1.0e4

        """
        Only use the relevant part of the atmospheric transmission spectrum
        """
        w0 = atm0['wav']
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
            watm = atm0['wav'][mask]
            trans = atm0['flux'][mask]

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
                       title=None, xlabel=None, ylabel=None,
                       mode='input'):
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

        """
        Set the flux array to be used for the scaling based on the passed mode
         variable.
        The default is to just use the unmodified input spectrum
         (mode='input')
        """
        spec = self.select_mode(mode)

        """ Set some plotting parameters """
        if label == 'default':
            plabel = 'atmTrans'
        elif label is None:
            plabel = None

        """ Now do the plotting """
        tmp = self.atm_trans.copy()
        tmp *= spec['flux'].max() * scale
        tmp += offset
        if plabel is not None:
            self.ax1.plot(self['wav'], tmp, color, linestyle=ls,
                          drawstyle='steps', label=plabel)
        else:
            self.ax1.plot(self['wav'], tmp, color, linestyle=ls,
                          drawstyle='steps')

        """ Label things if requested """
        if xlabel:
            self.ax1.xlabel(xlabel)
        if ylabel:
            self.ax1.ylabel(ylabel)
        if title:
            self.ax1.title(title)

        """ Clean up """
        del tmp

    # -----------------------------------------------------------------------

    def atm_corr(self, mode='input', airmass=1.0, atm='model', fwhm=15.,
                 model='default', airmass_std=1.0):
        """

        Does a correction for atmospheric transmission.
        For now this is done via the model spectrum

        """

        """ Make sure that there is an atmospheric spectrum to use """
        if self.atm_trans is None:
            if atm == 'model':
                """ Make a model spectrum if one doesn't exist"""
                self.make_atm_trans(fwhm=fwhm, modfile=model)

                """ Scale the atmospheric transmission for airmass(???) """
                atmflux = self.atm_trans**airmass
                atmvar = 0.
                # try:
                #     self.make_atm_trans(fwhm=fwhm, modfile=modfile)
                # except IOError:
                #     raise IOError
                # except ValueError:
                #     raise ValueError
            elif atm == 'telluric':
                atmflux = model['flux']
                atmvar = model['var']
                atmflux[atmflux == 0.] = 1.
                mask = (atmflux == 0.) | (atmvar <= 0.)
                atmvar[mask] = 5. * atmvar.max()
            else:
                print('')
                print('Warning: No atmospheric correction applied')
                print('Right now "model" and "telluric" are the only options')
                print(' for the atm parameter')
                print('')
                return

        """ Divide the input spectrum by the transmission """
        atmcorr = self['flux'] / atmflux
        atmcvar = atmcorr**2 * (self['var'] / self['flux']**2 +
                                atmvar / atmflux**2)

        """ Save the output in a Data1d container """
        self.atmcorr = df.Data1d(self['wav'], atmcorr, atmcvar,
                                 names=self.names0)

    # -----------------------------------------------------------------------

    def resp_corr(self, response, mode='input', action='multiply'):
        """

        Given a response curve, corrects the spectrum by either multiplying
         (the default) or dividing the spectrum by the response curve.  The
         version of the spectrum to correct is set by the mode parameter.
        The result is stored in the respcorr version of the spectrum.

        """

        """
        Set the arrays to use based on the passed mode variable.
        The default is to just use the unmodified input spectrum
         (mode='input')
        """
        spec = self.select_mode(mode)

        """ Correct the spectrum """
        if action == 'divide':
            spec['flux'] /= response
            spec['var'] /= response**2
        else:
            spec['flux'] *= response
            spec['var'] *= response**2

        """ Save the result """
        self.respcorr = df.Data1d(self['wav'], spec['flux'], spec['var'],
                                  names=self.names0)

    # -----------------------------------------------------------------------

    def plot(self, mode='input',
             xlabel='Wavelength (Angstroms)', ylabel='Relative Flux',
             title='default', docolor=True, color='b', linestyle='solid',
             showzero=True, model=None, modcolor='g',
             label=None, fontsize=12, rmscolor='r', rmsoffset=0, rmsls=None,
             add_atm_trans=False, atmscale=1.05, atmfwhm=15., atmoffset=0.,
             atmls='-', atmmodfile='default', usesmooth=False, verbose=True,
             fig=None, ax=None, **kwargs):
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

        """
        Set the flux and var arrays to use based on the passed mode variable.
        The default is to just use the unmodified input spectrum
         (mode='input')
        """
        if fig is None:
            self.fig = plt.figure()
        else:
            self.fig = fig
        if ax is not None:
            self.ax1 = ax
        else:
            self.ax1 = self.fig.add_subplot(111)
        if usesmooth:
            mode = 'smooth'
        spec = self.select_mode(mode)

        """ Set the arrays to be plotted """
        wav = spec['wav']
        flux = spec['flux']
        try:
            var = spec['var']
        except KeyError:
            var = None

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
            self.ax1.axhline(color='k')

        """ Set the label """
        if label is not None:
            plabel = label
        else:
            plabel = 'Flux'

        """ Plot the spectrum """
        drawstyle = 'steps'
        ls = "%s" % linestyle
        self.ax1.plot(self['wav'], flux, color, linestyle=ls,
                      drawstyle=drawstyle, label=plabel, **kwargs)
        self.ax1.tick_params(labelsize=fontsize)
        self.ax1.set_xlabel(xlabel, fontsize=fontsize)

        """
        Plot the model, given as an astropy.modeling model, if requested
        """
        if model is not None:
            fmod = model(wav)
            self.ax1.plot(wav, fmod, color=modcolor)

        """ Plot the RMS spectrum if the variance spectrum exists """
        if var is not None:
            rms = np.sqrt(var) + rmsoffset
            if rmsls is None:
                if docolor:
                    rlinestyle = 'solid'
                    rlw = 1
                else:
                    rlinestyle = 'dotted'
                    rlw = 2
            else:
                rlinestyle = '%s' % rmsls
            self.ax1.plot(self['wav'], rms, rmscolor, linestyle=rlinestyle,
                          drawstyle=drawstyle, label='RMS', lw=rlw)

        """ More plot labels """
        self.ax1.set_ylabel(ylabel, fontsize=fontsize)
        if title is not None:
            self.ax1.set_title(title)
        if(wav[0] > wav[-1]):
            self.ax1.set_xlim([wav[-1], wav[0]])
        else:
            self.ax1.set_xlim([wav[0], wav[-1]])
        # print(self['wav'][0], self['wav'][-1])

        """ Plot the atmospheric transmission if requested """
        if add_atm_trans:
            self.plot_atm_trans(mode=mode, ls=atmls, scale=atmscale,
                                offset=atmoffset, fwhm=atmfwhm,
                                modfile=atmmodfile)
        return self.fig
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
        elif 'var' in self.colnames:
            skyflux = np.sqrt(self['var'])
            print('Using RMS spectrum as a proxy for the sky spectrum')
            skylab = 'RMS spectrum'
        else:
            if verbose:
                print('')
                print('Cannot plot sky spectrum.')
                print('The spectrum must either have either a sky column or a'
                      ' a variance column')
                print('')
            raise KeyError('No sky or variance information in the spectrum')

        """ Set up for plotting """
        ls = '%s' % linestyle
        ds = 'steps'
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
            plt.plot(self['wav'], skyflux, ls=ls, ds=ds, color=color,
                     label=lab)
        else:
            plt.plot(self['wav'], skyflux, ls=ls, ds=ds, color=color)
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

    def smooth(self, filtwidth, smfunc='boxcar', mode='input', doplot=True,
               outfile=None, **kwargs):

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

        """ Select the spectrum to be smoothed """
        spec = self.select_mode(mode)

        """
        Smooth the spectrum using the requested smoothing function
         [For now, only boxcar smoothing is allowed]
        The smoothing functions are inherited from the Data1d class
        """
        if smfunc == 'boxcar':
            ysmooth, varsmooth = spec.smooth_boxcar(filtwidth, verbose=False)
        else:
            print('')
            print('For smoothing, smfunc can only be one of the following:')
            print("  'boxcar'")
            print('')
            raise ValueError

        """ Put the smoothed spectrum into a Data1d container """
        if varsmooth is None:
            names = self.names0[:-1]
        else:
            names = self.names0
        self.smospec = df.Data1d(self['wav'], ysmooth, varsmooth,
                                 names=names)

        """ Plot the smoothed spectrum if desired """
        if doplot:
            self.plot(mode='smooth', **kwargs)

    # -----------------------------------------------------------------------

    def load_linelist(self, linefile='default'):

        linefmt = [('name', 'S10'), ('wavelength', float), ('label', 'S10'),
                   ('dxlab', float), ('type', int), ('plot', bool)]
        lineinfo = np.array([
            ('He II',         256.32, 'HeII',    0.0, 2, True),
            ('He II',         303.78, 'HeII',    0.0, 2, True),
            ('He I',          537.03, 'HeI',     0.0, 2, True),
            ('He I',          584.33, 'HeI',     0.0, 2, True),
            ('Ly-gamma',      972.54, r'Ly$\gamma$',      0.0, 3, True),
            ('Ly-beta',      1025.7,  r'Ly$\beta$',      0.0, 3, True),
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
            ('Pa-delta',    10050.,   r'Pa$\delta$', 0.0, 3, True),
            ('Pa-gamma',    10940.,   r'Pa$\gamma$', 0.0, 3, True),
            ('Pa-beta',     12820.,   r'Pa$\beta$',  0.0, 3, True),
            ('Br-12',       15552.,   'Br12',        0.0, 3, True),
            ('Br-11',       15696.,   'Br11',        0.0, 3, True),
            ('Br-10',       15876.,   'Br10',        0.0, 3, True),
            ('Br-9',        16105.,   'Br9',         0.0, 3, True),
            ('Br-8',        16400.,   'Br8',         0.0, 3, True),
            ('Br-7',        16800.,   'Br7',         0.0, 3, True),
            ('Br-6',        17357.,   'Br6',         0.0, 3, True),
            ('Br-5',        18170.,   'Br5',         0.0, 3, True),
            ('Pa-alpha',    18750.,   r'Pa$\alpha$', 0.0, 3, True),
            ('Br-delta',    19440.,   r'Br$\delta$', 0.0, 3, True),
            ('Br-gamma',    21660.,   r'Br$\gamma$', 0.0, 3, True),
            ('Br-beta',     26250.,   r'Br$\beta$',  0.0, 3, True),
            ('Br-alpha',    40510.,   r'Br$\alpha$', 0.0, 3, True),
        ], dtype=linefmt)

        self.lineinfo = Table(lineinfo)

    # -----------------------------------------------------------------------

    def draw_tick(self, lam, linetype, ticklen, axes, usesmooth=False,
                  labww=20., tickfac=0.75):
        """
        This method is called by mark_lines
        It labels a spectral line by drawing a tickmark above or below the
         spectrum at the given wavelength (lam).
        """

        """ Choose whether to use the smoothed flux or not """
        if usesmooth:
            flux = self.smospec['flux']
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
        axes.plot([lam, lam], [tickstart, tickend], 'k')

        """ Return relevant info for plotting """
        return labstart, labva

    # -----------------------------------------------------------------------

    def mark_lines(self, linetype, z, usesmooth=False, marktype='tick',
                   labww=20., labfs=12, tickfrac=0.05, tickfac=0.75,
                   showz=True, zstr='z', zfs=16, labloc='default',
                   labcolor='k', namepos='top', markatm=True, fig=None):
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
        if fig is None:
            self.fig = plt.gcf()
        else:
            self.fig = fig
        self.ax = self.fig.gca()
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
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
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
                                   tickfac=tickfac, axes=self.ax)
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
                self.ax.axvline(xarr[i], color='k', ls='--')
                labha = 'right'
                if namepos == 'bottom':
                    labstart = y0 + 0.05 * ydiff
                else:
                    labstart = y1 - 0.05 * ydiff
                    labva = 'top'

            """ Label the lines """
            if info['plot']:
                self.ax.text(xarr[i] + info['dxlab'], labstart, info['label'],
                             rotation='vertical', ha=labha, va=labva,
                             color=labcolor, fontsize=labfs)

        """ Label the plot with the redshift, if requested """
        if showz:
            if labloc == 'topright':
                labx = 0.99
                laby = 0.9
                ha = 'right'
            else:
                labx = 0.01
                laby = 0.99
                ha = 'left'
            self.ax.text(labx, laby, '%s = %5.3f' % (zstr, z), ha=ha, va='top',
                         color=labcolor, fontsize=zfs,
                         transform=self.ax.transAxes)

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
        elif 'var' in self.colnames:
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
        print(deltaobs, deltamod)
        print(skyflux.mean(), skymod['flux'].mean())
        skymod['flux'] *= 0.75 * deltaobs / deltamod
        skymod['flux'] += skyflux.mean() - skymod['flux'].mean()

        """ Make the plot """
        wrange = waveobs.max() - waveobs.min()
        xmin = waveobs.min() - 0.05*wrange
        xmax = waveobs.max() + 0.05*wrange
        fig = plt.gcf()
        ax = plt.gca()
        skymod.plot(color='r', label='Model sky', fig=fig, ax=ax)
        plt.legend()
        plt.xlim(xmin, xmax)

        """ Clean up """
        del waveobs, skymod

    # -----------------------------------------------------------------------

    def resample(self, owave=None, verbose=True):
        """
        Resample the spectrum onto a new wavelength grid.
        There are two possibilities for the output wavelength vector that
         sets where the interpolation happens.  They are:
         1. owave = None [default]
            A linearized set of spacings between the minimum and maximum
            values in the input wavelength vector
         2. owave is set to an array
            A user-defined x array that has been passed through the owave
            parameter

        This is just a specialized call to the Data1d.resamp method

        """

        self.rswav, self.rsflux = self.resamp(xout=owave, verbose=verbose)

    # -----------------------------------------------------------------------

    def mask_line(self, linereg, bkgdwidth, mode='input', atm_corr=False,
                  **kwargs):
        """

        Replaces the region of a spectrum containing a spectral line with
        a simple model of the continuum level in the location of that line.
        This is done by fitting a 2nd order polynomial to small regions
        of the spectrum immediately to the blue and red sides of the line
        regions.

        Required inputs:
          linereg   - A two-element list, array, or tuple that contains the
                      starting and ending wavelenghts for the line region
          bkgdwidth - Width, in wavelength units, of the regions of the
                       spectrum immediately to the left and right of the line
                       region that is used to estimate the continuum level.
                      This can either be a single number (same width for
                       the continuum region on both sides) or a pair of
                       numbers in a list, array or tuple (one for the
                       blue-side width and one for the red-side width)
        """

        """
        Set the arrays to use based on the passed mode variable.
        The default is to just use the unmodified input spectrum
         (mode='input')
        """
        spec = self.select_mode(mode)

        """ Set up short-cut names for the relevant arrays """
        w = spec['wav']
        f = spec['flux']

        """
        Make the continuum region width into a two-element quantity if
        only a single width was provided
        """
        if isinstance(bkgdwidth, float) or isinstance(bkgdwidth, int):
            cwidth = [bkgdwidth, bkgdwidth]
        else:
            cwidth = bkgdwidth

        """ Select the regions that will be used to fit the continuum """
        lmin = linereg[0]
        lmax = linereg[1]
        maskb = (w > lmin-cwidth[0]) & (w < lmin)
        maskr = (w > lmax) & (w < lmax+cwidth[1])
        mask = maskb | maskr

        """ Fit a polynomial to the continuum """
        ww = w[mask]
        ff = f[mask]
        mod = np.polyfit(ww, ff, 2)
        fmod = np.polyval(mod, w)

        """ Replace the line flux with the corresponding continuum values """
        linemask = (w >= lmin) & (w <= lmax)
        spec['flux'][linemask] = fmod[linemask]

    # -----------------------------------------------------------------------

    def fit_continuum(self, mode='input', maskreg=None, smo=51,
                      atm_corr=False, atm='model', airmass=1.0):
        """

        Fits a continuum to a spectrum.  The steps involved are:
         1. Do an atmospheric transmission correction if requested
         2. Smooth the spectrum, with a kernel width set by the smo parameter
         3. Fit to the smoothed spectrum, excluding any regions set by
            the maskreg parameter

        Inputs:
           mode     - which spectrum to use
           maskreg  - region(s) of the spectrum that will NOT be used in
                      the fitting.
                      These regions are designated by (start, end) pairs.
                      For example:
                         maskreg = [(6850, 7000), (7540, 7610)]
                         maskreg = [[6850, 7000], [7540, 7610]]
           atm_corr - do an atmospheric correction before fitting to the
           atm      - the atmopheric transmission to use for the correction
        """

        """ Do the atmospheric correction if requested """
        if atm_corr:
            self.atmcorr(airmass, atm)

        """ Smooth the spectrum """
        """ [NOT DONE YET] """
        print('Warning: Not done yet')
        return

    # -----------------------------------------------------------------------

    def normalize(self, mode='smooth', smo=25, mask=None):
        """
        Normalizes the spectrum via one of the following methods:
          1. highly smoothing the spectrum and then dividing by that
          2. NOT YET AVAILABLE
        """

        """
        If the spectrum has been smoothed already, save the smoothed spectrum
        """
        if self.ysmooth is not None:
            tmpsmooth = self.ysmooth.copy()
        else:
            tmpsmooth = None
        if self.varsmooth is not None:
            tmpvsmooth = self.varsmooth.copy()
        else:
            tmpvsmooth = None

        """ Select the good data if a mask is set """
        if mask is not None:
            w = self['wav'][mask]
            f = self['flux'][mask]
            if 'var' in self.colnames:
                v = self['var'][mask]
            else:
                v = None
        else:
            w = self['wav'].copy()
            f = self['flux'].copy()
            if 'var' in self.colnames:
                v = self['var'].copy()
            else:
                v = None
        data = df.Data1d(w, f, v)

        """ Smooth the spectrum """
        ry, rv = data.smooth_boxcar(smo)

        return w, ry

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
          4. 'fitstab'  - produces a fits file with the spectral data stored
                          as a binary table in the FITSRec format.
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
            if 'var' in self.colnames:
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
            if 'var' in self.colnames:
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
            # CHANGE TO JUST USING THE WRITE() METHOD FOR THE TABLE HERE!
            if 'var' in self.colnames:
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

        elif outformat == 'fitstab':
            tmp = Table(self)
            tmp.write(outfile, format='fits', overwrite=True)

        else:
            print('')
            print('ERROR: outformat is not one of the recognized types')
            print('')
            raise ValueError

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
    print('')
    print((os.path.split(__file__))[0])
    moddir = os.path.split(__file__)[0]
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
        print('Model starting wavelength: %f' % wstart)
        print('Model ending wavelength:    %f' % wend)
        print('Model dispersion:             %f' % disp)

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
