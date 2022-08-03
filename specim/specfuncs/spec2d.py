"""
spec2d.py

This file contains the Spec2d class, which is used to process and plot
two-dimensional spectroscopic data.  This processing includes sky subtraction
and extracting a one-dimensional spectrum from the 2D spectrum.

Typical methods that are used to do an extraction:
   display_spec
   spatial_profile
   initial_model
   refined_model
   find_and_trace
   extract

"""

from math import sqrt, pi

import numpy as np
from scipy.ndimage import filters
from scipy.special import gamma
import matplotlib.pyplot as plt

from astropy.io import fits as pf
from astropy.table import Table
from astropy.modeling import models

from cdfutils import datafuncs as df
from .. import imfuncs as imf
from .spec1d import Spec1d

import sys
pyversion = sys.version_info.major

# ===========================================================================
#
# Start of Spec2d class
#
# ===========================================================================


class Spec2d(imf.Image):
    """
    A class to process 2-dimensional spectra, i.e., the CCD data that
     comes out of a typical spectrograph.
    The main purpose of this Spec2d class and its associated functions is to
     extract a 1-dimensional spectrum from a 2-dimensional spectrum.
     The extracted 1-dimensional spectrum will, in the end, be output into a
     file that can be analyzed using the Spec1d class.
    NOTE: Spec2d inherits the properties of the Image class that is defined
     in imfuncs.py

    Example of standard processing on a spectrum within this class:
      - myspec = Spec2d('myspec2d.fits')
      - myspec.display_spec()
      - myspec.spatial_profile()
    Depending on how the spatial profile looks one may or mayn't want to build
    an initial model. If one wants to build an initial model then should call
      - myspec.initial_model()
    If one further imposes constraints on initial model and wants to see how 
    it fits to spatial profile should call
      - myspec.refined_model(constrained_init_model)
    One can call find_and_trace method with or without a model. If no model
    is provided then a single gaussian profile is adopted as a model. Next
      - myspec.find_and_trace(refined_model)
      - myspec.extract(outfile='myspec1d.txt')
    """

    def __init__(self, inspec, hext=0, invar=None, varext=None,
                 xtrim=None, ytrim=None, transpose=False, fixnans=True,
                 nanval='sky', logwav=False, verbose=True,
                 wcsverb=False):
        """
        Reads in the 2-dimensional spectrum from an input fits file (or the
        HDUList from a previously loaded fits file) and
        stores it in a Spec2d class container.

        Required inputs:
            inspec  - The input spectrum.  This can either be:
                      1. a filename, the most common case
                            - or -
                      2. a HDU list.  It is sometimes the case, e.g., with
                      multi-extension fits files such as those produced by ESI,
                      that the fits file has already been loaded.  In this case
                      it is more efficient to first read the HDU list from the
                      input file (external to this class) and then pass that
                      HDU list and a desired HDU (set by the hext parameter,
                      see below) to Spec2d instead.
                      For example, the Esi2d class does this.

        Optional inputs:
            hext      - The header-data unit (HDU) that contains the
                        2-dimensional spectroscopic data.  The default value
                        (hdu=0) should work for most fits files.
            invar     - If the 2d variance spectrum has already been computed
                        by previous reduction steps and stored as a separate
                        external file, then it needs to be read in.
                        Default value is None, implying no external variance
                        file.
                        If set, this can either be a filename or a hdulist if
                        the file has already been opened.
            xtrim     - Change from the default value (None) if the input
                        spectrum needs to be trimmed along the x-axis.
                        Example format for trimming:  xtrim=[300,701]
            ytrim     - Change from the default value (None) if the input
                        spectrum needs to be trimmed along the y-axis.
                        Example format for trimming:  ytrim=[300,701]
            transpose - If transpose=True, transpose the x and y dimensions of
                        the input spectrum.  This is done, e.g., to change the
                        dispersion axis from vertical to horizontal.
                        NOTE: If transpose=True, the transpose action happens
                        AFTER any trimming that is done if xtrim and/or ytrim
                        have a value different from None
                        Default = False.
            verbose   - Set to True (the default) for information about the
                        input file to be printed.
        """

        """ Initialize some variables """
        self.dispaxis = 'x'
        self.specaxis = 1
        self.spaceaxis = 0
        self.vardata = None
        self.sky1d = None
        self.spec1d = None
        self.fitrange = None
        self.profile = None
        self.profcent = None
        self.ap = None
        self.apmin = -4.
        self.apmax = 4.
        self.muorder = 3
        self.sigorder = 3
        self.mod0 = None
        self.logwav = logwav

        """
        Read in the data and call the superclass initialization for useful
        Image attributes
        """
        if pyversion == 2:
            super(Spec2d, self).__init__(inspec, hext=hext, vardat=invar,
                                         varext=varext, verbose=verbose,
                                         wcsverb=wcsverb)
        else:
            super().__init__(inspec, hext=hext, vardat=invar,
                             varext=varext, verbose=verbose,
                             wcsverb=wcsverb)

        """ Set the portion of the input spectrum that should be used """
        nx = self.header['naxis1']
        ny = self.header['naxis2']
        trimmed = False
        if xtrim is not None:
            xmin = xtrim[0]
            xmax = xtrim[1]+1
            trimmed = True
        else:
            xmin = 0
            xmax = nx
        if ytrim is not None:
            ymin = ytrim[0]
            ymax = ytrim[1]+1
            trimmed = True
        else:
            ymin = 0
            ymax = ny

        """ Put the data in the appropriate container """
        data = self.data[ymin:ymax, xmin:xmax]
        if transpose:
            self.data = data.transpose()
        else:
            self.data = data

        """
        Do the same thing for the external variance file, if there is one
        ASSUMPTION: the external variance file (if it exists) is the same size
         as the data file and thus should be trimmed, transposed, etc. in the
         same way
        """
        if self['var'] is not None:
            vdata = self['var'].data[ymin:ymax, xmin:xmax]
            if transpose:
                self.vardata = vdata.transpose()
            else:
                self.vardata = vdata

        """ Set other variables and report results """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        if verbose:
            if self.infile is None:
                print('Read in 2-dimensional spectrum from HDU=%d' % hext)
            else:
                print('Read in 2-dimensional spectrum from %s (HDU=%d)' % \
                     (self.infile, hext))
            if trimmed:
                print('The input dataset was trimmed')
                print(' xrange: %d:%d.  yrange: %d:%d' % 
                      (xmin, xmax, ymin, ymax))
            if transpose:
                print('The input dataset was transposed')
            print('Final data dimensions (x y): %d x %d' % 
                  (self.data.shape[1], self.data.shape[0]))
        self.get_dispaxis(verbose)

        """
        Check for NaN's within the spectrum and replace them if they are there
        """
        if fixnans:
            self.fix_nans_spec(verbose=True)

    # -----------------------------------------------------------------------

    def get_dispaxis(self, verbose=True):
        """
        The dispersion axis is the axis corresponding to the spectral direction
        in a 2-dimensional spectrum.  get_dispaxis does the simple task of
        showing the current value of the dispaxis variable, either 'x' or 'y'
        """

        self.npix = self.data.shape[self.specaxis]
        self.nspat = self.data.shape[self.spaceaxis]
        if verbose:
            print('')
            print('Dispersion axis:              %s' % self.dispaxis)
            print('N_pixels along dispersion axis: %d' % self.npix)
            print('')

    # -----------------------------------------------------------------------

    def set_dispaxis(self, dispaxis):
        """
        The dispersion axis is the axis corresponding to the spectral direction
        in a 2-dimensional spectrum.
        set_dispaxis is used to change the value of the dispaxis variable.

        For example, if the 2d spectrum was loaded as:
          myspec = Spec2d('myspec.fits')
        then to change the dispersion axis from x to y (the only two choices)
        type:
          myspec.set_dispaxis('y')

        Required input:
            dispaxis - Dispersion axis: 'x' and 'y' are the only two possible
                          choices
        """

        oldval = self.dispaxis
        if dispaxis == 'x' or dispaxis == 'y':
            self.dispaxis = dispaxis
            if self.dispaxis == "y":
                self.specaxis = 0
                self.spaceaxis = 1
            else:
                self.specaxis = 1
                self.spaceaxis = 0
            print('')
            print('Old value of dispaxis: %s' % oldval)
            self.get_dispaxis()
            print('')
            return
        else:
            print('')
            print("ERROR: dispaxis must be either 'x' or 'y'")
            print('%s is not a valid value' % dispaxis)
            print('')
            print('Keeping current value for dispaxis:  %s' % self.dispaxis)
            print('')
            return

    # -----------------------------------------------------------------------

    def get_wavelength(self, verbose=False):
        """
        Gets a wavelength vector from the fits header, if it exists

        """

        if self.dispaxis == 'y':
            dim = 2
        else:
            dim = 1
        cdkey = 'cd%d_%d' % (dim, dim)
        crpix = 'crpix%d' % dim
        crval = 'crval%d' % dim
        hdr = self.header
        # print cdkey, crpix, crval
        self.has_cdmatx = True
        try:
            dw = hdr[cdkey]
        except KeyError:
            self.has_cdmatx = False
            dw = 1
        try:
            wstart = hdr[crval]
        except KeyError:
            self.has_cdmatx = False
            wstart = 0
        try:
            wpix = hdr[crpix] - self.xmin - 1
        except KeyError:
            self.has_cdmatx = False
            wpix = 0

        """ Create the wavelength vector from the information above """
        self.wavelength = wstart + (np.arange(self.npix) - wpix) * dw
        if self.logwav:
            self.wavelength = 10.**self.wavelength
        if verbose:
            print(self.wavelength)

    # -----------------------------------------------------------------------

    def fix_nans_spec(self, nanval='sky', verbose=False):
        """
        Detects NaN's within the spectrum and replaces them with real numbers
        if they are there.
        """

        nanmask = np.isnan(self.data)
        nnan = nanmask.sum()
        if nnan > 0:
            if verbose:
                print('Found %d NaNs in the two-dimensional spectrum' % nnan)

            """ First replace the NaNs with a temporary value """
            self.data[nanmask] = -999

            """
            Now find the median sky values by calling the subtract_sky_2d
            method
            """
            self.subtract_sky_2d()

            """
            Finally, replace the NaNs with the median sky for their row/column
            """
            self.data[nanmask] = self['sky2d'].data[nanmask]

    # -----------------------------------------------------------------------

    def subtract_sky_2d(self, outfile=None, outsky=None):
        """
        Given the input 2D spectrum, creates a median sky and then subtracts
        it from the input data.  Two outputs are saved: the 2D sky-subtracted
        data and a 1D sky spectrum.

        Optional inputs:
        data       - array containing the 2D spectrum
        outfile    - name for output fits file containing sky-subtracted
                     spectrum
        outskyspec - name for output 1D sky spectrum
        """

        """ Set the dispersion axis direction """
        if self.dispaxis == 'y':
            spaceaxis = 1
        else:
            spaceaxis = 0

        """ Take the median along the spatial direction to estimate the sky """
        if self.data.ndim < 2:
            print('')
            print('ERROR: subtract_sky needs a 2 dimensional data set')
            return
        else:
            pix = np.arange(self.npix)
            tmp1d = np.median(self.data, axis=spaceaxis)
            self.sky1d = Spec1d(wav=pix, flux=tmp1d)

        """ Turn the 1-dimension sky spectrum into a 2-dimensional form """
        # sky2d = np.zeros(self.data.shape)
        # for i in range(self.nspat):
        #     sky2
        sky2d = np.tile(self.sky1d['flux'].data,
                        (self.data.shape[spaceaxis], 1))
        self['sky2d'] = imf.WcsHDU(sky2d, wcsverb=False)

        """ Subtract the sky from the data """
        skysub = self.data - sky2d
        self['skysub'] = imf.WcsHDU(skysub, wcsverb=False)

        """ Clean up """
        del sky2d, skysub

    # -----------------------------------------------------------------------

    def szap(self, outfile, sigmax=5., boxsize=7):
        """

        Rejects cosmic rays from a 2D spectrum via the following
        1. Creates the median sky from the spectrum
        2. Subtracts the sky from the spectrum
        3. Divides the subtracted spectrum by the square root of the sky, which
           gives it a constant rms
        4. Rejects pixels that exceed a certain number of sigma
        5. Adds the sky back in

        """

        """ Subtract the sky  """
        self.subtract_sky_2d()
        skysub = self['skysub'].data.copy()

        """
        Divide the result by the square root of the sky to get a rms image
        """
        ssrms = skysub / np.sqrt(self['sky2d'].data)
        m, s = df.sigclip(ssrms)

        """ Now subtract a median-filtered version of the spectrum """
        tmpsub = ssrms.data - filters.median_filter(ssrms, boxsize)

        """
        Make a bad pixel mask that contains pixels in tmpsub with
         values > sigmax*s
        """
        mask = tmpsub > sigmax * s
        tmpsub[mask] = m

        """ Replace the bad pixels in skysub with a median-filtered value """
        self.sigma_clip('skysub')
        skysub[mask] = self.mean_clip
        ssfilt = filters.median_filter(skysub, boxsize)
        skysub[mask] = ssfilt[mask]

        """ Add the sky back in and save the final result """
        szapped = skysub + self['sky2d'].data
        pf.PrimaryHDU(szapped).writeto(outfile)
        print(' Wrote szapped data to %s' % outfile)

        """ Clean up """
        del skysub, ssrms, tmpsub, szapped

    # -----------------------------------------------------------------------

    def display_spec(self, doskysub=True):
        """
        Displays the two-dimensional spectrum and also, by default, the
        same spectrum after a crude sky subtraction.  To show only the
        input spectrum without the additional sky-subtracted version,
        just set doskysub=False

        Optional inputs:
            doskysub - If this is True (the default) then make a second
                       plot, showing the 2-D spectrum after a crude
                       sky-subtraction has been performed.
        """

        if doskysub:

            """ Subtract the sky if this has not already been done """
            if 'skysub' not in self.keys():
                self.subtract_sky_2d()

            """ Set the subplot designation for the main spectrum """
            pltnum_main = 411

            """ Get rid of the space between the subplots"""
            plt.subplots_adjust(hspace=0.001)

        else:
            """ If no sky subtraction, then we just have one plot """
            pltnum_main = 111

        """ Plot the input spectrum """
        ax1 = plt.subplot(pltnum_main)
        self.display(mode='xy', axlabel=False)

        """ Plot the subtracted sky spectrum if desired """
        if doskysub:
            """ First get rid of the x-axis tick labels for main plot """
            ax1.set_xticklabels([])

            """ Plot the sky-subtracted 2D spectrum """
            ax2 = plt.subplot(412, sharex=ax1, sharey=ax1)
            self.found_rms = False
            self.display(dmode='skysub', mode='xy')

            """ Plot an estimate of the 1D sky spectrum """
            ax3 = plt.subplot(212, sharex=ax1)
            self.sky1d.plot(title=None, xlabel='x (pix)', ax=ax3)
        self.found_rms = False

        """
        For ease of viewing, only display part of the spectrum if it is
        much longer in one dimension than the other
        """
        sfac = 7.5
        if self.npix > sfac * self.nspat:
            xmin = int(self.npix / 2. - (sfac/2.) * self.nspat)
            xmax = int(self.npix / 2. + (sfac/2.) * self.nspat)
            ax3.set_xlim(xmin, xmax)

            """ Scale the portion of the spectrum that is being displayed """
            w = self.sky1d['wav']
            flux = self.sky1d['flux'][(w >= xmin) & (w <= xmax)]
            ymin, ymax = ax3.get_ylim()
            ydiff = flux.max() - ymin
            ax3.set_ylim(ymin, (ymin + 1.05 * ydiff))

    # -----------------------------------------------------------------------

    def compress_spec(self, pixrange=None):
        """
        Compresses the spectrum along the spectral direction to get a
         spatial profile
        """

        """ Set the data range in which to find the trace """
        if pixrange is not None:
            if self.data.ndim < 2:
                tmpdat = self.data[pixrange[0]:pixrange[1]]
            else:
                if self.specaxis == 0:
                    tmpdat = self.data[pixrange[0]:pixrange[1], :]
                else:
                    tmpdat = self.data[:, pixrange[0]:pixrange[1]]
            # print pixrange
        else:
            tmpdat = self.data.copy()

        """
        Compress the data along the dispersion axis
        """
        if self.data.ndim < 2:
            pflux = tmpdat
        else:
            pflux = np.median(tmpdat, axis=self.specaxis)

        """ Return the compressed spectrum """
        return pflux
    
    # -----------------------------------------------------------------------

    def spatial_profile(self, pixrange=None, doplot=True, pixscale=None,
                        title='Spatial Profile', model=None, normalize=False,
                        showap=True, verbose=True, debug=False, **kwargs):
        """
        Compresses a 2d spectrum along the dispersion axis to create a spatial
         profile, and then plots it if requested
        """

        color = 'b'

        """
        Compress the data along the dispersion axis and find its max value
        """
        pflux = self.compress_spec(pixrange)
        pmax = pflux.max()
        if verbose:
            print('Profile max value (before normalization) %f' % pmax)

        """ Normalize the profile if requested """
        if normalize:
            pflux /= pmax
            pmax = 1.0

        """ Save the profile as a Spec1d instance """
        px = np.arange(pflux.shape[0])
        if pixscale is not None:
            wav = px * pixscale
            units = 'arcsec'
        else:
            wav = px
            units = 'pix'
        profile = Spec1d(wav=wav, flux=pflux, verbose=debug)

        """
        Plot the compressed spectrum, showing the best-fit Gaussian if
        requested
        """
        if(doplot):
            xlab = 'Spatial direction (%s)' % units
            profile.plot(color=color, title=title, xlabel=xlab, model=model,
                         showzero=False, **kwargs)
            if showap:
                if self.profcent is not None:
                    plt.axvline(self.profcent + self.apmin, color='k')
                    plt.axvline(self.profcent + self.apmax, color='k')

        return profile

    # -----------------------------------------------------------------------

    def initial_model(self, profile=None, verbose=True):

        """
        Creates an initial model to fit with spatial profile from user input
        of parameter values. Takes a polynomial upto 2nd degree and any number
        of Gaussian and/or Moffat profile. And iterates unitl the user wants
        to quit.

        Required input:
            profile : a spatial profile of the 2d spectra produced using the
                      function spatial_profile(). If no profile is
                      provided then profile inherited from spatial_profile()
                      is used.
        Output:
            mod : retuns the final fitted model
        """

        print("\nTo create an initial model you first need to enter degree " \
              "of background polynomial(<3) and number of Gaussian and Moffat " \
              "profile as integers.")

        while True:

            background_order = int(input('Order of background polynomial : '))
            num_gauss = int(input('Number of Gaussian profile : '))
            num_moffat = int(input('Number of Moffat profile : '))

            bg_const = []
            parm_gauss = []
            parm_moffat = []

            """Collect parameter inputs for background polynomial and build a
               polynomial from those inputs."""

            if background_order<3:
                print("\nNow enter constants of the background polynomial upto " \
                      "degree %d" %background_order)

                for i in range(background_order+1):
                    bg_const.append(float(input('c%d : '%i)))

                if background_order==0:
                    b = models.Polynomial1D(degree=0, c0=bg_const[0])

                elif background_order==1:
                    b = models.Polynomial1D(degree=1, c0=bg_const[0], c1=bg_const[1])

                elif background_order==2:
                    b = models.Polynomial1D(degree=2, c0=bg_const[0], c1=bg_const[1],
                                                                      c2=bg_const[2])
            else:
                print("\ndegree of background polynomial is expected to be less " \
                      "than 3...start over")
                continue

            """ Collect parameters for Gaussian profiles, and build and add Gaussian
                profiles using those input parameters."""

            if num_gauss>0:

                print("\nEnter parameter values for a Gaussian profile in " \
                      "following format...\namplitude_value, mu_value, sigma_value")

                for i in range(num_gauss):
                    amp, mu, sigma = input('\nParameters of Gaussian profile %d : '
                                                                 %(i+1)).split(',')
                    try:
                        p = [float(amp), float(mu), float(sigma)]
                        parm_gauss.append(p)
                        print('\namplitude=%f, mu=%f, sigma=%f' %(p[0], p[1], p[2]))
                    except:
                        print('Error : parameter values should be integer or float')

                for i, p in enumerate(parm_gauss):
                    if i==0:
                        g = models.Gaussian1D(amplitude=p[0], mean=p[1], stddev=p[2])
                    else:
                        g1 = models.Gaussian1D(amplitude=p[0], mean=p[1], stddev=p[2])

                        g += g1

            """ Collect parameters for Moffat profiles, and build and add Moffat
                profiles using those input parameters."""

            if num_moffat>0:

                print("\nEnter parameter values for a Moffat profile in following " \
                      "format...\namplitude_value, x_0_value, gamma_value, alpha_value")

                for i in range(num_moffat):
                    amp, x0, gam, alp = input('\nParameters of Moffat profile %d : '
                                                                    %(i+1)).split(',')
                    try:
                        p = [float(amp), float(x0), float(gam), float(alp)]
                        parm_moffat.append(p)
                        print('\namplitude=%f, x_0=%f, gamma=%f, alpha=%f'
                                                           %(p[0], p[1], p[2], p[3]))
                    except:
                        print('Error : parameter values should be integer or float')

                for i, p in enumerate(parm_moffat):
                    if i==0:
                        m = models.Moffat1D(amplitude=p[0], x_0=p[1], gamma=p[2],
                                                                        alpha=p[3])
                    else:
                        m1 = models.Moffat1D(amplitude=p[0], x_0=p[1], gamma=p[2],
                                                                        alpha=p[3])

                        m += m1

            """ Create a final model adding background polynomial, Gaussian and 
                Moffat profiles,"""

            if num_gauss>0 and num_moffat>0:
                init_mod = b + g + m

            else:
                if num_gauss>0:
                    init_mod = b + g
                else:
                    init_mod = b + m

            """It is expected that a spatial profile has been already created"""

            if profile is None:
                profile = self.profile

            """Fit the model to the spatial profile.""" 

            mod, fit_info = profile.fit_mod(init_mod, verbose=False)
            diff = profile.y - mod(profile.x)

            """Print and plot the fitted model"""

            if verbose:
                print('\nFitted model')
                print('-------------')
                print(mod)
                print('\n-------------------------------------------\n')

            xlab = 'Spatial direction (0-indexed)'
            title = 'Fit to Spatial Profile'
            fig = plt.figure()
            frame1=fig.add_axes((.1,.3,.8,.6))
            plt.plot(profile.x, profile.y, color='b', linestyle='solid',
                                  drawstyle='steps', label='Spatial profile')
            plt.plot(profile.x, mod(profile.x), color='g', drawstyle='steps',
                                                           label='model fit')
            plt.ylabel('Relative flux')
            plt.legend()
            plt.title(title)

            frame2=fig.add_axes((.1,.1,.8,.2))
            plt.plot(profile.x, diff, 'r', drawstyle='steps')
            plt.hlines(y=0, xmin=0, xmax=max(profile.x))
            plt.ylabel('Difference')
            plt.xlabel(xlab)
            #plt.show()

            plt.figure()
            plt.plot(profile.x, profile.y, color='b', linestyle='solid',
                                  drawstyle='steps', label='Spatial profile')
            label_g = True
            label_m = True
            for i, md in enumerate(mod):
                if isinstance(md, models.Gaussian1D):
                    if label_g:
                        plt.plot(profile.x, mod[i](profile.x), color='k',
                             drawstyle='steps', label='Gaussian prof. in fit')
                        label_g = False
                    else:
                        plt.plot(profile.x, mod[i](profile.x), color='k',
                                                              drawstyle='steps')
                elif isinstance(md, models.Moffat1D):
                    if label_m:
                        plt.plot(profile.x, mod[i](profile.x), color='r',
                                 drawstyle='steps', label='Moffat prof. in fit')
                        label_m = False
                    else:
                        plt.plot(profile.x, mod[i](profile.x), color='r',
                                                              drawstyle='steps')
            plt.legend()
            plt.xlabel('Spatial direction (0-indexed)')
            plt.ylabel('Relative Flux')
            plt.title('Individual profile component in fitted model')
            #plt.show()

            """Check whether user want to create a new model or done with building
               a model."""

            prompt = input('\nDo you want to start over ? : type yes/no\n'
                                                                   ).strip().lower()
            if prompt=='no':
                break

        self.mod0 = mod

        """Retrun the latest model"""
        return mod

    # -----------------------------------------------------------------------

    def locate_trace(self, pixrange=None, init=None, fix=None,
                     doplot=True, ngauss=1, axes=None,
                     title='Spatial Profile', verbose=True,
                     **kwargs):
        """
        Compresses a 2d spectrum along the dispersion axis so that
         the trace of the spectrum can be automatically located by fitting
         a gaussian + background to the spatial direction.  The function
         returns the parameters of the best-fit gaussian.
        The default dispersion axis is along the x direction.  To change this
         set the dispaxis to "y" with the set_dispaxis method in this Spec2d
         class.
        """

        """ Start by compressing the data, but don't show it yet """
        profile = self.spatial_profile(pixrange, doplot=False, 
                                       verbose=verbose)

        """
        Fit a shape -- most commonly a single Gaussian -- to the spatial
        profile
        """

        modtype = '1gauss'

        if modtype == '1gauss':
            """  Fit a Gaussian plus background to the profile """
            mod, fitinfo = profile.fit_gauss(mod0=init, verbose=verbose)
            self.profcent = mod[1].mean

        """ Now plot the spatial profile, showing the best fit """
        if doplot:
            if axes is not None:
                ax = axes[0]
            else:
                ax = plt.subplot(111)
            xlab = 'Spatial direction (0-indexed)'
            profile.plot(title=title, xlabel=xlab, model=mod, showzero=False,
                         ax=ax, **kwargs)
            ax.axvline(self.profcent + self.apmin, color='k')
            ax.axvline(self.profcent + self.apmax, color='k')

        """ Return the model """
        return profile, mod

    # -----------------------------------------------------------------------

    def fit_slices(self, mod0, stepsize, ncomp=1, bgorder=0, fitrange=None,
                   usevar=None, mu0arr=None, sig0arr=None, verbose=True,
                   debug=False):
        """

        Steps through the 2d spectrum, fitting a model to a series of
        spatial profiles, one for each slice.  The model outputs for each
        slice get stored in arrays that are returned by this method.

        """

        """
        Define the slices through the 2D spectrum that will be used as the
         inputs to the spatial profile modeling
        """
        if stepsize == 'default':
            nbins = 25
            stepsize = int(self.npix / nbins)
        if stepsize == 1:
            xstep = np.arange(self.npix)
        else:
            xstep = np.arange(0, self.npix-stepsize, stepsize)
        if verbose:
            print('Fitting to the trace at %d segments' % len(xstep))
            print('  of the 2D spectrum with stepsize=%d pix ...' % stepsize)

        """
        Set up containers for parameter values and covariances along the trace
        """
        partab = Table()
        partab['x'] = xstep
        for param in mod0.param_names:
            partab[param] = np.zeros(xstep.size)
        partab['flux'] = np.zeros(xstep.size)
        covar = []
        
        """ Make a temporary profile that will be used in the fitting  """
        tmpprof = Spec1d((Table(self.profile).copy()), verbose=False)

        """
        Step through the data, slice by slice, fitting to the spatial profile
        in each slice along the way and storing the results in the parameter
        table
        """
        for i, x in enumerate(xstep):
            """ Create the data slice"""
            pixrange = [x, x+stepsize]
            tmpprof.y = self.compress_spec(pixrange)

            """ Fix the input model parameters (this code will change) """
            if mu0arr is not None:
                mod0[1].mean = mu0arr[i]
            if sig0arr is not None:
                mod0[1].stddev = sig0arr[i]

            """ Do the model fitting """
            mod, fitinfo = tmpprof.fit_mod(mod0=mod0, usevar=usevar,
                                           verbose=debug)

            """
            Store the fitted parameters in the parameter table and the
            covariances in the covar list
            """
            for param, val in zip(mod.param_names, mod.parameters):
                partab[param][i] = val
            covar.append(fitinfo['param_cov'])

        """
        Compute the integrated fluxes and then return all the fitted
        parameters
        """
        # for i in range(1, ncomp+1):
        #     partab['flux_%d' % i] = sqrt(2. * pi) * partab['stddev_%d' %i] \
        #         * partab['amplitude_%d' %i]
        return partab, covar
    
    # -----------------------------------------------------------------------

    def fit_poly_to_trace(self, coarsepars, mod0, fitorder=None,
                          fitrange=None, nsig=3.0, doplot=True,
                          markformat='bo', ylabel='default',
                          title='default', axes=None, verbose=True):

        """
        Select the region of the spectrum to use when fitting polynomials
        to the trace parameters
        """
        x0 = coarsepars['x']
        if fitrange is None:
            xmask = np.ones(len(x0), dtype=bool)
            fittab = coarsepars.copy()
        else:
            xmask = np.logical_and(x0 >= fitrange[0], x0 < fitrange[1])
            fittab = coarsepars[xmask]

        """
        Set up containers to hold the polynomial fit parameters and the
        points that were excluded from the polynomial fits
        """
        polypars = {}
        exclude_masks = {}

        """
        Loop over the parameters and only fit to the ones that define
        centroid or shape
        """
        fitinfo = {}
        count = 1
        for p in coarsepars.colnames:

            """ Select the location and shape parameters only """
            if p[:4] == 'mean' or p[:6] == 'stddev' or p[:2] == 'x0':
                    
                """ Select the column to be fitted """
                data0 = coarsepars[p]
                
                """ Do a sigma clipping to reject clear outliers """
                dmu, dsig = df.sigclip(data0[xmask], nsig=nsig)
                goodmask = np.absolute(data0 - dmu) < nsig * dsig

                """
                Select the points to be fitted by combining the two masks
                """
                fitmask = np.logical_and(xmask, goodmask)
                exclude_masks[p] = np.logical_not(fitmask)
                x = coarsepars['x'][fitmask]
                data = data0[fitmask]
                fitinfo[p] = [x, data]

                """ Get the order of the polynomial to be fit """
                if isinstance(fitorder, dict) and p in fitorder.keys():
                    polyorder = int(fitorder[p])
                else:
                    polyorder = 3

                """ Fit a polynomial to the trace parameter """
                if polyorder == -1:
                    polypars[p] = np.nan
                    for n0, p0 in zip(mod0.param_names, mod0.parameters):
                        if n0 == p:
                            polypars[p] = np.array([p0])
                    if verbose and np.isfinite(polypars[p]):
                        print('Setting %s to value from initial fit to '
                              'the spatial profile' % p)
                else:
                    if verbose:
                        print('Fitting polynomial of degree %d to parameter:'
                              ' %s' % (polyorder, p))
                    polypars[p] = np.polyfit(x, data, polyorder)

        if doplot:
            print('Plotting centroid and width of model component 1')
            if axes is not None:
                ax1 = axes[1]
                ax2 = axes[2]
            else:
                ax1 = plt.subplot(111)
                ax2 = plt.subplot(111)
                
            """ Get the data for the centroid """
            pars = []
            modpars = []
            if 'mean_1' in fitinfo.keys():
                pars.append('mean_1')
                modpars.append('mean')
            elif 'x0_1' in fitinfo.keys():
                pars.append('x0_1')
                modpars.append('x0') 
            else:
                pars.append(None)
                modpars.append(None)

            """ Get the data for the width """
            if 'stddev_1' in fitinfo.keys():
                pars.append('stddev_1')
                modpars.append('stddev_1')
            else:
                pars.append(None)
                modpars.append(None)

            """  Plot the fits """
            xfit = np.arange(self.npix).astype(float)
            ptstyle = ['bo', 'g^']
            axislist = [ax1, ax2]
            for par, modpar, style, ax in zip(pars, modpars, ptstyle,
                                              axislist):
                if par is not None:
                    x1 = fitinfo[par][0]
                    y1 = fitinfo[par][1]
                    yfit = np.polyval(polypars[par], xfit)
                    # y0 = mod0[modpar].value
                    ax.plot(x1, y1, style)
                    # ax.axhline(y0, color='k', linestyle='--')
                    ax.plot(xfit, yfit, 'r')
            # if widpar is not None:
            #     x2 = fitinfo[widpar][0]
            #     y2 = fitinfo[widpar][1]
            #     yfit = np.polyval(polypars[widpar], xfit)
            #     ax2.plot(x2, y2, 'g^')
            #     ax2.plot(xfit, yfit, 'r')
            # if centpar is not None:
            #     x1 = fitinfo[centpar][0]
            #     y1 = fitinfo[centpar][1]
            #     yfit = np.polyval(polypars[centpar], xfit)
            #     y0 = mod0[centpar].value
            #     ax1.plot(x1, y1, 'bo')
            #     ax1.axhline(y0, color='k', linestyle='--'
            #     ax1.plot(xfit, yfit, 'r')
            # if widpar is not None:
            #     x2 = fitinfo[widpar][0]
            #     y2 = fitinfo[widpar][1]
            #     yfit = np.polyval(polypars[widpar], xfit)
            #     ax2.plot(x2, y2, 'g^')
            #     ax2.plot(xfit, yfit, 'r')
            ax2.set_xlabel('%s (pixels)' % self.dispaxis)

        return polypars, exclude_masks
                
        """ Fit a polynomial to the trace """

        if fitorder == -1:
            dpoly[0] = data0

        """
        Calculate the fitted function
        REPLACE THIS WITH np.polyval
        """
        fitx = np.arange(self.npix).astype(np.float32)
        fity = 0.0 * fitx
        for i in range(dpoly.size):
            fity += dpoly[i] * fitx**(dpoly.size - 1 - i)

        """ Plot the results """
        ymin = dmu - 4.5*dsig
        ymax = dmu + 4.5*dsig
        if doplot:
            if axes is not None:
                ax = axes[1]
            else:
                ax = plt.subplot(111)
            ax.plot(x, data, markformat)
            ax.set_xlabel("Pixel (dispersion direction)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)

            """ Show the value from the compressed spatial profile """
            ax.axhline(data0, color='k', linestyle='--')

            """ Mark the bad points that were not included in the fit """
            ax.plot(xbad, dbad, "rx", markersize=10, markeredgewidth=2)

            """ Show the fitted function """
            ax.plot(fitx, fity, "r")

            """
            Show the range of points included in the fit, if fitrange was set
            """
            if fitrange is not None:
                ax.axvline(fitrange[0], color='k', linestyle=':')
                ax.axvline(fitrange[1], color='k', linestyle=':')
                xtmp = 0.5 * (fitrange[1] + fitrange[0])
                xerr = xtmp - fitrange[0]
                ytmp = fity.min() - 0.2 * fity.min()
                ax.errorbar(xtmp, ytmp, xerr=xerr, ecolor="g", capsize=10)
            ax.set_xlim(0, self.npix)
            ax.set_ylim(ymin, ymax)

        """
        Return the parameters produced by the fit and the fitted function
        """
        print(dpoly)
        return dpoly, fity

    # -----------------------------------------------------------------------

    def trace_spectrum(self, mod0, ngauss=1, stepsize='default', meantol=0.5,
                       fitrange=None, fitorder={'mean_1': 3, 'stddev_1': 4},
                       doplot=True, axes=None, verbose=True, debug=False):
        """
        Fits a gaussian plus background to the spatial profile of the spectrum
         This is done in binned segments, because for nearly all cases the SNR
         in a single column (assuming that y is the spatial axis) is much too
         low to get a good fit to the profile.  The bin size, in pixels in the
         dispersion direction, is set by the stepsize parameter.  Setting
         stepsize='default' (which is the default) means that the stepsize
         (in pixels) is selected automatically to give 75 steps across the
         spectrum

        The steps in this method are as follow:
         1. Obtain the parameters of the spatial fit in each bin and save them
             in the mustep and sigstep arrays
         2. Under the assumption that the parameters that describe the spatial
             profile vary slowly with wavelength, fit a polynomial to the
             values in the mustep and sigstep arrays.
             This polynomial will then provide the profile-description
             parameters for each individual column (or row, depending on the
             dispersion direction) in the spectrum.
             The order of the polynomials are set by the muorder and sigorder
             parameters. NOTE: values of -1 for these parameters mean to just
             take the values from the overall spatial profile and to not do
             this fitting exercise.
        """

        """ Make sure that a first-guess model has been passed """
        if mod0 is None:
            msg = 'Invalid input model.  Run locate_trace first'
            raise ValueError(msg)

        """ Trace the spectrum down the chip """
        if verbose:
            print('')
            print('Running fit_trace')
            print('---------------------------------------------------------')

        coarsepars, covar = self.fit_slices(mod0, stepsize, ncomp=ngauss,
                                            verbose=verbose)
        if debug:
            print(coarsepars)
        if verbose:
            print('    Done')

        """ Fit a polynomial to the location of the trace """
        polypars, exclude_masks = \
            self.fit_poly_to_trace(coarsepars, mod0, fitrange=fitrange,
                                   fitorder=fitorder, axes=axes)
        return polypars
    
    # -----------------------------------------------------------------------

    def make_prof2d(self, polypars, mod0):
        """

        Creates a two-dimensional profile based on the fits to the
         parameters that describe the shape of the trace as a function
         of position on the chip.
        This profile is used as one component of the weighting in the
         extraction of the spectrum if the default 'optimal' extraction
         method is chosen.

        Inputs:
         polypars - parameters of the polynomials that were fit to the
                     trace parameters produced in the trace_spectrum
                     method.  Polynomials are only fit to the trace location
                     (e.g., Gaussian mean) and shape (e.g., Gaussian
                     sigma), but not to the background level or amplitude
         mod0     - Original model fit to the spatial profile of the
                     trace.  This is used to define which type of
                     components will be used to generate the 2d profile
        """

        """
        Set up parameters that will be used to generate and hold the
        two-dimension profile
        """
        x = np.arange(self.npix)
        y2d, x2d = np.indices(self['input'].data.shape)
        profdat = np.zeros(self['input'].data.shape)
        profmods = []

        """
        Loop on the number of components (e.g., Gaussian1D) that make
         up the model fit to the profile.  Ignore the Polynomial1D component
         since it is used to fit to the sky level and won't be used in
         the 2d profile created by this method.
        """
        amp0 = -1
        for i, mod in enumerate(mod0):
            if isinstance(mod, models.Gaussian1D):
                if amp0 == -1:
                    amp0 = mod.amplitude.value

                """
                Use the polynomial parameters to generate the profile
                values at each wavelength step along the chip
                """
                amp = np.ones(self.npix) * mod.amplitude.value / amp0
                mean = np.polyval(polypars['mean_%d' % i], x)
                stddev = np.polyval(polypars['stddev_%d' % i], x)

                """ Create a model set using the profile values """
                mod = models.Gaussian1D(amplitude=amp, mean=mean,
                                        stddev=stddev, n_models=self.npix)

                """
                Use the model set to generate the output 2D profile.
                NOTE: The two transposes (set by the .T calls) are needed
                because the 'orientation' of the model set is the 
                opposite to the orientation of the data
                """
                profdat += (mod(y2d.T)).T
                profmods.append(mod)

        """ Normalize the profile in the spatial direction """
        Pnorm = (profdat.sum(axis=self.spaceaxis))
        newdim = (self.npix, self.nspat)
        Pnorm = Pnorm.repeat(self.nspat).reshape(newdim).T
        profdat /= Pnorm
        
        """
        Clean up and return the profile and the model sets used to
        generate it
        """
        del x, x2d, y2d
        return profdat, profmods
    
    # -----------------------------------------------------------------------

    def find_and_trace(self, ngauss=1, bgorder=0, stepsize='default',
                       fitorder={'mean_1': 3, 'stddev_1': 4},
                       fitrange=None, doplot=True, do_subplot=True,
                       axes=None, verbose=True):

        """
        The first step in the spectroscopy reduction process.

        The find_and_trace function will:
          1. Locate roughly where the target object is in the spatial direction
              (usually the y axis is the spatial direction) by taking the
              median in the spectral direction so the peak in the spatial
              direction stands out.  This step provides the initial guesses
              for the location (mu0) and width (sig0) of the peak that are
              then used in the second step.

              * This step is done by a call to the locate_trace method

          2. Once the rough location of the peak has been found, determines how
              its location and width change in the spectral direction.
              That is, this will find where the peak falls in each column.
              It returns the position (pos) and width (width) of the peak as
              a function of x location

              * This step is done by a call to the trace_spectrum method

        Inputs:
           stepsize
           muorder
           sigorder
        """

        """ Set up the plotting parameters """
        if do_subplot and axes is None:
            ax1 = plt.axes([0.05, 0.05, 0.25, 0.9])
            ax2 = plt.axes([0.35, 0.55, 0.6, 0.4])
            ax3 = plt.axes([0.35, 0.05, 0.6, 0.4])
            axes = [ax1, ax2, ax3]
            
        self.profile, self.mod0 = \
            self.locate_trace(doplot=doplot, ngauss=ngauss, pixrange=fitrange,
                              axes=axes, verbose=verbose)

        polypars = \
            self.trace_spectrum(self.mod0, ngauss=ngauss, stepsize=stepsize,
                                fitorder=fitorder, fitrange=fitrange,
                                doplot=doplot, axes=axes, verbose=verbose)

        self.prof2d, self.profmods = self.make_prof2d(polypars, self.mod0)

        """
        The following two lines are here temporarily so that the extract
        method still works as it was previously written.  Once the
        extraction code has been updated to use the prof2d array, then
        these two lines will be deleted.
        """
        self.mu = self.profmods[0].mean.value
        self.sig = self.profmods[0].stddev.value

    # -----------------------------------------------------------------------

    def _extract_modelfit(self, usevar=True, extrange=None, verbose=True):
        """

        Does an extraction by fitting a n-component model + background to
         the spatial profile in each wavelength bin.
        For now, the components are only allowed to be Gaussian, but that
         may change in the future (e.g., by adding Moffat profiles)
        The expectation is that the SNR in any wavelength slice is too low
         to actually do a good job with a full model fitting, therefore
         most of the model components (e.g., Gaussian mean and sigma) will
         be fixed to the values described by the polynomial fits to the
         traces obtained from the trace_spectrum method.
        """

        """
        Make a copy of the profile shape to use for the model fitting,
        and then fix the mean and sigma parameters.  The model fitting will
        therefore just fit to the amplitudes of the profiles
        """
        mod0 = self.mod0.copy()
        ncomp = mod0.n_submodels() - 1
        if verbose:
            print('Fitting to %d components, plus a background' % ncomp)
        for i in range(1, ncomp+1):
            mod0[i].mean.fixed = True
            mod0[i].stddev.fixed = True

        """ Do the extraction by calling fit_slices """
        if verbose:
            print('Extracting the spectrum.  Please be patient')
            if extrange is None:
                print(' Extraction range (pixels): 0 - %d' % self.npix)
            else:
                print(' Extraction range (pixels): %d - %d' %
                      extrange[0], extrange[1])
        fitpars, covar = self.fit_slices(mod0, 1, mu0arr=self.mu,
                                         sig0arr=self.sig)
        return fitpars, covar

    # -----------------------------------------------------------------------

    def _extract_horne(self, profile, gain=1.0, rdnoise=0.0, extrange=None,
                       verbose=True):
        """

        STILL TO DO:
          - possibly pass to this method a previously
            generated profile instead of generating it here

        Implements the "optimal extraction" method of Keith Horne (1986,
        PASP, 98, 609) to extract a 1d spectrum from the 2d spectral data.
        This is just a weighted average, where the output flux at a given
        wavelength is
          f_lambda = (Sum_i w_i x_i) / Sum_i w_i
        and the index i runs along the spatial direction.
        The Horne method uses the spatial profile of the spectrum, which
         it implements as a probability density function (PDF), P to do
         this.  The x variable in the above equation is x_i = (D_i - S) / P_i,
         where D is the calibrated data in pixel i, S is the estimated sky
         value in that pixel, and P_i is the PDF value in the pixel.
         The weight factor is w_i = P_i^2 / V_i, where V_i is the 
         estimated variance in pixel i  based on counts and the detector gain
         and readnoise.
        The final extracted flux is thus given by:

                  Sum_i{ M_i * P_i * (D_i - S_i) / V_i }
              f = --------------------------------------
                     Sum_i{ M_i * P_i^2 / V_i }

         and the variance on the extracted flux, sigma_f^2, is

                            Sum_i{ M_i * P_i }
             sigma_f^2 = --------------------------
                         Sum_i{ M_i * P_i^2 / V_i }

         where M_i is 1 if the pixel is good or 0 if the pixel is bad 
         (e.g., from a cosmic ray, etc.)
        NOTE: P must be normalized for each wavelength

        There are, therefore, three components to the weighting:
         1. The aperture definition itself (stored in the apmin and apmax
             variables that are part of the Spec2d class definition).
             This weighting is, in fact, not really a weighting but just a mask
             set up so that a pixel will get a weight of 1.0 if it is inside
             the aperture and 0.0 if it is outside.
         2. The profile of the trace, P, i.e., aperture weighting.  This
             profile has been created by the make_prof2d method
         3. The statistical errors associated with the detector, etc.,
             in the form of inverse variance weighting.
            The variance can be provided as an external variance image,
             if the previous reduction / processing has provided this.
            If no external variance spectrum is provided, then the variance
             image will be constructed from the data counts (including counts
             from a 2d sky spectrum if the sky has already been subtracted
             from the data) plus the gain and readnoise of the detector.

        """

        """
        Set up arrays for coordinate system
        Remember, self.npix is the number of pixels in the spectral direction
         and self.nspat is the number of pixels in the spatial direction
        """
        x1d = np.arange(self.npix)
        y1d = np.arange(self.nspat)
        x, y = np.meshgrid(x1d, y1d)
        y = y.astype(float)

        """
        Define an array where the values are spatial-direction offsets from
        the center of the trace.

        To do this make the 1d mu polynomial into a 2d polynomial that varies
         along the spectral direction but has a fixed value along a given
         column.
        The transpose (.T) at the end is necessary because doing a np.repeat
         directly on the desired shape does not give the proper behavior
         (constant along columns in the right way).
        """
        
        """
        Set the aperture limits
        ----------------------------------------
        Put in the aperture limits, delimited by apmin and apmax
        """
        newdim = (self.npix, self.nspat)
        self.mu2d = self.mu.repeat(self.nspat).reshape(newdim).T
        ydiff = y - self.mu2d
        apmask = (ydiff > self.apmin - 1) & (ydiff < self.apmax)
        # bkgdmask = np.logical_not(apmask)

        """
        Third weighting: Inverse variance
        ---------------------------------
        Set up the variance based on the detector characteristics
         (gain and readnoise) if an external variance was not provided
        """
        if self.vardata is not None:
            varspec = self.vardata
        else:
            varspec = (gain * self.data + rdnoise**2) / gain**2

        """ Check for NaNs """
        nansci = np.isnan(self.data)
        nanvar = np.isnan(varspec)
        nanmask = np.logical_or(np.isnan(self.data), np.isnan(varspec))
        # nnans = nansci.sum()
        # nnanv = nanvar.sum()
        # nnan = nanmask.sum()

        """
        Set up a 2d background grid (think about doing this as part of a call
        to the sky subtraction routine in the future)
        """
        tmp = self.data.copy()
        tmp[apmask] = np.nan
        bkgd = np.nanmedian(tmp, axis=self.spaceaxis)
        bkgd2d = bkgd.repeat(self.nspat).reshape((self.npix, self.nspat)).T
        del tmp

        """
        Create the total weight array, combining (1) the aperture profile,
        (2) the aperture limits, and (3) the inverse variance weighting
        following the optimal extraction approach of Horne (1986) as described
        above.
        """
        self.extwt = np.zeros(self.data.shape)
        vmask = varspec <= 0.
        varspec[vmask] = 1.e8
        varspec[nanvar] = 1.e8
        invar = 1. / varspec
        self.extwt[apmask] = self.prof2d[apmask] * invar[apmask]
        self.extwt[nanmask] = 0.
        self.extwt[vmask] = 0.
        wtdenom = (self.prof2d * self.extwt).sum(axis=self.spaceaxis)
        # wtdenom *= apmask.sum(axis=self.spaceaxis)

        """ Compute the weighted sum of the flux """
        data = self.data
        data[nansci] = 0.
        wtdenom[wtdenom == 0] = 1.e9
        flux = ((self.data - bkgd2d) *
                self.extwt).sum(axis=self.spaceaxis) / wtdenom

        """
        Compute the proper variance.
        """
        var = self.prof2d.sum(axis=self.spaceaxis) / wtdenom

        """
        Fix any remaining NaNs (there shouldn't be any, but put this in
         just to be on the safe side
        """
        nansci = np.isnan(flux)
        nanvar = np.isnan(var)
        flux[nansci] = 0.
        var[nansci] = 1.e9
        var[nanvar] = 1.e9

        """ Get the wavelength/pixel vector """
        self.get_wavelength()

        """
        Save the result as a Spec1d instance
        """
        # print('*** Number of nans: %d %d %d ***' % (nnans, nnanv, nnan))
        if extrange is not None:
            extmin = extrange[0]
            extmax = extrange[1]
            owav = self.wavelength[extmin:extmax]
            oflux = flux[extmin:extmax]
            ovar = var[extmin:extmax]
            sky = bkgd[extmin:extmax]
        else:
            owav = self.wavelength.copy()
            oflux = flux
            ovar = var
            sky = bkgd
        self.spec1d = Spec1d(wav=owav, flux=oflux, var=ovar, sky=sky,
                             verbose=verbose)
        self.apmask = apmask

        """ Clean up """
        del(invar, flux, var, bkgd, owav, oflux, ovar, sky)

    # -----------------------------------------------------------------------

    def extract(self, method='optimal', weight='gauss', extrange=None,
                sky=None, usevar=True, gain=1.0, rdnoise=0.0,
                doplot=True, do_subplot=True, outfile=None,
                outformat='text', verbose=True, **kwargs):
        """
        Second step in reduction process.

        This function extracts a 1D spectrum from the input 2D spectrum
        It uses the information about the trace profile that has been generated
        by the trace_spectrum function and which is stored (for now) in the
        self.mu and self.sig arrays.
        """

        """ Extract the spectrum """
        if method == 'modelfit':
            self._extract_modelfit(usevar=usevar, extrange=extrange)
        else:
            self._extract_horne(weight, gain, rdnoise, extrange=extrange,
                                verbose=verbose)

        """ Plot the extracted spectrum if desired """
        if doplot:
            if verbose:
                print('')
                print('Plotting the spectrum')
            if self.has_cdmatx:
                xlab = 'Wavelength'
            else:
                xlab = 'Pixel number along the %s axis' % self.dispaxis
            self.spec1d.plot(xlabel=xlab, title='Extracted spectrum',
                             **kwargs)

        """ Save the extracted spectrum to a file if requested """
        if outfile is not None:
            self.spec1d.save(outfile, outformat=outformat)

    # -----------------------------------------------------------------------
