"""

ifucube.py

Generic code for handling a data cube from an IFU spectrograph.
Will serve as the basis for data from the OSIRIS spectrograph and KCWI

"""

import sys
from os import path
import numpy as np
from scipy.ndimage import filters

from astropy import wcs
from astropy.io import fits as pf

from cdfutils import datafuncs as df
from .. import imfuncs as imf
from .spec1d import Spec1d
from ..imfuncs.wcshdu import WcsHDU

pyversion = sys.version_info.major

# ===========================================================================


class IFUCube(imf.Image):
    """
    A class used to visualize and analyze IFU data
    """

    def __init__(self, indat, hext=0, maskfile=None, verbose=True,
                 wcsverb=True, **kwargs):
        """
        Loads in an OSIRIS data cube that was processed through the
        standard data reduction pipeline and, possibly, has had some
        additional processing done.

        Inputs:
          indat - either the name of an input fits file or a HDU
        """

        """ Read the data into an Image structure """
        if pyversion == 2:
            # Python 2.7
            super(IFUCube, self).__init__(indat, verbose=verbose, hext=hext,
                                          wcsverb=wcsverb, **kwargs)
        else:
            # Python 3 syntax
            super().__init__(indat, verbose=verbose, wcsverb=wcsverb, hext=hext,
                             **kwargs)

        """
        Make sure that the input file has 3 dimensions
        Future versions of this code may be able to handle more than 3, but for
         now exactly 3 dimensions are required
        """
        hdr = self.header
        if 'NAXIS' in hdr.keys():
            self.ndim = hdr['naxis']
        else:
            raise KeyError('No NAXIS keyword in fits header')
        if self.ndim != 3:
            print('')
            print('** ERROR (dimensions) **')
            print('')
            raise ValueError('Input file must have three dimensions')

        """ Get the axis order from the fits header """
        print('')
        print('Original WCS information')
        print('------------------------')
        print(self.wcsinfo)
        print('')
        print(' CTYPE   NAXIS arr_axis')
        print('-------- ----- --------')
        foundax = False
        for i, k in enumerate(self.wcsinfo.wcs.ctype):
            if k[:2] == 'RA':
                self.xinfo = self.set_axinfo(i + 1)
                print('%-8s   %d      %d' % (k, self.xinfo['axis'],
                                             self.xinfo['ndax']))
            if k[:3] == 'DEC':
                self.yinfo = self.set_axinfo(i + 1)
                print('%-8s   %d      %d' % (k, self.yinfo['axis'],
                                             self.yinfo['ndax']))
            if k[:3] == 'WAV':
                self.winfo = self.set_axinfo(i + 1)
                print('%-8s   %d      %d' % (k, self.winfo['axis'],
                                             self.winfo['ndax']))
        print(self.data.shape)
        self.pixscale = self['input'].pixscale
        print(self.pixscale)
        # print(self.wcsinfo.pixel_scale_matrix)
        # print(self.wcsinfo.get_axis_types())
        print('')
        # foo = self.wcsinfo.reorient_celestial_first()
        # print(foo)
        # print(self.data.shape)

        """ Set up the wavelength vector based on the header information """
        self.wav = np.arange(self.winfo['size'])
        self.wav = self.winfo['crval'] + self.wav * self.winfo['cdelt']

        """ Convert to angstroms if necessary """
        if self.winfo['unit'] == 'nm':
            self.wav *= 10.
            self.winfo['unit'] = 'Angstrom'
            self.winfo['crval'] *= 10.
            self.winfo['cdelt'] *= 10.
        elif self.winfo['unit'] == 'm':
            self.wav *= 1.e10
            self.winfo['unit'] = 'Angstrom'
            self.winfo['crval'] *= 1.e10
            self.winfo['cdelt'] *= 1.e10
        else:
            pass

        """
        Save the sizes of the array in easily accessible form
        Remember that python indexes the data in the opposite order than how
         the axes are listed in the fits header
        """
        if verbose:
            self.cubeinfo()

        """ Create arrays of (x,y) coordinate values """
        self.xcoords, self.ycoords = \
            np.indices((self.xinfo['size'], self.yinfo['size']))

        """ Get information about the observations """
        self.obsinfo()

        """ Set default values """
        self.cube = None
        self.mask = None
        self.moment0 = None
        self.meanspec = None
        self.varspec = None

        """ Load a mask file if one has been provided """
        if maskfile is not None:
            self.read_maskfile(maskfile, verbose=verbose)

    # -----------------------------------------------------------------------

    # Actually, don't use this syntax below unless required.  See the
    #  @property calls in dispparam.py for examples
    #
    # imslice = property(fget=get_imslice, fset=set_imslice)

    # -----------------------------------------------------------------------

    def set_axinfo(self, fitsax):
        """

        Creates a dictionary with information about a WCS/spectral axis

        """

        """ Initial setup """
        axinfo = {}
        axinfo['axis'] = fitsax
        axinfo['ndax'] = self.ndim - fitsax

        """ Add WCS / spectral info """
        hdr = self.header
        wcs = self.wcsinfo.wcs
        fitsindex = fitsax - 1
        axinfo['size'] = hdr['naxis%d' % fitsax]
        axinfo['unit'] = wcs.cunit[fitsindex]
        axinfo['crval'] = wcs.crval[fitsindex]
        axinfo['crpix'] = wcs.crpix[fitsindex]
        axinfo['cdelt'] = wcs.cdelt[fitsindex]

        return axinfo

    # -----------------------------------------------------------------------

    def cubeinfo(self):
        """

        Print information about the cube

        """
        xsize = self.xinfo['size']
        ysize = self.yinfo['size']
        print('')
        print('Size of each slice: %d x %d pix = %5.2f x %5.2f arcsec' %
              (xsize, ysize, xsize*self.pixscale[0], ysize*self.pixscale[1]))
        print('Number of wavelength slices: %d' % self.winfo['size'])
        print('Wavelength range: %8.2f-%8.2f Angstrom' %
              (self.wav[0], self.wav[-1]))

    # -----------------------------------------------------------------------

    def obsinfo(self):
        """

        Extracts the following parameters from the fits header, if available.
          obsdate (e.g., 20190908)
          lenslet size (e.g., 100)
          filter (e.g., Kn1)

        """

        hdr = self.header
        try:
            self.obsdate = (hdr['date-obs']).replace('-', '')
        except KeyError:
            self.obsdate = 'DateUnknown'
        try:
            self.filt = hdr['sfilter']
        except KeyError:
            self.filt = 'FiltUnknown'
        try:
            self.lenslet = int(float(hdr['sscale']) * 1000)
        except KeyError:
            self.lenslet = -999

    # -----------------------------------------------------------------------

    def read_maskfile(self, maskfile, maskdir='../Clean', verbose=True,
                      debug=False):
        """

        Reads an external file that will be used as a mask for the
        spatial data.  The data in the file should be set up so that
        values > 0 indicate good data, while 0 indicates bad data.
        The information gets saved as the 'mask' attribute of the class,
        which is a boolean array.

        """

        """ Set up the mask filename if 'default' was chosen """
        if maskfile == 'default':
            mfile = 'mask_%s_%s.fits' % (self.filt, self.lenslet)
            if maskdir is not None:
                maskfile = path.join(maskdir, mfile)
            else:
                maskfile = mfile
            if debug:
                print(maskfile)

        """
        Load the information from the file and convert the data into
        a boolean format
        """
        mhdu = WcsHDU(maskfile, verbose=verbose, wcsverb=False)
        self.mask = mhdu.data > 0.

    # -----------------------------------------------------------------------

    def brief_header(self, dmode='input'):
        """

        Returns an abbreviated version of the header, with a lot of the
        keywords associated with, e.g., instrument temperature, etc.,
        stripped out.

        NOTE: the returned header card do NOT include any associated with
        WCS information, since other methods associated with this class
        can change the WCS values.
        """

        """ Make a new blank header """
        outhdr = pf.header.Header()

        """ Set the original header """
        inhdr = self[dmode].header

        """
        Set the list of header cards to be saved, if they are present in
        the original header
        """
        klist = ['object', 'telescop', 'instrume', 'bunit', 'bscale', 'bzero',
                 'itime', 'coadds', 'sampmode','numreads', 'saturate',
                 'elaptime', 'date-obs', 'mjd-obs', 
                 'detgain', 'instr', 'pscale', 'pa_spec', 'sfilter', 'sscale',
                 'airmass', 'filter', 'rotposn', 'instangl', 'targwave',
                 'ra', 'dec', 'obfmxim', 'obfmyim', 'aotsx', 'aotsy']

        """ Copy the information from the original header """
        for k in klist:
            if k.upper() in inhdr.keys():
                outhdr[k] = inhdr[k]

        return outhdr

    # -----------------------------------------------------------------------

    def update_wcs_from_2d(self, crpix, crval, verbose='False'):
        """

        Updates the RA and Dec keywords based on passed CRPIX and CRVAL
        values that were obtained from a 2d image.
        Note that in the OSIRIS DRP format, the RA axis is axis 3, while in
        the standard 2d setup, RA is on axis 1.  This needs to be taken
        into account.

        """

        """ Set up axis mapping """
        # axes2d = [1, 2]
        axesdrp = [3, 2]

        """ Loop through """
        hdr = self.header
        for i in range(2):
            incrpix = crpix[i]
            incrval = crval[i]
            outcrpixkey = 'crpix%d' % axesdrp[i]
            outcrvalkey = 'crval%d' % axesdrp[i]
            hdr[outcrpixkey] = incrpix
            hdr[outcrvalkey] = incrval

    # -----------------------------------------------------------------------

    def make_wcs2dhdr(self, hdr='default'):
        """

        Takes the input WCS information for the cube (3 dimensions, including
        wavelength) and converts it into 2-dimensional spatial-only WCS
        information for an image slice or the cube compressed along the
        spatial direction.
        """

        """ Select the WCS information to use """
        if hdr == 'default':
            hdr = self.header
        wcsinfo = wcs.WCS(hdr)

        """ Get the WCS information """
        if self.xinfo['axis'] > self.yinfo['axis']:
            outwcs = wcsinfo.celestial.swapaxes(1, 0)
        else:
            outwcs = wcsinfo.celestial
        outhdr = outwcs.to_header()

        """ Add other important info """
        tmphdr = self.brief_header()
        for k in tmphdr.keys():
            outhdr[k] = tmphdr[k]

        return outhdr

    # -----------------------------------------------------------------------

    def sub_to_wcshdu(self, data, wcsinfo):
        """

        Takes the subset of the full cube, in the form of data and a
        (possibly updated) wcsinfo object, and creates
        a new WcsHDU object
        Here, "subset" can be either a 3d cube or a 2d slice

        """


    # -----------------------------------------------------------------------

    def set_imslice(self, imslice, dmode='input', display=True, mode='xy',
                    **kwargs):
        """
        Sets the 2-dimension slice to use for the display functions.

        Inputs:
          imslice - which image slice to use.
        """

        """
        Select the image slice to use.  Right now this assumes the standard
        axis order produced by the pipeline, namely that CTYPE1 is WAVE,
        CTYPE2 is RA, and CTYPE3 is DEC.  This means that the data array
        structure is hard-wired to this assumption
        """

        if imslice >= self.wsize:
            print('')
            print('ERROR: Requested an image slice outside the available '
                  'range')
            print('Maximum available slice value is %d' % (self.wsize-1))
            print('')
            raise IndexError

        """
        Make a wcs header for the slice, i.e., a set of 2d wcs information
        without the spectral information.
        """
        w2dhdr = self.make_wcs2dhdr()

        """
        Actually make the slice.
        The transpose is to get RA and Dec into the order that the WcsHDU
         container expects them to be
        The header information.
        """
        # self['slice'] = WcsHDU(np.transpose(self.data[:, :, imslice]))
        self['slice'] = WcsHDU(np.transpose(self[dmode].data[:, :, imslice]),
                               w2dhdr, wcsverb=False)

        """ Display the image slice if requested """
        self.found_rms = False
        if display:
            self.display(dmode='slice', mode=mode,
                         title='Image Slice %d (zero-indexed)' % imslice,
                         **kwargs)

    # -----------------------------------------------------------------------

    def slice_cube(self, wlim=None, dmode='input', outroot='slice',
                   debug=False):
        """

        Splits the cube into all of its individual slices and saves them
        to disk

        """

        """
        Get the range of wavelength slices to extract from the cube.
        The default is to use the full wavelength range.
        """
        if wlim is not None:
            wmin = wlim[0]
            wmax = wlim[1]
        else:
            wmin = 0
            wmax = self.wsize

        """
        Make a wcs header for the slice, i.e., a set of 2d wcs information
        without the spectral information.
        """
        w2dhdr = self.make_wcs2dhdr()

        """ Flip the data cube if it hasn't already been done """
        if 'xyz' not in self:
            data = self[dmode].data.swapaxes(0, 2)
            wcsinfo = self.wcsinfo.swapaxes(0,2)
            hdr = wcsinfo.to_header()
            self['xyz'] = WcsHDU(data, hdr, wcsverb=False)
        else:
            data = self['xyz'].data

        """ Extract the slices """
        for w in range(wmin, wmax):
            outname = '%s_%03d.fits' % (outroot, w)
            dat = data[w, :, :]
            pf.PrimaryHDU(dat, w2dhdr).writeto(outname, overwrite=True)

    # -----------------------------------------------------------------------

    def select_cube(self, wlim=None, xlim=None, ylim=None, wmode='slice',
                    dmode='input', verbose=False):
        """
        Creates a cube to analyze, defined by ranges in x, y, and wavelength.
        """

        """ Use default values if none are requested """
        if xlim is not None:
            xmin = xlim[0]
            xmax = xlim[1]
        else:
            xmin = 0
            xmax = self.xinfo['size']
        if ylim is not None:
            ymin = ylim[0]
            ymax = ylim[1]
        else:
            ymin = 0
            ymax = self.yinfo['size']
        if wlim is not None:
            wmin = wlim[0]
            wmax = wlim[1]
        else:
            wmin = 0
            wmax = self.winfo['size']

        if verbose:
            print('')
            print('Creating a cube from original data with ranges:')
            print('  x:      %d - %d' % (xmin, xmax))
            print('  y:      %d - %d' % (ymin, ymax))
            print('  lambda: %d - %d (slice number)' % (wmin, wmax))

        """
        Select the cube
        For now put in a kludge, which assumes that the array axis order
        is either (x, y, wavelength) or (wavelength, y, x).
        """
        if self.winfo['ndax'] == 2:
            cube = self[dmode].data[xmin:xmax, ymin:ymax, wmin:wmax]
        else:
            cube = self[dmode].data[wmin:wmax, ymin:ymax, xmin:xmax]
        cubehdr = self.header.copy()
        cubehdr['crpix%d' % self.winfo['axis']] = self.winfo['crpix'] - wmin
        cubehdr['crpix%d' % self.yinfo['axis']] = self.yinfo['crpix'] - ymin
        cubehdr['crpix%d' % self.xinfo['axis']] = self.xinfo['crpix'] - xmin

        """ Return the results """
        return cube, cubehdr

    # -----------------------------------------------------------------------

    def compress_spec(self, wlim=None, xlim=None, ylim=None, wmode='slice',
                      dmode='input', combmode='sum', display=True,
                      verbose=True, **kwargs):
        """

        Compresses the data cube along the spectral dimension, but only
        for image slices between some minimum and maximum wavelengths.
        These wavelength limits (wlim) can be set either by the slice
        number or the actual wavelength in Angstrom [wavelength mode is NOT
        yet implemented].
        Setting wlim=None (the default) will use the full wavelength range

        The compression can be done either as a sum or as a median.

        The result is a 2-dimensional spatial image, which is stored in the
        data container and, thus, can be easily displayed.
        """

        """ First select the image slices to use """
        if wmode == 'wavelength':
            print('NOTE: wavelength mode has not yet been implemented')
            return
        else:
            if wlim is None:
                wlim = (0, self.winfo['size']-1)
            minslice = wlim[0]
            maxslice = wlim[1]
            wavmin = self.wav[minslice]
            wavmax = self.wav[maxslice]

        if verbose:
            print('')
            print('Data cube will be compressed along the spectral direction')
            print('  Image slice range: %d - %d' % (minslice, maxslice))
            print('  Corresponding wavelength range: %8.2f - %8.2f'
                  % (wavmin, wavmax))

        """ Create a temporary cube container """
        cube, cubehdr = self.select_cube(wlim, xlim, ylim, dmode=dmode,
                                         verbose=verbose)

        """
        Make a wcs header for the slice that will be the output of the
        compression, i.e., a set of 2d wcs information without the spectral
        information.
        """
        w2dhdr = self.make_wcs2dhdr(hdr=cubehdr)

        """ Compress the temporary cube along the spectral axis """
        if combmode == 'median':
            tmpcube = np.median(cube, axis=self.winfo['ndax'])
            if self.xinfo['axis'] > self.yinfo['axis']:
                self['slice'] = WcsHDU(np.transpose(tmpcube), w2dhdr)
            else:
                self['slice'] = WcsHDU(tmpcube, w2dhdr)
        else:
            tmpcube = np.median(cube, axis=self.winfo['ndax'])
            if self.xinfo['axis'] > self.yinfo['axis']:
                self['slice'] = WcsHDU(np.transpose(tmpcube), w2dhdr)
            else:
                self['slice'] = WcsHDU(tmpcube, w2dhdr)
        del tmpcube

        """ Display the result if requested """
        if display:
            self.display(dmode='slice', **kwargs)

        """ Clean up """
        del(cube)

    # -----------------------------------------------------------------------

    def whitelight(self, combmode='sum', display=True, verbose=True,
                   **kwargs):
        """
        Creates, and displays if requested, the "white light" image.
        This is a specialized version of the compress_spec method, which
        compresses the cube along* the spectral direction, but for the full
        data cube.
        """

        if verbose:
            print('Creating white light image')

        """ Set up for running compress_spec on the full data cube """
        self.compress_spec(wmode='slice', combmode=combmode,
                           display=display, **kwargs)

    # -----------------------------------------------------------------------

    def make_moment0(self, wlim, xlim, ylim, wmode='slice', combmode='sum',
                     display=True, verbose=True, **kwargs):
        """

        Makes the moment 0 (i.e., total flux) map of the science region
        of the data cube.  The science region is defined by a wavelength
        range (wlim), x-pixel range (xlim), and y-pixel range (ylim).

        Essentially this method is just a front-end for the (barely) more
        generic compress_spec method, with a perhaps easier to remember
        name
        """

        """ Call compress_spec to create the 2-d moment0 image """
        self.compress_spec(wlim, xlim, ylim, wmode=wmode, 
                           combmode=combmode, display=display,
                           verbose=verbose, **kwargs)

        """ Save the data in a moment0 attribute """
        self.moment0 = self.data.copy()

    # -----------------------------------------------------------------------

    def smooth_xy(self, kwidth, smtype='median', outfile=None):
        """
        Smooths the cube over the two spatial dimensions.
        The type of smoothing set by the smtype parameter.  This could be one
         of the following:
           'gauss':   a circular gaussian with sigma=kwidth
           'median':  a median filter with side length=kwidth
           (more to come)
         kwidth is given in pixels
        """

        """ Smooth the data """
        data = self.data
        sm = smtype.lower()
        if sm == 'gauss' or sm == 'guass' or sm == 'gaussian':
            cube = filters.gaussian_filter(data, sigma=[kwidth, kwidth, 0])
            smotype = 'Gaussian'
        elif sm == 'median' or sm == 'medfilt':
            cube = filters.median_filter(data, size=[kwidth, kwidth, 1])
            smotype = 'Median filter'
        else:
            print('')
            print('Smoothing type %s has not been implemented' % smtype)
            print('')
            raise NameError

        """ Put the smoothed data into a new WcsHDU """
        hdr = self.header.copy()
        hdr['history'] = 'Data have been spatially smoothed'
        hdr['smotype'] = smotype
        hdr['smoothw'] = ('%5.1f' % kwidth,
                          'Smoothing kernel width')
        self['smooth'] = WcsHDU(cube, hdr)

        """ Save the smoothed cube in an output file if desired """
        if outfile:
            print('')
            print('Wrote smoothed data cube to %s' % outfile)
            self['smooth'].writeto(outfile, overwrite=True)
            print('')

    # -----------------------------------------------------------------------

    def slice_stats(self, imslice, dmode='input', nsig=3., verbose=False,
                    debug=False):
        """

        Calculates a mean and variance associated with the selected slice.
        The statistics are calculated within the good region of the slice,
         which is set by the mask if the mask has been loaded into the
         OsCube object.
        The returned values are the clipped mean and the square of the
         clipped rms, where "clipped" means that the statistics are
         calculated after a sigma-clipping routine that rejects obvious
         outliers has been run.

        Required inputs:
         imslice - the image slice for which the statistics are calculated

        Optional inputs:
         verbose - Report the image statistics?

        """

        """
        Get the 2-dimensional mask that is appropriate for this slice.
        NOTE: Ordinarily, if we were getting the mask as a 2d slice from a
         3d mask, we would have to transpose the mask to match the
         image slice, since the slices are generally set up to have RA
         along the x axis.  Here, however, since we just care about image
         statistics and not the WCS orientation, we don't transpose the
         image slices.  This does, however, mean that a 2d mask does
         need to be transposed, since it is assuming the standard WCS
         orientation.
        """
        if self.mask.ndim == 3:
            mask2d = self.mask[:, :, imslice]
        else:
            mask2d = np.transpose(self.mask)

        """
        Select the requested slice from the science data cube and calculate
         its statistics
        """
        # self.set_imslice(imslice, display=False)
        # self['slice'].sigma_clip(mask=mask2d, verbose=False)
        # mean = self['slice'].mean_clip
        # r = self['slice'].rms_clip
        data = self[dmode].data[:, :, imslice]
        mean, r = df.sigclip(data, nsig=nsig, mask=mask2d, verbose=False)
        var = r**2
        if debug:
            print('Total pixels in slice: %d' % self['slice'].data.size)
            print('Number of good pixels: %d' % mask2d.sum())
            self['slice'].sigma_clip(verbose=False)
            print('Unmasked rms: %f' % self['slice'].rms_clip)
            print('Masked rms:   %f' % r)
            print('')

        """
        Report statistics, if requested, and return the mean and variance
        """
        if verbose:
            print(imslice, mean, r)
        return mean, var

    # -----------------------------------------------------------------------

    def read_varspec(self, varfile, maskfile, informat='fitstab', **kwargs):
        """

        Reads a previously-generated variance spectrum into the OsCube object.
        The reason for having a method to do this is that the make_varspec
         code takes a long time to run (on a laptop) and rather than re-running
         make_varspec multiple times (e.g., when testing code changes), it
         is much faster to save the variance spectrum and then read it in
         for subsequent runs.

        Inputs:
         varfile  - file containing the variance spectrum
         maskfile - file containing the mask that was used when initially
                     creating the variance spectrum.  This is included because
                     other methods in the OsCube class assume that if a
                     variance spectrum exists then the mask file also exists.
         **kwargs - arguments associated with initializing a Spec1d object

        """

        self.varspec = Spec1d(varfile, informat=informat, **kwargs)
        self.read_maskfile(maskfile)

    # -----------------------------------------------------------------------

    def make_varspec(self, maskfile=None, outfile=None, outformat='fitstab',
                     doplot=False, verbose=False, **kwargs):
        """
        Steps through the spectral slices and computes the statistics of
        the illuminated region and stores the clipped mean and variance
        for each slice.

        If requested, this method also saves the variance spectrum in
        an external file, for later use.
        """

        """ Get the mask info """
        if maskfile is not None:
            self.read_maskfile(maskfile)
            maskdat = self.mask
        elif self.mask is not None:
            maskdat = self.mask
        else:
            raise ValueError('No mask info has been provided')

        """ Set up the containers to store the info """
        mean = np.zeros(self.wsize)
        var = np.zeros(self.wsize)

        """ Loop through the slices, calculating the statistics of each """
        print('Calculating variance spectrum.  Be patient.')
        for i in range(self.wsize):
            mean[i], var[i] = self.slice_stats(i, verbose=verbose)

        self.meanspec = mean
        self.varspec = Spec1d(wav=self.wav, flux=var)

        """ Plot the spectrum if requested """
        if doplot:
            self.varspec.plot(**kwargs)

        """ Save the variance spectrum, if requested """
        if outfile is not None:
            self.varspec.save(outfile, outformat=outformat)

    # -----------------------------------------------------------------------

    def _make_varcube_from_varspec(self, maskfile=None, **kwargs):

        """

        Takes the 1-dimensional variance spectrum and converts it into a
         3-dimensional data cube, where each slice is created by multiplying
         an integer version of the mask (where 1 is good and 0 and bad) by
         the variance that has been computed for that slice.
        This method requires that the make_varspec method has been run
         first.

        """

        """ Make the variance spectrum if it has not already been done """
        if self.varspec is None:
            self.make_varspec(maskfile, **kwargs)

        """ Step through the slices, making each one a 2d variance slice """
        data = self.data * 0.
        for imslice, var in enumerate(self.varspec['flux']):
            """
            Get the 2-dimensional mask that is appropriate for this slice
            """
            if self.mask.ndim == 3:
                mask2d = self.mask[:, :, imslice]
            else:
                mask2d = np.transpose(self.mask)

            """ Set the good pixels to the variance value for the slice """
            data[:, :, imslice][mask2d] = var

        return data

    # -----------------------------------------------------------------------

    def _make_varcube_from_darksub(self, darksubfile, texp=None, gain=None):

        """ Read in the dark-subtracted file """
        try:
            darksub = OsCube(darksubfile)
        except IOError:
            print('')
            print('Cannot find dark-subtracted file: %s' % darksubfile)
            print('')
            return None

        """ Get the exposure time and gain """
        if texp is None:
            try:
                texp = darksub.header['elaptime']
            except KeyError:
                print('')
                print('Missing ELAPTIME header keyword in %s' % darksubfile)
                print('')
                return None

        if gain is None:
            try:
                gain = darksub.header['sysgain']
            except KeyError:
                print('')
                print('Missing SYSGAIN header keyword in %s' % darksubfile)
                print('')
                return None

        """ Convert the dark-subtracted data to a variance """
        var = darksub.data / (texp * gain)
        del darksub
        return var

    # -----------------------------------------------------------------------

    def make_varcube(self, method, maskfile=None, dsfile=None,
                     outfile=None, **kwargs):

        if method == 'varspec':
            data = self._make_varcube_from_varspec(maskfile=maskfile, **kwargs)
        elif method == 'darksub':
            if dsfile is None:
                raise ValueError('make_varcube method "darksub" requires the'
                                 ' dsfile parameter to be set')
            data = self._make_varcube_from_darksub(dsfile, **kwargs)
        else:
            print('')
            raise ValueError('make_varcube method must be either "darksub"'
                             ' or "varspec"')

        """ Create the container for the variance cube """
        if data is not None:
            self['var'] = WcsHDU(self.data, self.header)
            self['var'].data = data
        else:
            raise ValueError('Could not create variance cube data')

        """ Save the variance cube if requested """
        if outfile is not None:
            self.save_drp(outfile, 'var')

    # -----------------------------------------------------------------------

    def clean(self, nsig1=5., nsig2=5., smtype='median', smsize=3,
              skysub=True, verbose=False):
        """

        Does a bad-pixel cleaning, slice by slice.  The basic algorithm
        is the following:
           1. Create a smoothed (on the slices but not in wavelength) version
              of the data using the smooth_xy method
           2. Use the pixel-to-pixel variance for each plane (obtained
              previously by running the make_varspec method) to flag pixels
              that satisfy both of the following conditions:
              A. Deviate from the clipped mean by more than nsig1 sigma
              B. Deviate from the smoothed version of the data by more than
                 nsig2 sigma
              This should flag bad pixels without incorrectly flagging
              pixels that are high due to the presence of a real astronomical
              object
           3. Replace the flagged pixels with the corresponding value in
              the smooth data

        """

        """ Make sure that the variance spectrum exists """
        if self.varspec is None:
            print('')
            print('In order to run the clean algorithm, you first have to '
                  'run make_varspec')
            print('')
            return

        """
        Create a smoothed version of the data and subtract it from the
        input data
        """
        data = self.data.copy()
        self.smooth_xy(smsize, smtype)
        diff = np.fabs(data - self['smooth'].data)

        """
        Step through the slices, flagging the pixels that differ too much
        from both the clipped mean value and the smoothed data
        """
        if verbose:
            print('')
            print('Slice N_flag')
            print('----- ------')
        rms = np.sqrt(self.varspec['flux'])
        for i, r in enumerate(rms):
            """ Subtract the sky if requested """
            if skysub:
                slmean = self.meanspec[i]
            else:
                slmean = 0.
            smdat = self['smooth'].data[:, :, i] - slmean
            mdiff = np.fabs(data[:, :, i] - slmean)
            data[:, :, i] -= slmean
            mask = (mdiff > nsig1 * r) & (diff[:, :, i] > nsig2 * r)
            data[:, :, i][mask] = smdat[mask]
            if verbose:
                print(' %3d  %5d' % (i, mask.sum()))

            """
            Make sure that the regions outside the illuminated part of the
            chip are set to zero, since they may have been set to a non-zero
            value in the sky subtraction
            """
            data[np.transpose(np.logical_not(self.mask))] = 0.

        """ Save the cleaned cube """
        self['clean'] = WcsHDU(data, self.header)

    # -----------------------------------------------------------------------

    def make_snrcube(self, maskfile=None):
        """

        Use the information in the variance spectrum to make a SNR cube

        """

        """ Make sure that the variance spectrum exists """
        if self.varspec is None:
            print('')
            print('In order to run the make_snrcube algorithm, you first have '
                  'to run make_varspec')
            print('')
            return

        """
        Make a copy of the data cube and step through the slices, dividing
        each by the associated rms value
        """
        cube = self.data.copy()
        rms = np.sqrt(self.varspec['flux'])
        mask = (np.transpose(self.mask)).astype(float)
        for i, r in enumerate(rms):
            cube[:, :, i] *= (mask / r)

        """ Save the result """
        self['snr'] = WcsHDU(cube, self.header)
        del(cube)

    # -----------------------------------------------------------------------

    def make_1dspec(self, reg, maskfile=None, display=True,
                    skyx=None, skyy=None, debug=False, **kwargs):
        """
        Takes a spatial region of the cube, designated by the reg parameter,
        and extracts the spectral information into a Spec1d container.
        Also plots the 1d spectrum if requested.

        The reg parameter can be one of the following:
          1. A single spaxel, designated by an (x,y) tuple or [x,y] list
          2. A rectangular region, designated by an ((x1,x2), (y1,y2)) tuple
             or a [[x1, x2], [y1, y2]] list
          3. A boolean mask array, with the spaxels that are set to True
             designating the region to use

        Returns: the Spec1d container (instance).

        """

        """ Set the data set to use """
        cube = self.data

        """ Make the variance spectrum if it doesn't exist """
        # if self.varspec is None:
        #     if maskfile is None:
        #         print('')
        #         raise ValueError('No maskfile given for make_varspec call')
        #     self.make_varspec(maskfile)

        """ Parse the reg parameter """
        mask = None
        if isinstance(reg, tuple) or isinstance(reg, list):
            x = reg[0]
            y = reg[1]

            if isinstance(x, int) and isinstance(y, int):
                # flux = cube[x, y, :]
                flux = cube[:, y, x]
                npix = 1

            elif (isinstance(x, list) or isinstance(x, tuple)) and \
                    (isinstance(y, list) or isinstance(y, tuple)):
                """ Need to check lengths """
                xmin = int(x[0])
                xmax = int(x[1])
                ymin = int(y[0])
                ymax = int(y[1])
                # flux = cube[xmin:xmax, ymin:ymax, :].sum(axis=0)
                # flux = flux.sum(axis=0)
                # npix = (xmax - xmin) * (ymax - ymin)
                mask = (self.xcoords >= xmin) & (self.xcoords < xmax) \
                    & (self.ycoords >= ymin) & (self.ycoords < ymax)

            if mask is not None:
                xx = self.xcoords[mask].flatten()
                yy = self.ycoords[mask].flatten()
                flux = np.zeros(self.winfo['size'])
                for i, j in zip(xx, yy):
                    # flux += cube[i, j, :]
                    flux += cube[:, j, i]
                npix = len(xx)

        if debug:
            print('npix: %d' % npix)
            print(self.wav.size, flux.size)

        """
        Make the variance spectrum if it has been requested by
        setting skyx and skyy
        """
        # if skyx is not None and skyy is not None:
        #     skycube, hdr = self.select_cube(xlim=skyx, ylim=skyy)
        #     var = npix * skycube.var(axis=(0, 1))
        # else:
        #     var = None

        """
        Make, and display if requested, the final spectrum.
        """
        # var = self.varspec['flux'] * npix
        var = None
        spec = Spec1d(wav=self.wav, flux=flux, var=var)

        if display:
            spec.plot(**kwargs)

        """ Clean up and return """
        del flux
        if var is not None:
            del var
        return spec

    # -----------------------------------------------------------------------

    def click_1dspec(self, xysmooth=1, **kwargs):
        """
        An interactive interface to set_1dspec.  Produces the 1D spectrum
        associated with the spaxel that is clicked.
        """

        self.start_interactive()
        self.set_1dspec(int(self.xclick), int(self.yclick), **kwargs)

    # -----------------------------------------------------------------------

    def save_drp(self, outfile, dmode):
        """

        Save the cube in a format that will be recognized by the OSIRIS DRP

        """

        phdu = pf.PrimaryHDU(self[dmode].data, self[dmode].header)
        hdulist = pf.HDUList(phdu)
        if self.drphdu1 is not None:
            hdulist.append(self.drphdu1)
        if self.drphdu2 is not None:
            hdulist.append(self.drphdu2)
        hdulist.writeto(outfile, overwrite=True)

    # -----------------------------------------------------------------------

    def save_xyl(self, outfits, dmode='input', outtext=None, **kwargs):
        """

        Saves the data cube as a fits file but with the spectral axis
         as axis 3 (in fits parlance) rather than axis 1.
         The OSIRIS data reduction pipeline produces a data cube with
         FITS data axis order of (lambda, y, x).  This method produces
         an output cube with a FITS data axis order of (x, y, lambda)
        If the optional outtext parameter is set to a filename, then an
         output text file will be created in the format expected by
         Francesca Rizzo's modeling code

        Required inputs:
          outfits - name of output fits file

        Optional inputs:
          outtext - if set to a string value, then save an output text file
                    in the format expected by Francesca Rizzo's modeling code.
                    The default value (None) means that no output file is
                    created.

          NOTE: This should be rewritten to take advantage of the new
          functionality in the WcsHDU writeto method
        """

        """
        Create a copy of the data cube and swap the axes
        """
        cube, hdr0 = self.select_cube(dmode=dmode, verbose=True, **kwargs)
        tmp = cube.copy()
        tmp2 = np.swapaxes(tmp, 0, 2)

        """
        Create a new header and write the output file
        """
        tmphdu = pf.PrimaryHDU(tmp2)
        hdr = tmphdu.header
        """ Non-coordinate keywords """
        klist = ['bunit', 'bscale', 'bzero', 'itime', 'coadds', 'sampmode',
                 'numreads', 'saturate', 'instr', 'pscale', 'object',
                 'pa_spec', 'sfilter', 'telescop', 'instrume', 'targwave',
                 'airmass', 'wcsaxes']
        for k in klist:
            try:
                hdr[k] = hdr0[k]
            except KeyError:
                continue
        """
        Coordinate keywords.
        For now we're assuming that the input was (Wavelength, Dec, RA) and
         the output will be (RA, Dec, Wavelength)
        """
        crlist = ['ctype', 'cunit', 'crval', 'crpix', 'cdelt']
        for k in crlist:
            hdr['%s1' % k] = hdr0['%s3' % k]
            hdr['%s2' % k] = hdr0['%s2' % k]
            hdr['%s3' % k] = hdr0['%s1' % k]
        """ Override 'Angstrom' to 'Ang' to help ds9 """
        hdr['cunit3'] = 'Ang'
        
        """ PC matrix """
        raaxis = self['input'].raaxis
        decaxis = self['input'].decaxis
        hdr['pc1_1'] = hdr0['pc%d_%d' % (raaxis, raaxis)]
        hdr['pc1_2'] = hdr0['pc%d_%d' % (raaxis, decaxis)]
        hdr['pc2_1'] = hdr0['pc%d_%d' % (decaxis, raaxis)]
        hdr['pc2_2'] = hdr0['pc%d_%d' % (decaxis, decaxis)]

        print('')
        print('Saving to output file %s' % outfits)
        tmphdu.writeto(outfits, overwrite=True)

        """ Create an output text file if requested """
        if outtext is not None:
            print('Creating kinematics data file %s' % outtext)
            f = open(outtext, 'w')
            f.write('#Cube\n')
            f.write('nch %d\n' % tmp2.shape[0])
            f.write('ctype wave\n')
            f.write('cunit angstrom\n')
            f.write('cd3 %f\n' % hdr['cdelt3'])
            f.write('cv3 %f\n' % hdr['crval3'])
            f.write('crpix3 %d\n' % hdr['crpix3'])
            f.write('crota %7.2f\n' % self.pa)
            f.close()

        """ Clean up before exiting """
        del(tmp, tmphdu)
