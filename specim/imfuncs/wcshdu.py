"""

wcshdu.py

Defines a new class that is essentially a fits PrimaryHDU class but with
some of the WCS information in the header split out into separate
attributes of the class

"""

import os
from math import fabs
import numpy as np
from scipy import ndimage
from scipy.ndimage import filters
from astropy import wcs
from astropy.io import fits as pf
from cdfutils import coords, datafuncs as df
from .imutils import open_fits

# ---------------------------------------------------------------------------


class WcsHDU(pf.PrimaryHDU):
    """

    This class is essentially a fits PrimaryHDU class but with some of the
    WCS information that may be in the header split out into separate
    attributes of the class.

    The class also contains methods to:
      * Compute statistics for all or part of the data
      * Make cutouts of the data based on either wcs or pixel information

    """

    def __init__(self, indat, inhdr=None, hext=0, wcsext=None, verbose=True,
                 wcsverb=True):
        """

        Inputs:
          indat - One of the following:
                   1. a filename for a fits file
                   2. a HDUList that has previously been read in
                   3. a single HDU
                   4. a numpy.ndarray, which will be interpreted as the data
                      portion of a data / header pair to be put into a HDU.
                      In this case, the header information, if it exists,
                      should be passed via the optional inhdr parameter.
                      If inhdr is None, then a minimal header will be
                      automatically generated.
        """

        """ Set some default values """
        self.infile = None
        self.wcsinfo = None
        self.radec = None
        self.pixscale = None
        self.impa = None
        self.radec = None
        self.raaxis = None
        self.decaxis = None
        self.found_rms = False

        """
        Check the format of the input info and get the data and header info
        """
        if isinstance(indat, str):
            hdu = self.read_from_file(indat, verbose=verbose)
            data = hdu[hext].data
            hdr = hdu[hext].header
        elif isinstance(indat, pf.HDUList):
            hdu = indat
            data = hdu[hext].data
            hdr = hdu[hext].header
        elif isinstance(indat, pf.PrimaryHDU) or \
                isinstance(indat, pf.ImageHDU):
            data = indat.data.copy()
            hdr = indat.header.copy()
        elif isinstance(indat, np.ndarray):
            data = indat.copy()
            if inhdr is not None:
                hdr = inhdr.copy()
            else:
                hdr = None
        else:
            print('')
            print('ERROR: The input for the WcsHDU class must be'
                  'one of the following:')
            print('  1. A filename (i.e. a string)')
            print('  2. A HDUList')
            print('  3. A single PrimaryHDU or ImageHDU')
            print('  4. A numpy data array (numpy.ndarray), which will be')
            print('     interpreted as the data portion of a HDU.  In this'
                  ' case, use the optional')
            print('     inhdr parameter to pass the header portion of the'
                  ' HDU, if it exists')
            print('')
            raise TypeError

        """
        Use the input data and header info to make the call to the super class
        """
        super(WcsHDU, self).__init__(data, hdr)

        """
        Now add WCS attributes to the class.
        NOTE, sometimes these are in a different header, which can be
        indicated by the wcsext parameter
        """
        if wcsext is not None:
            try:
                wcshdr = hdu[wcsext].header
            except UnboundLocalError:
                print('')
                print('ERROR: You cannot set wcsext parameter if the input'
                      ' is not a file or HDUList')
                print('')
                raise TypeError
        else:
            wcshdr = hdr
        try:
            self.read_wcsinfo(wcshdr, verbose=wcsverb)
        except KeyError:
            """ Just keep default values for WCS attributes, i.e., None """
            pass

    # -----------------------------------------------------------------------

    def read_from_file(self, infile, verbose=True):
        """
        Reads the image data from a file (as opposed to getting the data
        directly from an input HDUList
        """
        if verbose:
            print('')
            print('Loading file %s' % infile)
            print('-----------------------------------------------')

        if os.path.isfile(infile):
            try:
                hdu = open_fits(infile)
            except IOError:
                print(' ERROR. Problem in loading file %s' % infile)
                print(' Check to make sure filename matches an existing'
                      'file')
                print(' If it does, there may be something wrong with the'
                      ' fits header.')
                print('')
                raise IOError('Error in read_from_file')
        else:
            print('')
            print('ERROR. File %s does not exist.' % infile)
            print('')
            raise IOError('Error in read_from_file')

        """
        Set parameters related to image properties and return the hdulist
        """
        self.infile = infile
        return hdu

    # -----------------------------------------------------------------------

    def read_wcsinfo(self, wcshdr, verbose=True):
        """

        Reads in WCS information from the header and saves it, if it's
        there, in some attributes of the class

        """

        """ Get the WCS information out of the header if it's there """
        try:
            wcsinfo = wcs.WCS(wcshdr)
        except:
            if verbose:
                if self.infile is not None:
                    print('No WCS information in image header: %s',
                          self.infile)
                else:
                    print('No WCS information in image header')
            self.wcsinfo = None
            raise KeyError

        """
        Make sure that the WCS information is actually WCS-like and not,
        for example, pixel-based
        """

        imwcs = wcsinfo.wcs
        rafound = False
        decfound = False
        for count, ct in enumerate(imwcs.ctype):
            if ct[0:2] == 'RA':
                rafound = True
                raax = count
                raaxis = raax + 1
                rakey = 'naxis%d' % raaxis
            if ct[0:3] == 'DEC':
                decfound = True
                decax = count
                decaxis = decax + 1
                deckey = 'naxis%d' % decaxis
        if rafound is False or decfound is False:
            if verbose:
                print('No valid WCS information in image header')
                print(' CTYPE keys are not RA/DEC')
            self.wcsinfo = None
            raise KeyError

        """ Get the RA and Dec of the center of the image """
        xcent = wcshdr[rakey] / 2.
        ycent = wcshdr[deckey] / 2.
        imcent = np.ones((1, wcshdr['naxis']))
        imcent[0, raax] = xcent
        imcent[0, decax] = ycent
        imcentradec = wcsinfo.wcs_pix2world(imcent, 1)
        radec = coords.radec_to_skycoord(imcentradec[0, raax],
                                         imcentradec[0, decax])

        """ Get the pixel scale and image rotation """
        pixscale = wcs.utils.proj_plane_pixel_scales(wcsinfo.celestial) \
            * 3600.

        impa = coords.matrix_to_rot(wcsinfo.pixel_scale_matrix, raax=raax,
                                    decax=decax)

        """ Summarize the WCS information """
        if verbose:
            print('Pixel scale (x, y): (%7.3f, %7.3f) arcsec/pix' %
                  (pixscale[0], pixscale[1]))
            print('Instrument FOV (arcsec): %7.1f x %7.1f' %
                  (pixscale[0] * wcshdr[rakey],
                   pixscale[1] * wcshdr[deckey]))
            print('Image position angle (E of N): %+7.2f' % impa)

        """ Add the information to the object """
        self.wcsinfo = wcsinfo
        self.raaxis = raaxis
        self.decaxis = decaxis
        self.radec = radec
        self.pixscale = pixscale
        self.impa = impa

    # -----------------------------------------------------------------------

    def make_hdr_wcs(self, inhdr, wcsinfo, keeplist='all', debug=False):
        """

        Creates a new header that includes (possibly updated) wcs
        information to use for an output file/HDU.

        Inputs:
          inhdr    - Input header.  This could be just the header of the
                     HDU that was used to create this Image object, but it
                     could also be some modification of that header or even
                     a brand-new header
          wcsinfo  - WCS information, which may be just the information in
                     the input file, but may also be a modification
          keeplist - If set to 'all' (the default) then keep all of the
                     header cards in inhdr.  If not, then just keep the
                     header cards -- designated as strings -- in keeplist

        """

        """
        Eliminate, as much as possible, the WCS header keywords from
         the original header.  This is done to avoid possibly conflicting
         information, e.g., a CD matrix in the original header and then
         a CDELT + PC matrix from the cutout.
        """
        hdr = inhdr.copy()
        wcskeys = ['ra', 'dec', 'ctype1', 'ctype2', 'crval1', 'crpix1',
                   'crval2', 'crpix2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2',
                   'cdelt1', 'cdelt2', 'pc1_1', 'pc1_2', 'pc2_1', 'pc2_2']
        for key in wcskeys:
            if key.upper() in hdr.keys():
                del hdr[key]
                if debug:
                    print('Deleting original %s keyword' % key.upper())

        """ Create a new output header, according to keeplist """
        if keeplist != 'all':
            tmphdu = pf.PrimaryHDU()
            outhdr = tmphdu.header.copy()
            for key in keeplist:
                if key.upper() in hdr.keys():
                    outhdr[key] = hdr[key]
                    if debug:
                        print(key.upper())
        else:
            outhdr = hdr

        """ Add the WCS information to the header """
        if self.wcsinfo is not None:
            wcshdr = wcsinfo.to_header()
            for key in wcshdr.keys():
                outhdr[key] = wcshdr[key]
                if debug:
                    print(key.upper(), wcshdr[key], outhdr[key])

        return outhdr

    # -----------------------------------------------------------------------

    def update_crpix(self, crpixarr, verbose=True):
        """

        Updates the CRPIX array in the wcsinfo structure

        Inputs:
         crpixarr - a list, tuple, or numpy ndarray containing the new 
                     CRPIX values.  For most data, this parameter will
                     contain two elements, to replace CRPIX1 and CRPIX2

        """

        """ Check dimensionality """
        if len(crpixarr) != len(self.wcsinfo.wcs.crpix):
            raise IndexError(' Input crpix array length does not match'
                             ' length of current crpix array')

        """
        Update the CRPIX array assuming that the input is in the correct
        format
        """
        if isinstance(crpixarr, list) or isinstance(crpixarr, tuple) \
                or isinstance(crpixarr, np.ndarray):
            if verbose:
                print('Updating CRPIX array')
            for i in range(len(crpixarr)):
                if verbose:
                    print('  %8.2f  -->  %8.2f' % (self.wcsinfo.wcs.crpix[i],
                                                   crpixarr[i]))
                self.wcsinfo.wcs.crpix[i] = crpixarr[i]
        else:
            raise TypeError('crpixarr must be list, tuple, or ndarray')

    # -----------------------------------------------------------------------

    def update_crval(self, crvalarr, verbose=True):
        """

        Updates the CRVAL array in the wcsinfo structure

        Inputs:
         crvalarr - a list, tuple, or numpy ndarray containing the new 
                     CRVAL values.  For most data, this parameter will
                     contain two elements, to replace CRVAL1 and CRVAL2

        """

        """ Check dimensionality """
        if len(crvalarr) != len(self.wcsinfo.wcs.crval):
            raise IndexError(' Input crval array length does not match'
                             ' length of current crval array')

        """
        Update the CRVAL array assuming that the input is in the correct
        format
        """
        if isinstance(crvalarr, list) or isinstance(crvalarr, tuple) \
                or isinstance(crvalarr, np.ndarray):
            if verbose:
                print('Updating CRVAL array')
            for i in range(len(crvalarr)):
                if verbose:
                    print('  %f  -->  %f' % (self.wcsinfo.wcs.crval[i],
                                             crvalarr[i]))
                self.wcsinfo.wcs.crval[i] = crvalarr[i]
        else:
            raise TypeError('crvalarr must be list, tuple, or ndarray')

    # -----------------------------------------------------------------------

    def copy_wcsinfo(self, wcshdu):
        """

        Takes the wcsinfo from the given wcshdu and copies the information
        into the appropriate locations

        """

        self.wcsinfo = wcshdu.wcsinfo
        self.pixscale = wcshdu.pixscale
        self.impa = wcshdu.impa
        self.radec = wcshdu.radec
        self.raaxis = wcshdu.raaxis
        self.decaxis = wcshdu.decaxis

    # -----------------------------------------------------------------------

    def flip(self, method):
        """

        Performs a flip of the data, in order to correct for the way
        that certain detectors read out.

        Inputs:
         method - method to utilize in order to flip the data.  Possibilities
                  are:
                   'x'       - flip the x-axis
                   'y'       - flip the y-axis
                   'xy'      - flip both x and y axes
                   'pfcam'   - flip x and then rotate -90 [NOT YET IMPLEMENTED]
        """

        data = self.data.copy()
        if method == 'x':
            self.data = data[:, ::-1]
        elif method == 'y':
            self.data = data[::-1, :]
        elif method == 'xy':
            self.data = data[::-1, ::-1]
        else:
            raise ValueError
    
    # -----------------------------------------------------------------------

    def sigma_clip(self, nsig=3., statsec=None, mask=None,
                   verbose=False):
        """
        Runs a sigma-clipping on image data.  After doing outlier rejection
        the code returns the mean and rms of the clipped data.

        This method is just a minimal wrapper for the sigclip method in the
         cdfutils.datafuncs library.

        NOTE: The region used for determining these image statistics is set
        by the following decision path:
         - if statsec is not None, use statsec
         - else, use the entire image
        for the second option, an optional mask can be used to
        exclude known bad pixels from the calculation.

        Optional inputs:
          nsig     - Number of sigma from the mean beyond which points are
                      rejected.  Default=3.
          statsec  - Region of the input image to be used to determine the
                      image statistics, defined by the coordinates of its
                      corners (x1, y1, x2, y2).
                      If this variable is is None (the default value)
                      then the image statistics will be determined from:
                        - the subimage, if it has been set
                        - else, the entire image.
                      The format for statsec can be any of the following:
                        1. A 4-element numpy array
                        2. A 4-element list:  [x1, y1, x2, y2]
                        3. A 4-element tuple: (x1, y1, x2, y2)
                        4. statsec=None.  In this case, the region used for
                           determining the pixel statistics defaults to either
                           the subimage (if defined) or the full image (if no
                           subimage has been defined)
          mask     - If some of the input data are known to be bad, they can
                       be flagged before the inputs are computed by including
                       a mask.  This mask must be set such that True
                       indicates good data and False indicates bad data
          verbose  - If False (the default) no information is printed
        """

        """ Determine what the input data set is """
        scdata = self.data.copy()
        if statsec is not None:
            x1, y1, x2, y2 = statsec
            scdata = scdata[y1:y2, x1:x2]

        """ Find the clipped mean and rms """
        mu, sig = df.sigclip(scdata, nsig=nsig, mask=mask, verbose=verbose)

        """ Store the results and clean up """
        del scdata
        self.found_rms = True
        self.mean_clip = mu
        self.rms_clip = sig
        return

    # -----------------------------------------------------------------------

    def smooth(self, size, smtype='median', invar=False):
        """

        Smooths the data, using one of the following schemes:
          gaussian
          median filter
          (more to come)

        Inputs:
          size   - sets the smoothing scale.  For a Gaussian, this is the
                   sigma, for median filter this is the length of the square
                   box being used
          smtype - type of smoothing to be done.  Options are: 'gauss',
                   'median'

        """

        """ Smooth the data using the requested smoothing type """
        if smtype.lower() == 'gauss' or smtype.lower() == 'gaussian':
            smdata = filters.gaussian_filter(self.data, sigma=size)
        elif smtype.lower() == 'median' or smtype.lower() == 'medfilt':
            smdata = filters.median_filter(self.data, size=size)
        else:
            print('')
            print('Smoothing type %s has not been implemented' % smtype)
            print('')
            raise NameError

        """ Return the smoothed data set """
        return smdata
        
    # -----------------------------------------------------------------------

    def normalize(self, method='sigclip', mask=None):
        """

        Normalizes the data in the object.  Allowed methods are:
          'sigclip' - divide by the clipped mean (the default)
          'median'  - divide by the median
          'mean'    - divide by the mean
          'average' - alternate way of indicating divide by mean

        """

        method = method.lower()
        if mask is not None:
            data = self.data[mask]
        else:
            data = self.data
        if method == 'median':
            normfac = np.median(data)
        elif method == 'mean' or method[:3] == 'ave':
            normfac = data.mean()
        elif method == 'sigclip':
            self.sigma_clip(mask=mask)
            normfac = self.mean_clip
        else:
            raise ValueError('method must be one of "sigclip", "median" '
                             'or "mean"')
        self.data /= normfac
            
    # -----------------------------------------------------------------------

    def make_objmask(self, nsig=1.5, bpm=None):
        """

        Makes a mask that is intended to indicate regions of the image
        that contain flux from objects, as opposed to being blank sky.
        The mask is set so that pixels containing object emission are
        indicated by True, while blank sky pixels are indicated by False

        Inputs:
         nsig  - number of sigma above the noise level a smoothed image
                  must have to be considered object emission. Default=1.5
         bpm   - bad pixel mask to apply.  Default=None

        Output:
         objmask - object mask

        """

        """ Compute the clipped rms in the image """
        self.sigma_clip(mask=bpm)
        print(self.mean_clip, self.rms_clip)

        """ Median smooth the image and set initial object mask """
        med = self.smooth(3, smtype='median')
        objmask = np.where((med - self.mean_clip)/self.rms_clip > nsig, 1, 0)

        """ Reject isolated cosmic rays via a minimum filter """
        objmask = ndimage.minimum_filter(objmask, 5)

        """ Grow the mask regions to encompass low SNR regions """
        objmask = ndimage.maximum_filter(objmask, 11)
        objmask = ndimage.maximum_filter(objmask, 11)

        return objmask
    
    # -----------------------------------------------------------------------

    def subim_bounds_xy(self, centpos, imsize):
        """
        Takes a requested image center (xcent, ycent) and requested image size
        (all in pixels) and returns
        the coordinates of the lower left corner (x1, y1) and the upper right
        corner (x2, y2).

        Inputs:
          centpos - (x, y) coordinates of cutout center, in pixels
                    centpos can take any of the following formats:
                       1. A 2-element numpy array
                       2. A 2-element list:  [xsize, ysize]
                       3. A 2-element tuple: (xsize, ysize)
                       4. centpos=None.  In this case, the center of the
                          cutout is just the center of the full image
          imsize  - size of cutout (postage stamp) image, in pixels
                    imsize can take any of the following formats:
                       1. A single number (which will produce a square image)
                       2. A 2-element numpy array
                       3. A 2-element list:  [xsize, ysize]
                       4. A 2-element tuple: (xsize, ysize)
                       5. imsize=None.  In this case, the full image is used
        """

        """ Get the full size of the image """
        hdr = self.header
        nx = hdr['naxis1']
        ny = hdr['naxis2']

        """
        If the subimage center is not already set, then define it as the
        center of the full data set
        """
        if centpos is None:
            xcent = int((nx + 1.) / 2.)
            ycent = int((ny + 1.) / 2.)
        else:
            if centpos[0] is None:
                xcent = int((nx + 1.) / 2.)
            else:
                xcent = centpos[0]
            if centpos[1] is None:
                ycent = int((ny + 1.) / 2.)
            else:
                ycent = centpos[1]

        """
        Define limits of subimage
        For now does not deal with regions partially outside the input file
        """
        if imsize is not None:
            subxy = np.atleast_1d(imsize)  # Converts imsize to a np array
            subx = subxy[0]
            if subxy.size > 1:
                suby = subxy[1]
            else:
                suby = subxy[0]
            halfx = int(subx/2.0)
            halfy = int(suby/2.0)
            x1 = int(xcent - halfx)
            x2 = int(x1 + subx)
            y1 = int(ycent - halfy)
            y2 = int(y1 + suby)
        else:
            x1 = 0
            x2 = nx
            y1 = 0
            y2 = ny

        return x1, y1, x2, y2

    # -----------------------------------------------------------------------

    def cutout_xy(self, x1, y1, x2, y2, nanval=0., fixnans=False,
                  verbose=True):
        """

        Selects the data in the subimage defined by the bounds x1, x2, y1, y2.
        These were either set directly (e.g., by a call to imcopy) or by the
        subim_bounds_xy function (which takes a subimage center and size)

        Inputs:
            verbose - Print out useful information if True (the default)
        """

        ## NEED TO ADD A CHECK FOR VALUES OF X1, X2, ETC. ##
        
        """
        Cut out the subimage based on the bounds.
        Note that radio images often have 4 dimensions (x, y, freq, stokes)
         so for those just take the x and y data
        """
        inhdr = self.header
        if inhdr['naxis'] == 4:
            data = self.data[0, 0, y1:y2, x1:x2].copy()
        else:
            data = self.data[y1:y2, x1:x2].copy()

        """ Fix NaNs """
        if fixnans:
            if nanval == 'max':
                goodmask = np.isfinite(data)
                dmax = data[goodmask].max()
                nanval = 10. * dmax
            data[~np.isfinite(data)] = nanval

        """ Set output image properties """
        nx = x2 - x1
        ny = y2 - y1
        subcentx = 0.5 * (x1 + x2)
        subcenty = 0.5 * (y1 + y2)

        """ Set up new WCS information if the image has WCS """
        hdr = inhdr.copy()
        hdr['naxis'] = 2
        hdr['naxis1'] = nx
        hdr['naxis2'] = ny
        for key in ['naxis3', 'naxis4']:
            if key in hdr:
                del hdr[key]
        if self.wcsinfo is not None:
            # xy = np.array([[subcentx, subcenty]])
            # radec = self.wcsinfo.wcs_pix2world(xy, 0)[0]
            # hdr['crpix1'] = nx / 2.
            # hdr['crpix2'] = ny / 2.
            # hdr['crval1'] = radec[0]
            # hdr['crval2'] = radec[1]
            hdr['crpix1'] -= x1
            hdr['crpix2'] -= y1

        """ Save the new data and header in a PrimaryHDU format """
        subim = pf.ImageHDU(data, hdr)

        """ Print out useful information """
        if verbose:
            print('')
            print('Cutout image center (x, y): (%d, %d)' %
                  (subcentx, subcenty))
            print('Cutout image size (x y): %dx%d' % (nx, ny))

        """
        Update the header info, including updating the CRPIXn values if they
        are present.
        """
        if self.infile is not None:
            subim.header['ORIG_IM'] = 'Copied from %s' % self.infile
        subim.header['ORIG_REG'] = \
            'Region in original image [xrange, yrange]: [%d:%d,%d:%d]' % \
            (x1, x2, y1, y2)

        """ Return the new HDU """
        return subim

    # -----------------------------------------------------------------------

    def cutout_radec(self, imcent, imsize, outscale=None, fixnans=False,
                     nanval=0., theta_tol=1.e-5, verbose=True, debug=True):
        """
        Makes a cutout of the data based on a image center in (ra, dec)
        and image size in arcseconds.

        The vast majority of the code is Matt Auger's (his image_cutout in
        imagelib.py).
        Some modifications have been made by Chris Fassnacht.

        Inputs:
          ra       - Central right ascension in decimal degrees
          dec      - Central declination in decimal degrees
          imsize   - size of the subimage to be displayed in arcsec
                     The default, designated by imsize=None, is to display
                     the entire image.  The imsize parameter can take
                     any of the following formats:
                        1. A single number (which will produce a square image)
                        2. A 2-element numpy array
                        3. A 2-element list:  [xsize, ysize]
                        4. A 2-element tuple: (xsize, ysize)
          outscale - Output image pixel scale, in arcsec/pix.
                      If outscale is None (the default) then the output image
                      scale will be the same as the input image scale
          verbose  - Print out informational statements if True (default=True)
        """

        """
        The following assignments get used whether a subimage has been
        requested or not
        """
        hdr = self.header.copy()
        indata = self.data
        nx = indata.shape[1]
        ny = indata.shape[0]

        """
        Check to make sure that a subimage is even requested.
        If not, then just create a "cutout" that is the full size of the
         image.  This cutout process is needed to set some of the header
         parameters of the displayed HDU correctly
        """
        if imsize is None:
            subim = self.cutout_xy(0, 0, nx, ny, fixnans=fixnans,
                                   nanval=nanval, verbose=False)
            return subim

        """
        If a sub-image is required, set up an astropy WCS structure that will
        be used for the calculations
        """
        w = self.wcsinfo

        """
        If the passed imcent is None, then just take the central pixel
         of the image as the requested center.
        If not, then we need to do some calculations
        """
        if imcent is None:
            x = nx / 2.
            y = ny / 2.

        else:
            """
            We have to convert the requested (RA, dec) center into the
             associated pixel values.
            """
            centradec = np.zeros(hdr['naxis'])
            centradec[0] = imcent[0]
            centradec[1] = imcent[1]
            xy = w.all_world2pix([centradec], 1)[0]
            x = xy[0]
            y = xy[1]

        """ Set the cutout size """
        xysize = np.atleast_1d(imsize)  # Converts imsize to a np array
        xsize = xysize[0]
        if xysize.size > 1:
            ysize = xysize[1]
        else:
            ysize = xysize[0]
        inpixxsize = int((xsize / self.pixscale[0]) + 0.5)
        inpixysize = int((ysize / self.pixscale[1]) + 0.5)

        """ Summarize the request """
        if verbose:
            radec = coords.radec_to_skycoord(imcent[0], imcent[1])
            print('------------------')
            rastr = '%02d %02d %06.3f' % \
                (radec.ra.hms.h, radec.ra.hms.m,
                 radec.ra.hms.s)
            decstr = '%+03d %02d %05.2f' % \
                (radec.dec.dms.d, radec.dec.dms.m,
                 radec.dec.dms.s)
            print(' Requested center (RA, dec): %11.7f    %+10.6f' %
                  (radec.ra.deg, radec.dec.deg))
            print(' Requested center (RA, dec):  %s %s' % (rastr, decstr))
            print(' Requested center (x, y):     %8.2f %8.2f' % (x, y))
            print(' Requested image size (arcsec): %6.2f %6.2f' %
                  (xsize, ysize))
            print(' Requested size in input pixels: %d %d' %
                  (inpixxsize, inpixysize))

        """
        The simple approach -- only valid in some cases

        At this point, see if the following are satisified:
          1. input image PA is zero, or close enough to it (defined by the
              value of theta_tol).
          2. outscale is None

        If they are then just do a pixel-based cutout rather than the more
         complex interpolation that is required otherwise.
        """
        if fabs(self.impa) < theta_tol and outscale is None:
            if verbose:
                print(' ------------------')
                print(' Image PA is effectively zero, so doing pixel-based '
                      'cutout')
            centpos = (x, y)
            imsize = (inpixxsize, inpixysize)
            x1, y1, x2, y2, = self.subim_bounds_xy(centpos, imsize)
            # print(x1, y1, x2, y2)
            subim = self.cutout_xy(x1, y1, x2, y2, nanval=nanval,
                                   fixnans=fixnans, verbose=verbose)
            return subim

        """

        The more complex case -- if outscale is not None or the input image
        does not have the standard rotation, or both

        First set the output scale and number of output pixels
        """
        if outscale is None:
            oscale = [self.pixscale, self.pixscale]
        elif (np.atleast_1d(outscale).size < 2):
            oscale = [outscale, outscale]
        else:
            oscale = outscale
        nx_out = int((xsize / oscale[0]) + 0.5)
        ny_out = int((ysize / oscale[1]) + 0.5)

        """
        Set up the output wcs information.

        Note that the output image will have the standard orientation
        with PA=0
        """
        whdr, subwcs = coords.make_header(imcent, oscale, nx_out, ny_out)

        """ Do the coordinate transform preparation """
        icoords = np.indices((ny_out, nx_out), dtype=np.float32)
        skycoords = subwcs.wcs_pix2world(icoords[1], icoords[0], 0)
        ccdcoords = w.wcs_world2pix(skycoords[0], skycoords[1], 0)
        icoords[0] = ccdcoords[1]
        icoords[1] = ccdcoords[0]
        self.coords = icoords.copy()

        # *** Now need to deal with regions that extend outside the data
        # should be doable, since map_coordinates just takes coordinate pairs
        # so masking the inputted ccdcoords arrays should be possible for
        # coordinates < 0 or > nx, ny

        """ Get the data to be resampled """
        if hdr['naxis'] == 4:
            data = indata[0, 0, :, :].copy()
        else:
            data = indata.copy()

        """ Transform the coordinates """
        if fixnans:
            if nanval == 'max':
                goodmask = np.isfinite(data)
                dmax = data[goodmask].max()
                nanval = 10. * dmax
            data[np.isnan(data)] = nanval
        outdata = ndimage.map_coordinates(data, icoords, order=5)

        """ Save the output as a ImageHDU object """
        outhdr = self.make_hdr_wcs(hdr, subwcs, debug=False)
        if self.infile is not None:
            outhdr['ORIG_IM'] = 'Copied from %s' % self.infile
        subim = pf.ImageHDU(outdata, outhdr)

        """ Clean up and exit """
        del data, icoords, skycoords, ccdcoords
        return subim

