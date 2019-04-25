"""

wcshdu.py

Defines a new class that is essentially a fits PrimaryHDU class but with
some of the WCS information in the header split out into separate
attributes of the class

"""

import os
from math import fabs
import numpy as np
from astropy import wcs
from astropy.io import fits as pf
from cdfutils import coords
from .imutils import open_fits

# ---------------------------------------------------------------------------


class WcsHDU(pf.PrimaryHDU):
    """

    This class is essentially a fits PrimaryHDU class but with some of the
    WCS information that may be in the header split out into separate
    attributes of the class

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
        pixscale = 3600. * wcs.utils.proj_plane_pixel_scales(wcsinfo)

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

    def cutout_xy(self, x1, y1, x2, y2, nanval=0., verbose=True):
        """

        Selects the data in the subimage defined by the bounds x1, x2, y1, y2.
        These were either set directly (e.g., by a call to imcopy) or by the
        subim_bounds_xy function (which takes a subimage center and size)

        Inputs:
            verbose - Print out useful information if True (the default)
        """

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
            xy = np.array([[subcentx, subcenty]])
            radec = self.wcsinfo.wcs_pix2world(xy, 0)[0]
            hdr['crpix1'] = nx / 2.
            hdr['crpix2'] = ny / 2.
            hdr['crval1'] = radec[0]
            hdr['crval2'] = radec[1]

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

    def cutout_radec(self, imcent, imsize, outscale=None,
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
            subim = self.cutout_xy(0, 0, nx, ny, verbose=False)
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
            xy = w.wcs_world2pix([centradec], 0)[0]
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
            subim = self.cutout_xy(x1, y1, x2, y2, nanval=nanval,
                                   verbose=verbose)
            return subim

        """

        The more complex case -- if outscale is not None or the input image
        does not have the standard rotation, or both

        First set the output scale and number of output pixels
        """
        if outscale is None:
            oscale = self.pixscale
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
        # so masking the inputte ccdcoords arrays should be possible for
        # coordinates < 0 or > nx, ny

        """ Get the data to be resampled """
        if hdr['naxis'] == 4:
            data = indata[0, 0, :, :].copy()
        else:
            data = indata.copy()

        """ Transform the coordinates """
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

