"""
imfuncs.py - A library of functions to do various basic image processing
                 operations

NB: Some of these functions are (slowly) being incorporated into the
Image class, the code for which is at the beginning of this file.

Methods (functions) in the Image class
--------------------------------------
Lots of functionality now, but not yet documented well.

Stand-alone functions
---------------------
    open_fits         - opens a fits file, incorporating the possibility of
                         the "missing end" problem that affects some of the
                         Keck NIR data
    imcopy            - copies a portion of a fits file given the corners (in
                         pixels)
    poststamp         - copies a portion of a fits file given the center and
                         size (in pixels)
    image_cutout      - copies a portion of a fits file given the center and
                         size (center in RA, dec)
    make_snr_image    - calculates the rms noise in an image and divides by
                         that value to create a SNR image, which can be
                         written as an output fits file.
    overlay_contours  - plots a greyscale image and overlays contours from a
                         second image
    calc_sky_from_seg - calculates sky level using SExtractor segmentation map
    quick_display     - displays an image with most of the display parameters
                         set to their default values
    plot_cat          - given a fits image and a object catalog, marks the
                         positions of the catalog objects.
"""

import os
from math import log, log10, sqrt, pi, fabs, atan
from math import cos as mcos, sin as msin
try:
    from astropy.io import fits as pf
except:
    import pyfits as pf
try:
    from astropy.coordinates import SkyCoord
except ImportError:
    from astropy.coordinates import ICRS as SkyCoord
from astropy import wcs
from astropy import units as u
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import datafuncs as df

# -----------------------------------------------------------------------


# class Image(pf.HDUList):#
class Image:

    def __init__(self, indat, datahext=0, hdrhext=0, wcshext=0, verbose=True):
        """
        This method gets called when the user types something like
            myim = Image(infile)

        Reads in the image data from an input fits file or a HDUList from a
         previously loaded fits file (or [not yet implemented] a PrimaryHDU).
        The image data is stored in a Image class container.

        Required inputs:
            indat - The input image data.  This can either be:
                     1. a filename, the most common case
                           - or -
                     2. a HDU list.
        """

        """ Check the format of the input image data """
        if isinstance(indat, str):
            informat = 'file'
        elif isinstance(indat, pf.HDUList):
            informat = 'hdulist'
        else:
            print('')
            print('ERROR: The input image data for the Image class must be'
                  'one of the following:')
            print('  1. A filename (i.e. a string)')
            print('  2. A HDUList')
            print('')
            raise TypeError

        """ Load the hdu information """
        if informat == 'hdulist':
            self.hdu = indat
            self.infile = None

        else:
            try:
                self.read_from_file(indat, verbose=verbose)
            except IOError:
                return None

        if verbose:
            self.hdu.info()

        """ Set up pointers to the default data and header """
        self.data = self.hdu[datahext].data.copy()
        self.hdr = self.hdu[hdrhext].header.copy()

        """ Do an initial import of the WCS information from the header """
        self.found_wcs = False
        self.radec = None
        self.pixscale = None
        try:
            self.get_wcs(hext=wcshext, verbose=verbose)
        except:
            if verbose:
                print('Could not load WCS info')
            self.found_wcs = False

        """ Initialize figures """
        self.fig1 = None
        self.fig2 = None

        """
        Initialize default display parameters

         - The scale for the display (i.e., the data values that correspond
            to full black and full white on a greyscale display) are
            (by default) set in terms of the "clipped mean" and "clipped rms".
            Those values are the mean and rms of the data after a sigma
            clipping algorithm has been applied to reject outliers.
         - The display min and max values are stored as self.fmin and self.fmax
         - For more information see the set_display_limits method
        """
        self.found_rms = False       # Have clipped rms / mean been calculated?
        self.mean_clip = 0.0         # Value of the clipped mean
        self.rms_clip = 0.0          # Value of the clipped rms
        self.fmin = None             # Lower flux limit used in image display
        self.fmax = None             # Upper flux limit used in image display
        self.fscale = 'linear'       # Flux scaling for display
        self.statsize = 2048         # Stats region size if image is too big
        self.statsec = None          # Region to use for pixel statistics
        self.zoomsize = 31           # Size of postage-stamp zoom
        self.mode = 'radec'          # Default display units are arcsec offsets
        self.extval = None           # Just label the axes by pixels
        self.cmap = plt.cm.YlOrBr_r  # This corresponds to the 'gaia' cmap

        """ Initialize contouring parameters """
        self.contbase = sqrt(3.)
        self.clevs = None
        self.overlay_im = None  # Not currently used

        """ Initialize radplot parameters """
        self.radplot_center = True
        self.radplot_maxshift = 5.
        self.radplot_file = None

        """ Initialize other parameters """
        self.reset_subim()
        self.reset_imex()

    # -----------------------------------------------------------------------

    def read_from_file(self, infile, verbose=True):
        """
        Reads the image data from a file (as opposed to getting the data
        directly from an input HDUList
        """
        if verbose:
            print('')
            print "Loading file %s" % infile
            print "-----------------------------------------------"

        if os.path.isfile(infile):
            try:
                self.hdu = open_fits(infile)
            except:
                print(' ERROR. Problem in loading file %s' % infile)
                print(' Check to make sure filename matches an existing'
                      'file')
                print(' If it does, there may be something wrong with the'
                      ' fits header.')
                print('')
                raise IOError
        else:
            print('')
            print('ERROR. File %s does not exist.' % infile)
            print('')
            raise IOError

        """ Set parameters related to image properties """
        self.infile = infile

    # -----------------------------------------------------------------------

    def set_data(self, imslice=0, raax=0, decax=1, specax=2, hext=0):
        """
        Sets the 2-dimension slice to use for the display functions.
        For nearly all imaging data, there is only one image slice, and so
        the default behavior is fine.  If, however, the input data are from
        an IFU and so there is a real data cube, then the choice of which
        image slice to use makes a difference.

        Inputs:
          imslice - which image slice to use.  The default value (imslice=0)
                    is fine for nearly all standard imaging.  Only for, e.g.,
                    IFU data, might you choose a different value.
        """

        """ Get the number of dimensions in the input image """
        # hdr = self.hdu[hext].header
        # if 'naxis' in hdr.keys():
        #     ndim = hdr['naxis']
        # else:
        #     raise KeyError

        """ Select the image slice to use """

    # -----------------------------------------------------------------------

    def reset_subim(self):
        """
        Returns the sub-image variables to their initial, unset, state
        """

        self.subim = None
        self.subcentx = None
        self.subcenty = None
        self.subsizex = None
        self.subsizey = None

    # -----------------------------------------------------------------------

    def reset_imex(self):
        """
        Resets the parameters that are associated with the imexam-like
         processing
        """

        """ Initialize coordinates and data """
        self.imex_x = None
        self.imex_y = None
        self.imex_data = None

        """ Initialize moment calculations """
        self.imex_mux = None
        self.imex_muy = None
        self.imex_sigxx = None
        self.imex_sigyy = None
        self.imex_sigxy = None

    # -----------------------------------------------------------------------

    def close(self):
        """
        Closes the image
        """

        self.hdu.close()
        return

    # -----------------------------------------------------------------------

    def sigma_clip(self, nsig=3., statsec=None, mask=None, hext=0,
                   verbose=False):
        """
        Runs a sigma-clipping on image data.  The code iterates over
        the following steps until it has converged:
            1. Compute mean and rms noise in the (clipped) data set
            2. Reject (clip) data points that are more than nsig * rms from
                the newly computed mean (using the newly computed rms)
            3. Repeat until no new points are rejected
        Once convergence has been reached, the final clipped mean and clipped
        rms are stored in the mean_clip and rms_clip variables

        NOTE: The region used for determining these image statistics is set
        by the following decision path:
         - if statsec is not None, use statsec
         - else, if the subim has been set, use the subim
         - else, use the entire image
        for the second and third options, an optional mask can be used to
        exclude known bad pixels from the calculation.

        Optional inputs:
          nsig    - Number of sigma from the mean beyond which points are
                     rejected.  Default=3.
          statsec - Region of the input image to be used to determine the
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
          mask    - If some of the input data are known to be bad, they can
                      be flagged before the inputs are computed by including
                      a mask.  This mask must be set such that True
                      indicates good data and False indicates bad data
          hext    - Image HDU containing the data.  The default (hext=0)
                      should work for all single-extension fits files and may
                      work for some multi-extension files.
                      NOTE: hext is ignored if the subim variable is already
                      set by, e.g., set_subim_xy or def_subim_radec
          verbose - If False (the default) no information is printed
        """

        """ Determine what the input data set is """
        if statsec is not None:
            x1, y1, x2, y2 = statsec
            data = self.hdu[hext].data[y1:y2, x1:x2]
        elif self.data is not None:
            data = self.data.copy()
        else:
            data = self.hdu[hext].data.copy()

        """ Find the clipped mean and rms """
        mu, sig = df.sigclip(data, nsig=nsig, mask=mask, verbose=verbose)

        """ Store the results and clean up """
        del data
        self.mean_clip = mu
        self.rms_clip = sig
        return

    # -----------------------------------------------------------------------

    def make_var(self, objmask=None, returnvar=True, hext=0, verbose=True):
        """
        Make a variance image under the assumption that the data in the
        current image is (1) in units of electrons, and (2) has not been
        sky subtracted.  The basic procedure is as follows:
          1. Find the clipped mean and clipped rms through a call to sigma_clip
          2. Sanity check by comparing the clipped rms to the square root of
              the clipped mean.
          3. Set all pixels in the variance image to the square of the clipped
              rms.
          4. If an object mask is set, then set the pixels in the variance
              image that within the mask to the pixel values in the input data
              (i.e., assume Poisson statistics here) as long as they are larger
              than the clipped rms.
              NOTE: The object mask is defined to have a value of 1 where the
              objects are and 0 otherwise.
        """

        """ Convert the input mask into the format needed for sigma_clip """
        if objmask is not None:
            mask = objmask == 0
        else:
            mask = None

        """ Get the clipped mean and rms """
        self.sigma_clip(mask=mask, hext=hext)

        """ Sanity check """
        sqrtmean = sqrt(self.mean_clip)
        check = sqrtmean / self.rms_clip
        if check < 0.9 or check > 1.1:
            print('Warning: %s - ratio of sqrt(mean) to rms is more than'
                  ' 10 percent from unity' % self.infile)
            print(' sqrt(mean) = %7.2f, rms = %7.2f' %
                  (sqrtmean, self.rms_clip))

        """ Create the base variance image """
        data = self.hdu[hext].data
        varval = (self.rms_clip)**2
        var = np.ones(data.shape) * varval

        """
        If an object mask exists, replace the pixel values in the variance
        image with the values from the image data which, under the assumption
        of Poisson statistics, should represent the associated pixel
        uncertainties.
        Only do this replacement if the data values are greater than the
        rms**2 values currently in the variance image
        """
        if objmask is not None:
            omask = (objmask > 0) & (data > varval)
            var[omask] = data[omask]

        """ Store the variance map, and also return it if requested """
        self.var = var
        if returnvar:
            return var

    # -----------------------------------------------------------------------

    def start_interactive(self):
        self.xmark = None
        self.ymark = None
        self.cid_mouse = self.fig1.canvas.mpl_connect('button_press_event',
                                                      self.onclick)
        self.cid_keypress = self.fig1.canvas.mpl_connect('key_press_event',
                                                         self.keypress)
        self.keypress_info()
        return

    # -----------------------------------------------------------------------

    def keypress_info(self):
        """
        Prints useful information about what the key presses do
        """
        print('')
        print('Actions available by pressing a key in the Figure 1 window')
        print('----------------------------------------------------------')
        print('Key        Action')
        print('-------  ---------------------------------')
        print('[click]  Report (x, y) position, and (RA, dec) if file has WCS')
        print('    m      Mark the position of an object')
        print('    z      Zoom in at the position of the cursor')
        print('    q      Quit and close the window')
        print('    x      Quit but do not close the window')
        print('')

    # -----------------------------------------------------------------------

    def onclick(self, event):
        """
        Actions taken if a mouse button is clicked.  In this case the
        following are done:
          (1) Store and print (x, y) value of cursor
          (2) If the image has wcs info (i.e., if found_wcs is True) then
                store and print the (RA, dec) value associated with the (x, y)
        """
        self.xclick = event.xdata
        self.yclick = event.ydata
        print('')
        print 'Mouse click x, y:    %7.1f %7.1f' % (self.xclick, self.yclick)

        """
        Also show the (RA, dec) of the clicked position if the input file has
         a WCS solution
        NOTE: This needs to be handled differently if the displayed image has
         axes in pixels or in arcsec offsets
        """
        if self.found_wcs:
            if self.mode == 'xy':
                pix = np.zeros((1, self.wcsinfo.naxis))
                pix[0, 0] = self.xclick
                pix[0, 1] = self.yclick
                radec = self.wcsinfo.wcs_pix2world(pix, 0)
                self.raclick = radec[0, 0]
                self.decclick = radec[0, 1]
            else:
                """ For now use small-angle formula """
                cosdec = mcos(self.radec.dec.radian)
                self.raclick = self.radec.ra.degree + \
                    (self.xclick + self.zeropos[0]) / (3600. * cosdec)
                self.decclick = self.radec.dec.degree + self.yclick/3600. + \
                    self.zeropos[1]
            print 'Mouse click ra, dec: %11.7f %+11.7f' % \
                (self.raclick, self.decclick)
        return

    # -----------------------------------------------------------------------

    def keypress(self, event):
        """
        Actions taken if a key on the keyboard is pressed
        """

        if event.key == 'f':
            """
            Change the display range
            """
            print('')
            self.set_display_limits(fmax=None, funits='abs')

        if event.key == 'm':
            """
            Mark an object.  Hitting 'm' saves the (x, y) position into
            the xmark and ymark variables
            """
            global xmark, ymark
            print('')
            print('Marking position %8.2f %8.2f' % (event.xdata, event.ydata))
            print('')
            self.xmark = event.xdata
            self.ymark = event.ydata
            imsize = (self.zoomsize, self.zoomsize)
            imcent = (self.xmark, self.ymark)
            self.display(imcent=imcent, imsize=imsize, mode=self.mode,
                         show_xyproj=True)

        if event.key == 'z':
            """
            Zoom in by a factor of two at the location of the cursor
            """
            xzoom, yzoom = event.xdata, event.ydata
            xl1, xl2 = self.ax1.get_xlim()
            yl1, yl2 = self.ax1.get_ylim()
            dx = (xl2 - xl1)/4.
            dy = (yl2 - yl1)/4.
            xz1 = min((max(xl1, (xzoom - dx))), (xzoom - 1.))
            xz2 = max((min(xl2, (xzoom + dx))), (xzoom + 1.))
            yz1 = min((max(yl1, (yzoom - dy))), (yzoom - 1.))
            yz2 = max((min(yl2, (yzoom + dy))), (yzoom + 1.))
            self.ax1.set_xlim(xz1, xz2)
            self.ax1.set_ylim(yz1, yz2)
            self.fig1.show()
            return

        if event.key == 'x':
            print('')
            print('Stopping interactive mode')
            print('')
            if self.fig1:
                self.fig1.canvas.mpl_disconnect(self.cid_mouse)
                self.fig1.canvas.mpl_disconnect(self.cid_keypress)
            if self.fig2:
                self.fig2.canvas.mpl_disconnect(self.cid_keypress2)
            return

        if event.key == 'q':
            print('')
            print 'Closing down'
            print('')
            if self.fig1:
                self.fig1.canvas.mpl_disconnect(self.cid_mouse)
                self.fig1.canvas.mpl_disconnect(self.cid_keypress)
            if self.fig2:
                self.fig2.canvas.mpl_disconnect(self.cid_keypress2)
            for ii in plt.get_fignums():
                plt.close(ii)
            return

        self.keypress_info()
        return

    # -----------------------------------------------------------------------

    def radec_to_skycoord(self, ra, dec):
        """
        Converts a (RA, dec) pair into the astropy.coordinates SkyCoord
        format

        Required inputs:
          ra  - RA in one of three formats:
                 Decimal degrees: ddd.ddddddd  (as many significant figures
                   as desired)
                 Sexigesimal:     hh mm ss.sss (as many significant figures
                   as desired)
                 Sexigesimal:     hh:mm:ss.sss (as many significant figures
                   as desired)
          dec - Dec in one of three formats:
                 Decimal degrees: sddd.ddddddd, where "s" is + or -
                 Sexigesimal:     sdd mm ss.sss (as many significant figures
                    as desired)
                 Sexigesimal:     sdd:mm:ss.sss (as many significant figures
                    as desired)
        """

        """ Get RA format """
        if type(ra) == float or type(ra) == np.float32 or \
                type(ra) == np.float64:
            rafmt = u.deg
        else:
            rafmt = u.hourangle

        """ Dec format is always degrees, even if in Sexigesimal format """
        decfmt = u.deg

        """ Do the conversion """
        radec = SkyCoord(ra, dec, unit=(rafmt, decfmt))
        return radec

    # -----------------------------------------------------------------------

    def get_wcs(self, hext=0, verbose=True):
        """
        Reads in WCS information from the header and stores it in
        new wcsinfo (see below) and pixscale variables.
        NOTE: This used to use Matt Auger's wcs library, but it has
         been converted to the astropy wcs module

        Inputs:
            hext - Header extension that contains the WCS info.  Default=0

        """

        """ Read in the header and use it to set the WCS information"""
        hdr = self.hdu[hext].header
        try:
            self.wcsinfo = wcs.WCS(hdr)
        except:
            if verbose:
                print('get_wcs: No WCS information in image header')
            self.found_wcs = False
            return

        """
        Make sure that the WCS information is actually WCS-like and not,
        for example, pixel-based
        """

        imwcs = self.wcsinfo.wcs
        rafound = False
        decfound = False
        count = 0
        for ct in imwcs.ctype:
            if ct[0:2] == 'RA':
                rafound = True
                raax = count
                rakey = 'naxis%d' % (count + 1)
            if ct[0:3] == 'DEC':
                decfound = True
                decax = count
                deckey = 'naxis%d' % (count + 1)
            count += 1
        if rafound is False or decfound is False:
            if verbose:
                print('get_wcs: No valid WCS information in image header')
                print('         RA and/or DEC information is missing')
            self.found_wcs = False
            return

        """ Get the RA and Dec of the center of the image """
        xcent = hdr[rakey] / 2.
        ycent = hdr[deckey] / 2.
        imcent = np.ones((1, hdr['naxis']))
        imcent[0, raax] = xcent
        imcent[0, decax] = ycent
        imcentradec = self.wcsinfo.wcs_pix2world(imcent, 1)
        self.radec = self.radec_to_skycoord(imcentradec[0, raax],
                                            imcentradec[0, decax])

        """ Calculate the pixel scale and image PA (E of N) """
        self.found_wcs = True
        w = self.wcsinfo.wcs
        rad2deg = 180. / pi
        if imwcs.has_cd():
            self.pixscale = sqrt(w.cd[raax, raax]**2 +
                                 w.cd[decax, raax]**2) * 3600.
            rot1 = atan(-1. * w.cd[raax, decax] / w.cd[decax, decax])
            rot2 = atan(w.cd[decax, raax] / w.cd[raax, raax])
        elif imwcs.has_pc():
            self.pixscale = sqrt(w.pc[raax, raax]**2 + w.pc[decax, raax]**2) *\
                w.cdelt[raax] * 3600.
            rot1 = atan(-1. * w.cdelt[raax] * w.pc[raax, decax] /
                        (w.cdelt[decax] * w.pc[decax, decax]))
            rot2 = atan(w.cdelt[decax] * w.pc[decax, raax] /
                        (w.cdelt[raax] * w.pc[raax, raax]))
        elif isinstance(imwcs.cdelt, np.ndarray):
            self.pixscale = abs(w.cdelt[raax]) * 3600.
        else:
            print 'Warning: no WCS info in header %d' % hext
            self.found_wcs = False

        """ Save the PA information """
        rot1 *= rad2deg
        rot2 *= rad2deg
        if fabs(rot1 - rot2) < 1.:
            self.instpa = (rot1 + rot2) / 2.
        else:
            self.instpa = np.array([rot1, rot2])

        if self.found_wcs and verbose:
            print('Pixel scale: %7.3f arcsec/pix' % self.pixscale)
            print('Instrument FOV (arcsec): %7.1f %7.1f' %
                  (self.pixscale * hdr[rakey], self.pixscale * hdr[deckey]))
            if isinstance(self.instpa, np.ndarray):
                print('Instrument position angles (E of N): %+7.2f %+7.2f' %
                      (self.instpa[0], self.instpa[1]))
            else:
                print('Instrument position angle (E of N): %+7.2f' %
                      self.instpa)

    # -----------------------------------------------------------------------

    def set_pixscale(self):
        """
        Interactively set the pixel scale
        """

        print('')
        self.pixscale = \
            float(raw_input('Enter the pixel scale for the image '
                            'in arcsec/pix: '))

    # -----------------------------------------------------------------------

    def make_header(self, radec, pixscale, nx, ny=None, rot=None):
        """

        Makes a header with wcs information.

        Inputs:
          radec    - The desired (RA, Dec) pair to be put into the CRVAL
                      keywords.
                      NOTE: This should be in the SkyCoord format defined in
                       astropy.coordinates.  To convert a "normal" pair of
                       numbers / sexigesimal strings to SkyCoord format, use
                       the radec_to_skycoord method in this Image class.
          pixscale - Desired pixel scale in arcsec/pix
          nx       - image size along the x-axis --or-- if the image is square
                       (indicated by ny=None) then this is also the y-axis size
          ny       - [OPTIONAL] y-axis size, if different from the x-axis size
                       ny=None means that the two axes have the same size
          rot      - [OPTIONAL] desired rotation angle, in degrees E of N.
                       NOT IMPLEMENTED YET
        """

        """ Create a blank 2d WCS container """
        w = wcs.WCS(naxis=2)

        """ Get the image size and central pixel """
        cp1 = nx / 2.
        if ny is None:
            cp2 = cp1
        else:
            cp2 = ny / 2.

        """ Fill it in with appropriate values and save it """
        px = pixscale / 3600.
        w.wcs.crpix = [cp1, cp2]
        w.wcs.crval = [radec.ra.degree, radec.dec.degree]
        w.wcs.cdelt = [(-1.*px), px]
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        w.wcs.equinox = 2000.
        self.subim_wcs = w

        """ Convert to a fits header format """
        hdr = w.wcs.to_header()
        return hdr

    # -----------------------------------------------------------------------

    def set_wcsextent(self, hext=0, zeropos=None):
        """
        For making plots with WCS information, it is necessary to define
        the boundaries in terms of RA and Dec offsets from the center, in
        arcsec.  For this purpose, the imshow and contour methods in
        matplotlib.pyplot have an 'extent' parameter.

        This set_wcsextent method will use the WCS information in the fits
        header to properly set the extent parameter values and return them.
        These are put into the "extval" container, which is part of the Image
        class.  extval is a four-element tuple containing the coordinates of
        the lower left and upper right corners, in terms of RA and Dec
        offsets.

        Optional inputs:
          hext    - HDU containing the WCS info.  Default=0
          zeropos - By default, which happens when zeropos=None, the (0, 0)
                     point on the output image, as designated by the image
                     axis labels, will be at the center of the image.
                     However, you can shift the (0, 0) point to be somewhere
                     else by setting zeropos.  For example, zeropos=(0.5, 0.3)
                     will shift the origin to the point that would have been
                     (0.5, 0.3) if the origin were at the center of the image
        """

        self.get_wcs(hext)
        coords = np.indices(self.data.shape).astype(np.float32)
        pltc = np.zeros(coords.shape)
        pltc[0] = (coords[0] - self.data.shape[0] / 2.) * self.pixscale
        pltc[1] = (coords[1] - self.data.shape[1] / 2.) * self.pixscale
        pltc[1] *= -1.
        maxi = np.atleast_1d(self.data.shape) - 1
        extx1 = pltc[1][0, 0]
        exty1 = pltc[0][0, 0]
        extx2 = pltc[1][maxi[0], maxi[1]] - self.pixscale
        exty2 = pltc[0][maxi[0], maxi[1]] + self.pixscale

        if zeropos is not None:
            dx = zeropos[0]
            dy = zeropos[1]
        else:
            dx = 0.
            dy = 0.
        extx1 -= dx
        extx2 -= dx
        exty1 -= dy
        exty2 -= dy

        """ Set the extval values, and also record the zerpos values used """
        self.extval = (extx1, extx2, exty1, exty2)
        self.zeropos = (dx, dy)

    # -----------------------------------------------------------------------

    def im_moments(self, x0, y0, rmax=10., detect_thresh=3., skytype='global',
                   hext=0, verbose=False):
        """
        Given an initial guess of a centroid position, calculates the
        flux-weighted first and second moments within a square centered
        on the initial guess point and with side length of 2*rmax + 1.
        The moments will be estimates of the centroid and sigma of the
        light distribution within the square.

        Inputs:
          x0      - initial guess for x centroid
          y0      - initial guess for y centroid
          rmax    - used to set size of image region probed, which will be a
                     square with side = 2*rmax + 1. Default=10
          skytype - set how the sky/background level is set.  Three options:
                     'global' - Use the clipped mean as determined by the
                                sigma_clip method.  This is the default.
                     'local'  - Use a region that surrounds the source
                                NOT YET IMPLEMENTED
                     None     - Don't do any sky/background subtraction
          hext    - HDU extension that contains the data.  Default = 0
        """

        """ Define the data and the coordinate arrays """
        data = self.hdu[hext].data
        y, x = np.indices(data.shape)

        """
        Select the data within the square of interest
        """
        x1, x2 = x0-rmax-1, x0+rmax+1
        y1, y2 = y0-rmax-1, y0+rmax+1
        pixmask = (x > x1) & (x < x2) & (y > y1) & (y < y2)
        if skytype is None:
            f = data[pixmask]
        else:
            if self.found_rms is False:
                self.sigma_clip(verbose=verbose)
                self.found_rms = True
            print self.mean_clip, self.rms_clip
            f = data[pixmask] - self.mean_clip
        self.imex_x = x[pixmask]
        self.imex_y = y[pixmask]

        """
        Calculate the flux-weighted moments
         NOTE: Do the moment calculations relative to (x1, y1) -- and then add
          x1 and y1 back at the end -- in order to avoid rounding errors (see
          SExtractor user manual)
        """
        objmask = f > self.mean_clip + detect_thresh * self.rms_clip
        fgood = f[objmask]
        """
        """
        xgood = self.imex_x[objmask] - x1
        ygood = self.imex_y[objmask] - y1
        fsum = fgood.sum()
        mux = (fgood * xgood).sum() / fsum
        muy = (fgood * ygood).sum() / fsum
        self.imex_mux = mux + x1
        self.imex_muy = muy + y1
        self.imex_sigxx = (fgood * xgood**2).sum() / fsum - mux**2
        self.imex_sigyy = (fgood * ygood**2).sum() / fsum - muy**2
        self.imex_sigxy = (fgood * xgood*ygood).sum() / fsum - mux*muy
        print self.imex_mux, self.imex_muy
        print sqrt(self.imex_sigxx), sqrt(self.imex_sigyy), self.imex_sigxy

    # -----------------------------------------------------------------------

    def eval_gauss_1d_r_plus_bkgd(self, p, r, y):
        """
        Compares the data to the model in the case of a one-sided gaussian
         fit to a radial profile, where there is also a constant background
         level.
        In this case, the mean is fixed to be mu=0.
        The parameter values are:
         p[0] = sigma
         p[1] = amplitude
         p[2] = background
        """

        """ Unpack p """
        sigma = p[0]
        amp = p[1]
        bkgd = p[2]

        """
        Compute the difference between model and real values
        """

        ymod = bkgd + amp * np.exp(-0.5 * (r/sigma)**2)
        diff = y - ymod

        return diff

    # -----------------------------------------------------------------------

    def eval_gauss_1d_r(self, p, r, y):
        """
        Compares the data to the model in the case of a one-sided gaussian
         fit to a radial profile, where there is no background
        In this case, the mean is fixed to be mu=0.
        The parameter values are:
         p[0] = sigma
         p[1] = amplitude
        """

        """ Unpack p """
        sigma = p[0]
        amp = p[1]

        """
        Compute the difference between model and real values
        """

        ymod = amp * np.exp(-0.5 * (r/sigma)**2)
        diff = y - ymod

        return diff

    # -----------------------------------------------------------------------

    def fit_gauss_1d_r(self, r, flux, bkgd=None):
        """
        Fits a 1-d gaussian to a flux profile that is a function OF RADIUS.
        In other words, this is a one-sided fit, where the expected mu=0, so
        all that is being fit for is the amplitude and sigma.
        """

        from scipy import optimize

        """ Set up the defaults """
        amp0 = max(flux)
        sig0 = max(r[flux > (amp0 / 2.)])
        mf = 100000  # Maximum number of evaluations

        """ Set up for the cases with and without a background """
        if bkgd is not None:
            p = [sig0, amp0, bkgd]
            p_out, ier = optimize.leastsq(self.eval_gauss_1d_r_plus_bkgd, p,
                                          (r, flux), maxfev=mf)
            self.rprof_sig = p_out[0]
            self.rprof_amp = p_out[1]
            self.rprof_bkgd = p_out[2]
        else:
            p = [sig0, amp0]
            p_out, ier = optimize.leastsq(self.eval_gauss_1d_r, p, (r, flux),
                                          maxfev=mf)
            self.rprof_sig = p_out[0]
            self.rprof_amp = p_out[1]
            self.rprof_bkgd = 0.

        self.rprof_mu = 0.

    # -----------------------------------------------------------------------

    def circ_profile(self, r, flux, rmax_fit=5., verbose=True):
        """
        Computes a circularly averaged profile from the provided radius and
        flux vectors.

        Required inputs:
          r     - pre-computed radius vector, i.e., distances of the pixels
                     from some reference pixel
          flux - fluxes at the points in the r vector

        """
        max_r = np.floor(max(r)+1)
        rcirc = np.arange(1, max_r)
        r_ann0 = np.zeros(rcirc.size)
        f_ann0 = np.zeros(rcirc.size)
        for i in range(rcirc.size):
            if i == 0:
                mask = r <= rcirc[i]
            else:
                mask = (r > rcirc[i-1]) & (r <= rcirc[i])
            npts = mask.sum()
            if npts == 0:
                f_ann0[i] = -99
            else:
                f_ann0[i] = flux[mask].sum() / npts
                r_ann0[i] = ((r[mask] * flux[mask]).sum()) / flux[mask].sum()
        mask2 = f_ann0 > 0.
        self.rcirc = r_ann0[mask2]
        self.fcirc = f_ann0[mask2]

        """
        Fit a 1 dimensional Gaussian to the circularly averaged profile, only
         fitting out to rmax_fit
        """
        r2 = self.rcirc[self.rcirc < rmax_fit]
        f2 = self.fcirc[self.rcirc < rmax_fit]
        self.fit_gauss_1d_r(r2, f2)

        if verbose:
            print self.rcirc
            print self.fcirc
            print self.rprof_amp, self.rprof_sig

    # -----------------------------------------------------------------------

    def radplot(self, x0, y0, rmax, center=True, imex_rmax=10., maxshift=5.,
                skylevel=0., zp=None, runit='pixel', logr=False, hext=0,
                doplot=True, normalize=False, outfile=None,
                outtype='radplot'):
        """
        Given a position in the image file (the x0 and y0 parameters), makes
         a plot of image flux as a function of distance from that (x, y)
         position, out to a maximum distance of rmax.
        The default is to make the plot in flux units (or ADU or counts or
         counts/sec).  However, if the zero point parameter (zp) is set
         then the values in the data array will be converted into surface
         brightness in magnitudes, via the usual formula:
            mu = -2.5 log10(data) + zp

        Required inputs:
          x0    - x coordinate
          y0    - y coordinate
          rmax - maximum radius, in pixels, for the plot
        Optional inputs:
          center    - If True (the default) then there will be a call to
                       im_moments to re-calculate the center position based on
                       the initial guess provided by x0 and y0.
          imex_rmax - Maximum radius used for computing object moments in the
                       call to im_moments
          maxshift  - Only used if center is True.  Maximum shift from the
                       original(x0, y0) guess (in pixels) that is allowed.
                       If im_moments returns a new central position that is
                       more than maxshift from the original guess, then that
                       new solution is rejected and the original guess is used
                       instead.
                       Default = 5 pixels
          skylevel  - If the sky has not been subtracted from the data, then
                       the integrated counts, surface brightness in
                       mag/arcsec**2, and integrated magnitude will all be
                       wrong.
                       Set this parameter to the rough sky level to address
                       these issues.
                       The default (skylevel=0) is appropriate if the sky
                       _has_ been subtracted.
          zp        - zero point.  If this parameter is set, then the output
                       plot will be in magnitude units (i.e., surface
                       brightness) rather than the default flux-like units
                       (ADU, counts, counts/sec, etc.)
          runit     - units for the x-axis of the plot.  The only options are
                       'pixel' (the default) or 'arcsec'
          logr      - If False (the default) then x-axis is linear. If true,
                       then it is in log
          hext      - HDU extension that contains the data.  Default = 0
          doplot    - Sets whether a plot is made or not.  Default is
                       doplot=True
          normalize - Normalize so that central value of the profile is 1.0?
                       Default is normalize=False
          outfile   - Write radplot data to an output file if set. The default
                       value (outfile=None) means that no output file will be
                       written
          outtype   - Output file data time.  The options are 'radplot' or
                       'fcirc'.
                       Choosing 'radplot' (the default) writes out
                        the radial position and flux for all pixels with in
                        the requested region.
                       Choosing 'fcirc' writes out a circularly averaged
                       profile.
        """

        """ Define the data and the coordinate arrays """
        data = self.hdu[hext].data
        y, x = np.indices(data.shape)

        """ Recenter from the initial guess using the flux distribution """
        if center:
            self.im_moments(x0, y0, rmax=imex_rmax)
            xc = self.imex_mux
            yc = self.imex_muy
            # STILL IMPLEMENT CHECK FOR MAXSHIFT
        else:
            xc = x0
            yc = y0

        """
        Find the offsets of each pixel in the data array from the central
         location
        For better speed, only actually do the computations for pixels that
         might be in the correct area
        """
        x1, x2 = xc-rmax-1, xc+rmax+1
        y1, y2 = yc-rmax-1, yc+rmax+1
        pixmask = (x > x1) & (x < x2) & (y > y1) & (y < y2)
        dx = x[pixmask] - xc
        dy = y[pixmask] - yc
        r = np.sqrt(dx**2 + dy**2)

        """
        Get the pixel scale if needed, which it will be if either runit==arcsec
         or if zp is set (and thus the first plot is mag/arcsec**2).
        """
        if zp or (runit == 'arcsec'):
            if self.pixscale is None:
                self.set_pixscale()
            print 'Using pixel scale of %6.3f arcsec/pix' % self.pixscale

        """ Select the points within rmax and convert to mags if desired """
        ii = np.argsort(r)
        if runit == 'arcsec':
            rr = r[ii] * self.pixscale
            xlab = 'r (arcsec)'
            rmax *= self.pixscale
        else:
            rr = r[ii]
            xlab = 'r (pixels)'
        rflux = (data[pixmask])[ii] - skylevel
        if zp:
            domega = self.pixscale**2
            mu = -2.5 * np.log10(rflux/domega) + zp
            ftype = 'Surface Brightness'
            ttype = 'Magnitude'
            flab = 'Surface rrightness (mag/arcsec**2)'
            tlab = 'Integrated magnitude within r (mag)'
        else:
            ftype = 'Counts'
            ttype = 'Counts'
            flab = 'Counts / pixel'
            tlab = 'Integrated counts within r'

        """ Compute the circularly averaged flux profile """
        self.circ_profile(rr, rflux)
        rfit = np.linspace(0, rmax, 500)
        ffit = self.rprof_bkgd + self.rprof_amp * \
            np.exp(-0.5 * (rfit / self.rprof_sig)**2)
        if normalize:
            rflux /= ffit[0]
            self.fcirc /= ffit[0]
            ffit /= ffit[0]

        """ Plot the surface brightness if requested """
        if doplot:
            ax1 = plt.subplot(211)
            if zp:
                if logr:
                    plt.semilogx(rr, mu, '+')
                else:
                    plt.plot(rr, mu, '+')
                yl1, yl2 = plt.ylim()
                plt.ylim(yl2, yl1)
            else:
                if logr:
                    plt.semilogx(rr, rflux, '+')
                    plt.semilogx(self.rcirc, self.fcirc, 'r', lw=2)
                    plt.semilogx(rfit, ffit, 'k', ls='dashed')
                else:
                    plt.plot(rr, rflux, '+')
                    plt.plot(self.rcirc, self.fcirc, 'r', lw=2)
                    plt.plot(rfit, ffit, 'k', ls='dashed')
            plt.xlim(0, rmax)
            plt.title('%s Profile centered at (%6.1f,%6.1f)' % (ftype, xc, yc))
            plt.xlabel(xlab)
            plt.ylabel(flab)

            """ Plot the integrated flux/mag """
            plt.subplot(212, sharex=ax1)
            ftot = np.cumsum(rflux)
            if zp:
                m = -2.5 * np.log10(ftot) + zp
                if logr:
                    plt.semilogx(rr, m, '+')
                else:
                    plt.plot(rr, m, '+')
                yl1, yl2 = plt.ylim()
                plt.ylim(yl2, yl1)
            else:
                if logr:
                    plt.semilogx(rr, ftot, '+')
                else:
                    plt.plot(rr, ftot, '+')
            plt.xlim(0, rmax)
            plt.title('Integrated %s centered at (%6.1f, %6.1f)' %
                      (ttype, xc, yc))
            plt.xlabel(xlab)
            plt.ylabel(tlab)

        """ Save the output if desired """
        if outfile is not None:
            if outtype == 'fcirc':
                out = np.zeros((self.rcirc.size, 2))
                out[:, 0] = self.rcirc
                out[:, 1] = self.fcirc
            else:
                out = np.zeros(rr.size, 2)
                out[:, 0] = rr
                out[:, 1] = rflux
            np.savetxt(outfile, out, fmt='%7.3f %f')

    # -----------------------------------------------------------------------

    def set_contours(self, rms=None, hext=0, verbose=True):
        """
        Sets the contouring levels for an image.  If a subimage (i.e., cutout)
        has already been defined, then its properties are used.  Otherwise,
        the full image is used.

        The levels are set in terms of an rms, which is either passed
        explicitly via the optional rms parameter, or is determined from
        the properties of the data themselves (if rms=None).  The contours
        are multiples of (1) the rms, and (2) the contour base level
        (contbase), which has a default value of sqrt(3).  Thus:

         clev = [-contbase**2, contbase**2, contbase**3, contbase**4,...] * rms

        Optional inputs:
         rms      - If rms is None (the default), then use the data to
                     determine the rms.  If it is not None, then use the
                     passed value.
         hext     - Image HDU containing the data.  The default (hext=0)
                     should work for all single-extension fits files and may
                     work for some multi-extension files.
                     NOTE: hext is ignored if the subim variable is already
                     set by, e.g., set_subim_xy or def_subim_radec
         verbose - Report contour levels if True (the default)
        """

        """
        Set the portion of the data to be used.  This may already have been
        set before calling set_contours.  If it has already been set, then
        self.data will contain the data and the hext parameter will be
        ignored.
        If it has not been set, i.e., if self.data is None, then set the
        data to be the full image.
        """
        if self.data is None:
            self.set_subim()

        """
        If no rms value has been requested, calculate the rms from the data
        """
        if rms is None:
            self.sigma_clip()
            rms = self.rms_clip

        """ Set the contours based on the rms and the contour base """
        maxcont = int(log((self.data.max() / rms), self.contbase))
        if maxcont < 3:
            self.clevs = np.array([-3., 3., self.contbase**3])
        else:
            poslevs = np.logspace(2., maxcont, maxcont-1, base=self.contbase)
            self.clevs = np.concatenate(([-self.contbase**2], poslevs))

        if verbose:
            print "Contour levels: %f *" % rms
            print self.clevs
        self.clevs *= rms

    # -----------------------------------------------------------------------

    def plot_contours(self, color='r', rms=None, hext=0, overlay=True,
                      verbose=True):
        """

        Plots contours based on the flux (counts) in the image.

        """

        """ Set the contour levels if this has not already been done """
        if self.clevs is None:
            self.set_contours(rms, hext, verbose)

        """ Plot the contours """
        if overlay:
            plt.contour(self.data, self.clevs, colors=color,
                        extent=self.extval)
        else:
            plt.contour(self.data, self.clevs, colors=color,
                        extent=self.extval, origin='lower')

    # -----------------------------------------------------------------------

    def get_subim_bounds(self, centpos, imsize, hext=0):
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
        hdr = self.hdu[hext].header
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
            self.subsizex = int(subx)
            self.subsizey = int(suby)
        else:
            x1 = 0
            x2 = nx
            y1 = 0
            y2 = ny
            self.subsizex = nx
            self.subsizey = ny

        return x1, y1, x2, y2

    # -----------------------------------------------------------------------

    def set_subim_xy(self, x1, y1, x2, y2, hext=0, verbose=True):
        """

        Selects the data in the subimage defined by the bounds x1, x2, y1, y2.
        These were either set directly (e.g., by a call to imcopy) or by the
        get_subim_bounds function (which takes a subimage center and size)

        Inputs:
            hext     - Image HDU number that contains the full image
            verbose - Print out useful information if True (the default)
        """

        """
        Cut out the subimage based on the bounds.
        Note that radio images often have 4 dimensions (x, y, freq, stokes)
         so for those just take the x and y data
        """
        hdr = self.hdu[hext].header
        if hdr['naxis'] == 4:
            self.data = self.hdu[hext].data[0, 0, y1:y2, x1:x2].copy()
        else:
            self.data = self.hdu[hext].data[y1:y2, x1:x2].copy()
        self.data[~np.isfinite(self.data)] = 0.
        self.subimhdr = self.hdu[hext].header.copy()
        self.subcentx = 0.5 * (x1 + x2)
        self.subcenty = 0.5 * (y1 + y2)

        """ Print out useful information """
        if verbose:
            print('')
            print('Cutout image center (x, y): (%d, %d)' %
                  (self.subcentx, self.subcenty))
            print('Cutout image size (x y): %dx%d' %
                  (self.subsizex, self.subsizey))

        """
        Update the header info, including updating the CRPIXn values if they
        are present.
        """
        if self.infile:
            self.subimhdr['ORIG_IM'] = 'Copied from %s' % self.infile
        self.subimhdr['ORIG_REG'] = \
            'Region in original image [xrange, yrange]: [%d:%d,%d:%d]' % \
            (x1, x2, y1, y2)

        """ Update the headers to reflect the cutout center"""
        if 'crpix1' in self.subimhdr.keys():
            self.subimhdr['CRPIX1'] -= self.x1

        if 'crpix2' in self.subimhdr.keys():
            self.subimhdr['CRPIX2'] -= self.y1

    # -----------------------------------------------------------------------

    def def_subim_radec(self, ra, dec, xsize, ysize=None, outscale=None,
                        docdmatx=True, hext=0, dext=0, verbose=True):
        """
        Selects the data in the subimage defined by ra, dec, xsize, and ysize.

        The vast majority of the code is Matt Auger's (his image_cutout in
        imagelib.py).
        Some modifications have been made by Chris Fassnacht.

        Inputs:
          ra       - Central right ascension in decimal degrees
          dec      - Central declination in decimal degrees
          xsize    - Output image x size in arcsec
          ysize    - Output image y size in arcsec
                      If ysize is None (the default) then use the same size
                      for y as is being used for x (i.e., ysize=xsize)
          outscale - Output image pixel scale, in arcsec/pix.
                      If outscale is None (the default) then the output image
                      scale will be the same as the input image scale
          docdmatx - If set to True (the default), then put the output image
                      scale in terms of a CD matrix.  If False, then use the
                      CDELT and PC matrix formalism instead.
          hext     - Input file HDU number that contains the WCS info
                        (default 0)
          dext     - Input file HDU number that contains the image data
                        (default 0)
          verbose  - Print out informational statements if True (default=True)
        """

        """
        The following assignments get used whether a subimage has been
        requested or not
        """
        self.subimhdr = self.hdu[hext].header.copy()
        nx = self.hdu[hext].data.shape[1]
        ny = self.hdu[hext].data.shape[0]

        """ Check to make sure that a subimage is even requested """
        if (ra is None or dec is None) and xsize is None:
            self.data = self.hdu[hext].data.copy()
            self.subsizex = nx
            self.subsizey = ny
            return

        """
        If a sub-image is required, set up an astropy WCS structure that will
        be used for the calculations
        """
        w = wcs.WCS(self.subimhdr)

        """
        If the passed ra or dec is None, then just take the central pixel
         of the image as the requested center.
        If not, then we need to do some calculations
        """
        if ra is None or dec is None:
            x = nx / 2.
            y = ny / 2.

        else:
            """
            We have to convert the requested (RA, dec) center into the
             associated pixel values.
            The first step is to convert ra and dec into astropy.coordinates
             SkyCoord format
            """
            self.radec = self.radec_to_skycoord(ra, dec)

            """
            Calculate the (x, y) that is associated with the requested center
            """
            radec = np.zeros(self.subimhdr['naxis'])
            radec[0] = self.radec.ra.degree
            radec[1] = self.radec.dec.degree
            xy = w.wcs_world2pix([radec], 0)[0]
            x = xy[0]
            y = xy[1]

        """
        Get rough image size in pixels for the segment of input image, since
        the pixel scale for the output image does not necessarily match that
        of the input image.
        """
        if ysize is None:
            ysize = xsize
        inpixxsize = int(xsize / self.pixscale)
        inpixysize = int(ysize / self.pixscale)
        if outscale is None:
            outscale = self.pixscale
        self.subsizex = int(xsize / outscale)
        self.subsizey = int(ysize / outscale)

        """ Summarize the request """
        if verbose:
            print('')
            rastr = '%02d %02d %06.3f' % \
                (self.radec.ra.hms.h, self.radec.ra.hms.m,
                 self.radec.ra.hms.s)
            decstr = '%+03d %02d %05.2f' % \
                (self.radec.dec.dms.d, self.radec.dec.dms.m,
                 self.radec.dec.dms.s)
            print " Requested center (RA, dec): %11.7f    %+10.6f" % \
                (self.radec.ra.deg, self.radec.dec.deg)
            print " Requested center (RA, dec):  %s %s" % (rastr, decstr)
            print " Requested center (x, y):     %8.2f %8.2f" % (x, y)
            print " Requested image size (arcsec): %6.2f %6.2f" % \
                (xsize, ysize)
            print " Requested size in input pixels: %d %d" % \
                (inpixxsize, inpixysize)

        # """
        # In order to account for rotations, etc., when cutting out the
        # desired image section, start with a region that is larger
        # (by a factor of 2, if the image is large enough).
        # """
        # x0 = max(0, int(x-inpixxsize))
        # x1 = min(self.subimhdr['naxis1'], int(x+inpixxsize))
        # y0 = max(0, int(y-inpixysize))
        # y1 = min(self.subimhdr['naxis2'], int(y+inpixysize))
        # if verbose:
        #     print(" Cutting out image with x=%d--%d, y=%d--%d" %
        #             (x0, x1, y0, y1)
        #
        # """ Actually get the data in the large region """
        # if  self.subimhdr['naxis'] == 4:
        #     #data = self.hdu[dext].data[0, 0, y0:y1, x0:x1].copy()
        #     data = self.hdu[dext].data[0, 0,:,:].copy()
        # else:
        #     #data = self.hdu[dext].data[y0:y1, x0:x1].copy()
        #     data = self.hdu[dext].data.copy()
        # data[~np.isfinite(data)] = 0.
        #
        # """ Update the headers to reflect the cutout center"""
        # try:
        #     self.subimhdr['CRPIX1'] -= x0
        # except:
        #     print 'Warning: CRPIX1 header not found in %s' % self.infile
        #     pass
        # try:
        #     self.subimhdr['CRPIX2'] -= y0
        # except:
        #     print 'Warning: CRPIX2 header not found in %s' % self.infile
        #     pass

        """
        Set up the output header and do the coordinate transform preparation
        """
        if self.subimhdr['naxis'] == 4:
            data = self.hdu[dext].data[0, 0, :, :].copy()
        else:
            data = self.hdu[dext].data.copy()
        outhdr = self.make_header(self.radec, outscale, self.subsizex,
                                  self.subsizey)
        coords = np.indices((self.subsizey, self.subsizex)).astype(np.float32)
        skycoords = self.subim_wcs.wcs_pix2world(coords[1], coords[0], 0)
        ccdcoords = w.wcs_world2pix(skycoords[0], skycoords[1], 0)
        coords[0] = ccdcoords[1]
        coords[1] = ccdcoords[0]
        self.coords = coords.copy()

        # *** Now need to deal with regions that extend outside the data
        # should be doable, since map_coordinates just takes coordinate pairs
        # so masking the inputte ccdcoords arrays should be possible for
        # coordinates < 0 or > nx, ny
        #
        # Also, still need to try doing the integer pixel cutout for subimages
        #  that are both large enough and have PA=0.

        """ Transform the coordinates """
        self.data = ndimage.map_coordinates(data, coords, output=np.float64,
                                            order=5)
        self.data[np.isnan(self.data)] = 0.
        self.subimhdr = outhdr

        """ Clean up """
        del data, coords, skycoords, ccdcoords

    # -----------------------------------------------------------------------

    def get_rms(self, centpos, size, hext=0, verbose=True):
        """
        Calculates the rms (by calling the sigma_clip method) by calculating
        the pixel value statistics in a region of the image defined by its
        center (centpos parameter) and its size (size parameter).

        Inputs:
            centpos - (x, y) coordinates of the center of the region, in pixels
                      centpos can take any of the following formats:
                       1. A 2-element numpy array
                       2. A 2-element list:  [xsize, ysize]
                       3. A 2-element tuple: (xsize, ysize)
                       4. centpos=None.  In this case, the center of the cutout
                          is just the center of the full image
            size    - size of cutout (postage stamp) image, in pixels
                      size can take any of the following formats:
                       1. A single number (which will produce a square image)
                       2. A 2-element numpy array
                       3. A 2-element list:  [xsize, ysize]
                       4. A 2-element tuple: (xsize, ysize)
                       5. size=None.  In this case, the full image is used
            hext    - HDU containing the image data in the input image
                      (default=0)
        """

        """
        Convert the center and size paramters into the coordinates of the
        corners of the region
        """
        statsec = self.get_subim_bounds(centpos, size, hext)
        print('')
        print statsec

        """ Get the pixel statistics """
        self.sigma_clip(statsec=statsec)
        if verbose:
            print 'RMS = %f' % self.rms_clip
        return self.rms_clip

    # -----------------------------------------------------------------------

    def poststamp_xy(self, centpos, imsize, outfile=None, hext=0,
                     verbose=True):
        """
        Creates a subimage that is a cutout of the original image.  For
        this method, the image center is defined by its (x, y) coordinate
        rather than (ra, dec).
        The image is written to an output fits file if the outfile parameter
        is set.

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
            outfile - name of optional output file (default=None)
            hext    - HDU containing the image data in the input image
                      (default=0)
        """

        """ Make the cutout """
        x1, y1, x2, y2, = self.get_subim_bounds(centpos, imsize, hext)
        self.set_subim_xy(x1, y1, x2, y2, hext, verbose=verbose)

        """ Write to the output file if requested """
        if outfile:
            print('')
            if self.infile:
                print('Input file:  %s' % self.infile)
            print 'Output file: %s' % outfile
            pf.PrimaryHDU(self.data, self.subimhdr).writeto(outfile,
                                                            overwrite=True)
            print "Wrote postage stamp cutout to %s" % outfile

    # -----------------------------------------------------------------------

    def poststamp_radec(self, ra, dec, xsize, ysize, scale, outfile,
                        docdmatx=True, hext=0, dext=0, verbose=True):
        """
        Given a central coordinate (RA, dec), a size in pixels, and a pixel
        scale, creates an output cutout image

        The majority of the code is Matt Auger's (his image_cutout in
        imagelib.py).
        Some modifications have been made by Chris Fassnacht.

        Inputs:
            ra       - Central right ascension in decimal degrees
            dec      - Central declination in decimal degrees
            xsize    - Output image x size in arcsec
            ysize    - Output image y size in arcsec
            scale    - Output image pixel scale, in arcsec/pix
            outfile  - Output file name
            docdmatx - If set to True (the default), then put the output image
                        scale in terms of a CD matrix.  If False, then use the
                        CDELT and PC matrix formalism instead.
            hext     - Input file HDU number that contains the WCS info
                        (default 0)
            dext     - Input file HDU number that contains the image data
                        (default 0)
        """

        """ Create the postage stamp data """
        self.def_subim_radec(ra, dec, xsize, ysize, scale, docdmatx, hext,
                             dext, verbose)

        """
        Put the new WCS information into the original header, along with some
        additional info.
        """
        newhdr = self.hdu[hext].header.copy()
        wcskeys = \
            ['ra', 'dec', 'ctype1', 'ctype2', 'crval1', 'crpix1', 'crval2',
             'crpix2']
        if docdmatx:
            for i in ('cd1_1', 'cd1_2', 'cd2_1', 'cd2_2'):
                wcskeys.append(i)
        else:
            for i in ('cdelt1', 'cdelt2', 'pc1_1', 'pc1_2', 'pc2_1', 'pc2_2'):
                wcskeys.append(i)

        for i in wcskeys:
            newhdr.update(i, self.subimhdr[i])
        if self.infile:
            newhdr.update('ORIG_IM', self.infile)

        """ Write the postage stamp to the output file """
        pf.PrimaryHDU(self.data, newhdr).writeto(outfile, overwrite=True)
        print "Wrote postage stamp cutout to %s" % outfile

    # -----------------------------------------------------------------------

    def imcopy(self, x1, x2, y1, y2, outfile, hext=0):
        """
        Description: Given the x and y coordinates of
        the lower left corner and the upper right corner, creates a new
        fits file that is a cutout of the original image.

        Inputs:
          x1:      x coordinate of the lower left corner of desired region
          x2:      x coordinate of the upper right corner of desired region
          y1:      y coordinate of the lower left corner of desired region
          y2:      y coordinate of the upper right corner of desired region
          outfile: file name of output image
          hext:    HDU containing the image data in the input image
                   (default=0)
        """

        """ Get info about input image """
        inhdr = self.hdu[hext].header.copy()
        xmax = inhdr["NAXIS1"]
        ymax = inhdr["NAXIS2"]
        print('')
        print "imcopy: Input image has dimensions %d x %d" % \
            (xmax, ymax)

        """Check to make sure that requested corners are inside the image"""

        """ Make sure that everything is in integer format """
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        """
        Cut the file, and then update the CRPIXn header cards if they're there
        """
        print('imcopy: Cutting out region between ((%d,%d)) and ((%d,%d))' %
              (x1, y1, x2, y2))
        outdat = self.hdu[hext].data[y1:y2, x1:x2].copy()
        if self.infile:
            inhdr['ORIG_IM'] = 'Copied from %s with region[%d:%d,%d:%d]' % \
                (self.infile, x1, x2, y1, y2)
        print('')
        print "Updating CRPIXn header cards if they exist"
        print "------------------------------------------"
        try:
            crpix1 = inhdr['crpix1']
        except:
            print "    No CRPIX1 header found"
            crpix1 = np.nan
        try:
            crpix2 = inhdr['crpix2']
        except:
            print "    No CRPIX2 header found"
            crpix2 = np.nan
        if np.isnan(crpix1):
            pass
        else:
            inhdr['crpix1'] -= x1
            print "    Updating CRPIX1:  %8.2f --> %8.2f" % \
                (crpix1, inhdr['crpix1'])
        if np.isnan(crpix2):
            pass
        else:
            inhdr['crpix2'] -= y1
            print "    Updating CRPIX2:  %8.2f --> %8.2f" % \
                (crpix2, inhdr['crpix2'])

        """ Write to output file and clean up """
        outhdu = pf.PrimaryHDU(data=outdat, header=inhdr)
        outhdu.verify('fix')
        print "imcopy: Writing to output file %s" % outfile
        outhdu.writeto(outfile, overwrite=True)
        del outdat

    # -----------------------------------------------------------------------

    def snr_image_xy(self, centpos, imsize, statcent=None, statsize=None,
                     outfile=None, hext=0, verbose=True):
        """
        Creates a signal-to-noise image by first calculating the image RMS
        within a region defined by statcent and statsize.  If both of these
        parameters are None, then the region used to calculate the RMS is the
        same as the image defined by the required centpos and imsize
        parameters.
        The result can be written to an output file if outfile is not None,
        otherwise the numpy array containing the SNR information is returned.

        Inputs:
            centpos  - (x, y) coordinates of cutout center, in pixels
                       centpos can take any of the following formats:
                         1. A 2-element numpy array
                         2. A 2-element list:  [xsize, ysize]
                         3. A 2-element tuple: (xsize, ysize)
                         4. centpos=None.  In this case, the center of the
                            cutout is just the center of the full image
            imsize   - size of cutout (postage stamp) image, in pixels
                       imsize can take any of the following formats:
                         1. A single number (which will produce a square image)
                         2. A 2-element numpy array
                         3. A 2-element list:  [xsize, ysize]
                         4. A 2-element tuple: (xsize, ysize)
                         5. imsize=None.  In this case, the full image is used
            statcent - (x, y) coordinates of the center of the region to be
                        used to compute the image statistics.
                       statcent can take any of the following formats:
                        1. A 2-element numpy array
                        2. A 2-element list:  [xsize, ysize]
                        3. A 2-element tuple: (xsize, ysize)
                        4. statcent=None.  In this case, the statcent defaults
                           to the value of centpos
            statsize - size, in pixels, of the region to be used to compute the
                        image statistics
                        statsize can take any of the following formats:
                         1. A single number (which will produce a square image)
                         2. A 2-element numpy array
                         3. A 2-element list:  [xsize, ysize]
                         4. A 2-element tuple: (xsize, ysize)
                         5. statsize=None.  In this case, the statsize defaults
                            to the value of imsize
            outfile  - name of optional output file (default=None)
            hext     - HDU containing the image data in the input image
                          (default=0)
        """

        """ First get the rms in the requested region """
        if statcent is not None:
            xytmp = statcent
        else:
            xytmp = centpos
        if statsize is not None:
            sztmp = statsize
        else:
            sztmp = imsize
        imrms = self.get_rms(xytmp, sztmp, hext=hext, verbose=verbose)

        """ Now create the SNR image """
        self.poststamp_xy(centpos, imsize, hext=hext, outfile=None,
                          verbose=verbose)
        self.data /= imrms
        if outfile is not None:
            if verbose:
                print 'Output SNR file: %s' % outfile
            pf.PrimaryHDU(self.data, self.subimhdr).writeto(outfile,
                                                            overwrite=True)

    # -----------------------------------------------------------------------

    def mark_fov(self, ra, dec, size, pa=0.0, color='g', lw=1):
        """
        Draws a rectangle on the currently displayed image data.  This
        rectangle can represent the FOV of a camera or, for example, the
        slit for a spectrograph.

        Required inputs:
          ra    - RA of the center of the rectangle
          dec   - Dec of the center of the rectangle
          size  - size of the rectangle IN ARCSEC.  This can be in one of the
                  following formats:
                    1. A single number (which will produce a square image)
                    2. A 2-element numpy array
                    3. A 2-element list:  [xsize, ysize]
                    4. A 2-element tuple: (xsize, ysize)
                  NOTE: The convention for a slit is for the narrow dimension
                   (i.e., the slit width) to be given as the xsize and the
                   long dimension (the slit length) to be given as the ysize

        Optional inputs:
          pa     - Position angle of the FOV, in units of degrees E of N
                     Default value of 0.0 will produce a vertical slit
          color - Line color for drawing the rectangle.  Default='g'
          lw     - Line width for drawing the rectangle.  Default=1
        """

        """
        This function is meaningless if the input image does not have WCS
        information in it.  Check on this before proceeding
        """

        if self.found_wcs is False:
            print('')
            print('ERROR: Requested a FOV plot, but input image'
                  ' does not have WCS information in it.')
            print('')
            exit()

        """ Set the rectangle size """
        imsize = np.atleast_1d(size)  # Converts size to a numpy array
        xsize = imsize[0]
        if imsize.size > 1:
            ysize = imsize[1]
        else:
            ysize = xsize

        """
        Set the original vertices of the FOV marker, in terms of dx and dy
        """
        dw = 1. * xsize / 2.
        dh = 1. * ysize / 2.
        dx0 = np.array([dw, dw, -dw, -dw, dw])
        dy0 = np.array([-dh, dh, dh, -dh, -dh])

        """
        Rotate the vertices.
        NOTE: With the standard convention of RA increasing to the left (i.e.,
         north up, east left) and the PA defined as north through east, we
         have to set the user's PA to its negative to get what the user wanted
         in this astronomical convention.
        """
        parad = -1. * pa * pi / 180.
        cpa = mcos(parad)
        spa = msin(parad)
        dx = dx0 * cpa - dy0 * spa
        dy = dx0 * spa + dy0 * cpa

        """
        Find the center point of the FOV.
        For now assume that it is close enough to the center point of the
         image that we can use the small-angle approximation to calculate
         the offsets.
        Note that we have to include the zeropos offset to get the alignment
         to be correct, since the origin of the axes may not be at the center
         pixel.
        """
        cosdec = mcos(self.radec.dec.radian)
        fovx0 = 3600. * cosdec * (ra - self.radec.ra.deg) - self.zeropos[0]
        fovy0 = 3600. * (dec - self.radec.dec.deg) - self.zeropos[1]
        fovx = fovx0 + dx
        fovy = fovy0 + dy

        """ Plot the FOV """
        xlim = plt.xlim()
        ylim = plt.ylim()
        plt.plot(fovx, fovy, color=color, lw=lw)
        plt.xlim(xlim)
        plt.ylim(ylim)

    # -----------------------------------------------------------------------

    def set_display_limits(self, fmin=-1., fmax=10., funits='sigma', hext=0,
                           verbose=False):
        """

        The method used to set the flux limits for the image display.  The
         two numbers that are generated by this method will be used for the
         vmin and vmax values when the actual call to imshow (from
         matplotlib.pyplot) is made.  The two values will be stored within the
         Image class as fmin and fmax.

        Inputs:
          fmin   - Value that is used to set the minimum of the displayed flux
                    range, where the actual value depends on the
                    value of the funits paramters (see below).
                   NOTE: If fmin is None then switch to interactive mode
          fmax   - Value that is used to set the maximum of the displayed flux
                    range, where the actual value depends on the
                    value of the funits paramters (see below).
                   NOTE: If fmin is None then switch to interactive mode
          funits - Either 'sigma' (the default) or 'abs'. Used to determine
                    the method of setting fmin and fmax.
                   If funits is 'abs' then the two numbers in the disprange
                    list just get stored as fmin and fmax.
                   If funits is 'sigma' (the default) then the two numbers
                    in disprange represent the numbers of clipped standard
                    devitations relative to the clipped mean.  In that case,
                    the method will first calculate the clipped mean and
                    standarddeviations and then multiply them by the passed
                    values.

        """

        """
        If funits is 'abs', then just set self.fmin and self.fmax directly from
         the disprange values if those are set. Otherwise, query the user for
         the values.
        """
        if funits == 'abs':

            """ If disprange was set, then just transfer the values """
            if fmin is not None and fmax is not None:
                self.fmin = fmin
                self.fmax = fmax

            else:  # Otherwise, query the user
                """
                Set some default values if there aren't already some in the
                 fmin and fmax containers
                """
                if self.fmin is None or self.fmax is None:
                    if self.found_rms is False:
                        self.sigma_clip(verbose=verbose)
                        self.found_rms = True
                    self.fmin = self.mean_clip - 1.*self.rms_clip
                    self.fmax = self.mean_clip + 10.*self.rms_clip
                """ Query the user for new values """
                tmpmin = self.fmin
                tmpmax = self.fmax
                tmp = raw_input('Enter minimum flux value for display [%f]: '
                                % tmpmin)
                if len(tmp) > 0:
                    self.fmin = float(tmp)
                tmp = raw_input('Enter maximum flux value for display [%f]: '
                                % tmpmax)
                if len(tmp) > 0:
                    self.fmax = float(tmp)
            print 'fmin:  %f' % self.fmin
            print 'fmax:  %f' % self.fmax

        else:
            """
            If funits is not 'abs', then it must be 'sigma', which is the only
            other possibility, and the default value for funits.  In that case,
            set the display limits in terms of the clipped mean and sigma
            """

            """ Start by calculating the clipped statistics if needed """
            if self.found_rms is False:
                print "Calculating display limits"
                print "--------------------------"
                print self.data.size
                print self.data.shape
                self.sigma_clip(hext=hext, verbose=verbose)
                self.found_rms = True

            """ If disprange is not set, then query the user for the range """
            if fmin is None or fmax is None:
                fmin = -1.
                fmax = 10.
                tmp = raw_input('Enter min flux for display in terms of sigma '
                                'from mean [%f]: ' % fmin)
                if len(tmp) > 0:
                    fmin = float(tmp)
                tmp = raw_input('Enter max flux for display in terms of sigma '
                                'from mean [%f]: ' % fmax)
                if len(tmp) > 0:
                    fmax = float(tmp)

            """ Set fmin and fmax in terms of clipped mean and sigma"""
            self.fmin = self.mean_clip + fmin * self.rms_clip
            self.fmax = self.mean_clip + fmax * self.rms_clip
            print " Clipped mean: %f" % self.mean_clip
            print " Clipped rms:  %f" % self.rms_clip
            s1 = '-' if fmin < 0. else '+'
            s2 = '-' if fmax < 0. else '+'
            print(' fmin (mean %s %3d sigma):  %f' %
                  (s1, fabs(fmin), self.fmin))
            print(' fmax (mean %s %3d sigma):  %f' %
                  (s2, fabs(fmax), self.fmax))

    # -----------------------------------------------------------------------

    def set_cmap(self, cmap='gaia'):
        """

        Sets the color map for the image display.

        Inputs:
         cmap - name of the color map to use.  There are only a limited
                    number of choices:
                    ---
                    None
                    'gaia' (default)
                    'gray' or 'grey'
                    'gray_inv' or 'grey_inv'
                    'heat' or 'hot'
                    'jet'
        """

        if cmap == 'gray' or cmap == 'grey':
            self.cmap = plt.cm.gray
        elif cmap == 'gray_inv' or cmap == 'grey_inv':
            self.cmap = plt.cm.gray_r
        elif cmap == 'heat' or cmap == 'hot':
            self.cmap = plt.cm.hot
        elif cmap == 'Yl_Or_Br' or cmap == 'gaia':
            self.cmap = plt.cm.YlOrBr_r
        elif cmap == 'jet':
            self.cmap = plt.cm.jet
        else:
            print ' WARNING - Requested unknown color map.  Using gaia colors'
            self.cmap = plt.cm.YlOrBr_r

    # -----------------------------------------------------------------------

    def set_subim(self, hext=0, mode='xy', imcent=None, imsize=None,
                  verbose=False):
        """
        Sets the region of the image to be displayed
        """

        """ First check to see if any modification needs to be made """
        if imcent is None and imsize is None:
            if self.data is None:
                # NOTE: fix this to use set_data once that method is written
                self.data = self.hdu[hext].data.copy()
            else:
                return

        """
        The definition of the subimage depends on whether the requested
        cutout is based on WCS coordinates (RA and Dec) or pixels (x and y),
        where this choice is made through the mode parameter.
        Treat each case appropriately.
        """
        if mode == 'radec':
            # NOTE: need to check to see if found_wcs is True

            """ Set the display center"""
            if imcent is None:
                ra = None
                dec = None
            else:
                ra = imcent[0]
                dec = imcent[1]

            """ Set the display size """
            if imsize is None:
                xsize = None
                ysize = None
            else:
                xsize = imsize[0]
                ysize = imsize[1]
            self.def_subim_radec(ra, dec, xsize, ysize, hext=hext,
                                 verbose=verbose)

        else:
            """
            If not requesting a (RA, dec) cutout, the code is simpler
            """
            self.poststamp_xy(imcent, imsize, hext=hext)
        print('')

    # -----------------------------------------------------------------------

    def display_setup(self, hext=0, cmap='gaia', fmin=-1., fmax=10.,
                      funits='sigma', fscale='linear', statsize=2048,
                      title=None,  mode='xy', zeropos=None, verbose=False):
        """
        Sets parameters within the Image class that will be used to actually
         display the image or the requested part of it.
        NOTE: This method is usually called from the display method, and is
         not meant to be used in a stand-alone manner
        For more information about the parameters, etc., please see the
         help information for the display method.
        """

        """ Set the displayed axes to be in WCS offsets, if requested """
        self.mode = mode
        if self.mode == 'radec':
            if not self.found_wcs:
                print('')
                print("WARNING: mode='radec' but no WCS info in image"
                      "header")
                print 'Using pixels instead'
                print('')
                self.mode = 'xy'
                self.extval = None
            else:
                self.set_wcsextent(hext, zeropos)
        else:
            self.extval = None

        """ Set the image flux display limits """
        self.set_display_limits(fmin, fmax, funits, hext=hext, verbose=verbose)
        self.fscale = fscale

        """ Set the color map """
        self.set_cmap(cmap)

        """ Set other display parameters """
        self.title = title

    # -----------------------------------------------------------------------

    def scale_data(self):
        """
        Sets the scaling for the display, which depends on the fmin and fmax
        parameters _and_ the choice of scaling (for now either 'log' or
        'linear', with 'linear' being the default).  Then, scale the data
        appropriately if something besides 'linear' has been chosen

        The method returns the scaled data and the values of vmin and vmax to
        be used in the call to imshow.
        """

        fdiff = fabs(self.fmax - self.fmin)
        bitscale = 255.  # For 8-bit display

        if self.fscale == 'log':
            """
            For the log scaling, some thought needs to go into this.
            The classic approach is to choose the display range in the
             following way:
                vmin = log10(self.fmin - self.submin.min() + 1.)
                vmax = log10(self.fmax - self.submin.min() + 1.)
             where the "+ 1." is put in so that you are not trying to take the
             log of zero.  This seems to work well when the imaged to be
             displayed is in counts, where, e.g., the sky can be in the tens or
             hundreds of counts and the bright objects have thousands of
             counts.
            However, this does not work so well for situations such as when the
             units of the image are in, e.g., counts/s or e-/s, in which case
             taking the classic approach will typically make the display range
             between log(1+a) and log(1+b) where both a and b are small values.
             In this case, the log curve is essentially linear and the display
             does not look much different than choosing the "linear" option
             for display.
            Therefore, follow the lead of the ds9 display tool, which takes the
             requested display range and maps it onto the range 1-255.  This
             should provide decent dynamic range, even for the case where the
             units are counts/s or e-/s, and should more closely match the
             display behavior that the user wants.
            """
            # data = self.subim.copy() - self.subim.min()
            data = self.data.copy() - self.data.min()

            """ Now rescale from 1-255 in requested range """
            data[data >= 0] = ((bitscale - 1) * data[data >= 0] / fdiff) + 1.
            vmin = 0
            vmax = log10(bitscale)
            data[data <= 0.] = 1.
            data = np.log10(data)
            print 'Using log scaling: vmin = %f, vmax = %f' % (vmin, vmax)
            print data.min(), data.max()

        else:
            """ Linear scaling is the default """
            # data = self.subim
            data = self.data
            vmin = self.fmin
            vmax = self.fmax

        """ Return the values """
        return data, vmin, vmax

    # -----------------------------------------------------------------------

    def display_implot(self, show_xyproj=False, axlabel=True, fontsize=None):
        """

        NOTE: DO NOT USE this routine/method unless you know exactly what
        you are doing.  It is meant to be called from the display() method,
        as well as in a few other specialized cases, and is NOT meant to
        have stand-alone functionality.

        Please see the help for the display method for more information.
        """

        """
        Set up for displaying the image data
         - If show_xyproj is False (the default), then just show self.data
         - If show_xyproj is True, then make a three panel plot, with
             Panel 1: self.data (i.e., what you would see in the default
              behavior)
             Panel 2: Projection of data in self.data onto the x-axis
             Panel 3: Projection of data in self.data onto the x-axis
          - Setting show_xyproj=True is most useful when evaluating, e.g., a
             star in the image data.  The projections along the two axes of the
             cutout can be useful for evaluating whether the object is a star
             and/or whether it is saturated
        """
        if show_xyproj:
            self.fig2 = plt.figure(figsize=(10, 3))
            self.fig2.add_subplot(131)
        else:
            self.fig1 = plt.gcf()
            self.ax1 = plt.gca()

        """ Set the actual range for the display """
        data, vmin, vmax = self.scale_data()

        """ Display the image data """
        plt.imshow(data, origin='lower', cmap=self.cmap, vmin=vmin,
                   vmax=vmax, interpolation='nearest', extent=self.extval,
                   aspect='equal')

        """ Label the plot, if requested """
        if axlabel is True:
            if self.mode == 'radec':
                xlabel = 'Offset (arcsec)'
                ylabel = 'Offset (arcsec)'
                # if fontsize is not None:
                #     plt.xlabel(r"$\Delta \alpha$ (arcsec)",
                #         fontsize=fontsize)
                #     plt.ylabel(r"$\Delta \delta$ (arcsec)",
                #         fontsize=fontsize)
                # else:
                #     plt.xlabel(r"$\Delta \alpha$ (arcsec)")
                #     plt.ylabel(r"$\Delta \delta$ (arcsec)")
            else:
                xlabel = 'x (pix)'
                ylabel = 'y (pix)'
                # if fontsize is not None:
                #     plt.xlabel('x (pix)', fontsize=fontsize)
                #     plt.ylabel('y (pix)', fontsize=fontsize)
                # else:
                #     plt.xlabel('x (pix)')
                #     plt.ylabel('y (pix)')
            if fontsize is not None:
                plt.xlabel(xlabel, fontsize=fontsize)
                plt.ylabel(ylabel, fontsize=fontsize)
            else:
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
        if self.title is not None:
            plt.title(self.title)

        """
        Now add the x and y projections if requested (i.e., if show_xyproj
        is True)
        """
        if show_xyproj:
            self.fig2.add_subplot(132)
            xsum = self.data.sum(axis=0)
            plt.plot(xsum)
            plt.xlabel('Relative x Coord')
            self.fig2.add_subplot(133)
            ysum = self.data.sum(axis=1)
            plt.plot(ysum)
            plt.xlabel('Relative y Coord')
            self.cid_keypress2 = \
                self.fig2.canvas.mpl_connect('key_press_event',
                                             self.keypress)
            self.fig2.show()

    # -----------------------------------------------------------------------

    def display(self, hext=0, cmap='gaia',
                fmin=-1., fmax=10., funits='sigma', fscale='linear',
                statsize=2048, title=None,
                mode='radec', imcent=None, imsize=None,
                zeropos=None, axlabel=True, fontsize=None,
                mask=None, show_xyproj=False, verbose=False):
        """
        The main way to display the image data contained in the Image class.
        The default is to display the entire image, but it is possible to
        display cutouts (subimages), which can be defined either by (RA, dec)
        or (x, y)

        Optional inputs:
          mode    - Either 'radec' (the default) or 'xy'.  Replaces the
                    obsolete subimdef and dispunits parameters
          imsize - size of the subimage to be displayed, either in pixels
                       (the default) or arcsec (if mode='radec').  The
                       default, designated by imsize=None, is to display
                       the entire image.  The imsize parameter can take
                       any of the following formats:
                         1. A single number (which will produce a square image)
                         2. A 2-element numpy array
                         3. A 2-element list:  [xsize, ysize]
                         4. A 2-element tuple: (xsize, ysize)
          zeropos   - NOTE: Only used if mode='radec'
                       By default, which happens when zeropos=None, the (0, 0)
                       point on the output image, as designated by the image
                       axis labels, will be at the center of the image.
                       However,you can shift the (0, 0) point to be somewhere
                       else by setting zeropos.
                       For example, zeropos=(0.5, 0.3) will shift the origin
                       to the point that would have been (0.5, 0.3) if the
                       origin were at the center of the image

        Obsolete inputs [not used any more -- just kept for legacy purposes]
          subimdef
          dispunits
        """
        print('')
        if self.infile:
            print('Input file:  %s' % self.infile)

        """
        Select the region of the image to be displayed, which may be the
        full input image
        """
        self.mode = mode
        self.set_subim(hext, mode, imcent, imsize, verbose=verbose)

        """ Set up the parameters that will be needed to display the image """
        self.display_setup(hext=hext, cmap=cmap,
                           fmin=fmin, fmax=fmax, funits=funits, fscale=fscale,
                           statsize=statsize, title=title, mode=mode,
                           zeropos=zeropos, verbose=verbose)

        """ Now display the data """
        self.display_implot(show_xyproj, axlabel, fontsize)

    # -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Stand-alone functions below this point
# -----------------------------------------------------------------------


def get_rms(infile, xcent, ycent, xsize, ysize=None, hext=0, outfile=None,
            verbose=True):
    """

    Gets the rms in the requested region of the input file.  The region is
    defined by its central pixel location (xcent, ycent) and its size in
    pixels (xsize and, optionally, ysize).  If ysize is not given, then the
    region is assumed to be square, with the length of a side given by xsize.

    The rms value that is returned is the "clipped rms" for the region, i.e.,
    the rms after an iterative sigma clipping algorithm has been run on the
    data in the region.

    Required Inputs:
      infile - name of the input fits file
      xcent  - x coordinate of the pixel that defines the center of the region
      ycent  - y coordinate of the pixel that defines the center of the region
      xsize  - width of the region or, if ysize is not given, length of a side
                of a square region.

    Optional Inputs:
      ysize    - height of the region.  If not given, then ysize=xsize
      hext     - the HDU containing the data.  The default (hext=0) should
                  work for most data files.
      outfile - output file to store the data [NOT YET IMPLEMENTED]
      verbose - set to False to surpress output
    """

    im = Image(infile, verbose=verbose)
    if ysize is not None:
        im.poststamp_xy((xcent, ycent), (xsize, ysize), hext=hext,
                        verbose=verbose)
    else:
        im.poststamp_xy((xcent, ycent), xsize, hext=hext, verbose=verbose)
    im.sigma_clip()
    rms = im.rms_clip
    if verbose:
        print('')
        print '%s: RMS in requested region is %f' % (infile, rms)
    del im
    return rms

# -----------------------------------------------------------------------


def make_cutout(infile, ra, dec, imsize, scale, outfile, whtsuff=None,
                makerms=False, rmssuff='_rms', hext=0, dext=0, verbose=True):
    """
    Makes a cutout from an input image, based on a requested (RA, dec) center
    and an image size in arcsec.
    Additional, optional functionality:
        - Makes the same-sized cutout for the associated weight file.  Done if
          whtsuff is not None
        - Makes an RMS image following Matt Auger's recipe for the NIRC2 data
          This only happens if BOTH whtsuff is not None AND makerms is True

    Inputs:
      infile  - input file
      ra      - RA of cutout center, in decimal degrees
      dec     - Dec of cutout center, in decimal degrees
      imsize  - output image size, in arcsec
      scale   - pixel scale of output image, in arcsec/pix
      outfile - output file name
      whtsuff - suffix for input weight file, if a cutout of the weight file
                 is also desired.  If whtsuff is None (the default) then no
                 weight-file cutout is made.
                 Example: whtsuff='_wht' means that for infile='foo.fits' the
                  weight file is called 'foo_wht.fits'
      makerms - Set to True to make, in addition, an output rms file
                 following Matt Auger's prescription for the NIRC2 data.
                 Default is False
                 NB: Both whtsuff being something other than None and
                  makerms=True are required for an output rms file to be
                  created.
      rmssuff - Suffix for output rms file.  Default='_rms' means that for
                   infile='foo.fits', the output file will be 'foo_rms.fits'
      hext    - Input file HDU number that contains the WCS info (default 0)
      dext    - Input file HDU number that contains the image data (default 0)
    """

    """ Make the input file cutout """
    infits = Image(infile)
    infits.poststamp_radec(ra, dec, imsize, imsize, scale, outfile, hext=hext,
                           dext=dext, verbose=verbose)

    """ Make the weight file cutout, if requested """
    if whtsuff is not None:
        whtfile = infile.replace('.fits', '%s.fits' % whtsuff)
        outwht = outfile.replace('.fits', '%s.fits' % whtsuff)
        whtfits = Image(whtfile)
        whtfits.poststamp_radec(ra, dec, imsize, imsize, scale, outwht,
                                hext=hext, dext=dext, verbose=verbose)

    """ Make output RMS file, if requested """
    # CODE STILL TO COME

    """ Clean up """
    infits.close()
    if whtsuff is not None:
        whtfits.close()

# -----------------------------------------------------------------------


def open_fits(infile, mode='copyonwrite'):
    """
    Opens a fits file, allowing for the possibility of the missing end that
    plagues some of the NIR instruments on Keck.

    Inputs:
        infile - input file name
        mode    - [OPTIONAL] mode of opening the file.  Note that the default
                    value ('copyonwrite') is the pyfits default value.  Look at
                    the help information for pyfits open for other options.
    """

    try:
        hdulist = pf.open(infile, mode=mode)
    except:
        try:
            """ Try to get rid of read-in warnings """
            import warnings
            warnings.filterwarnings('ignore')
            hdulist = pf.open(infile, mode=mode, ignore_missing_end=True)
        except:
            print('')
            print "ERROR. Could not open fits file %s" % infile
            return None

    return hdulist

# ---------------------------------------------------------------------------


def imcopy(infile, x1, x2, y1, y2, outfile):
    """
    Description: Given a fits file name, the x and y coordinates of
      the lower left corner and the upper right corner, creates a new
      fits file that is a cutout of the original image.

    Inputs:
      infile:  file name of input image
      x1:        x coordinate of the lower left corner of desired region
      x2:        x coordinate of the upper right corner of desired region
      y1:        y coordinate of the lower left corner of desired region
      y2:        y coordinate of the upper right corner of desired region
      outfile: file name of output image
    """

    hdu_list = open_fits(infile)
    if hdu_list is None:
        return

    """ Get info about input image """
    inhdr = hdu_list[0].header
    xmax = inhdr["NAXIS1"]
    ymax = inhdr["NAXIS2"]
    print('')
    print('imcopy: Input image %s has dimensions %d x %d' %
          (infile, xmax, ymax))

    """Check to make sure that requested corners are inside the image"""

    """ Make sure that everything is in integer format """
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    """
    Cut the file, and then update the CRPIXn header cards if they're there
    """
    print('imcopy: Cutting out region between (%d,%d) and (%d,%d)' %
          (x1, y1, x2, y2))
    outdat = hdu_list[0].data[y1:y2, x1:x2]
    print('')
    print "Updating CRPIXn header cards if they exist"
    print "------------------------------------------"
    try:
        crpix1 = inhdr['crpix1']
    except:
        print "    No CRPIX1 header found"
        crpix1 = np.nan
    try:
        crpix2 = inhdr['crpix2']
    except:
        print "    No CRPIX2 header found"
        crpix2 = np.nan
    if np.isnan(crpix1) is False:
        inhdr['crpix1'] -= x1
        print('    Updating CRPIX1:  %8.2f --> %8.2f' %
              (crpix1, inhdr['crpix1']))
    if np.isnan(crpix2) is False:
        inhdr['crpix2'] -= y1
        print ('    Updating CRPIX2:  %8.2f --> %8.2f' %
               (crpix2, inhdr['crpix2']))

    """ Write to output file """
    outhdu = pf.PrimaryHDU(data=outdat, header=inhdr)
    outhdu.verify('fix')
    print "imcopy: Writing to output file %s" % outfile
    outhdu.writeto(outfile, overwrite=True)

    return

# ---------------------------------------------------------------------------


def poststamp(infile, centx, centy, xsize, ysize, outfile):
    """
    Description: Given a fits file name, an image center, and an image size,
      creates a new fits file that is cutout of part of the original image.
    """

    im = Image(infile)
    im.poststamp_xy((centx, centy), (xsize, ysize), outfile=outfile)
    im.close()
    del im

    return

# -----------------------------------------------------------------------


def del_history_cards(hdu):
    """
    Deletes the history cards from the input HDU
    """

    print "del_history_cards is not yet implemented"

# -----------------------------------------------------------------------


def make_snr_image(infile, outcent=None, outsize=None, statcent=None,
                   statsize=None, outfile=None, hext=0):
    """
    Reads in the data from the input image and estimates the RMS noise
    in the region defined by statcent and statsize (default is to use the
    entire image).
    Divides the image by that number and either writes the output SNR image
    to a file or returns it as a data array.
    """

    """ Read in the data """
    im = Image(infile)

    """ Make the SNR image """
    im.snr_image_xy(outcent, outsize, statcent=statcent, statsize=statsize,
                    outfile=outfile, hext=hext)

    if outfile is not None:
        print('')
        return im.subim

# ---------------------------------------------------------------------------


def overlay_contours(infile1, infile2, ra, dec, imsize, pixscale=None,
                     zeropos=None, fmax=10., hext1=0,
                     hext2=0, rms2=None, ccolor2='r',
                     infile3=None, hext3=0, rms3=None, ccolor3='b',
                     title=None, showradec=True,
                     verbose=True):
    """
    Creates a postage-stamp cutout (of size imgsize arcsec) of the data in the
     Image class and then overlays contours from the second image (infile2).

    Required inputs:
      infile1 - fits file containing the data for the first image
      infile2 - fits file containing the data for the second image
      ra      - single number containing RA for image center
                 (best if in decimal degrees)
      dec     - single number containing Dec for image center
                 (best if in decimal degrees)
      imsize  - length of one side of output image, in arcsec
    Optional inputs:
      pixscale  - pixel scale of output image, in arcsec/pix
                   If pixscale is None (the default) then just use the
                   native pixel scale of each of the input images.
      zeropos   - By default, which happens when zeropos=None, the (0, 0)
                   point on the output image, as designated by the image
                   axis labels, will be at the center of the image.  However,
                   you can shift the (0, 0) point to be somewhere else by
                   setting zeropos.  For example, zeropos=(0.5, 0.3) will
                   shift the origin to the point that would have been
                   (0.5, 0.3) if the origin were at the center of the image
      fmax      - upper range for plotting the greyscale in the first image,
                   expressed as the number of sigma above the clipped mean.
                   Default = 10.
      hext1     - HDU containing the actual image data for the first file
                   default=0
      hext2     - HDU containing the actual image data for the second file
                   default=0
      rms2      - user-requested rms for data in the second image. If set to
                   None (the default) then calculate rms from the cutout data
                   themselves
      ccolor2   - color for the contours from infile2.  Default='r'
      infile3   - OPTIONAL name of a third image, to be used for a second set
                   of contours in a different line style.  Default=None
      hext3     - HDU containing the actual image data for the third file
                   default=0
      rms3      - user-requested rms for data in the optional third image. If
                   set to  None (the default) then calculate rms from the
                   cutout data themselves
      ccolor3   - color for the contours from infile3.  Default='b' (black)
      title     - title for the figure.  The default value (None) will show
                   no title
      showradec - print the RA and Dec of the center of the image, in decimal
                   degrees, on the figure.  Default=True
      verbose   - print out useful information while running.  Default=True
    """

    """ Read the input images """
    try:
        im1 = Image(infile1)
    except:
        print('')
        print 'ERROR: Could not properly open %s' % infile1
        return
    print "    .... Done"
    try:
        im2 = Image(infile2)
    except:
        print('')
        print 'ERROR: Could not properly open %s' % infile2
        return
    print "    .... Done"

    """
    Make cutouts of the appropriate size for each of the input images
    For the first image this is done via a call to display
    """
    im1.display(hext=hext1, cmap='gray_inv', mode='radec',
                imcent=(ra, dec), imsize=(imsize, imsize),
                fmax=fmax, zeropos=zeropos)
    im2.def_subim_radec(ra, dec, imsize, outscale=pixscale)

    """ Set contour levels for the second image """
    im2.set_contours(rms2)

    """
    Change the axis labels to be offsets in arcseconds from a fiducial point
    in the image, which is set to be the origin.
    Default value for the origin is the center of the image.
    Override the default value by setting the zeropos parameter
    """
    im2.set_wcsextent(zeropos=zeropos)

    """ Plot the contours """
    plt.contour(im2.subim, im2.clevs, colors=ccolor2, extent=im2.extval)

    """ If there is a third image, plot contours from it """
    if infile3 is not None:
        try:
            im3 = Image(infile3)
        except:
            print('')
            print 'ERROR: Could not properly open %s' % infile3
            return
        im3.def_subim_radec(ra, dec, imsize, outscale=pixscale)
        im3.set_contours(rms3)
        im3.set_wcsextent(zeropos=zeropos)
        plt.contour(im3.subim, im3.clevs, colors=ccolor3, extent=im3.extval)

    """ Clean up """
    im1.close()
    im2.close()
    del im1, im2
    if infile3 is not None:
        im3.close()
        del im3

# -----------------------------------------------------------------------


def calc_sky_from_seg(infile, segfile):
    """
    Description: Calculates the sky level in an image using only
        those regions in which SExtractor's segmentation file has
        a value of 0.

    Inputs:
     infile:    input fits file
     segfile:  SExtractor segmentation file associated with infile.
    """

    """ Load data """
    indat = pf.getdata(infile)
    segdat = pf.getdata(segfile)

    """ Set mask region and select associated regions of indat """
    mask = segdat == 0
    sky = indat[mask]
    # NB: These preceding 2 lines could have been combined as
    #    sky = indat[segdat==0]

    """ Calculate statistics """
    print "Statistics of sky outside masked regions"
    print "----------------------------------------"
    print "  N_pix  = %d" % sky.size
    print "  Median = %f" % np.median(sky)
    print "  Mean    = %f" % np.mean(sky)

    return

# -----------------------------------------------------------------------


def quick_display(infile, hext=0, cmap='gaia', fmin=-1.0, fmax=10.0,
                  funits='sigma'):
    """
    Displays the image data contained in an input fits file.
    Does this through a call to the Image class, which is returned.
    Note that most of the display parameters, which normally are accessed
    through the Image class, are fixed to their default values.  The only
    parameters that can be modified by this function are:
      hext, cmap, fmin, fmax, and funits
    For other functionality, set up an Image class structure and access the
    display method within it rather than using this quick function.
    """

    """ Load the image """
    image = Image(infile)

    """ Display the image. """
    image.display(hext=hext, cmap=cmap, fmin=fmin, fmax=fmax, funits=funits)

    return image

# -----------------------------------------------------------------------


def plot_cat(fitsfile, catfile, xcol=0, ycol=1, marksize=20., markcolor='g',
             inhdu=0, cmap='gray', fmin=-1., fmax=10., funits='sigma'):
    """
    Plots a catalog (e.g., one generated by SExtractor), on top of a fits
    image.

    Inputs:
      fitsfile  - input fits data file containing the image
      catfile   - input file containing the object catalog
      xcol      - column in the input file with the object x coordinates
                   (remember that the first column corresponds to xcol=0)
                   default value: 0
      ycol      - column in the input file with the object y coordinates
                   (remember that the second column corresponds to ycol=1)
                   default value: 1
      marksize  - size of circles marking the objects on the image, in points
                   default value: 20.0
      markcolor - color of circles marking the objects
                   default value: 'g'
      inhdu     - header-data unit containing the image data in the input
                   fits image.  The default value of 0 is appropriate for all
                   simple fits images (i.e., those without multiple
                   extensions).
                   default value: 0
      cmap      - color map used to present the image data
                   default value: 'gray'
      fmin      - sets display range for input image
                   default value: -1.0 (1-sigma below the clipped mean)
      fmax      - sets display range for input image
                   default value: 10.0 (10-sigma below the clipped mean)
      funits    - units for display range.  Default value = 'sigma', to
                   calculate range in terms of sigma above and below the
                   clipped mean
    """

    """ Plot the image """
    try:
        quick_display(fitsfile, inhdu=inhdu, cmap=cmap, fmin=fmin, fmax=fmax,
                      funits=funits)
    except:
        print('')
        print "Image display failed when called from plot_cat."
        print('')
        return
    nx = pf.getval(fitsfile, 'naxis1')
    ny = pf.getval(fitsfile, 'naxis2')

    """ Read in the catalog and extract the x and y coordinates """
    data = np.loadtxt(catfile)
    x = data[:, xcol]
    y = data[:, ycol]

    """ Mark the catalog objects """
    plt.plot(x, y, 'o', ms=marksize, mec=markcolor, mfc="none")
    plt.xlim(0, nx - 1)
    plt.ylim(0, ny - 1)
