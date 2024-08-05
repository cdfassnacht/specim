"""

wcshdu.py

Defines a new class that is essentially a fits PrimaryHDU class but with
some of the WCS information in the header split out into separate
attributes of the class

"""

import os
import sys
from math import fabs
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from scipy import ndimage
from scipy.ndimage import filters

from astropy import wcs
from astropy import units as u
from astropy.io import fits as pf
from astropy.coordinates import SkyCoord

from cdfutils import coords, datafuncs as df
from .imutils import open_fits

pyversion = sys.version_info.major

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

        """
        Check the format of the input info and get the data and header info
        """
        hdu = None
        infile = None
        if isinstance(indat, str):
            hdu = self.read_from_file(indat, verbose=verbose)
            data = hdu[hext].data
            hdr = hdu[hext].header
            infile = indat
        elif isinstance(indat, pf.HDUList):
            hdu = indat
            data = hdu[hext].data
            hdr = hdu[hext].header
            infile = indat.filename()
        elif isinstance(indat, (pf.PrimaryHDU, pf.ImageHDU)):
            data = indat.data.copy()
            hdr = indat.header.copy()
            try:
                infile = indat.fileinfo()['file'].name
            except TypeError:
                infile = None
        elif isinstance(indat, WcsHDU):
            data = indat.data.copy()
            hdr = indat.header.copy()
            infile = indat.infile
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
        if pyversion == 2:
            super(WcsHDU, self).__init__(data, hdr)
        else:
            super().__init__(data, hdr)
        self.infile = infile
        if infile is not None:
            self.basename = os.path.basename(infile)
            dirname = os.path.dirname(infile)
            if dirname == '':
                self.dirname = None
            else:
                self.dirname = dirname
        else:
            self.basename = None
            self.dirname = None

        """ Set some general default values """
        self.found_rms = False
        self.fftconj = None

        """ Set some WCS-related default values """
        self.wcsinfo = None
        self.raaxis = None
        self.decaxis = None
        self.radec = None
        self.pixscale = None
        self.impa = 0.
        self.radec = None
        self.crpix = None
        self.crval = None
        self.cdelt = None
        self.pc = None
        self.pointing = None

        """
        Now add WCS attributes to the class.
        NOTE, sometimes these are in a different header, which can be
        indicated by the wcsext parameter
        """
        self.hext = hext
        if wcsext is not None and hdu is not None:
            try:
                wcshdr = hdu[wcsext].header
            except UnboundLocalError:
                print('')
                print('ERROR: You cannot set wcsext parameter if the input'
                      ' is not a file or HDUList')
                print('')
                raise TypeError
        else:
            wcshdr = self.header
        try:
            self.read_wcsinfo(wcshdr, verbose=wcsverb)
        except KeyError:
            """ Just keep default values for WCS attributes, i.e., None """
            if wcsverb:
                print('Warning: Could not find WCS information')

    # -----------------------------------------------------------------------

    """
    Use properties to set many/most of the attributes that are associated
     with this class.

    For an attribute called x, use two declarations as can be seen below:
     @property
     @x.setter
    These two will then allow you to set the value of x via
      foo.x = [value]
    but the @property construction also allows you to add some evaluation
    of what the value is and how to respond.  For example, if x needs to
    be non-negative, then with the appropriate coding, foo.x = -10 will
    actually set foo.x = 0, in a way that is transparent to the user.
    """
    # -----------------------------------

    """ Pixel scale """
    @property
    def pixscale(self):
        return self._pixscale

    @pixscale.setter
    def pixscale(self, pixscale):
        """
        Set the pixscale attribute and also update the WCS information

        Input:
           pixscale - desired pixel scale IN ARCSEC

        """

        """ Parse the input"""
        if isinstance(pixscale, float):
            newscale = np.ones(2) * pixscale
        elif isinstance(pixscale, (tuple, list)):
            newscale = np.array(pixscale)
        elif isinstance(pixscale, np.ndarray) or pixscale is None:
            newscale = pixscale
        else:
            raise TypeError('pixscale must be a scalar or one of: '
                            'list, numpy array, or tuple')

        """ Set shortcuts """
        raax = self.raaxis
        decax = self.decaxis

        """
        Update the values in the WcsHDU object.  Assume for now that the
        values are in (RA, Dec) order.
        Note that what part of the wcsinfo structure gets modified depends on
         whether the coordinate tranform is stored in a (CDELT, PC matrix)
         format or in a (CD matrix) format.
        """
        self._pixscale = newscale

        """
        Convert to a standard format within the wcsinfo object whatever the
        input format may have been, if the passed value is not None
        """
        if self.wcsinfo is not None and newscale is not None:
            hdr = self.header

            """ First get the PA """
            pa = coords.matrix_to_rot(self.wcsinfo.pixel_scale_matrix,
                                      raax=raax-1, decax=decax-1)

            """
            Convert the PA to a PC matrix that just reflects the rotation and
            put this information into both the wcsinfo object and the header
            """
            pc = coords.rot_to_pcmatrix(pa, verbose=False)
            self.wcsinfo.wcs.pc = pc
            hdr['PC%d_%d' % (raax, raax)] = pc[0, 0]
            hdr['PC%d_%d' % (raax, decax)] = pc[0, 1]
            hdr['PC%d_%d' % (decax, raax)] = pc[1, 0]
            hdr['PC%d_%d' % (decax, decax)] = pc[1, 1]

            """
            Get rid of the old CD matrix, if it exists, to avoid conflicts
            """
            if self.wcsinfo.wcs.has_cd():
                del self.wcsinfo.wcs.cd

            """ Update the CDELT values """
            for i, ax in enumerate([self.raaxis, self.decaxis]):
                if ax == self.raaxis:
                    sgn = -1.
                else:
                    sgn = 1.
                hdr['cdelt%d' % ax] = sgn * newscale[i] / 3600.
                self.wcsinfo.wcs.cdelt[(ax - 1)] = sgn * newscale[i] / 3600.

                self.wcsinfo.wcs.cdelt[(ax-1)] = sgn * newscale[i] / 3600.

    # -----------------------------------

    """ CRPIX values """
    @property
    def crpix(self):
        return self._crpix

    @crpix.setter
    def crpix(self, crpix):
        """

        Updates the crpix attribute and also updates the crpix values in:
          1. the header
          2. the wcsinfo object

        """

        """ If crpix is None then don't change anything """
        if crpix is None:
            self._crpix = None
            return

        """
        Update the CRPIX array assuming that the input is in the correct
        format
        """
        if isinstance(crpix, (list, tuple, np.ndarray)):
            """ Check dimensionality """
            if len(crpix) != len(self.wcsinfo.wcs.crpix):
                raise IndexError(' Input crpix array length does not match'
                                 ' length of current crpix array')
            """ Set the CRPIX values in both the header and wcsinfo"""
            for i in range(len(crpix)):
                self.wcsinfo.wcs.crpix[i] = crpix[i]
                self.header['crpix%d' % (i+1)] = crpix[i]
        else:
            raise TypeError('crpixarr must be list, tuple, or ndarray')

        """ Update the attribute"""
        self._crpix = np.array(crpix)

    # -----------------------------------

    """ CRVAL values """
    @property
    def crval(self):
        return self._crval

    @crval.setter
    def crval(self, crval):
        """

        Updates the crval attribute and also updates the crval values in:
          1. the header
          2. the wcsinfo object

        """

        """ If crval is None then don't change anything """
        if crval is None:
            self._crval = None
            return

        """
        Update the CRVAL array assuming that the input is in the correct
        format
        """
        if isinstance(crval, (list, tuple, np.ndarray)):
            """ Check dimensionality """
            if len(crval) != len(self.wcsinfo.wcs.crval):
                raise IndexError(' Input crpix array length does not match'
                                 ' length of current crval array')
            """ Set the CRVAL values in both the header and wcsinfo """
            for i in range(len(crval)):
                self.wcsinfo.wcs.crval[i] = crval[i]
                self.header['crval%d' % (i+1)] = crval[i]
        else:
            raise TypeError('crvalarr must be list, tuple, or ndarray')

        """ Update the attribute"""
        self._crval = np.array(crval)

    # -----------------------------------

    """ PC matrix values """
    @property
    def pc(self):
        return self._pc

    @pc.setter
    def pc(self, pc):
        """

        Updates the pc attribute and also updates the pc matrix values in:
          1. the header
          2. the wcsinfo object

        """

        """ If pc is None then don't change anything """
        if pc is None:
            self._pc = None
            return

        """
        Update the pc array assuming that the input is in the correct
        format
        """
        if isinstance(pc, np.ndarray):
            """ Check dimensionality """
            if pc.size != self.wcsinfo.wcs.pc.size:
                raise IndexError(' Input PC matrix size does not match'
                                 ' size of current PC matrix')
            """ Set the PC values in both the header and wcsinfo """
            self.wcsinfo.wcs.pc = pc
            hdr2 = self.wcsinfo.to_header()
            for k in hdr2.keys():
                if k[:2] == 'PC':
                    self.header[k] = hdr2[k]
        else:
            raise TypeError('pc must be a numpy ndarray')

        """ Update the attribute"""
        self._pc = pc

    # -----------------------------------------------------------------------

    def read_from_file(self, infile, verbose=True):
        """
        Reads the image data from a file (as opposed to getting the data
        directly from an input HDUList
        """
        if verbose:
            print('Loading file %s' % infile)

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
        self.infile = os.path.basename(infile)
        return hdu

    # -----------------------------------------------------------------------

    def read_wcsinfo(self, wcshdr, verbose=True):
        """

        Reads in WCS information from the header and saves it, if it's
        there, in some attributes of the class

        """

        """ Set some defaults """
        raax = 0
        decax = 1
        rakey = 'naxis1'
        deckey = 'naxis2'

        """ Get the WCS information out of the header if it's there """
        try:
            wcsinfo = wcs.WCS(wcshdr)
        except:
            if verbose:
                if self.infile is not None:
                    print('No WCS information in image header: %s' %
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
                rakey = 'naxis%d' % (raax + 1)
            if ct[0:3] == 'DEC':
                decfound = True
                decax = count
                deckey = 'naxis%d' % (decax + 1)
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
        impa = coords.matrix_to_rot(wcsinfo.pixel_scale_matrix, raax=raax,
                                    decax=decax)
        pixscale = wcs.utils.proj_plane_pixel_scales(wcsinfo.celestial) \
            * 3600.

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
        self.raaxis = raax + 1
        self.decaxis = decax + 1
        self.radec = radec
        self.pixscale = pixscale
        self.impa = impa
        self.crpix = wcsinfo.wcs.crpix
        self.crval = wcsinfo.wcs.crval
        if wcsinfo.wcs.has_pc():
            self.pc = wcsinfo.wcs.pc

    # -----------------------------------------------------------------------

    def get_pointing(self, rakey='RA', deckey='DEC'):
        """

        For many ground-based telescopes, raw data files from the telescope
        do not include WCS information in the FITS standard format.
        However, many include the coordinates at which the telescope control
        thinks the telescope is pointing.  This method extracts this information
        from the fits header if it exists.

        """

        if rakey in self.header.keys() and deckey in self.header.keys():
            ra = self.header[rakey]
            dec = self.header[deckey]
            if isinstance(ra, str) and isinstance(dec, str):
                self.pointing = \
                    SkyCoord('%s %s' % (ra, dec), unit=(u.hourangle, u.deg))
            elif isinstance(ra, float) and isinstance(dec, float):
                self.pointing = \
                    SkyCoord('%f %f' % (ra, dec), unit=(u.deg, u.deg))
        else:
            self.pointing = None

    # -----------------------------------------------------------------------

    def __add__(self, other):
        """

        Adds either a constant or another WcsHDU or other flavor of HDU to
        the data in this WcsHDU object

        """

        """ Get the data and header """
        data = self.data.copy()
        hdr = self.header.copy()

        """ Do the addition """
        if isinstance(other, (float, int)):
            data += other
        elif isinstance(other, (WcsHDU, pf.PrimaryHDU, pf.ImageHDU)):
            data += other.data
        else:
            raise TypeError('\nAdded object must be one of: int, float, '
                            'WcsHDU, PrimaryHDU, or ImageHDU')

        """ Return a new WcsHDU object """
        return WcsHDU(data, inhdr=hdr, verbose=False, wcsverb=False)

    # -----------------------------------------------------------------------

    def __sub__(self, other):
        """

        Adds either a constant or another WcsHDU or other flavor of HDU to
        the data in this WcsHDU object

        """

        """ Get the data and header """
        data = self.data.copy()
        hdr = self.header.copy()

        """ Do the addition """
        if isinstance(other, (float, int)):
            data -= other
            subitem = other
        elif isinstance(other, (WcsHDU, pf.PrimaryHDU, pf.ImageHDU)):
            data -= other.data
            if other.infile is not None:
                subitem = other.infile
            else:
                other.sigma_clip()
                subitem = 'a file with mean %9.3f' % other.mean_clip
        else:
            raise TypeError('\nAdded object must be one of: int, float, '
                            'WcsHDU, PrimaryHDU, or ImageHDU')

        """ Return a new WcsHDU object """
        hdr['subinfo'] = 'Subtracted %s' % subitem
        return WcsHDU(data, inhdr=hdr, verbose=False, wcsverb=False)

    # -----------------------------------------------------------------------

    def __mul__(self, other):
        """

        Adds either a constant or another WcsHDU or other flavor of HDU to
        the data in this WcsHDU object

        """

        """ Get the data and header """
        data = self.data.copy()
        hdr = self.header.copy()

        """ Do the addition """
        if isinstance(other, (float, int)):
            data *= other
        elif isinstance(other, (WcsHDU, pf.PrimaryHDU, pf.ImageHDU)):
            data *= other.data
        elif isinstance(other, np.ndarray):
            if other.shape != data.shape:
                raise ValueError('\nThe WcsHDU.data and numpy ndarray'
                                 ' have different sizes.\n\n')
            else:
                data *= other
        else:
            raise TypeError('\nMultiplied object must be one of the following:'
                            '\n  int\n  float\n  WcsHDU\n  PrimaryHDU\n'
                            '  ImageHDU\n  numpy ndarray')

        """ Return a new WcsHDU object """
        return WcsHDU(data, inhdr=hdr, verbose=False, wcsverb=False)

    # -----------------------------------------------------------------------

    def __truediv__(self, other):
        """

        Adds either a constant or another WcsHDU or other flavor of HDU to
        the data in this WcsHDU object

        """

        """ Get the data and header """
        data = self.data.copy()
        hdr = self.header.copy()

        """ Do the addition """
        if isinstance(other, (float, int)):
            data /= other
        elif isinstance(other, (WcsHDU, pf.PrimaryHDU, pf.ImageHDU)):
            data /= other.data
        else:
            raise TypeError('\nAdded object must be one of: int, float, '
                            'WcsHDU, PrimaryHDU, or ImageHDU')

        """ Return a new WcsHDU object """
        return WcsHDU(data, inhdr=hdr, verbose=False, wcsverb=False)

    # -----------------------------------------------------------------------

    def copy(self):
        """
        Returns a copy of the WcsHDU object
        """

        """ Use the built-in copy methods for the data and header """
        data = self.data.copy()
        hdr = self.header.copy()

        """ Return a new WcsHDU object """
        newhdu = WcsHDU(data, inhdr=hdr, verbose=False, wcsverb=False)
        newhdu.infile = self.infile
        return newhdu

    # -----------------------------------------------------------------------

    def cross_correlate(self, other, padfrac=0.6, shift=True, datacent=None,
                        datasize=None, othercent=None, hext=0,
                        reset_fft=False):
        """

        Cross correlates the image data, or a subset of them, with the
        image data in another object.

        Inputs:
          other - Other data set with which to correlate.  Can be one of the
                   following: a numpy array, a PrimaryHDU, an ImageHDU,
                   or a HDUList

        Output:
          xcorr  - cross-correlated data, returned as a WcsHDU object

        """

        """ Select the portion of the data to be used """
        if datacent is not None:
            if datasize is None:
                raise ValueError('\nif datacent is set, then datasize must '
                                 'also be set')
            x1 = int(datacent[0] - datasize/2.)
            x2 = x1 + datasize
            y1 = int(datacent[1] - datasize/2.)
            y2 = y1 + datasize
            # NEED TO ADD CHECKS
            data = self.data[y1:y2, x1:x2]
        else:
            data = self.data.copy()

        """ Set the size of the images to correlate, including padding """
        ysize, xsize = data.shape
        pad = int(padfrac * max(xsize, ysize))
        padsize = int(2 * pad + max(xsize, ysize))
        
        """
        If the conjugate FFT doesn't already exist, take the FFT of the
        selected data and then take its conjugate
        """
        if self.fftconj is None or reset_fft:
            f1 = np.zeros((padsize, padsize))
            f1[pad:pad+ysize, pad:pad+xsize] = data
            F1 = fft2(f1)
            F1c = np.conjugate(F1)
            del f1, F1
        else:
            F1c = self.fftconj

        """ Get the data the other data set """
        if isinstance(other, np.ndarray):
            odata2 = other
        elif isinstance(other, (pf.PrimaryHDU, pf.ImageHDU, WcsHDU)):
            odata2 = other.data
        if othercent is not None:
            if datasize is None:
                raise ValueError('\nif othercent is set, then datasize must '
                                 'also be set')
            x1 = int(othercent[0] - datasize/2.)
            x2 = x1 + datasize
            y1 = int(othercent[1] - datasize/2.)
            y2 = y1 + datasize
            # NEED TO ADD CHECKS
            data2 = odata2[y1:y2, x1:x2]
        else:
            data2 = odata2.copy()

        """ Make the FFT of the other data set """
        f2 = np.zeros((padsize, padsize))
        f2[pad:pad+ysize, pad:pad+xsize] = data2
        F2 = fft2(f2)
        
        """ Do the cross correlation and return the results as a WcsHDU """
        if shift:
            xc = fftshift(ifft2(F1c * F2)).real
        else:
            xc = ifft2(F1c * F2).real
        return WcsHDU(xc, verbose=False, wcsverb=False)
            
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
        wcskeys = []
        if self.wcsinfo is not None:
            for j in range(1, self.wcsinfo.wcs.naxis + 1):
                for key in ['ctype', 'crpix', 'crval', 'cunit', 'crota']:
                    wcskeys.append('%s%d' % (key, j))
                for k in range(1, self.wcsinfo.wcs.naxis + 1):
                    for key in ['pc', 'cd']:
                        wcskeys.append('%s%d_%d' % (key, j, k))

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
        This method is no longer used, but it remains in the code for
        legacy reasons.

        Instead of using update_crpix, just set the crpix values directly:
          e.g.,  myim.crpix = [1023.5, 1327.8]
        This syntax will call the crpix attribute setter

        Inputs:
         crpixarr - a list, tuple, or numpy ndarray containing the new
                     CRPIX values.  For most data, this parameter will
                     contain two elements, to replace CRPIX1 and CRPIX2

        """

        self.crpix = crpixarr

    # -----------------------------------------------------------------------

    def update_cdelt(self, cdeltarr, verbose=True):
        """

        Updates the CDELT array in the wcsinfo structure

        Inputs:
         cdeltarr - a list, tuple, or numpy ndarray containing the new
                     CDELT values.  For most data, this parameter will
                     contain two elements, to replace CDELT1 and CDELT2

        """

        """ Check dimensionality """
        if len(cdeltarr) != len(self.wcsinfo.wcs.cdelt):
            raise IndexError(' Input crpix array length does not match'
                             ' length of current crpix array')
        else:
            print('')
            print('update_cdelt is not yet properly implemented.')
            print('')

        """
        Here do something similar to what is done for the update_crpix method
        """

    # -----------------------------------------------------------------------

    def update_crval(self, crvalarr):
        """

        Updates the CRVAL array in the wcsinfo structure.
        *** This has been supplemented by the crval @property code, but
            is being kept in for legacy reasons ***

        Inputs:
         crvalarr - a list, tuple, or numpy ndarray containing the new 
                     CRVAL values.  For most data, this parameter will
                     contain two elements, to replace CRVAL1 and CRVAL2

        """

        self.crval = crvalarr

    # -----------------------------------------------------------------------

    def copy_wcsinfo(self, wcshdu):
        """

        Takes the wcsinfo from the given wcshdu and copies the information
        into the appropriate locations

        """

        self.wcsinfo = wcshdu.wcsinfo
        self.raaxis = wcshdu.raaxis
        self.decaxis = wcshdu.decaxis
        self.pixscale = wcshdu.pixscale
        self.impa = wcshdu.impa
        self.radec = wcshdu.radec

    # -----------------------------------------------------------------------

    def make_ones(self):
        """

        Creates a copy of this WcsHDU object, but with the data replaced by
        an array of 1.0 values.  Because this new WcsHDU object maintains
        the WCS information of the original object, it can be used for,
        e.g., an input to swarp.

        """

        newhdu = self.copy()
        newhdu.data = np.ones(self.data.shape)

        return newhdu

    # -----------------------------------------------------------------------

    def make_texp(self, texpkey):
        """

        Creates an exposure time map.
        Because this new WcsHDU object maintains
        the WCS information of the original object, it can be used for,
        e.g., an input to swarp.

        """

        newhdu = self.copy()
        newhdu.data = np.ones(self.data.shape)

        if texpkey.upper() in self.header.keys():
            newhdu.data *= self.header[texpkey]
        else:
            raise KeyError('Exposure time keyword %s not found in header'
                           % texpkey)

        return newhdu

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
                   'pfcam'   - flip x and then rotate -90
        """

        data = self.data.copy()
        hdr = self.header
        if 'CRPIX1' in hdr.keys() and 'CRPIX2' in hdr.keys():
            do_update = True
            crpix1 = hdr['crpix1']
            crpix2 = hdr['crpix2']
        else:
            do_update = False
            crpix1 = None
            crpix2 = None

        if method == 'x':
            self.data = data[:, ::-1]
            if do_update:
                crpix1 = hdr['naxis1'] - hdr['crpix1']
                crpix2 = hdr['crpix2']
        elif method == 'y':
            self.data = data[::-1, :]
            if do_update:
                crpix1 = hdr['crpix1']
                crpix2 = hdr['naxis2'] - hdr['crpix2']
        elif method == 'xy':
            self.data = data[::-1, ::-1]
            if do_update:
                crpix1 = hdr['naxis1'] - hdr['crpix1']
                crpix2 = hdr['naxis2'] - hdr['crpix2']
        elif method == 'pfcam':
            self.data = data.T[::-1,::-1]
            # NOTE: Still missing correct setting of crpix values
        else:
            raise ValueError('Flip method %s is not recognized' % str(method))

        if do_update:
            self.crpix = [crpix1, crpix2]
            # self.update_crpix([crpix1, crpix2], verbose=False)
    
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

    def get_rms(self, statcent, statsize, centtype, sizetype=None,
                verbose=True):
        """

        A front-end to sigma_clip that is used primarily to find the rms in
        a certain part of the data.  The main point of this method is just
        to set the "statsec" region that then gets passed to the
        sigma_clip method.

        """

        """ Get the center of the region to be used for the statistics """
        if centtype == 'radec':
            if self.wcsinfo is not None:
                w =self.wcsinfo
                xy = w.all_world2pix([statcent], 1)[0]
            else:
                raise ValueError('\nType "radec" chosen but no wcs info '
                                 'in this file.\n\n')
        elif centtype == 'xy' or centtype == 'pix':
            xy = statcent
        else:
            raise ValueError('\nstatcent parameter must be "radec" or "xy"\n')

        """ Get the size of the region to be used for the statistics """
        if sizetype is None:
            sizetype = centtype
        if sizetype == 'radec':
            if self.wcsinfo is not None:
                xysize = statsize / self.pixscale
            else:
                raise ValueError('\nType "radec" chosen but no wcs info '
                                 'in this file.\n\n')
        elif sizetype == 'xy':
            xysize = statcent
        else:
            raise ValueError('\nsizecent parameter must be "radec",  "xy"'
                             ' or None')

        """ Set up the region for calculating the statistics"""
        statsec = self.subim_bounds_xy(xy, xysize)

        """ Calculate the rms """
        self.sigma_clip(statsec=statsec)
        if verbose:
            print('RMS calculated in region [x1, y1, x2, y2]: ', statsec)
            print('RMS = %f' % self.rms_clip)

        return {'rms': self.rms_clip, 'statcent': xy, 'statsec': statsec}

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
        return normfac
            
    # -----------------------------------------------------------------------

    def sky_to_zero(self, method='sigclip', mask=None, verbose=False):
        """

        Subtracts a constant "sky value" from data in the object.
        The allowed methods for determining the sky value are:
          'sigclip' - the clipped mean (the default)
          'median'  - the median

        """

        method = method.lower()
        if mask is not None:
            data = self.data[mask]
        else:
            data = self.data
        if method == 'median':
            skyval = np.median(data)
        elif method == 'sigclip':
            self.sigma_clip(mask=mask)
            skyval = self.mean_clip
        else:
            raise ValueError('method must be one of "sigclip" or "median" ')
        self.data -= skyval
        if verbose:
            if self.infile is not None:
                descript = self.infile
            else:
                descript = 'the data'
            print('   Subtracted value of %f from %s' % (skyval, descript))
        return skyval
            
    # -----------------------------------------------------------------------

    def make_bpm(self, type, nsig=10., goodval=1, smosize=5, smtype='median',
                 var=None, outfile=None, outobj=None):
        """

        Makes a bad pixel mask (bpm) based on the data in this WcsHDU object.
        The treatment is different depending on whether the data is a
        dark or bias frame (type='dark') or is a science image (type='sci').

        Inputs:
          type    - type of data.  Right now only types 'dark' or 'sci'
                    are supported
          nsig    - number of sigma deviation from the clipped mean to indicate
                    a bad pixel.  Default=10.
          goodval - The value (1 or 0) that indicates a good pixel in the
                    pixel mask.  Bad pixels will be indicated by the opposite
                    value.  Default=1
        """

        """ Set up the baseline mask, with all pixels set to the good value """
        if goodval == 1:
            bpm = np.ones(self.data.shape)
            badval = 0
        elif goodval == 0:
            bpm = np.zeros(self.data.shape)
            badval = 1
        else:
            raise ValueError('goodval must be either 1 or 0')

        """
        Do a sigma clipping of the data.
        This will definitely be used for type "dark" and may be used for  
        """
        self.sigma_clip()

        """ Check type and act accordingly """
        varmask = None
        smodat = None
        if type.lower() == 'dark':
            """
            Get the difference image and the rms to use for the n-sigma
            comparison
            """
            diff = np.fabs(self.data - self.mean_clip)
            rms = self.rms_clip

        elif type.lower() == 'sci':
            """ Smooth the science image """
            smodat = self.smooth(smosize, smtype=smtype)
            """ 
            Calculate the difference between the smoothed and unsmoothed data
            """
            diff = self.data - smodat
            """
            Calculate the rms using either the variance data
            (strongly preferred) or the overall sky rms for the image (not
            preferred)
            """
            if var is not None:
                if isinstance(var, str):
                    vardata = pf.getdata(var)
                elif isinstance(var, np.ndarray):
                    vardata = var.copy()
                elif isinstance(var, (WcsHDU, pf.PrimaryHDU, pf.ImageHDU)):
                    vardata = var.data.copy()
                else:
                    raise TypeError('var parameter is not an accepted data '
                                    'type: (filename, numpy array, PrimaryHDU,'
                                    ' ImageHDU, or WcsHDU')
                varmask = vardata <= 0.
                vardata[varmask] = 1.e-10
                rms = np.sqrt(vardata)
            else:
                rms = self.rms_clip

        else:
            raise TypeError('Currently only types "dark" or "sci" are'
                            ' supported')

        """ Flag the pixels that are more than n sigma too high """
        bpm[np.fabs(diff) > nsig * rms] = badval
        if varmask is not None:
            bpm[varmask] = badval

        """
        Make cosmetic fixes on the image if it is a science image.  
        These updates should not affect the final coadded image, since they
         will be associated with zero weight, but will make the individual
         exposures look better.
        """
        if type.lower() == 'sci':
            badmask = bpm == badval
            if smodat is not None:
                self.data[badmask] = smodat[badmask]
                if self.infile is not None:
                    self.writeto(self.infile)

        if outfile is not None:
            phdu = pf.PrimaryHDU(bpm)
            if outobj is not None:
                phdu.header['object'] = outobj
            phdu.writeto(outfile, overwrite=True)
        else:
            return bpm

    # -----------------------------------------------------------------------

    def make_objmask(self, nsig=0.7, init_kernel=3, bpm=None, flat=None):
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

        """ Median smooth the image and set initial object mask """
        med = self.smooth(init_kernel, smtype='median')
        objmask = \
            np.where((med - self.mean_clip) / self.rms_clip > nsig, 1, 0)

        """ Reject isolated cosmic rays via a minimum filter """
        objmask = ndimage.minimum_filter(objmask, (init_kernel+2))

        """ Grow the mask regions to encompass low SNR regions """
        growkernel = int(init_kernel * 3 + 2)
        objmask = ndimage.maximum_filter(objmask, growkernel)
        objmask = ndimage.maximum_filter(objmask, growkernel)

        return objmask
    
    # -----------------------------------------------------------------------

    def apply_pixmask(self, mask, badval=0, maskval=np.nan):
        """

        Applies a mask to the data by setting the pixels identified by
         the mask to the requested value (default value is NaN).
        If the mask is boolean, then the pixels to be masked are identified
         by True.
        If the mask is integer-valued, then the pixels to be masked are
         identified as those that are set to badval.  For example, many
         bad pixel masks have goodval=1 and badval=0.

        """

        """ Check the mask type """
        if mask.dtype == int:
            pixmask = mask == badval
        elif mask.dytpe == float:
            pixmask = (int(mask) == badval)
        elif mask.dtype == bool:
            pixmask = mask.copy()
        else:
            raise TypeError('Mask elements must be either int or bool')

        """ Apply the mask """
        self.data[pixmask] = maskval
            
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

        # NEED TO ADD A CHECK FOR VALUES OF X1, X2, ETC. ##
        
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
            crpix = [hdr['crpix1'] - x1, hdr['crpix2'] - y1]
        else:
            crpix = None

        """ Put the new data and header into a WcsHDU format """
        subim = WcsHDU(data, hdr, verbose=False, wcsverb=False)
        if crpix is not None:
            subim.crpix = crpix

        """ Print out useful information """
        if verbose:
            print('   Cutout data in section [xrange,yrange]:  '
                  '[%d:%d,%d:%d]' % (x1, x2, y1, y2))
            print('   Cutout image center (x, y): (%d, %d)' %
                  (subcentx, subcenty))
            print('   Cutout image size (x y): %dx%d' % (nx, ny))

        """
        Update the header info, including updating the CRPIXn values if they
        are present.
        """
        if self.infile is not None:
            subim.header['ORIG_IM'] = 'Copied from %s' % self.infile
        subim.header['trim'] = \
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
            if fabs(self.impa) < theta_tol:
                subim = self.cutout_xy(0, 0, nx, ny, fixnans=fixnans,
                                       nanval=nanval, verbose=False)
                return subim
            else:
                imsize = [nx * self.pixscale[0], ny * self.pixscale[1]]

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
            imcent = w.all_pix2world([[x, y]], 1)[0]
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
            oscale = self.pixscale
        elif np.atleast_1d(outscale).size < 2:
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

        """
        Replace any NaNs in the image.  This needs to be done even if
        fixnans is set to False (the default) since if there are any NaNs
        in the image, then the interpolation below will set everything to
        NaN.  Therefore, set the value with which to replace all NaNs
        """

        if fixnans:
            if nanval == 'max':
                goodmask = np.isfinite(data)
                dmax = data[goodmask].max()
                nanval = 10. * dmax
        else:
            self.sigma_clip()
            nanval = self.mean_clip
        data[np.isnan(data)] = nanval

        """ Transform the coordinates """
        outdata = ndimage.map_coordinates(data, icoords, order=5,
                                          cval=np.nan)

        """ Save the output as a ImageHDU object """
        outhdr = self.make_hdr_wcs(hdr, subwcs, debug=False)
        if self.infile is not None:
            outhdr['ORIG_IM'] = 'Copied from %s' % self.infile
        subim = WcsHDU(outdata, outhdr, wcsverb=False)

        """ Clean up and exit """
        del data, icoords, skycoords, ccdcoords
        return subim

    # -----------------------------------------------------------------------

    def process_data(self, trimsec=None, bias=None, gain=-1., texp=-1.,
                     flat=None, fringe=None, darkskyflat=None, zerosky=None,
                     flip=None, pixscale=0.0, rakey='ra', deckey='dec',
                     verbose=True):

        """

        This function applies calibration corrections to the data.
        All of the calbration steps are by default turned off (keywords set
         to None).
        To apply a particular calibration step, set the appropriate keyword.
        The possible steps, along with their keywords are:

          Keyword      Calibration step
          ----------  ----------------------------------
          bias          Bias subtraction
          gain          Convert from ADU to electrons if set to value > 0
                          NB: Gain must be in e-/ADU
          flat          Flat-field correction
          fringe        Fringe subtraction
          darksky      Dark-sky flat correction
          skysub        Subtract mean sky level if keyword set to True
          texp_key     Divide by exposure time (set keyword to fits header
                        keyword name, e.g., 'exptime')
          flip          0 => no flip
                        1 => PFCam style (flip x then rotate -90), 
                        2 => P60 CCD13 style (not yet implemented)
                        3 => flip x-axis
          pixscale     If >0, apply a rough WCS using this pixel scale (RA and
                         Dec come from telescope pointing info in fits header)
          rakey        FITS header keyword for RA of telescope pointing.
                         Default = 'ra'
          deckey       FITS header keyword for Dec of telescope pointing.
                         Default = 'dec'
        
         Required inputs:
          frame

         Optional inputs (additional to those in the keyword list above):
          trimsec - a four-element list or array: [x1, y1, x2, y2] if something
                    smaller than the full frame is desired.  The coordinates
                    define the lower-left (x1, y1) and upper right (x2, y2)
                    corners of the trim section.

        """

        """ Set up convenience variables """
        hext = self.hext
        hdustr = 'HDU%d' % self.hext

        if verbose:
            startstr = 'Processing data'
            if self.infile is not None:
                startstr = '%s: %s' % (startstr, self.infile)
            print(startstr)
            if 'OBJECT' in self.header.keys():
                print(self.header['object'])

        """ Trim the data if requested """
        if trimsec is not None:
            x1, y1, x2, y2 = trimsec
            tmp = self.cutout_xy(x1, y1, x2, y2, verbose=verbose)
        else:
            tmp = WcsHDU(self.data, self.header, verbose=False, wcsverb=False)

        """ Bias-subtract if requested """
        if bias is not None:
            tmp -= bias
            biasmean = bias.data.mean()
            if hext == 0:
                keystr = 'biassub'
            else:
                keystr = 'biassub%d' % self.hext
            tmp.header[keystr] = 'Bias frame for %s is %s with mean %f' % \
                (hdustr, bias.infile, biasmean)
            print('   Subtracted bias frame %s' % bias.infile)
    
        """ Convert to electrons if requested """
        if gain > 0:
            tmp.data *= gain
            tmp.header['gain'] = (1.0, 'Units are now electrons')
            if hext == 0:
                keystr = 'ogain'
            else:
                keystr = 'ogain%d' % hext
            tmp.header.set(keystr, gain, 'Original gain for %s in e-/ADU'
                           % hdustr, after='gain')
            tmp.header['bunit'] = ('Electrons',
                                   'Converted from ADU in raw image')
            if self.hext == 0:
                keystrb1 = 'binfo_1'
            else:
                keystrb1 = 'binfo%d_1' % self.hext
            keystrb1 = keystrb1.upper()
            tmp.header[keystrb1] = \
                'Units for %s changed to e- using gain=%6.3f e-/ADU' % \
                (hdustr, gain)
            if verbose:
                print('   Converted units to e- using gain = %f' % gain)
        else:
            keystrb1 = None
    
        """ Divide by the exposure time if requested """
        if texp > 0.:
            tmp.data /= texp
            if hext == 0:
                keystr = 'binfo_2'
            else:
                keystr = 'binfo'+str(hext)+'_2'
            keystr = keystr.upper()
            tmp.header['gain'] = (texp,
                                  'If units are e-/s then gain=t_exp')
            tmp.header['bunit'] = ('Electrons/sec', 'See %s header' % keystr)
            if keystrb1 is not None:
                if keystrb1 in tmp.header.keys():
                    afterkey = keystrb1
                else:
                    afterkey = 'gain'
            else:
                afterkey = 'gain'
            tmp.header.set(keystr,
                           'Units for %s changed from e- to e-/s using '
                           'texp=%7.2f' % (hdustr, texp), after=afterkey)
            print('   Converted units from e- to e-/sec using exposure '
                  'time %7.2f' % texp)
    
        """ Apply the flat-field correction if requested """
        if flat is not None:
            tmp /= flat
            
            """
            Set up a bad pixel mask based on places where the 
            flat frame = 0, since dividing by zero gives lots of problems
            """
            flatdata = flat.data
            zeromask = flatdata == 0
            
            """ Correct for any zero pixels in the flat-field frame """
            tmp.data[zeromask] = 0
            flatmean = flatdata.mean()
            if hext == 0:
                keystr = 'flatcor'
            else:
                keystr = 'flatcor%d' % hext
            tmp.header[keystr] = \
                'Flat field image for %s is %s with mean=%f' % \
                (hdustr, flat.infile, flatmean)
            if verbose:
                print('   Divided by flat-field image: %s' % flat.infile)
    
        """ Apply the fringe correction if requested """
        if fringe is not None:
            fringedata = fringe.data
            tmp.data -= fringedata
            if self.hext == 0:
                keystr = 'fringcor'
            else:
                keystr = 'frngcor'+str(hext)
            fringemean = fringedata.mean()
            tmp.header[keystr] = \
                'Fringe image for %s is %s with mean=%f' % \
                (hdustr, fringe.infile, fringemean)
            print('   Subtracted fringe image: %s' % fringe.infile)
    
        """ Apply the dark sky flat-field correction if requested """
        if darkskyflat is not None:
            dsflatdata = darkskyflat.data
            tmp.data /= dsflatdata
            dsflatmean = dsflatdata.mean()
            """
            Set up a bad pixel mask based on places where the flat frame = 0,
            since dividing by zero gives lots of problems
            """
            dszeromask = dsflatdata == 0
            """ Correct for any zero pixels in the flat-field frame """
            tmp.data[dszeromask] = 0
            if hext == 0:
                keystr = 'dsflat'
            else:
                keystr = 'dflat'+str(hext)
            tmp.header[keystr] = \
                'Dark-sky flat image for %s is %s with mean=%f' % \
                (hdustr, darkskyflat.infile, dsflatmean)
            print('    Divided by dark-sky flat: %s' %
                  darkskyflat.infile)

        """ Subtract the sky level if requested """
        if zerosky is not None:
            skylev = tmp.sky_to_zero(method=zerosky, verbose=verbose)
            if hext == 0:
                keystr = 'zerosky'
            else:
                keystr = 'zerosky'+str(hext)
            tmp.header[keystr] = ('%s: subtracted constant sky level of %f' %
                                  (hdustr, skylev))
    
        """ Flip if requested """
        if flip is not None:
            tmp.flip(flip)
            print('   Flipped image using method %s' % flip)

        """ Add a very rough WCS if requested """
        if pixscale > 0.0:
            tmp.pixscale = pixscale

        return tmp
    
    # -----------------------------------------------------------------------

    def writeto(self, outfile=None, keeplist='all'):
        """

        Saves the (possibly modified) HDU to an output file

        """
        """ Put the possibly updated wcs info into the header """
        outhdr = self.make_hdr_wcs(self.header, self.wcsinfo, keeplist)

        """
        Create a new PrimaryHDU object and write it out, possibly
        overwriting the image from which this WcsHDU object was read
        """
        if outfile is None:
            if self.infile is None:
                raise ValueError('No output file specified and no file'
                                 ' information in current WcsHDU')
            outfile = self.infile

        pf.PrimaryHDU(self.data, outhdr).writeto(outfile, overwrite=True)

    # -----------------------------------------------------------------------

    def save(self, outfile=None, keeplist='all'):
        """

        Saves the (possibly modified) HDU to an output file

        """

        self.writeto(outfile=outfile, keeplist=keeplist)
