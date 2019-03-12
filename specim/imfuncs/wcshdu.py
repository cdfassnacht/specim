"""

wcshdu.py

Defines a new class that is essentially a fits PrimaryHDU class but with
some of the WCS information in the header split out into separate 
attributes of the class

"""

import os
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

    def __init__(self, indat, hext=0, wcsext=None, verbose=True):
        """

        Inputs:
          indat - either a filename for a fits file or a HDUList that has
                  previously been read in
        """

        """ Set some default values """
        self.wcsinfo = None
        self.radec = None
        self.pixscale = None
        self.impa = None
        self.radec = None
        self.raaxis = None
        self.decaxis = None

        """ Check the format of the input info """
        if isinstance(indat, str):
            informat = 'file'
            try:
                hdu = self.read_from_file(indat, verbose=verbose)
            except IOError:
                raise IOError
        elif isinstance(indat, pf.HDUList):
            informat = 'hdulist'
            hdu = indat
        else:
            print('')
            print('ERROR: The input for the WcsHDU class must be'
                  'one of the following:')
            print('  1. A filename (i.e. a string)')
            print('  2. A HDUList')
            print('')
            raise TypeError

        """
        Select the request HDU and then use the super-class to give
        it useful properties
        """
        data = hdu[hext].data
        hdr = hdu[hext].header
        super(WcsHDU, self).__init__(data, hdr)

        """
        Now add WCS attributes to the class.
        NOTE, sometimes these are in a different header, which can be
        indicated by the wcsext parameter
        """
        if wcsext is not None:
            wcshdr = hdu[wcsext].header
        else:
            wcshdr = hdu[hext].header
        try:
            self.read_wcsinfo(wcshdr, verbose=verbose)
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
