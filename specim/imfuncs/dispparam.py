"""

dispparam.py

Definition of a class that sets the parameters to be used when an 
image is displayed

"""

from matplotlib import pyplot as plt

# -----------------------------------------------------------------------


class DispParam(object):
    """

    A DispParam object sets and stores parameters that are used to govern
    what a displayed image looks like


    The code here was originally part of the Image class, but has been
    split out for clarity.

    """

    def __init__(self):
        """

        Initiates a DispParam object and initializes all of the 
        attributes

        """

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
        self.statsize = 2048         # Stats region size if image is too big
        self.statsec = None          # Region to use for pixel statistics
        self.mode = 'radec'          # Default display units are arcsec offsets
        self.extval = None           # Just label the axes by pixels
        self.cmap = plt.cm.YlOrBr_r  # This corresponds to the 'gaia' cmap
        self.title = None            # Title on displayed image

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
            print(' WARNING - Requested unknown color map.  Using gaia'
                  ' colors')
            self.cmap = plt.cm.YlOrBr_r

    # -----------------------------------------------------------------------

    def set_wcsextent(self, zeropos=None):
        """

        For making plots with WCS information, the display style is to set
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
          zeropos - By default, which happens when zeropos=None, the (0, 0)
                     point on the output image, as designated by the image
                     axis labels, will be at the center of the image.
                     However, you can shift the (0, 0) point to be somewhere
                     else by setting zeropos.  For example, zeropos=(0.5, 0.3)
                     will shift the origin to the point that would have been
                     (0.5, 0.3) if the origin were at the center of the image
        """

        # self.get_wcs(self['plotim'].header)
        data = self['plotim'].data
        icoords = np.indices(data.shape).astype(np.float32)
        pltc = np.zeros(icoords.shape)
        pltc[0] = (icoords[0] - data.shape[0] / 2.) * self['input'].pixscale[1]
        pltc[1] = (icoords[1] - data.shape[1] / 2.) * self['input'].pixscale[0]
        pltc[1] *= -1.
        maxi = np.atleast_1d(data.shape) - 1
        extx1 = pltc[1][0, 0]
        exty1 = pltc[0][0, 0]
        extx2 = pltc[1][maxi[0], maxi[1]] - self['input'].pixscale[1]
        exty2 = pltc[0][maxi[0], maxi[1]] + self['input'].pixscale[0]

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

    def set_extval(self, mode):
        """

        """
    # -----------------------------------------------------------------------

    def set_flux_limits(self, fmin=-1., fmax=10., funits='sigma',
                        mask=None, verbose=False, debug=False):
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
                        self.sigma_clip('plotim', verbose=verbose)
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
            print('fmin:  %f' % self.fmin)
            print('fmax:  %f' % self.fmax)

        else:
            """
            If funits is not 'abs', then it must be 'sigma', which is the only
            other possibility, and the default value for funits.  In that case,
            set the display limits in terms of the clipped mean and sigma
            """

            """ Start by calculating the clipped statistics if needed """
            if self.found_rms is False:
                print('Calculating display limits')
                print('--------------------------')
                if mask is not None:
                    print('Using a mask')
                if debug:
                    tmpverb = True
                else:
                    tmpverb = False
                self.sigma_clip('plotim', verbose=tmpverb, mask=mask)
                self.found_rms = True

            """ If disprange is not set, then query the user for the range """
            if fmin is None or fmax is None:
                fmin = -1.
                fmax = 10.
                tmp = raw_input('Enter min flux for display in terms of sigma'
                                ' from mean [%f]: ' % fmin)
                if len(tmp) > 0:
                    fmin = float(tmp)
                tmp = raw_input('Enter max flux for display in terms of sigma'
                                ' from mean [%f]: ' % fmax)
                if len(tmp) > 0:
                    fmax = float(tmp)

            """ Set fmin and fmax in terms of clipped mean and sigma"""
            self.fmin = self.mean_clip + fmin * self.rms_clip
            self.fmax = self.mean_clip + fmax * self.rms_clip
            print(' Clipped mean: %f' % self.mean_clip)
            print(' Clipped rms:  %f' % self.rms_clip)
            s1 = '-' if fmin < 0. else '+'
            s2 = '-' if fmax < 0. else '+'
            print(' fmin (mean %s %3d sigma):  %f' %
                  (s1, fabs(fmin), self.fmin))
            print(' fmax (mean %s %3d sigma):  %f' %
                  (s2, fabs(fmax), self.fmax))



