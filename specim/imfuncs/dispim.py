"""

dispim.py

Definition of a class that will take care of the mechanics of displaying
an image and interacting with the displayed image

"""

import sys
from math import fabs, log10
from math import cos as mcos
import numpy as np
from matplotlib import pyplot as plt
from .wcshdu import WcsHDU
from .dispparam import DispParam

pyversion = sys.version_info.major

# -----------------------------------------------------------------------


class DispIm(WcsHDU):
    """

    The DispIm class is invoked to display a 2d image and handle any
    interactions with the displayed image.

    The code here was originally part of the Image class, but has been
    split out for clarity and to simplify some aspects of the interactive
    code

    """

    def __init__(self, inhdu):
        """

        Initialize an object by taking the data to be displayed, information
        about the data, and parameters that govern how to display the data.

        """

        """ Link to the superclass """
        if pyversion == 2:
            super(DispIm, self).__init__(inhdu, wcsverb=False)
        else:
            super().__init__(inhdu, wcsverb=False)

        """ Initialize some parameters """
        self.dpar = None
        self.ax1 = None
        self.fig1 = None
        self.fig2 = None
        self.zoomsize = 31           # Size of postage-stamp zoom
        self.intlabs = ['tltext', 'tctext', 'trtext',
                        'bltext', 'bctext', 'brtext']
        self.intlabpos = [(0.05, 0.9), (0.5, 0.9), (0.95, 0.9),
                          (0.05, 0.1), (0.5, 0.1), (0.95, 0.1)]
        self.intalign = ['left', 'center', 'right', 'left', 'center', 'right']

    # -----------------------------------------------------------------------

    def make_dpar(self):
        """

        Loads parameters that govern how the image will be displayed

        """

        dpar = DispParam(self)
        dpar.display_setup(mode=mode, verbose=verbose, debug=debug, **kwargs)

    # -----------------------------------------------------------------------

    def scale_data(self, fscale='linear'):
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

        if fscale == 'log':
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
            data = self.data.copy() - self.data.min()

            """ Now rescale from 1-255 in requested range """
            data[data >= 0] = ((bitscale - 1) * data[data >= 0] / fdiff) + 1.
            vmin = 0
            vmax = log10(bitscale)
            data[data <= 0.] = 1.
            data = np.log10(data)
            print('Using log scaling: vmin = %f, vmax = %f' % (vmin, vmax))
            print(data.min(), data.max())

        else:
            """ Linear scaling is the default """
            data = self.data.copy()
            vmin = self.fmin
            vmax = self.fmax

        """ Return the values """
        return data, vmin, vmax

    # -----------------------------------------------------------------------

    @staticmethod
    def add_scalebar(ax, dpar):
        """

        Adds a scalebar to the figure

        """

        if dpar.mode == 'radec':
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            x1 = xmin + 0.1 * (xmax - xmin)
            y1 = ymin + 0.1 * (ymax - ymin)
            x2 = x1 - dpar['barlength']  # RA runs in opposite to x
            ax.plot([x1, x2], [y1, y1], color=dpar['barcolor'], lw=1)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    # -----------------------------------------------------------------------

    @staticmethod
    def add_crosshair(ax, dpar):
        """

        Adds a crosshair at the position in dpar['crosshair']

        """

        """ Check the format of dpar['crosshair'] """
        if isinstance(dpar['crosshair'], (tuple, list)):
            if len(dpar['crosshair']) != 2:
                raise ValueError('\nThe "crosshair" parameter must have two'
                                 ' elements: (x,y) or (RA,Dec)\n')
        else:
            raise TypeError('\nThe "crosshair" parameter must be either a'
                            ' tuple or a list, with 2 elements')

        ax.axvline(dpar['crosshair'][0], color=dpar['xhaircolor'],
                   lw=dpar['xhairlw'], ls=dpar['xhairls'])
        ax.axhline(dpar['crosshair'][1], color=dpar['xhaircolor'],
                   lw=dpar['xhairlw'], ls=dpar['xhairls'])

    # -----------------------------------------------------------------------

    def display(self, ax=None, axlabel=True, fontsize=None,
                show_xyproj=False, mode='radec', dpar=None, debug=False):
        """

        Display the image data through a call to matplotlib.pyplot.imshow
        Many of the parameters that govern how the data will be displayed
         are contained in the dpar parameter, which is a DispParam object
         (see specim.imfuncs.dispparam)

        """

        """
        Set up for displaying the image data
         - If show_xyproj is False (the default), then just show
            self.data
         - If show_xyproj is True, then make a three panel plot, with
             Panel 1: self.data (i.e., what you would see in the
              default behavior)
             Panel 2: Projection of data in self['plotim'].data onto the
                      x-axis
             Panel 3: Projection of data in self['plotim'].data onto the
                      y-axis
          - Setting show_xyproj=True is most useful when evaluating, e.g., a
             star in the image data.  The projections along the two axes of the
             cutout can be useful for evaluating whether the object is a star
             and/or whether it is saturated
        """
        if show_xyproj:
            self.fig2 = plt.figure(figsize=(10, 3))
            self.fig2.add_subplot(131)
        elif ax is not None:
            self.fig1 = plt.gcf()
            ax1 = ax
        else:
            self.fig1 = plt.gcf()
            ax1 = plt.gca()

        """
        If no display parameters were passed via dpar, then set up the
        default display values
        """
        if dpar is None:
            dpar = DispParam(self)

        """ Set figure / axes attributes based on the dpar values """
        # self.fig1.set_dpi(dpar.dpi)
        # self.fig1.set_facecolor(dpar.facecolor)

        """ Set the actual range for the display """
        self.fmin = dpar.fmin
        self.fmax = dpar['fmax']
        data, vmin, vmax = self.scale_data(dpar.fscale)

        """ Display the image data """
        self.mode = mode
        ax1.imshow(data, origin='lower', cmap=dpar.cmap, vmin=vmin,
                   vmax=vmax, interpolation='nearest', extent=dpar.extval,
                   aspect='equal')

        """ Provide exterior labels for the plot, if requested """
        if dpar.axlab == 'off':
            ax1.set_xticks([])
            ax1.set_yticks([])
        elif axlabel is True:
            if mode == 'radec':
                xlabel = 'Offset (arcsec)'
                ylabel = 'Offset (arcsec)'
            else:
                xlabel = 'x (pix)'
                ylabel = 'y (pix)'
            if fontsize is not None:
                plt.xlabel(xlabel, fontsize=fontsize)
                plt.ylabel(ylabel, fontsize=fontsize)
            else:
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
        if dpar.title is not None:
            plt.title(dpar.title)

        """ Provide interior labels for the plot, if requested """
        for i, k in enumerate(self.intlabs):
            if k in dpar.keys():
                ax1.text(self.intlabpos[i][0], self.intlabpos[i][1], dpar[k],
                         horizontalalignment=self.intalign[i],
                         color=dpar['intlabcolor'], transform=ax.transAxes,
                         fontsize=10)

        """ Add a scalebar if requested """
        if 'scalebar' in dpar.keys():
            self.add_scalebar(ax1, dpar)

        """ Add a crosshair if requested """
        if 'crosshair' in dpar.keys():
            self.add_crosshair(ax1, dpar)

        """ Save the current axis that was used to make the plot """
        self.ax1 = ax1

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
        print('   m     Mark the position of an object')
        print('   z     Zoom in at the position of the cursor')
        print('   q     Quit and close the window')
        print('   x     Quit but do not close the window')
        print('')

    # -----------------------------------------------------------------------

    def onclick(self, event):
        """
        Actions taken if a mouse button is clicked.  In this case the
        following are done:
          (1) Store and print (x, y) value of cursor
          (2) If the image has wcs info (i.e., if wcsinfo is not None) then
                store and print the (RA, dec) value associated with the (x, y)
        """
        self.xclick = event.xdata
        self.yclick = event.ydata
        print('')
        print('Mouse click x, y:    %7.1f %7.1f' %
              (self.xclick, self.yclick))

        """
        Also show the (RA, dec) of the clicked position if the input file has
         a WCS solution
        NOTE: This needs to be handled differently if the displayed image has
         axes in pixels or in arcsec offsets
        """
        if self.wcsinfo is not None:
            if self.mode == 'xy':
                pix = np.zeros((1, self.wcsinfo.naxis))
                pix[0, 0] = self.xclick
                pix[0, 1] = self.yclick
                radec = self.wcsinfo.wcs_pix2world(pix, 1)
                self.raclick = radec[0, 0]
                self.decclick = radec[0, 1]
            else:
                """ For now use small-angle formula """
                radec = self.radec
                cosdec = mcos(radec.dec.radian)
                self.raclick = radec.ra.degree + \
                    (self.xclick + self.zeropos[0]) / (3600. * cosdec)
                self.decclick = radec.dec.degree + self.yclick/3600. + \
                    self.zeropos[1]
            print('Mouse click ra, dec: %11.7f %+11.7f' %
                  (self.raclick, self.decclick))
        return

    # -----------------------------------------------------------------------

    def keypress(self, event):
        """
        Actions taken if a key on the keyboard is pressed
        """

        # print(event.key)
        if event.key == 'b':
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
            plt.axvline(self.xmark, color='g')
            plt.axhline(self.ymark, color='g')
            plt.draw()
            # imsize = (self.zoomsize, self.zoomsize)
            # imcent = (self.xmark, self.ymark)
            # self.display(imcent=imcent, imsize=imsize, mode=self.mode,
            #              show_xyproj=True)

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
            # self.fig1.canvas.draw_idle()
            plt.draw()
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
            print('Closing down')
            print('')
            if self.fig1:
                self.fig1.canvas.mpl_disconnect(self.cid_mouse)
                self.fig1.canvas.mpl_disconnect(self.cid_keypress)
            if self.fig2:
                self.fig2.canvas.mpl_disconnect(self.cid_keypress2)
            for ii in plt.get_fignums():
                plt.close(ii)
            return

        # self.keypress_info()
        return
