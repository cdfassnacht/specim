"""

imfit.py - define the ImFit class, which is used to compute statistics and
           fit models to (portions of) image data

"""

from math import sqrt
import numpy as np
from astropy.modeling import models, fitting

from cdfutils import datafuncs as df

# -----------------------------------------------------------------------

def tie_alpha(model):
    return model.alpha_0
    
# -----------------------------------------------------------------------

def tie_gamma(model):
    return model.gamma_0

# ---------------------------------------------------------------------------


class ImFit(object):
    """
    ImFit class - Used to compute statistics for and to fit models to
      (portions of) image data
    """

    def __init__(self, data):
        """

        Initialize the object by loading the portion of the image to be
        analyzed.

        """
        
        """ Put the input data into the class """
        self.data = data.copy()

        """ Set some default parameters """
        self.rms_clip = None
        self.mean_clip = None

        """
        It may be that the astropy model fitters don't work when the input
        data has NaN's in it.  Therefore replace any NaN's or infinities with
        Gaussian noise
        """
        goodmask = np.isfinite(data)
        if goodmask.sum() < data.size:
            self.mean_clip, self.rms_clip = df.sigclip(data)
            noise = np.random.normal(self.mean_clip, self.rms_clip, data.shape)
            badmask = np.logical_not(goodmask)
            self.data[badmask] = noise[badmask]
            del badmask

        
        """ Define the coordinate arrays """
        self.y, self.x = np.indices(data.shape)

        """ Clean up """
        del goodmask


    # -----------------------------------------------------------------------

    def moments(self, x0, y0, rmax=10., detect_thresh=3., skytype='global',
                verbose=False):
        """


        flux-weighted first and second moments within a square centered
        on the initial guess point and with side length of 2*rmax + 1.
        The moments will be estimates of the centroid and sigma of the
        light distribution within the square.

        Inputs:
          x0      - initial guess for x centroid
          y0      - initial guess for y centroid
          rmax    - used to set size of image region probed, which will be a
                     square with side = 2*rmax + 1. Default=10
          skytype - set how the sky/background level is set.  The options are:
                     'global' - Use the clipped mean as determined by the
                                sigma_clip method.  This is the default.
                     None     - Don't do any sky/background subtraction
        """

        """
        Select the data within the square of interest
        """
        x1, x2 = x0-rmax-1, x0+rmax+1
        y1, y2 = y0-rmax-1, y0+rmax+1
        pixmask = (self.x > x1) & (self.x < x2) & (self.y > y1) & \
            (self.y < y2)

        """ Subtract the sky level if requested """
        if skytype is None:
            f = self.data[pixmask]
        else:
            if self.rms_clip is None:
                self.mean_clip, self.rms_clip = df.sigclip(self.data)
            if verbose:
                print(self.mean_clip, self.rms_clip)
            f = self.data[pixmask] - self.mean_clip

        """ Get the x and y coordinates associated with the region """
        x = self.x[pixmask]
        y = self.y[pixmask]

        """
        Select pixels that are significantly above background for the
        calculation
        """
        objmask = f > self.mean_clip + detect_thresh * self.rms_clip
        fgood = f[objmask]
        
        """
        Calculate the flux-weighted moments
         NOTE: Do the moment calculations relative to (x1, y1) -- and then add
          x1 and y1 back at the end -- in order to avoid rounding errors (see
          SExtractor user manual)
        """
        xgood = x[objmask] - x1
        ygood = y[objmask] - y1
        fsum = fgood.sum()
        mux = (fgood * xgood).sum() / fsum
        muy = (fgood * ygood).sum() / fsum
        sigxx = (fgood * xgood**2).sum() / fsum - mux**2
        sigyy = (fgood * ygood**2).sum() / fsum - muy**2
        sigxy = (fgood * xgood * ygood).sum() / fsum - mux * muy
        mux += x1
        muy += y1
        if verbose:
            print(mux, muy)
            print(sqrt(sigxx), sqrt(sigyy), sigxy)

        """ Package the results in a dictionary, clean up, and return  """
        del x, y, xgood, ygood, fgood, pixmask, objmask
        outdict = {'mux': mux, 'muy': muy, 'sigxx': sigxx, 'sigxy': sigxy,
                   'sigyy': sigyy}
        return outdict

    # -----------------------------------------------------------------------

    def gaussians(self, x0, y0, fwhmpix=3., dxymax=5., fitbkgd=True,
                  usemoments=True, verbose=True, **kwargs):
        """

        Simultaneously fits N Gaussian profiles to N different locations in
         the image.
        The number of components to be fit is set by the number of elements
         of the x0 (and y0) arrays.

        """

        """ Get the number of components to fit """
        if isinstance(x0, int):
            x0 = float(x0)
        if isinstance(y0, int):
            y0 = float(y0)
        xinit = np.atleast_1d(x0)
        yinit = np.atleast_1d(y0)
        if xinit.size != yinit.size:
            raise ValueError('x0 and y0 are not the same size')
        ngauss = xinit.size

        """ Create the Gaussian profiles with initial guess parameters """
        x2 = np.zeros(ngauss)
        y2 = np.zeros(ngauss)
        for i in range(ngauss):
            """
            First refine the position guess by calling the moments method
            """
            if usemoments:
                objstats = self.moments(xinit[i], yinit[i], **kwargs)
                x2[i] = objstats['mux']
                y2[i] = objstats['muy']
            else:
                x2[i] = xinit[i]
                y2[i] = yinit[i]

            """
            Do a crude sky subtraction to estimate the starting amplitude.
            Note that the mean_clip attribute will have been set through
            the call to self.moments
            """
            if fitbkgd:
                amp0 = self.data[int(y2[i]), int(x2[i])] - self.mean_clip
            else:
                amp0 = self.data[int(y2[i]), int(x2[i])]
                
            """
            Create a 2d Gaussian profile with initial guess values
            """
            stddev0 = fwhmpix / 2.355
            tmpmod = models.Gaussian2D(amplitude=amp0, x_mean=x2[i],
                                       y_mean=y2[i], x_stddev=stddev0,
                                       y_stddev=stddev0, theta=0.1)

            """ Set bounds for the parameters """
            tmpmod.x_mean.bounds = (x2[i] - dxymax, x2[i] + dxymax)
            tmpmod.y_mean.bounds = (y2[i] - dxymax, y2[i] + dxymax)
            tmpmod.amplitude.bounds = (0., None)

            """
            Tie the alpha and gamma parameters together, and add this model
            to the compound model
            """
            if i==0:
                mod = tmpmod
            else:
                mod += tmpmod
                
        """ Fit the model to the data """
        fit = fitting.LevMarLSQFitter()
        outmod = fit(mod, self.x, self.y, self.data)
        # outmod = fit(mod, self.x, self.y, self.data, weights=1.0/rms)

        """ Report on fit if requested """
        if verbose:
            print('  Initial           Refined         Final')
            print('-------------   -------------   -------------')
            for i in range(ngauss):
                if ngauss > 1:
                    print('%6.2f %6.2f   %6.2f %6.2f   %6.2f %6.2f' %
                          (xinit[i], yinit[i], x2[i], y2[i],
                           outmod[i].x_mean.value, outmod[i].y_mean.value))
                else:
                    print('%6.2f %6.2f   %6.2f %6.2f   %6.2f %6.2f' %
                          (xinit[i], yinit[i], x2[i], y2[i],
                           outmod.x_mean.value, outmod.y_mean.value))
        return outmod
    
    # -----------------------------------------------------------------------

    def moffats(self, x0, y0, fwhmpix=3., dxymax=5., verbose=True, **kwargs):
        """

        Fits N Moffat profiles to N different locations in the image.
        This multiple-component fitting might be done, for example, to fit
         to the multiple lensed quasar images in a gravitational lens system,
         or to a set of stars in an image
        The number of components to be fit is set by the number of elements
         of the x0 (and y0) arrays.

        """

        """ Get the number of components to fit """
        if isinstance(x0, int):
            x0 = float(x0)
        if isinstance(y0, int):
            y0 = float(y0)
        xinit = np.atleast_1d(x0)
        yinit = np.atleast_1d(y0)
        if xinit.size != yinit.size:
            raise ValueError('x0 and y0 are not the same size')
        nmoffat = xinit.size

        """ Create the Moffat profiles with initial guess parameters """
        x2 = np.zeros(nmoffat)
        y2 = np.zeros(nmoffat)
        for i in range(nmoffat):
            """
            First refine the position guess by calling the moments method
            """
            objstats = self.moments(xinit[i], yinit[i], **kwargs)
            x2[i] = objstats['mux']
            y2[i] = objstats['muy']

            """
            Do a crude sky subtraction to estimate the starting amplitude.
            Note that the mean_clip attribute will have been set through
            the call to self.moments
            """
            amp0 = self.data[int(y2[i]), int(x2[i])] - self.mean_clip
            
            """
            Create a 2d Moffat profile with initial guess values
            The initial value for what astropy calls alpha and others
             call beta, i.e., 4.765, comes from Trujillo et al. 2001
            The initial value for what astropy calls gamma and others 
             call alpha is related to the FWHM and (alpha/beta)
            """
            alpha0 = 4.765
            gamma0 = fwhmpix / (2. * sqrt(2.**(1./alpha0) - 1.))
            tmpmod = models.Moffat2D(amplitude=amp0, x_0=x2[i], y_0=y2[i],
                                     alpha=alpha0, gamma=gamma0)

            """ Set bounds for the parameters """
            tmpmod.x_0.bounds = (x2[i] - dxymax, x2[i] + dxymax)
            tmpmod.y_0.bounds = (y2[i] - dxymax, y2[i] + dxymax)
            tmpmod.amplitude.bounds = (0., None)

            """
            Tie the alpha and gamma parameters together, and add this model
            to the compound model
            """
            if i==0:
                mod = tmpmod
            else:
                mod += tmpmod
                mod[i].alpha.tied = tie_alpha
                mod[i].gamma.tied = tie_gamma
                
        """ Fit the model to the data """
        fit = fitting.LevMarLSQFitter()
        outmod = fit(mod, self.x, self.y, self.data)
        # outmod = fit(mod, self.x, self.y, self.data, weights=1.0/rms)

        """ Report on fit if requested """
        print('  Initial           Refined         Final')
        print('-------------   -------------   -------------')
        for i in range(nmoffat):
            print('%6.2f %6.2f   %6.2f %6.2f   %6.2f %6.2f' %
                  (xinit[i], yinit[i], x2[i], y2[i], outmod[i].x_0.value,
                   outmod[i].y_0.value))
        return outmod
