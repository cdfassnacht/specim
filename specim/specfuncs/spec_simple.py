"""
spec_simple.py - A library of functions to do various basic spectroscopy
 processing and plotting operations

Classes and their most useful methods
  Spec1d
     plot
     smooth
     plot_sky
     apply_wavecal_linear (NEEDS TO BE CHANGED TO ACCEPT A POLYFIT POLYNOMIAL)
     mark_lines
     save

Stand-alone functions:
    combine_spectra   - takes a list of input spectra and does a inverse-
                          variance weighted sum of the flux densities
    plot_model_sky_ir - plots the NIR sky transmission as well as the night-sky
                          emission lines to aid in planning NIR spectroscopy
                          observations.
    plot_blue_and_red - Plots a blue-side and red-side spectrum on one plot
    xxxx              - more descriptions here

NOTE: More and more of the previous stand-alone functionality has been moved
 into either the Spec1d or  Spec2d class.  However, some stand-alone functions
 will probably not be moved.

"""

import numpy as np
from scipy import optimize, ndimage
import matplotlib.pyplot as plt
try:
    from astropy.io import fits as pf
except ImportError:
    import pyfits as pf
from .spec1d import Spec1d, make_sky_model
from .spec2d import Spec2d

# -----------------------------------------------------------------------


def clear_all():
    """
    Clears all of the open figures
    """

    for i in plt.get_fignums():
        plt.figure(i)
        plt.clf()


# ===========================================================================
#
# Below here are:
#    1. a few stand-alone functions that do not fit particularly well into
#        either the Spec1d or Spec2d class and so will be maintained in
#        their stand-alone format
#    2. mostly old functions that will be kept for legacy purposes but will
#        slowly be modified to function in exactly the way but via calls to
#        the Spec2d and Spec1d classes.  This will be done in such a way to
#        be transparent to the user.
#    3. code that really should be incorporated into the Spec1d or Spec2d
#        classes.  This code will slowly disappear as it gets integrated
#        into the new classes.
#
# ===========================================================================

# -----------------------------------------------------------------------


def load_2d_spectrum(filename, hdu=0):
    """
    Reads in a raw 2D spectrum from a fits file

    NOTE: This has now been replaced by the Spec2d class.
    """
    data = Spec2d(filename, hext=hdu)
    print('')
    print 'Loaded a 2-dimensional spectrum from %s' % filename
    print 'Data dimensions (x y): %dx%d' % (data.shape[1], data.shape[0])
    return data

# -----------------------------------------------------------------------


def read_spectrum(filename, informat='text', varspec=True, verbose=True):
    """
    Reads in an extracted 1D spectrum and possibly the associated variance
    spectrum.  There are two possible input file formats:
      mwa:  A multi-extension fits file with wavelength info in the fits header
              Extension 1 is the extracted spectrum (flux)
              Extension 3 is the variance spectrum
      text: An ascii text file with information in columns:
              Column 1 is the wavelength
              Column 2 is the extracted spectrum
              Column 3 (optional) is the variance spectrum

    Inputs:
        filename - input file name
        informat - format of input file ("mwa" or "text")
        varspec  - if informat is text, then this sets whether to read in a
                      variance spectrum.  Default: varspec = True
    """

    if verbose:
        print ""
        print "Reading spectrum from %s" % filename

    """ Set default for variance spectrum """
    var = None

    """ Read in the input spectrum """
    if informat == "mwa":
        hdulist = pf.open(filename)
        flux = hdulist[1].data.copy()
        var = hdulist[3].data.copy()
        varspec = True
        wavelength = np.arange(flux.size)
        hdr1 = hdulist[1].header
        wavelength = hdr1['crval1'] + wavelength*hdr1['cd1_1']
        del hdulist
    else:
        spec = np.loadtxt(filename)
        wavelength = spec[:, 0]
        flux = spec[:, 1]
        if varspec:
            if spec.shape[1] < 3:
                print('')
                print 'Warning: varspec=True but only 2 columns in input file'
                print('')
            else:
                var = spec[:, 2]
        del spec

    if verbose:
        print " Spectrum Start: %8.2f" % wavelength[0]
        print " Spectrum End:    %8.2f" % wavelength[-1]
        print " Dispersion (1st pixel): %6.2f" % (wavelength[1]-wavelength[0])
        print " Dispersion (average):    %6.2f" % \
            ((wavelength[-1]-wavelength[0])/(wavelength.size-1))
        print ""

    return wavelength, flux, var

# -----------------------------------------------------------------------


def plot_blue_and_red(bluefile, redfile, outfile=None, smooth_width=7,
                      bscale=10., xlim=[3000., 9500.], title='default',
                      z=None, mark_em=False, mark_abs=True, informat='text'):
    """
    Creates a single plot given files that contain data from the blue and red
    sides of a spectrograph.

    Required inputs:
     bluefile - file containing the blue-side spectral data, which (for now)
                is expected to be a text file containing two or three columns:
                wavelength flux [variance]
     redfile  - file containing the red-side spectral data.  Same format
                as for bluefile
    Optional inputs:
     outfile      - Name of output file containing the plot.  The default
                     value (None) means no output file.
     smooth_width - The input spectra will be smoothed by a boxcar of
                     width = smooth_width (default value = 7).  If no
                     smoothing is desired, then set smooth_width=None
     bscale       - Factor to multiply the blue-side flux by in order to
                     get it to match (roughly) the red-side flux where
                     the two spectra meet/overlap. Default value = 10.
     xlim         - x-axis range to be shown on the plot.
                     Default=[3000.,9500.]
     z            - Redshift of the object.  Default (None) means that no
                     spectral lines are marked.  If z is not None, then
                     mark spectral lines, as set by the mark_em and mark_abs
                     parameters
     mark_em      - Mark emission lines on the spectrum. Default=False
     mark_abs     - Mark absorption lines on the spectrum. Default=True
    """

    """ Read in the spectra, including the scaling for the blue side """
    bspec = Spec1d(infile=bluefile, informat=informat)
    rspec = Spec1d(infile=redfile, informat=informat)
    bspec['flux'] *= bscale
    if bspec['var'] is not None:
        bspec['var'] *= bscale

    """ Smooth the data """
    if smooth_width is None:
        usesmooth = False
    else:
        bspec.smooth(smooth_width)
        rspec.smooth(smooth_width)
        usesmooth = True

    """ Plot the spectra """
    bspec.plot(rmscolor='k', usesmooth=usesmooth)
    rspec.plot(speccolor='r', rmscolor='k', usesmooth=usesmooth)
    plt.xlim(xlim[0], xlim[1])

    """ Mark the lines if the redshift is given """
    if z is not None:
        is_z_shown = False
        w = np.concatenate((bspec['wav'], rspec['wav']))
        if smooth_width is None:
            f = np.concatenate((bspec['flux'], rspec['flux']))
        else:
            f = np.concatenate((bspec.ysmooth, rspec.ysmooth))
        combspec = Spec1d(wav=w, flux=f)
        if mark_em:
            combspec.mark_lines('em', z)
            is_z_shown = True
        if mark_abs:
            if is_z_shown:
                combspec.mark_lines('abs', z, showz=False)
            else:
                combspec.mark_lines('abs', z)


# -----------------------------------------------------------------------


def subtract_sky(data, outfile, outskyspec, dispaxis='x', doplot=True):
    """
    Given the input 2D spectrum, creates a median sky and then subtracts
    it from the input data.  Two outputs are saved: the 2D sky-subtracted
    data and a 1D sky spectrum.

    Inputs:
     data         - array containing the 2D spectrum
     outfile     - name for output fits file containing sky-subtracted spectrum
     outskyspec - name for output 1D sky spectrum
    """

    """ Set the dispersion axis direction """
    if dispaxis == "y":
        spaceaxis = 1
    else:
        spaceaxis = 0

    """ Take the median along the spatial direction to estimate the sky """
    if data.ndim < 2:
        print ""
        print "ERROR: subtract_sky needs a 2 dimensional data set"
        return
    else:
        sky1d = np.median(data, axis=spaceaxis)
    sky = np.zeros(data.shape)
    for i in range(data.shape[spaceaxis]):
        if spaceaxis == 1:
            sky[:, i] = sky1d
        else:
            sky[i, :] = sky1d

    """ Plot the sky if desired """
    x = np.arange(sky1d.size)
    if doplot:
        if spaceaxis == 1:
            xlab = 'Row'
        else:
            xlab = 'Column'
        plot_spectrum_array(x, sky1d, xlabel=xlab, ylabel='Median Counts',
                            title='Median sky spectrum')

    """ Subtract the sky  """
    skysub = data - sky

    """ Save the sky-subtracted spectrum and the median sky """
    pf.PrimaryHDU(skysub).writeto(outfile)
    print ' Wrote sky-subtracted data to %s' % outfile
    save_spectrum(outskyspec, x, sky1d)
    print ' Wrote median sky spectrum to %s' % outskyspec
    print('')

    """ Clean up """
    del sky, sky1d, skysub

# -----------------------------------------------------------------------


# def make_gauss(x, mu, sigma, amp, bkgd):
def make_gauss(x, p):
    """
    Creates a model comprised of one or more Gaussian profiles plus a
     (for now) constant background.
    NOTE: the only oddity is that, if there are more than one Gaussian in
     the profile, then the "mu" term for the subsequent Gaussians
     (i.e., p[4], p[7], ..) are actually _offsets_ between the mean of the
     subsequent Gaussian and the mean of the first.  For example,
     mu_2 = p[0] + p[4]

    Inputs:
     x  - The independent variable that is used to generate the model, ymod(x)
     p  - The parameter values.  The length of this vector will be 1+3*n, where
           n>=1, for one constant background parameter (p[0]) plus one or more
           Gaussian parameters, which come in sets of three.  Thus, p can
           be decomposed as follows:
             p[0] - background: required
             p[1] - mu_1: required
             p[2] - sigma_1: required
             p[3] - amplitude_1: required
             p[4] - offset between mu_2 and mu_1: optional
             p[5] - sigma_2: optional
             p[6] - amplitude_2: optional
             ... etc. for as many Gaussians are used to construct the profile
    """

    """ Calculate the number of Gaussians in the model """
    ngauss = int((p.size-1)/3)
    if p.size - (ngauss*3+1) != 0:
        print('')
        print('ERROR: Gaussian model contains the incorrect number of'
              'parameters')
        print('')
        return np.nan

    """ Calculate y_mod using current parameter values """
    ymod = np.zeros(x.size) + p[0]
    for i in range(ngauss):
        ind = i*3+1
        if i == 0:
            mu = p[ind]
        else:
            mu = p[1] + p[ind]

        ymod += p[ind+2] * np.exp(-0.5 * ((x - mu)/p[ind+1])**2)

    return ymod

# -----------------------------------------------------------------------


def fit_gauss(p, x, y, p_init, fitind):
    """
    Compares the data to the model.  The model consists of at least one
     gaussian plus a constant background and is created by a call to
     make_gauss.
     Thus the comparison is between ymod(x) and y, where the latter is the
     measured quantity.

    NOTE: the only oddity in the model is that, if there are more than one
     Gaussian in the profile, then the "mu" term for the subsequent Gaussians
     (i.e., p[4], p[7], ..) are actually _offsets_ between the mean of the
     subsequent Gaussian and the mean of the first.  For example,
     mu_2 = p[0] + p[4]

    Inputs:
     p  - The parameter values.  The length of this vector will be 1+3*n, where
           n>=1, for one constant background parameter (p[0]) plus one or more
           Gaussian parameters, which come in sets of three.  Thus, p can
           be decomposed as follows:
             p[0] - background: required
             p[1] - mu_1: required
             p[2] - sigma_1: required
             p[3] - amplitude_1: required
             p[4] - offset between mu_2 and mu_1: optional
             p[5] - sigma_2: optional
             p[6] - amplitude_2: optional
             ... etc. for as many Gaussians are used to construct the profile
     x  - The independent variable that is used to generate the model, ymod(x)
     y  - The measured data, to be compared
    """

    """
    Create the full list of model parameters by combining the fitted parameters
    and the fixed parameters
    """
    pfull = p_init.copy()
    pfull[fitind] = p

    """
    Compute the difference between model and real values
    """
    ymod = make_gauss(x, pfull)
    diff = y - ymod

    return diff

# -----------------------------------------------------------------------


def fit_gpb_fixmusig(p, x, y, mu, sigma):
    """
    Compares the data to the model.  The model is a gaussian plus a
     constant background.  In the fit, mu and sigma are held fixed.
    The parameter values are:
     p[0] = background level
     p[1] = amplitude
    """

    """
    Put parameters into the form that is expected by make_gauss
    """
    ptmp = np.array([p[0], mu, sigma, p[1]])

    """
    Compute the difference between model and real values
    """

    ymod = make_gauss(x, ptmp)
    diff = y - ymod

    return diff

# -----------------------------------------------------------------------


def fit_gpb_fixmu(p, x, y, mu):
    """
    Compares the data to the model.  The model is a gaussian plus a
     constant background.  In the fit, mu is held fixed.
    The parameter values are:
     p[0] = background level
     p[1] = amplitude
     p[2] = sigma
    """

    """ Unpack p """
    bkgd = p[0]
    amp = p[1]
    sigma = p[2]
    if len(p) > 3:
        nps = (len(p)-1)/2
        for inpsf in range(1, nps):
            amp = np.append(amp, p[2*inpsf+1])
            sigma = np.append(sigma, p[2*inpsf+2])

    """
    Compute the difference between model and real values
    """
    if np.shape(amp) != ():
        mu = np.ones(len(bkgd))*mu
    ymod = make_gauss(x, mu, sigma, amp, bkgd)
    diff = y - ymod

    return diff

# -----------------------------------------------------------------------


def plot_spatial_profile(infile, dispaxis="x"):
    """
    Given an input fits file with (by assumption) a 2d spectrum, this
    function will compress the spectrum along the dispersion direction
    to get an average spatial profile.

    Inputs:
        infile    - input file containing the 2d spectrum
        dispaxis - axis corresponding to the dispersion direction (i.e.,
                      the wavelength axis)
    """

    """ Read the data """
    spec = Spec2d(infile)

    """ Set the dispersion axis direction """
    spec.set_dispaxis(dispaxis)

    """ Plot the spatial profile """
    spec.spatial_profile()

# -----------------------------------------------------------------------


def find_peak(data, dispaxis="x", mu0=None, sig0=None, fixmu=False,
              fixsig=False, showplot=True, do_subplot=False, verbose=True,
              apmin=-4., apmax=4.):
    """
     Compresses a 2d spectrum along the dispersion axis so that
      the trace of the spectrum can be automatically located by fitting
      a gaussian + background to the spatial direction.  The function
      returns the parameters of the best-fit gaussian.
     The default dispersion axis is along the x direction.  To change this
      set the optional parameter dispaxis to "y"
    """

    # Set the dispersion axis direction
    if dispaxis == "y":
        specaxis = 0
    else:
        specaxis = 1
    # print "specaxis = %d" % specaxis

    """ Compress the data along the dispersion axis and find the max value """
    if data.ndim < 2:
        cdat = data.copy()
    else:
        cdat = np.median(data, axis=specaxis)
        cdat.shape
    x = np.arange(1, cdat.shape[0]+1)

    # Set initial guesses

    if fixmu:
        if mu0 is None:
            print ""
            print "ERROR: find_peak.  mu is fixed, but no value for mu0 given"
            return
        fixmunote = "**"
    else:
        if mu0 is None:
            i = cdat.argsort()
            mu0 = 1.0 * i[i.shape[0]-1]
        fixmunote = " "
    if fixsig:
        if sig0 is None:
            print ""
            print('ERROR: find_peak.  sigma is fixed, but no value for sig0'
                  'given')
            return
        fixsignote = "**"
    else:
        if sig0 is None:
            sig0 = 3.0
        fixsignote = " "
    amp0 = cdat.max()
    bkgd0 = np.median(data, axis=None)
    if(verbose):
        print ""
        print "Initial guesses for Gaussian plus background fit"
        print "------------------------------------------------"
        print " mu         = %7.2f%s" % (mu0, fixmunote)
        print " sigma      =    %5.2f%s" % (sig0, fixsignote)
        print " amplitude  = %f" % amp0
        print " background = %f" % bkgd0
        print "Parameters marked with a ** are held fixed during the fit"
        print ""

    # Fit a Gaussian plus a background to the compressed spectrum
    mf = 100000
    if fixmu and fixsig:
        p = [bkgd0, amp0]
        pt, ier = optimize.leastsq(fit_gpb_fixmusig, p, (x, cdat, mu0, sig0),
                                   maxfev=mf)
        p_out = [pt[0], mu0, sig0, pt[1]]
    # p_out, ier = optimize.leastsq(fit_gpb_fixmu, p, (x, cdat, mu0),
    #    maxfev=mf)
    # p_out, ier = optimize.leastsq(fit_gpb_fixsig, p, (x, cdat, sig0),
    #    maxfev=mf)
    else:
        p = [bkgd0, mu0, sig0, amp0]
        p_out, ier = optimize.leastsq(fit_gauss, p, (x, cdat), maxfev=mf)

    # Give results
    if(verbose):
        print "Fitted values for Gaussian plus background fit"
        print "----------------------------------------------"
        print " mu            = %7.2f%s" % (p_out[1], fixmunote)
        print " sigma        =    %5.2f%s" % (p_out[2], fixsignote)
        print " amplitude  = %f" % p_out[3]
        print " background = %f" % p_out[0]
        print "Parameters marked with a ** are held fixed during the fit"
        print ""

    # Plot the compressed spectrum
    if(showplot):
        if(do_subplot):
            plt.subplot(221)
        else:
            plt.figure(1)
            plt.clf()
        plt.plot(x, cdat, linestyle='steps')
        xmod = np.arange(1, cdat.shape[0]+1, 0.1)
        ymod = make_gauss(xmod, p_out[1], p_out[2], p_out[3], p_out[0])
        plt.plot(xmod, ymod)
        plt.axvline(p_out[1]+apmin, color='k')
        plt.axvline(p_out[1]+apmax, color='k')
        plt.xlabel('Pixel number in the spatial direction')
        plt.title('Compressed Spatial Plot')

    return p_out

# -----------------------------------------------------------------------


def extract_wtsum_col(spatialdat, mu, apmin, apmax, weight='gauss', sig=1.0,
                      gain=1.0, rdnoise=0.0, sky=None):
    """
    Extracts the spectrum from one row/column in the wavelength direction
    via a weighted sum.  The choices for the weighting are:
       'gauss'    - a Gaussian, where mu and sigma of the Gaussian are fixed.
       'uniform' - uniform weighting across the aperture, which is centered
                         at mu

    Inputs:
      spatialdat - a one-dimensional array, corresponding to a cut in
                    the spatial direction from the 2-d spectrum
      mu         - the fixed centroid of the trace
      apmin      - the lower bound, with respect to mu, of the aperture to be
                    extracted
      apmax      - the upper bound, with respect to mu, of the aperture to be
                    extracted
      weight     - the weighting scheme to be used.  Valid choices are:
                    'gauss'    (the default value)
                    'uniform'
      sig        - the fixed sigma of the Gaussian fit to the trace
      gain       - CCD gain - used to compute the variance spectrum
      rdnoise    - CCD readnoise  - used to compute the variance spectrum
      sky        - sky value for this wavelength (default=None).  Used only
                   if the spectrum passed to this function has already been
                   background-subtracted.
    """

    """ Define aperture and background regions """
    y = np.arange(spatialdat.shape[0])
    ydiff = y - mu
    apmask = (ydiff > apmin - 1) & (ydiff < apmax)
    bkgdmask = np.logical_not(apmask)
    # print bkgdmask.sum(), apmask.sum()

    """ Estimate the background """
    bkgd = np.median(spatialdat[bkgdmask], axis=None)
    # print "Background level is %7.2f" % bkgd

    """ Make the weight array """
    if(weight == 'uniform'):
        gweight = np.zeros(y.size)
        gweight[apmask] = 1.0
    else:
        gweight = make_gauss(y, mu, sig, 1.0, 0.0)

    """ Do the weighted sum """
    wtsum = ((spatialdat - bkgd)*gweight)[apmask].sum() / gweight[apmask].sum()

    """ Calculate the variance """
    if sky is None:
        varspec = (gain * spatialdat + rdnoise**2) / gain**2
        var = (varspec * gweight)[apmask].sum() / gweight[apmask].sum()
    else:
        varspec = (gain * (spatialdat + sky) + rdnoise**2)/gain**2
        var = (varspec * gweight)[apmask].sum() / gweight[apmask].sum()

    return wtsum, var, bkgd

# -----------------------------------------------------------------------


def combine_spectra(file_list, outfile, informat='text', xlabel='Pixels'):
    """
    Given a list of input spectra, reads in
    the files and does an inverse-variance weighted combination of the flux
    """

    """ Read in the input spectra """
    inspec = []
    for f in file_list:
        tmpspec = Spec1d(f, informat)
        inspec.append(tmpspec)

    """ Initialize """
    nx = inspec[0]['wav'].size
    wtflux = np.zeros(nx)
    skysum = np.zeros(nx)
    wtsum = np.zeros(nx)

    """ Create the weighted sum """
    print ""
    for spec in inspec:
        wt = np.zeros(nx)
        wt[spec['var'] != 0] = 1.0 / spec['var']
        wtflux += wt * spec['flux']
        if spec['sky'] is not None:
            skysum += spec['sky']
        wtsum += wt
        del wt

    """
    Normalize the flux, and calculate the variance of the coadded spectrum.
    Note that the equation below for the variance only works for the case
     of inverse variance weighting.
    """
    wtflux[wtsum == 0] = 0
    wtsum[wtsum == 0] = 1
    outflux = wtflux / wtsum
    outvar = 1.0 / wtsum
    if inspec[0].sky is None:
        outsky = None
    else:
        outsky = skysum / len(inspec)

    """ Create a Spec1d structure for the output spectrum """
    outspec = Spec1d(wav=inspec[0]['wav'], flux=outflux, var=outvar,
                     sky=outsky)

    """ Plot the combined spectrum """
    outspec.plot(title='Combined spectrum', xlabel=xlabel)

    """ Save the combined spectrum """
    outspec.save(outfile, outformat=informat)

# -----------------------------------------------------------------------


def planck_spec(wavelength, T=1.0e4, waveunit='Angstrom'):
    """
    Given a wavelength vector and an input temperture, generates a thermal
     spectrum over the input wavelength range.
    The input spectrum is B_lambda(T) and NOT B_nu(T)
    """

    # Define the constants in the Planck function in SI units
    c = 3.0e8
    h = 6.626e-34
    k = 1.38e-23

    # Convert the wavelength into meters (default assumption is that the
    #  input wavelength is in Angstroms
    wtmp = wavelength.copy()
    if waveunit[0:6].lower() == 'micron':
        print "Converting wavelength from microns to meters"
        wtmp *= 1.0e-6
    elif waveunit[0:5].lower() == 'meter':
        wtmp *= 1.0
    else:
        print "Converting wavelength from Angstroms to meters"
        wtmp *= 1.0e-10

    # Generate the Planck function, and then scale it so that its mean matches
    #  the mean of the observed spectrum, just for ease in plotting.
    from math import e
    denom = e**(h * c / (wtmp * k * T)) - 1.0
    B_lam = 2.0 * h * c**2 / (wtmp**5 * denom)

    return B_lam

# -----------------------------------------------------------------------


def response_ir(infile, outfile, order=6, fitrange=None, filtwidth=9):
    """
    Calculates the response function for an IR spectral setup using
    observations of a standard star.  The assumption is that the standard is a
    hot star (A or B), and therefore its spectrum is just a power law in the
    NIR.
    The steps are:
      (1) Divide the model spectrum by the observed spectrum
      (2) Fit a polynomial to the result
      (3) Write out the result to the output file
    The response function in the output file can then be multiplied by other
    spectra to do a response correction (an approximation of flux calibration).

    Inputs:
     infile:    Input file containing observed spectrum of the hot star
                This file should have 3 columns (wavelength, flux, variance)
     outfile:   Output file that will contain the response function
     order:     Order of polynomial fit (default = 6)
     fitrange:  A list of 2-element lists, where each of the smaller lists
                 contains the starting and ending values for a range of
                 good data to include in the fit.
                 E.g., fitrange=[[20150., 21500.],[22000., 25000.]]
                The default (fitrange=None) uses the full wavelength range
                 in the fit.
     filtwidth: Width of box used in the maximum filtering step, which is
                 used to minimize the number of absorption lines in the
                 spectrum before fitting a low-order polynomial to the result
                 (default = 9)
    """

    # Read the input spectrum
    wave, fluxobs, var = read_spectrum(infile)
    rms = np.sqrt(var)

    # Generate the thermal spectrum and normalize it
    B_lam = planck_spec(wave)
    B_lam *= fluxobs.mean() / B_lam.mean()

    # The features in the observed spectrum that deviate from a thermal
    #  spectrum should only be absorption lines.  Therefore, run a maximum
    #  filter

    flux = ndimage.filters.maximum_filter(fluxobs, filtwidth)

    # Calculate the observed response function
    respobs = B_lam / flux

    # Show some plots
    plt.figure(1)
    plt.clf()
    plt.plot(wave, fluxobs)
    plt.plot(wave, B_lam)
    plt.plot(wave, flux)
    plt.plot(wave, rms)
    plt.figure(2)
    plt.clf()
    plt.plot(wave, respobs)

    # Define the spectral range to be included in the fit
    if fitrange is not None:
        mask = np.zeros(respobs.size, dtype=np.bool)
        fitr = np.atleast_2d(np.asarray(fitrange))
        for i in range(fitr.shape[0]):
            wmask = np.logical_and(wave > fitr[i, 0], wave < fitr[i, 1])
            mask[wmask] = True
        wavegood = wave[mask]
        respgood = respobs[mask]
    else:
        wavegood = wave
        respgood = respobs

    # Fit a polynomial to the observed response function
    fpoly = np.polyfit(wavegood, respgood, order)
    print ""
    print "Fit a polynomial of order %d to curve in Figure 2." % order
    print "Resulting coefficients:"
    print "-----------------------"
    print fpoly

    # Convert polynomial into a smooth response function
    p = np.poly1d(fpoly)
    resp = p(wave)

    # Add the smooth response to the plot and show corrected curve
    plt.plot(wave, resp, 'r')
    fc = fluxobs * resp
    plt.figure(3)
    plt.clf()
    plt.plot(wave, fc)
    plt.plot(wave, B_lam)

    # Write smooth response to output file
    out = np.zeros((wave.size, 2))
    out[:, 0] = wave
    out[:, 1] = resp
    np.savetxt(outfile, out, '%8.3f  %.18e')

# -----------------------------------------------------------------------


def response_correct(infile, respfile, outfile):
    """
    Applies a response correction, calculated previously by response_ir
    or another function, to the input file.  The output is placed in
    outfile.

    Inputs:
        infile:    Input spectrum
        respfile: Response correction spectrum
        outfile:  Output spectrum
    """

    # Read input files
    try:
        w, f, v = np.loadtxt(infile, unpack=True)
    except IOError:
        print ""
        print "ERROR: response_correct.  Unable to read input spectrum %s" \
            % infile
        return
    try:
        wr, resp = np.loadtxt(respfile, unpack=True)
    except IOError:
        print ""
        print "ERROR: response_correct.  Unable to read response spectrum %s" \
            % respfile
        return

    # Apply the response correction and save the spectrum
    f *= resp
    v *= resp**2
    save_spectrum(outfile, w, f, v)

# -----------------------------------------------------------------------


def normalize(infile, outfile, order=6, fitrange=None, filtwidth=11):
    """
    Normalizes a spectrum by fitting to the continuum and then dividing the
     input spectrum by the fit.

    Inputs:
        infile:    File containing the input spectrum
                   This file should have 3 columns (wavelength, flux, variance)
        outfile:   Output file that will contain the normalized spectrum
        order:     Order of polynomial fit (default = 6)
        fitrange:  A list of 2-element lists, where each of the smaller lists
                   contains the starting and ending values for a range of
                   good data to include in the fit.
                   E.g., fitrange=[[20150., 21500.],[22000., 25000.]]
                   The default (fitrange=None) uses the full wavelength range
                   in the fit.
        filtwidth: Width of box used in the boxcar smoothing step, which is
                   used to minimize the number of outlier points in the input
                   spectrum before fitting the polynomial to the spectrum
                   (default = 9)
    """

    # Read the input spectrum
    wave, fluxobs, var = read_spectrum(infile)

    # Try to minimize outliers due to both emission and absorption
    #  lines and to cosmetic features (cosmic rays, bad sky-line subtraction).
    #  Do this by doing a inverse-variance weighted boxcar smoothing.

    wt = 1.0 / var
    yin = wt * fluxobs
    flux = ndimage.filters.uniform_filter(yin, filtwidth)
    flux /= ndimage.filters.uniform_filter(wt, filtwidth)

    # Show some plots
    plt.figure(1)
    plt.clf()
    plt.plot(wave, fluxobs)
    plt.plot(wave, flux, 'r')

    # Define the spectral range to be included in the fit
    if fitrange is not None:
        mask = np.zeros(flux.size, dtype=np.bool)
        fitr = np.atleast_2d(np.asarray(fitrange))
        for i in range(fitr.shape[0]):
            wmask = np.logical_and(wave > fitr[i, 0], wave < fitr[i, 1])
            mask[wmask] = True
        wavegood = wave[mask]
        fluxgood = flux[mask]
    else:
        wavegood = wave
        fluxgood = flux

    # Fit a polynomial to the observed response function
    fpoly = np.polyfit(wavegood, fluxgood, order)
    print ""
    print "Fit a polynomial of order %d to the red curve in Figure 1." % order
    print "Resulting coefficients:"
    print "-----------------------"
    print fpoly

    # Convert polynomial into a smooth response function
    p = np.poly1d(fpoly)
    cfit = p(wave)

    # Add the smooth response to the plot and show corrected curve
    plt.plot(wave, cfit, 'k')
    fc = fluxobs / cfit
    vc = var / cfit**2
    plt.figure(2)
    plt.clf()
    plt.plot(wave, fc)

    # Write normalized spectrum to output file
    save_spectrum(outfile, wave, fc, vc)

# -----------------------------------------------------------------------


def plot_atm_trans(w, flux=None, return_atm=False, title=None, scale=1.,
                   **kwargs):
    """
    Given an input spectrum, represented by the wavelength (w) and flux (spec)
    vectors, and a rough fwhm (in Angstrom), smooths and resamples the
    atmospheric transmission spectrum for the NIR and plots it.

    kwargs
      fwhm
      modfile
      scale
      offset
      color
      linestyle
    """

    """ Make a spectrum based on the input wavelength and, perhaps, flux """
    if flux is None:
        flux = np.ones(w.size)
    tmpspec = Spec1d(wav=w, flux=flux)

    """
    Plot the atmospheric transmission over the requested wavelength range
    """
    tmpspec.plot_atm_trans(scale=scale, **kwargs)

    """ Clean up and return """
    if return_atm:
        flux = tmpspec.atm_trans.copy()
        atm = Spec1d(wav=w, flux=flux)
        del tmpspec
        return atm
    else:
        del tmpspec

# -----------------------------------------------------------------------


def plot_model_sky_ir(z=None, wmin=10000., wmax=25651., smooth=25.):
    """
    Calls plot_atm_trans and make_sky_model to make a combined plot for
    the NIR sky that can be used to judge whether expected spectral lines
    will fall in good parts of the sky
    """

    """
    Create the transmission and night-sky emission line spectra over the
    requested wavelength range
    """
    wsky = np.arange(wmin, wmax)
    flux = np.ones(wsky.size)
    tmpspec = Spec1d(wav=wsky, flux=flux)
    # tmpspec.make_atm_trans()
    skymod = make_sky_model(wsky, smooth=smooth)
    skymod['flux'] /= skymod['flux'].max()

    """ Set limits to improve appearance of the plot """
    xscale = 0.02
    wrange = wmax - wmin
    xmin = wmin - xscale*wrange
    xmax = wmax + xscale*wrange

    """ Get rid of the space between the subplots"""
    # plt.figure(1)
    plt.subplots_adjust(hspace=0.001)

    """ Plot the atmospheric transmission spectrum """
    ax1 = plt.subplot(211)
    tmpspec.plot_atm_trans(title='Near IR Sky', ylabel='Transmission',
                           scale=1.)

    """
    Plot the locations of bright emission features at the requested redshift
    """
    if z is not None:
        tmpspec.mark_lines('strongem', z, marktype='line', showz=False)
    plt.xlim(xmin, xmax)
    plt.ylim(-0.15, 1.1)
    plt.setp(ax1.get_xticklabels(), visible=False)

    """ Plot the night-sky emission lines """
    plt.subplot(212, sharex=ax1)
    skymod.plot(title=None, ylabel='Sky Emission')

    """
    Plot the locations of bright emission features at the requested redshift
    """
    if z is not None:
        tmpspec.mark_lines('strongem', z, marktype='line', zfs=12)
    plt.xlim(xmin, xmax)
    dy = 0.05 * skymod['flux'].max()
    plt.ylim(-dy, (skymod['flux'].max()+dy))

    """ Clean up """
    del tmpspec, skymod


# -----------------------------------------------------------------------


def calc_lineflux(wavelength, flux, bluemin, bluemax, redmin, redmax, var=None,
                  showsub=False):
    """
    Given vectors of flux and wavelength, interactively calculates the
    integratedflux in an emission line.  The user enters the wavelength ranges
    to use for the continuum on both the blue (bluemin and bluemax) and red
    (redmin and redmax) sides of the line.  The function will do a first order
    fit  (i.e., a line) to the continuum using these ranges, subtract the
    continuum from the data, and then numerically integrate the flux/counts in
    the line.
    """

    """ Plot the data over this spectral range """
    mask = (wavelength > bluemin) & (wavelength <= redmax)
    tmplamb = wavelength[mask].copy()
    tmpflux = flux[mask].copy()
    if(var):
        tmpvar = var[mask].copy()
        plot_spectrum_array(tmplamb, tmpflux, tmpvar)
    else:
        plot_spectrum_array(tmplamb, tmpflux)

    """ Find a linear fit to the background regions """
    bkgdmask = ((tmplamb > bluemin) & (tmplamb < bluemax)) | \
        ((tmplamb > redmin) & (tmplamb < redmax))
    bkgdwave = tmplamb[bkgdmask].copy()
    bkgdflux = tmpflux[bkgdmask].copy()
    bkgdpoly = np.polyfit(bkgdwave, bkgdflux, 1)
    continuum = tmplamb*bkgdpoly[0] + bkgdpoly[1]
    plt.plot(tmplamb, continuum, 'r')
    plt.axvline(bluemin, color='k')
    plt.axvline(bluemax, color='k')
    plt.axvline(redmin, color='k')
    plt.axvline(redmax, color='k')
    plt.xlim(tmplamb[0], tmplamb[tmplamb.size - 1])

    """ Calculate the subtracted spectrum, and plot it if desired """
    subflux = tmpflux - continuum

    if(showsub):
        plt.figure()
        plt.clf()
        plt.plot(tmplamb, subflux)
        plt.xlim(tmplamb[0], tmplamb[tmplamb.size - 1])

    """ Numerically integrate the flux/counts in the line region """
    linemask = np.logical_not(bkgdmask)
    linewave = tmplamb[linemask].copy()
    lineflux = subflux[linemask].copy()
    # Assume that the wavelength scale is linear
    delwave = linewave[1] - linewave[0]
    print delwave
    intflux = (lineflux * delwave).sum()
    print intflux
