# ===========================================================================
#
# Code below here consists of functions that were originally the basis of
# the spec_simple code but which now have been incorporated into the
# (relatively) new Spec2d and Spec1d classes.
#
# ===========================================================================

# -----------------------------------------------------------------------


def plot_sky(infile):
    """
    Given an input 2-dimensional fits file for which the sky has NOT been
    subtracted, this function will take the median along the spatial axis
    to produce a sky spectrum.

    ** NB: Right now this ASSUMES that the dispersion is in the x direction **
    """

    data = pf.getdata(infile)
    sky = np.median(data, axis=0)
    pix = np.arange(sky.size)
    plot_spectrum_array(pix, sky, xlabel='Pixels', title='Sky Spectrum')

# -----------------------------------------------------------------------


def apply_wavecal(infile, outfile, lambda0, dlambda, varspec=True):
    """
    Given an input file containing 2 columns (x and flux), and the y-intercept
    and slope of the x-lambda transformation, convert x to wavelength units
    and write the output as outfile.
    """

    """ Read the input file """
    x, flux, var = read_spectrum(infile, varspec=varspec)

    """ Convert x from pixels to wavelength units """
    wavelength = lambda0 + dlambda * x

    """ Plot and save the results """
    if varspec:
        plot_spectrum_array(wavelength, flux, var=var,
                            title="Wavelength-calibrated spectrum")
        save_spectrum(outfile, wavelength, flux, var)
    else:
        plot_spectrum_array(wavelength, flux,
                            title="Wavelength-calibrated spectrum")
        save_spectrum(outfile, wavelength, flux)

# -----------------------------------------------------------------------


def atm_trans(w, fwhm=15., flux=None, scale=1., offset=0.0, modfile='default'):
    """
    Creates a Spec1d instance (i.e., a 1-dimensional spectrum) containing the
    transmission of the Earth's atmosphere as a function of wavelength in
    the near-infrared (NIR) part of the spectrum.  The returned spectrum
    is for the wavelength range specified by the required w parameter, which
    is a wavelength vector.

    Inputs:
        w       - wavelength vector whose min and max values set the wavelength
                  range of the returned atmospheric transmission spectrum
        fwhm    - smoothing parameter for the output spectrum
        modfile - the full path+name of the file containing the atmospheric
                  transmission data.  The default location is in the Data
                  subdirectory contained within the directory in which this
                  code is found.
    """

    """ Read in the atmospheric transmission data"""
    if modfile != 'default':
        infile = modfile
    else:
        if __file__ == 'spec_simple.py':
            moddir = '.'
        else:
            moddir = __file__.split('/spec_simple')[0]
        infile = '%s/Data/atm_trans_maunakea.fits' % moddir
    print('Loading atmospheric data from %s' % infile)
    atm0 = Spec1d(infile, informat='fitstab')
    atm0['wav'] *= 1.0e4

    """ Only use the relevant part of the atmospheric transmission spectrum"""
    mask = np.where((atm0['wav'] >= w.min()) & (atm0['wav'] <= w.max()))
    watm = atm0['wav'][mask]
    trans = atm0['flux'][mask]
    del atm0

    """ Smooth the spectrum """
    trans = ndimage.gaussian_filter(trans, fwhm)

    """ Resample the smoothed spectrum """
    tmpspec = Spec1d(wav=watm, flux=trans)
    tmpspec.resample(w)

    """ Store result as a Spec1d instance """
    atm = Spec1d(wav=tmpspec.rswav, flux=tmpspec.rsflux)

    """
    If an input spectrum has been given, then rescale the trans spectrum
    """
    if flux is not None:
        atm['flux'] *= scale * flux.max()
    else:
        atm['flux'] *= scale

    """ Add any requested vertical offset """
    atm['flux'] += offset

    """ Return the transmission spectrum """
    del watm, trans, tmpspec
    return atm

