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
