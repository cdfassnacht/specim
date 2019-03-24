"""

A collection of useful functions that can be used in support of code
in the Image and WcsHDU classes

"""

import warnings
from astropy.io import fits as pf


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

    """ Try to get rid of read-in warnings """
    warnings.filterwarnings('ignore')

    try:
        hdulist = pf.open(infile, mode=mode)
    except IOError:
        try:
            hdulist = pf.open(infile, mode=mode, ignore_missing_end=True)
        except IOError:
            print('')
            print('ERROR. Could not open fits file %s' % infile)

    return hdulist
