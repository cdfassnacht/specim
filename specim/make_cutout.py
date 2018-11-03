"""
Script to make a cutout of a fits image.  The user requests a center
(RA, Dec) and an output image size in arcseconds.  The input fits
file must have valid WCS information.

Usage: python make_cutout.py [infile] [ra_cent] [dec_cent] [imsize] [outfile]

Example: python make_cutout.py bigim.fits 272.345 62.5432 4.5 cutout.fits

Inputs:
 1. infile   - input fits file with WCS information
 2. ra_cent  - requested central RA for cutout, in decimal degrees
 3. dec_cent - requested central Dec for cutout, in decimal degrees
 4. imsize   - size of cutout, in arcsec
 5. outfile  - name of output fits file

"""

import sys
import imfuncs as imf

# ---------------------------------------------------------------------------

def help_message():
    """
    Prints a helpful message in case of syntax error
    """
    print('')
    print('Usage: python make_cutout.py [infile] [ra_cent] [dec_cent] '
          '[imsize] [outfile]')
    print('')
    print('Example: python make_cutout.py bigim.fits 272.345 62.5432'
          '4.5 cutout.fits')
    print('')
    print('Inputs:')
    print(' 1. infile   - input fits file with WCS information')
    print(' 2. ra_cent  - requested central RA for cutout, in decimal degrees')
    print(' 3. dec_cent - requested central Dec for cutout, in decimal '
          'degrees')
    print(' 4. imsize   - size of cutout, in arcsec')
    print(' 5. outfile  - name of output fits file')
    print('')

# -------------------------------------------------------------------------

""" Check command line syntax """
if len(sys.argv) < 6:
    help_message()
    exit()

""" Assign variables based on command-line input """
infile = sys.argv[1]
try:
    racent = float(sys.argv[2])
except ValueError:
    print('')
    print('ERROR: ra_cent must be a number')
    help_message()
    exit()
try:
    deccent = float(sys.argv[3])
except ValueError:
    print('')
    print('ERROR: dec_cent must be a number')
    help_message()
    exit()
try:
    imsize = float(sys.argv[4])
except ValueError:
    print('')
    print('ERROR: imsize must be a number')
    help_message()
    exit()
outfile = sys.argv[5]

""" Load the input file and make the cutout """
inim = imf.Image(infile)
if inim.found_wcs == False:
    print('')
    print('ERROR: input file %s does not have valid WCS information' % infile)
    print('')
    inim.close()
    del(inim)
    exit()
else:
    inim.poststamp_radec((racent,deccent), imsize, outfile=outfile)

inim.close()
del(inim)

