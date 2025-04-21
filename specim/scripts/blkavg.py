"""
blkavg.py

Description: This bit of code replicates (more or less) the blkavg task
 within iraf/pyraf.  The purpose is to take an input fits file and to
 create an output fits file that is smaller by an integer factor (N).
 The code takes NxN blocks of pixels from the input file and creates 1
 pixel in the output file.  Therefore, unlike ndimage.zoom or 
 ndimage.map_coordinates, there is no interpolation and therefore no
 introduction of correlated noise between the pixels.

NOTE: This code has taken the resamp function directly from Matt
Auger's indexTricks library.  Right now there is NO ERROR CHECKING
in that part of the code.  User beware!

Usage: python blkavg.py [input_fits] [output_fits] [blkavg_factor]
 Example:  python blkavg.py big_file.fits ave_file.fits 3
"""

import sys
from specim import imfuncs as imf

#-----------------------------------------------------------------------

""" Main program """

""" Check command-line format """
if len(sys.argv)<4:
    print('')
    print('Usage: python blkavg.py [input_fits] [output_fits] [blkavg_factor]')
    print('  Example:  python blkavg.py big_file.fits ave_file.fits 3')
    print('')
    exit()

"""
Get information from the command line
"""

infits  = sys.argv[1]
outfits = sys.argv[2]
try:
    blkfact = int(sys.argv[3])
except ValueError:
    print('')
    print('ERROR. Third command-line input cannot be converted to an integer')
    print('')
    exit()

""" 
Do the averaging.
NOTE: For now assume data are in HDU 0
"""

indat = imf.Image(infits)
indat.blkavg(blkfact, outfile=outfits)


""" Clean up, and exit """
del indat
