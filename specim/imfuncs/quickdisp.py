"""
Code to do a quick look at a fits file.  The file gets displayed with
default parameters.

Usage: python quickdisp.py [fitsfile]

             -- or --

   python quickdisp.py [-f [flatfile] [-fmax fmax] [fitsfile(s)]

Input parameters:
    fitsfile - Name of input fits file to be displayed
"""

import sys
import numpy as np
from astropy import wcs
from astropy.io import fits as pf
from matplotlib import pyplot as plt
from specim import imfuncs as imf

""" Check command line syntax """
if len(sys.argv) < 2:
    print('')
    print('Usage:')
    print(' python quickdisp.py (flag1 flag1val flag2 flag2val...)'
          ' [fitsfile]')
    print('')
    print('Required inputs:')
    print(' fitsfile - Name of fits file to be displayed')
    print('')
    print('OPTIONAL FLAGS and associated parameters')
    print('  -p [pixscale]    - pixel scale in arcsec/pix')
    print('  -flat [flatfile] - flat-field file to be applied to the input'
          ' fits file')
    print('  -fmax [fmax]     - maximum flux value, in sigma above mean,'
          ' for display')
    print('                     Default value: 10')
    print('  -subtract [file] - subtract the designated file from the input')
    print('                     fits file before displaying.')
    print('')
    exit()

""" Set up variables for later use """
filestart = 1
pixscale = None
fmin = -1.
fmax = 10.
flatfile = None
subfile = None
start_files = False
subimsize = 21
no_error = True

""" Parse the command line """
while start_files is False and no_error:
    if sys.argv[filestart] == '-p':
        try:
            pixscale = float(sys.argv[filestart+1])
        except ValueError:
            msg = 'ERROR: pixel scale is not a floating point number'
            no_error = False
        except IndexError:
            msg = 'ERROR: -p used but no pixel scale given'
            no_error = False
        filestart += 2
    elif sys.argv[filestart] == '-flat':
        try:
            flatfile = sys.argv[filestart+1]
        except IndexError:
            msg = 'ERROR: -flat used but no flat-field file is given'
            no_error = False
        filestart += 2
    elif sys.argv[filestart] == '-fmax':
        try:
            fmax = float(sys.argv[filestart+1])
        except ValueError:
            msg = 'ERROR: fmax is not a floating point number'
            no_error = False
        except IndexError:
            msg = 'ERROR: -fmax used but no fmax value was given'
            no_error = False
        filestart += 2
    elif sys.argv[filestart] == '-fmin':
        try:
            fmin = float(sys.argv[filestart+1])
        except ValueError:
            msg = 'ERROR: fmin is not a floating point number'
            no_error = False
        except IndexError:
            msg = 'ERROR: -fmin used but no fmin value is given'
            no_error = False
        filestart += 2
    elif sys.argv[filestart] == '-subtract':
        try:
            subfile = sys.argv[filestart + 1]
        except IndexError:
            msg = 'ERROR: -subtract used but no filename was given'
            no_error = False
        filestart += 2
    else:
        start_files = True

if no_error is not True:
    print('')
    print('%s' % msg)
    print('')
    exit()

""" Create the input file list """
# if len(sys.argv) > filestart + 1:
#    files = sys.argv[filestart:]
# else:
#     files = [sys.argv[filestart],]
infile = sys.argv[filestart]

""" Read in the data to be subtracted """
if subfile is not None:
    sub = pf.getdata(subfile)
    print('')
    print('Loading data to be subtracted from %s' % subfile)
    print(sub.size)
else:
    sub = 0.

""" Read in the flat-field data """
if flatfile is not None:
    flat = pf.getdata(flatfile)
    print('')
    print('Using flat-field file: %s' % flatfile)
else:
    flat = 1.

""" Open and display the image """
im1 = imf.Image(infile)
im1.data -= sub
im1.data /= flat
im1.zoomsize = subimsize
im1.display(fmax=fmax, fmin=fmin, mode='xy', title=im1.infile)

""" Run the interactive zooming and marking """
im1.start_interactive()
plt.show()

del im1
