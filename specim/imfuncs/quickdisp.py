"""
Code to do a quick look at a fits file.  The file gets displayed with
default parameters.

Usage: python quickdisp [fitsfile]

             -- or --

   python quickdisp [-f [flatfile] [-fmax fmax] [fitsfile(s)]

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
if len(sys.argv)<2:
    print('')
    print('Usage:')
    print(' python quickdisp [fitsfile]')
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
    print('')
    exit()

""" Set up variables for later use """
filestart = 1
pixscale = None
fmax = 10.
flat = None
flatfile = None
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
            msg = 'ERROR: -fmax used but no fmax value is given'
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

""" Read in the flat-field data """
if flatfile is not None:
    flat = pf.getdata(flatfile)
    print('')
    print('Using flat-field file: %s' % flatfile)

""" Open and display the image """
im1 = imf.Image(infile)
if flat is not None:
    im1.data /= flat
im1.zoomsize = subimsize
im1.display(fmax=fmax, mode='xy', title=im1.infile)

""" Run the interactive zooming and marking """
im1.start_interactive()
plt.show()

del(im1)
