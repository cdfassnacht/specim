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


# -----------------------------------------------------------------------

""" Main program """

""" Check command-line format """
if len(sys.argv) < 4:
    print('')
    print('Usage: python blkavg.py [input_fits] [output_fits] [blkavg_factor]')
    print('  Example:  python blkavg.py orig_file.fits binned_file.fits 3')
    print('')
    exit()

""" Check command line syntax """
if len(sys.argv) < 2:
    print('')
    print('Usage:')
    print(' python blkavg.py (flag1 flag1val flag2 flag2val...)'
          ' [orig_file] [binned_file] [binfactor]')
    print('')
    print('Required inputs:')
    print(' orig_file   - Name of original fits file to be binned')
    print(' binned_file - Name of output binned fits file')
    print(' binfactor   - Binning factor, e.g. 2 means 2x2 binning')
    print('                (must be an integer)')
    print('')
    print('OPTIONAL FLAGS and associated parameters')
    print('  -mode ["sum"/"ave"]   - Method used to bin. Can only be "sum" '
          'or "ave"')
    print('                      Default value is "ave"')
    print('  -rms [rmsfile]        - Also bin an associated rms file, '
          'taking into')
    print('                      account the proper error propagation')
    print('  -verbose [True/False] - Report on binning? Default=True')
    print('')
    exit()

""" Set up variables for later use """
filestart = 1
mode = 'ave'
rmsfile = None
verbose = True
start_files = False
no_error = True
msg = ''

""" Parse the command line """
while start_files is False and no_error:
    if sys.argv[filestart] == '-mode':
        try:
            mode = sys.argv[filestart+1]
        except IndexError:
            msg = 'ERROR: -mode flag but no mode was given'
            no_error = False
        filestart += 2
    elif sys.argv[filestart] == '-rms':
        try:
            rmsfile = sys.argv[filestart+1]
        except IndexError:
            msg = 'ERROR: -rms flag used but no rms filename provided'
            no_error = False
        filestart += 2
    elif sys.argv[filestart] == '-verbose':
        try:
            verbose = bool(sys.argv[filestart+1])
        except ValueError:
            msg = 'ERROR: verbose value given is not a boolean (True or False)'
            no_error = False
        except IndexError:
            msg = 'ERROR: -vebose flag used but no verbose value was given'
            no_error = False
        filestart += 2
    else:
        start_files = True

if no_error is not True:
    print('')
    print('%s' % msg)
    print('')
    exit()

"""
Get the remaining information from the command line
"""
infits = sys.argv[filestart]
outfits = sys.argv[filestart+1]
try:
    blkfact = int(sys.argv[filestart+2])
except ValueError:
    print('')
    print('ERROR. Last command-line input cannot be converted to an integer')
    print('')
    exit()

""" Summarize the inputs """
print('')
print('Original file: %s' % infits)
print('Binned output file: %s' % outfits)
if rmsfile is not None:
    print('Original RMS file: %s' % rmsfile)
    rmsoutfile = rmsfile.replace('.fits', '%d_%d.fits' % (blkfact, blkfact))
    print('Binned output RMS file: %s' % rmsoutfile)

""" 
Do the binning for the input file
NOTE: For now assume data are in HDU 0
"""
indat = imf.Image(infits)
indat.blkavg(blkfact, mode=mode, intype='sci', outfile=outfits)
del indat

""" If requested, do the binning for the rms file as well """
if rmsfile is not None:
    inrms = imf.Image(rmsfile)
    inrms.blkavg(blkfact, mode=mode, intype='rms', outfile=rmsoutfile)
    del inrms
