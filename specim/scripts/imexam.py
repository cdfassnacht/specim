"""

Interactive program to get radial profiles, etc., of the input image.
The functionality mimics a subset of the iraf imexam function

Usage: python imexam.py (-fmax [fmax]) [fitsfile]


Input parameters:
    fmax        - [OPTIONAL] maximum display value, in sigma above mean
                  Default value if this is not set is fmax=10
    fitsfile    - Name of the input fits file

"""

import sys
from matplotlib import pyplot as plt
from specim.imfuncs import image as imf

""" Check command line syntax """
if len(sys.argv)<2:
    print('')
    print('Usage:')
    print(' python imexam.py (flag1 flag1val'
          ' flag2 flag2val...) [fitsfile]')
    print('')
    print('Required inputs:')
    print(' fitsfile    - Input filename')
    print('')
    print('OPTIONAL FLAGS and associated parameters')
    print('  -fmax [fmax]     - maximum flux value, in sigma above mean,'
          ' for display')
    print('                     Default value: 10')
    print('')
    exit()

""" Set up variables for later use """
filestart = 1
start_files = False
no_error = True
fmax = 10.

""" Parse command line """
while start_files is False and no_error:
    if sys.argv[filestart] == '-fmax':
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

infile = sys.argv[filestart]

""" Open and display the input file and """
im = imf.Image(infile)
im.display(fmax=fmax, mode='xy', title=im.infile)

""" Start the interactive display """
im.start_interactive()
plt.show()

