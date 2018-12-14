import os, re
from os import sys
from distutils.core import setup
from distutils.command.install import INSTALL_SCHEMES
from imp import find_module

for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

""" Check for required modules """
try:
    find_module('numpy')
except:
    sys.exit('### Error: python module numpy not found')
    
try:
    find_module('scipy')
except:
    sys.exit('### Error: python module scipy not found')
    
try:
    find_module('astropy')
except ImportError:
    try:
        find_module('pyfits')
    except ImportError:
        sys.exit('### Error: Neither astropy nor pyfits found.')

try:
    find_module('matplotlib')
except ImportError:
    sys.exit('### Error: python module matplotlib not found')

try:
    find_module('cdfutils')
except ImportError:
    sys.exit('### Error: python module cdfutils not found. '
             'Download and install from github cdfassnacht/cdfutils')


#try: find_module('MySQLdb')
#except: sys.exit('### Error: python module MySQLdb not found')


verstr = "unknown"
try:
    parentdir = os.getcwd()+'/'
    verstrline = open(parentdir+'/specim/_version.py', "rt").read()
except EnvironmentError:
    pass # Okay, there is no version file.
else:
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        verstr = mo.group(1)
    else:
        raise RuntimeError("unable to find version in " + parentdir + "+specim/_version.py")


setup(
    name = 'specim',
    version = verstr,#'0.1.3',
    author = 'Chris Fassnacht',
    author_email = 'cdfassnacht@ucdavis.edu',
    scripts=[],
    license = 'LICENSE.txt',
    description = 'Code for visualizing fits images and for'
    'extracting and plotting spectra',
    #long_description = open('README.txt').read(),
    requires = ['numpy','scipy','astropy','matplotlib','cdfutils'],
    packages = ['specim', 'specim.imfuncs', 'specim.specfuncs'],
    #package_dir = {'':'src'},
    package_data = {'specim.specfuncs' : ['Data/*']}
)
