import os
import sys
import glob
try:
   from astropy.io import fits as pyfits
except ImportError:
   import pyfits

""" Get the list of requested files """
if len(sys.argv) > 1:
   files = sys.argv[1:]
else:
   files = glob.glob('*fits')

""" Check the input file list """
if len(files) == 0:
   print('\nERROR: No files match requested wildcard pattern\n')
   exit()

""" Report information about each file """
print('#   File            size           Object       t_exp    Instrument '
      'Filter  ')
print('#---------------  ---------  ----------------- --------- ---------- '
      '---------')

for f in files:

   # Open the fits file
   try:
      hdulist = pyfits.open(f)
   except IOError:
      try:
         hdulist = pyfits.open(f,ignore_missing_end=True)
      except IOError:
         print('Unable to open file %s' % f)
         continue

   if f[-5:] == ".fits":
      fname = f[:-5]
   elif f[-4:] == ".FIT":
      fname = f[:-4]
   else:
      fname = f

   # Get information that is likely to be in the PHDU
   # Perhaps most important is the instrument
   hdr = hdulist[0].header

   # Instrument
   try:
      inst = hdr['instrume'].strip()
   except:
      try:
         inst = hdr['currinst'].strip()
      except:
         inst = None

   # Set some default values for instrument-specific parameters
   usecoadd  = False
   inst2     = None
   objname   = 'object'
   texpname  = 'exptime'
   texpname2 = 'exptime'
   filtname  = 'filter'
   coaddname = 'coadds'
   tscale    = 1.0

   # Set instrument-specific parameter value
   if inst is not None:

      """ HST Instruments """
      if inst == 'ACS':
         objname = 'targname'
         filtname = 'filter2'
      if inst == 'NICMOS':
         objname = 'targname'
         inst2   = 'aperture'
      elif inst == 'WFC3':
         objname = 'targname'
      elif inst == 'WFPC2':
         objname = 'targname'
         filtname = 'filtnam1'

      ### Keck Instruments ###
      elif inst[0:3] == 'ESI':
         inst = 'ESI'
         texpname = 'ttime'
         filtname = 'dwfilnam'
      elif inst[0:4] == 'LRIS':
         inst = 'LRIS'
         texpname = 'ttime'
         filtname = 'redfilt'
      elif inst == 'OSIRIS':
         usecoadd = True
         texpname = 'itime'
         tscale = 1000.0
         filtname = 'filter'
      elif inst == 'NIRC2' or inst == 'NIRES':
         usecoadd = True
         texpname = 'itime'
      elif inst == 'NIRSPEC':
         usecoadd = True
         texpname = 'itime2'
         filtname = 'filname'
         coaddname = 'coadds2'

      ### Subaru Instruments ###
      elif inst == 'MOIRCS':
         filtname = 'filter01'
      elif inst == 'SuprimeCam':
         filtname = 'filter01'

      ### Other Instruments ###
      elif inst == 'NIRI':
         usecoadd  = True
         texpname  = 'coaddexp'
         filtname  = 'filter1'

   # Get object name
   try:
      obj = hdr[objname].strip()
   except:
      obj = '(No_object)'

   # Get data size
   if len(hdulist) > 1:
      hext = 1
   else:
      hext = 0
   try:
      n1 = hdulist[hext].header['naxis1']
   except:
      n1 = 0
   try:
      n2 = hdulist[hext].header['naxis2']
   except:
      n2 = 0

   # Get exposure time, possibly including coadds
   try:
      exptime = hdr[texpname] / tscale
   except:
      try:
         exptime = hdr[texpname2] / tscale
      except:
         exptime = float('NaN')
   if usecoadd:
      try:
         coadd = hdr[coaddname]
      except:
         coadd = 1
      texp = '%2dx%-6.2f' % (coadd,exptime)
   else:
      texp = '%7.2f  ' % exptime

   # Get filter
   try:
      filt = hdr[filtname]
   except:
      filt = 'N/A'

   # Get instrument-specific cards

   if inst == 'NICMOS':
      try:
         aper = hdr['aperture'][0:4]
      except:
         aper = ''
      inst = '%s-%s' % (inst,aper)
   if(inst == 'NIRC2'):
      try:
         cam = hdr['camname']
      except:
         cam = 'N/A'
      cam = cam[0:4]
      inst = "%s-%s" % (inst,cam)

   """ Get rid of unwanted spaces """
   if inst is not None:
      inst = inst.replace(' ', '')
   obj = obj.replace(' ', '')
   filt = filt.replace(' ', '')

   # Print out final info
   fonly = fname.split('/')[-1]
   print('%-16s  %4dx%-4d  %-17s %-9s %-10s %s' % 
         (fonly,n1,n2,obj,texp,inst,filt))

