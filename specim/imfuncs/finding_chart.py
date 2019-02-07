import numpy as np
from matplotlib import pyplot as plt
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord
from cdfutils import coords
from . import image as imf

def make_fc(srcname, infile, imcent, imsize, zoomsize, outfile=None,
            zoomim=None, slitsize=None, slitpa=0., posfile=None,
            slitcent='default', starpos=None, rstar=1., **kwargs):
    """

    Makes a finding chart with both a wide-field and zoomed-in image

    """

    """ First make the wide-field image """
    fcim = imf.Image(infile)
    fig = plt.figure(figsize=(8,10))
    fig.add_axes([0.1,0.3,0.7,0.7])
    title = '%s Finding Chart' % srcname
    fcim.display(imcent=imcent, imsize=imsize, cmap='grey_inv', title=title)
    plt.axvline(0, ls='dotted', color='k')
    plt.axhline(0, ls='dotted', color='k')

    """ Get slit and star information from external file if requested """
    if slitcent == 'file' or starpos == 'file':
        if posfile is None:
            print('')
            print('ERROR: position file requested but none provided via the'
                  'posfile parameter')
            print('')
            raise IOError
        postab = ascii.read(posfile)
        if len(postab.colnames) >= 7:
            names = ['name', 'hr', 'min', 'sec', 'deg', 'amin', 'asec']
            for nin, nout in zip(postab.colnames, names):
                postab.rename_column(nin, nout)
        ra = []
        dec = []
        for info in postab:
            ra.append('%d %d %f' % (info['hr'], info['min'], info['sec']))
            dec.append('%+d %d %f' % (info['deg'], info['amin'], info['asec']))
        radec = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))

    """ Add a circle for the offset/TT star position """

    """ Make the zoomed-in image """
    if zoomim is not None:
        zim = imf.Image(zoomim)
    else:
        zim = fcim
    fig.add_axes([0.1,0.05,0.25,0.25])
    zim.display(imcent=imcent, imsize=zoomsize, cmap='grey_inv')
    plt.axvline(0, ls='dotted', color='k')
    plt.axhline(0, ls='dotted', color='k')
    if slitsize is not None:
        slitra = []
        slitdec = []
        if slitcent == 'default':
            slitra.append((imcent[0]))
            slitdec.append((imcent[1]))
        elif slitcent == 'file':
            for info, pos in zip(postab, radec):
                name = info['name'].lower()
                if name[:4] == 'star' or name[-4:] == 'star':
                    pass
                else:
                    slitra.append(pos.ra.degree)
                    slitdec.append(pos.dec.degree)
        else:
            slitra.append((slitcent[0]))
            slitdec.append((slitcent[1]))
        for sra, sdec in zip(slitra, slitdec):
            zim.mark_fov(sra, sdec, slitsize, pa=slitpa)

    """ Print some useful info on the plot (still to come) """

    """ Save the result """
    if outfile is not None:
        plt.savefig(outfile)
    else:
        plt.show()
