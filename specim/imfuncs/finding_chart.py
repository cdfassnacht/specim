# import numpy as np
from matplotlib import pyplot as plt
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord
# from cdfutils import coords
from . import image as imf


def read_posfile(posfile, verbose=True):
    """

    Reads one or more (RA, Dec) positions from an input file

    """

    if verbose:
        print('Reading central position from %s' % posfile)
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

    return postab, radec

# ---------------------------------------------------------------------------


def make_fc(srcname, infile, imcent, imsize, zoomsize, outfile=None,
            zoomim=None, slitsize=None, slitpa=0., slitfile=None,
            slitcent='default', starpos=None, rstar=1., **kwargs):
    """

    Makes a finding chart with both a wide-field and zoomed-in image

    """

    """ Get the image center """
    if isinstance(imcent, tuple):
        cent = imcent
    elif isinstance(imcent, str):
        imtab, impos = read_posfile(imcent)
        cent = (impos[0].ra.degree, impos[0].dec.degree)

    """ Make the wide-field image """
    fcim = imf.Image(infile)
    fig = plt.figure(figsize=(8, 10))
    fig.add_axes([0.1, 0.3, 0.7, 0.7])
    title = '%s Finding Chart' % srcname
    fcim.display(imcent=cent, imsize=imsize, cmap='grey_inv', title=title)
    plt.axvline(0, ls='dotted', color='k')
    plt.axhline(0, ls='dotted', color='k')

    """ Get slit and star information from external file if requested """
    if slitcent == 'file' or starpos == 'file':
        if slitfile is None:
            print('')
            print('ERROR: slit position file requested but none provided.')
            print('Please use the slitfile parameter')
            print('')
            raise IOError
        postab, radec = read_posfile(slitfile)

    """ Add a circle for the offset/TT star position """
    if starpos is not None:
        starra = []
        stardec = []
        if starpos == 'file':
            for info, pos in zip(postab, radec):
                name = info['name'].lower()
                if name[:4] == 'star' or name[-4:] == 'star':
                    starra.append(pos.ra.degree)
                    stardec.append(pos.dec.degree)
                else:
                    pass
        else:
            starra.append((starpos[0]))
            stardec.append((starpos[1]))
        for sra, sdec in zip(starra, stardec):
            fcim.plot_circle(sra, sdec, rstar, crosshair=True)

    """ Make the zoomed-in image """
    if zoomim is not None:
        zim = imf.Image(zoomim)
    else:
        zim = fcim
    fig.add_axes([0.1, 0.05, 0.25, 0.25])
    zim.display(imcent=cent, imsize=zoomsize, cmap='grey_inv', **kwargs)
    plt.axvline(0, ls='dotted', color='k')
    plt.axhline(0, ls='dotted', color='k')
    if slitsize is not None:
        slitra = []
        slitdec = []
        if slitcent == 'default':
            slitra.append((cent[0]))
            slitdec.append((cent[1]))
        elif slitcent == 'file':
            for info, pos in zip(postab, radec):
                name = info['name'].lower()
                if name[:4] == 'star' or name[-4:] == 'star' or \
                        name == 'center':
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
