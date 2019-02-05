from matplotlib import pyplot as plt
from . import image as imf

def make_fc(srcname, infile, imcent, imsize, zoomsize, outfile, zoomim=None,
            slitsize=None, slitcent='default', slitpa=0., starpos=None,
            rstar=1., **kwargs):
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
        if slitcent == 'default':
            slitra = imcent[0]
            slitdec = imcent[1]
        else:
            slitra = slitcent[0]
            slitdec = slitcent[1]
        zim.mark_fov(slitra, slitdec, slitsize, pa=slitpa)

    """ Print some useful info on the plot (still to come) """

    """ Save the result """
    plt.savefig(outfile)
