print 'import imfuncs'
import imfuncs
print 'import spec_simple'
import spec_simple

__version__ = 'unknown'
try:
    from _version import __version__
except ImportError:
    pass

