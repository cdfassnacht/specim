from .spec1d import Spec1d
from .spec2d import Spec2d

__version__ = 'unknown'
try:
    from _version import __version__
except ImportError:
    pass

