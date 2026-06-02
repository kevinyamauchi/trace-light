"""a light raytracing library"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("trace-light")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Kevin Yamauchi"
__email__ = "kevin.yamauchi@gmail.com"
