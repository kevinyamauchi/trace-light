"""A light raytracing library."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("trace-light")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Kevin Yamauchi"
__email__ = "kevin.yamauchi@gmail.com"

from trace_light import backends
from trace_light.kernels import _trace_surfaces
from trace_light.rays import Rays, _Params, _Structure

__all__ = [
    "Rays",
    "_Params",
    "_Structure",
    "_trace_surfaces",
    "backends",
]
