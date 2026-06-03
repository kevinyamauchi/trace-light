"""A light raytracing library."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("trace-light")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Kevin Yamauchi"
__email__ = "kevin.yamauchi@gmail.com"

from optisketch import analysis, backends, fluorescence, optimize, viz
from optisketch.kernels import _trace_surfaces, trace
from optisketch.rays import (
    Rays,
    Surface,
    System,
    _Params,
    _Structure,
    load_system,
    save_system,
)
from optisketch.sources import Source, collimated_source, emit, point_source
from optisketch.systems import SystemBuilder

__all__ = [
    "Rays",
    "Source",
    "Surface",
    "System",
    "SystemBuilder",
    "_Params",
    "_Structure",
    "_trace_surfaces",
    "analysis",
    "backends",
    "collimated_source",
    "emit",
    "fluorescence",
    "load_system",
    "optimize",
    "point_source",
    "save_system",
    "trace",
    "viz",
]
