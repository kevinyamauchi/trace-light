"""A light raytracing library."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("trace-light")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Kevin Yamauchi"
__email__ = "kevin.yamauchi@gmail.com"

from trace_light import analysis, backends, fluorescence, optimize, viz
from trace_light.kernels import _trace_surfaces, trace
from trace_light.rays import (
    Rays,
    Surface,
    System,
    _Params,
    _Structure,
    load_system,
    save_system,
)
from trace_light.sources import Source, collimated_source, emit, point_source
from trace_light.systems import SystemBuilder

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
