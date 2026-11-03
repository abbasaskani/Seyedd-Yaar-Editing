"""Coordinate-aware resampling of Copernicus rasters into Seyd-Yaar's grid.

GridSpec longitude/latitude extrema describe *cell centres* in EPSG:4326.
The target row order is north-to-south, matching the exported f32 BINs.
GDAL/rasterio reads a NetCDF subdataset with its own affine transform; do
not use array-size-based resizing, which silently relocates coasts and islands.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

from .utils_geo import GridSpec


def target_transform(grid: GridSpec) -> Affine:
    """Pixel-edge geotransform for a grid whose min/max refer to centres."""
    if grid.width < 2 or grid.height < 2:
        raise ValueError("Grid dimensions must both be >=2")
    return from_bounds(
        grid.lon_min - grid.dx / 2,
        grid.lat_min - grid.dy / 2,
        grid.lon_max + grid.dx / 2,
        grid.lat_max + grid.dy / 2,
        grid.width,
        grid.height,
    )


def align_raster_dataset(ds: rasterio.io.DatasetReader, grid: GridSpec) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reproject one georeferenced band using nearest-neighbour to grid centres.

    NetCDF input may have ascending or descending latitude: GDAL's transform
    handles orientation. It must be georeferenced. Missing and out-of-coverage
    cells are NaN; ocean data are never extrapolated into land.
    """
    transform = ds.transform
    if transform is None or transform.almost_equals(Affine.identity()):
        raise ValueError(f"{ds.name}: missing source georeference (identity transform)")
    if ds.count != 1:
        raise ValueError(f"{ds.name}: expected one time/depth raster band, got {ds.count}; subset one timestamp/depth")
    crs = ds.crs
    # CMEMS subsets are geographic lon/lat. The GDAL NetCDF driver may leave
    # the EPSG code unset while still exposing a valid geographic transform.
    if crs is None:
        b = ds.bounds
        if not (-360 <= b.left <= 360 and -360 <= b.right <= 360 and
                -90 <= b.bottom <= 90 and -90 <= b.top <= 90):
            raise ValueError(f"{ds.name}: missing CRS, bounds do not resemble lon/lat: {b}")
        crs = rasterio.crs.CRS.from_epsg(4326)
    source = ds.read(1).astype(np.float32, copy=False)
    if ds.nodata is not None:
        source[source == np.float32(ds.nodata)] = np.nan
    source[~np.isfinite(source) | (np.abs(source) > np.float32(1e6))] = np.nan
    destination = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    reproject(
        source=source,
        destination=destination,
        src_transform=transform,
        src_crs=crs,
        src_nodata=np.nan,
        dst_transform=target_transform(grid),
        dst_crs="EPSG:4326",
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
        init_dest_nodata=True,
        num_threads=1,
    )
    valid = np.isfinite(destination)
    info = {
        "src_width": int(ds.width),
        "src_height": int(ds.height),
        "src_crs": str(crs),
        "src_transform": [float(v) for v in transform[:6]],
        "src_bounds": [float(ds.bounds.left), float(ds.bounds.bottom),
                       float(ds.bounds.right), float(ds.bounds.top)],
        "dst_valid_fraction": float(valid.mean()),
        "method": "georeferenced_nearest",
        "dst_grid": "EPSG:4326 north-first centres",
    }
    if not valid.any():
        raise ValueError(f"{ds.name}: no georeferenced pixels intersect the target grid")
    return destination, info


def read_netcdf_variable_on_grid(path: Path, var: str, grid: GridSpec) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Open one GDAL NetCDF subdataset, return (H,W) and provenance."""
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    with rasterio.open(f'NETCDF:"{path}":{var}') as ds:
        return align_raster_dataset(ds, grid)
