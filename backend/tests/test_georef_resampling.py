"""Offline checks with known georeferencing and a synthetic land/no-data island.

Run from repository root: PYTHONPATH=backend pytest -q backend/tests
"""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds, from_origin

from seydyaar.utils_geo import GridSpec
from seydyaar.utils_resample import align_raster_dataset, target_transform


@pytest.fixture
def grid():
    return GridSpec(lon_min=53.5, lon_max=54.5, lat_min=12, lat_max=13,
                    width=11, height=11)


def test_destination_pixel_centres_match_grid(grid):
    tr = target_transform(grid)
    assert np.isclose(tr * (0.5, 0.5), (grid.lon_min, grid.lat_max)).all()
    assert np.isclose(tr * (10.5, 10.5), (grid.lon_max, grid.lat_min)).all()


def test_resampling_references_actual_lon_lat_not_array_position(grid):
    # Native grid is wider than requested coverage and has a different resolution.
    native = np.full((20, 18), 9.0, dtype=np.float32)
    tr = from_bounds(53.0, 11.5, 55.0, 13.5, 18, 20)
    # Make a small island of invalid cells around true (54E,12.5N).
    for y in range(native.shape[0]):
        for x in range(native.shape[1]):
            lon, lat = tr * (x + 0.5, y + 0.5)
            if abs(lon-54)<0.16 and abs(lat-12.5)<0.16:
                native[y,x] = -9999
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=20, width=18, count=1,
                      dtype="float32", crs="EPSG:4326", transform=tr,
                      nodata=-9999) as wr:
            wr.write(native,1)
        with mem.open() as src:
            dest, provenance = align_raster_dataset(src, grid)
    assert dest.shape == (11,11)
    assert np.isnan(dest[5,5]), 'island should stay at true location'
    assert dest[0,0] == 9.0
    assert dest[-1,-1] == 9.0
    assert 0 < provenance['dst_valid_fraction'] < 1
    assert provenance['method'] == 'georeferenced_nearest'


def test_source_latitude_orientation_south_up(grid):
    # Increasing latitude by row must be recognized from geotransform.
    src = np.arange(121, dtype=np.float32).reshape((11,11))
    tr = from_origin(grid.lon_min-grid.dx/2, grid.lat_min-grid.dy/2,
                     grid.dx, -grid.dy)
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=11, width=11, count=1,
                      dtype="float32", crs="EPSG:4326", transform=tr) as wr:
            wr.write(src,1)
        with mem.open() as ds:
            dest, _ = align_raster_dataset(ds, grid)
    assert np.array_equal(dest, src[::-1,:]), 'south-up source should be flipped into north-first output'


def test_missing_georeference_rejected(grid):
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=3, width=3, count=1,
                      dtype="float32") as wr:
            wr.write(np.ones((3,3),dtype=np.float32),1)
        with mem.open() as ds:
            with pytest.raises(ValueError, match='georeference'):
                align_raster_dataset(ds, grid)
