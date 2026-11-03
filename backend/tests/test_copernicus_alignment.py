"""Offline end-to-end ingestion test with mock Copernicus NetCDF files."""
from __future__ import annotations

import sys
import types
import numpy as np
import pytest

from seydyaar.utils_geo import GridSpec
from seydyaar.pipeline import run_daily as pipeline


def test_copernicus_path_uses_coordinate_aligned_inputs(tmp_path, monkeypatch):
    scipy = pytest.importorskip("scipy.io", reason="optional test fixture writer")
    from scipy.io import netcdf_file

    def subset(*, dataset_id, variables, output_filename, **kwargs):
        # Different products have DIFFERENT native resolution, extents and
        # latitude order, yet the missing-data island must stay in one place.
        is_chl = "chl" in variables
        H,W = (9,7) if is_chl else (17,13)
        lats=np.linspace(11.65,13.35,H,dtype=np.float32) if is_chl else np.linspace(13.35,11.65,H,dtype=np.float32)
        lons=np.linspace(53.3,54.7,W,dtype=np.float32)
        with netcdf_file(output_filename, "w") as f:
            f.createDimension("latitude",H)
            f.createDimension("longitude",W)
            lat=f.createVariable("latitude","f4",("latitude",)); lat[:]=lats; lat.units="degrees_north"
            lon=f.createVariable("longitude","f4",("longitude",)); lon[:]=lons; lon.units="degrees_east"
            for name in variables:
                v=f.createVariable(name,"f4",("latitude","longitude"))
                data=np.full((H,W), .8 if name=="chl" else 1.1, dtype=np.float32)
                if name=="thetao": data[:]=26.0
                if name=="zos": data[:]=0.2
                if name=="VHM0": data[:]=0.6
                if name in ("uo","vo"): data[:]=0.05
                for y in range(H):
                    for x in range(W):
                        if abs(lats[y]-12.5)<.12 and abs(lons[x]-54)<.12:
                            data[y,x]=-9999.0
                v._FillValue=np.float32(-9999.)
                v[:]=data

    monkeypatch.setenv("COPERNICUS_MARINE_USERNAME","test-username")
    monkeypatch.setenv("COPERNICUS_MARINE_PASSWORD","test-password")
    monkeypatch.setenv("SEYDYAAR_TMPDIR",str(tmp_path/"tmp"))
    monkeypatch.setenv("SEYDYAAR_LOG_DIR",str(tmp_path/"logs"))
    monkeypatch.setitem(sys.modules,"copernicusmarine",types.SimpleNamespace(subset=subset))
    monkeypatch.setattr(pipeline._DepthResolver,"closest_depth",lambda *a,**kw: None)
    grid=GridSpec(lon_min=53.5,lon_max=54.5,lat_min=12,lat_max=13,width=11,height=11)
    cfg={
        "sst":{"dataset_id":"dummy_sst","variable":"thetao"},
        "chl":{"dataset_id":"dummy_chl","variable":"chl"},
        "ssh":{"dataset_id":"dummy_ssh","variable":"zos"},
        "currents":{"dataset_id":"dummy_cur","variables":["uo","vo"]},
        "waves":{"dataset_id":"dummy_waves","variable":"VHM0"},
    }
    out,status=pipeline._try_copernicus_layers(grid,(53.5,12,54.5,13),"2026-11-03T00:00:00Z",cfg)
    assert status["ok"], status["errors"]
    assert out["sst_c"].shape == (11,11)
    assert out["chl_mg_m3"].shape == (11,11)
    assert np.isfinite(out["sst_c"][0,0])
    assert out["qc_chl"][5,5] == 0
    assert out["conf"][5,5] == 0
    assert out["qc_chl"][0,0] == 1
    assert status["reprojection"]["chl:chl"]["src_height"] == 9
    assert status["reprojection"]["sst:thetao"]["src_height"] == 17
