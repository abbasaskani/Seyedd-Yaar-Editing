"""No-network regression tests for selecting Copernicus surface depth."""
from __future__ import annotations

import json
import sys
import types
import numpy as np
import pytest

from seydyaar.pipeline import run_daily as pipeline
from seydyaar.utils_geo import GridSpec


def test_parse_depth_range_is_strict_about_dimension():
    err = RuntimeError(
        "Some of your subset selection [0.0, 0.0] for the depth dimension "
        "exceed the dataset coordinates [0.49402499198913574, 5727.9169921875]"
    )
    assert pipeline._surface_depth_from_error(err, 0.0) == pytest.approx(0.49402499198913574)
    assert pipeline._surface_depth_from_error(err, 100.0) == pytest.approx(100.0)
    assert pipeline._surface_depth_from_error(err, 9000.0) == pytest.approx(5727.9169921875)
    assert pipeline._surface_depth_from_error(RuntimeError("connection failure"), 0.0) is None
    assert pipeline._surface_depth_from_error(
        RuntimeError("time dimension exceed the dataset coordinates [0.494, 99.0]"), 0.0
    ) is None


def test_catalogue_parser_ignores_latitudes_and_times():
    doc = {"products": [{"datasets": [{"versions": [{"parts": [{"services": [{
        "variables": [{"short_name": "thetao", "coordinates": [
            {"coordinate_id": "latitude", "minimum_value": -80, "maximum_value": 90},
            {"coordinate_id": "depth", "minimum_value": 0.49402499198913574,
             "maximum_value": 5727.9169921875},
            {"coordinate_id": "longitude", "minimum_value": -180, "maximum_value": 180},
        ]}]
    }]}]}]}]}]}
    assert pipeline._depth_minima_from_catalogue(doc) == pytest.approx([0.49402499198913574])


def test_catalogue_cli_selects_dataset_and_returns_nearest_surface(monkeypatch):
    data = {"products": [{"coordinates": [{"coordinate_id": "depth", "minimum_value": 0.494}]}]}
    commands = []
    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return types.SimpleNamespace(stdout=json.dumps(data))
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    resolver = pipeline._DepthResolver()
    assert resolver.closest_depth("demo", 0.0) == pytest.approx(0.494)
    assert resolver.closest_depth("demo", 10.0) == pytest.approx(10.0)
    assert len(commands) == 1
    assert "--dataset-id" in commands[0]
    assert "-c" not in commands[0]


def test_copernicus_retries_depth_from_server_error(tmp_path, monkeypatch):
    calls = []
    depth_min = 0.49402499198913574
    def subset(*, dataset_id, output_filename, minimum_depth=None, maximum_depth=None, **kwargs):
        calls.append((dataset_id, minimum_depth, maximum_depth))
        if minimum_depth == 0.0:
            raise RuntimeError(
                "Some of your subset selection [0.0, 0.0] for the depth dimension "
                f"exceed the dataset coordinates [{depth_min}, 5727.9169921875]"
            )
        if dataset_id in ("height", "wave"):
            assert minimum_depth is None and maximum_depth is None
        else:
            assert minimum_depth == pytest.approx(depth_min)
            assert maximum_depth == pytest.approx(depth_min)
        from pathlib import Path
        Path(output_filename).write_bytes(b"mock-subset")

    monkeypatch.setenv("COPERNICUS_MARINE_USERNAME", "demo")
    monkeypatch.setenv("COPERNICUS_MARINE_PASSWORD", "demo")
    monkeypatch.setenv("SEYDYAAR_TMPDIR", str(tmp_path / "nc"))
    monkeypatch.setenv("SEYDYAAR_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setitem(sys.modules, "copernicusmarine", types.SimpleNamespace(subset=subset))
    monkeypatch.setattr(pipeline._DepthResolver, "closest_depth", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "read_netcdf_variable_on_grid",
                        lambda p, v, g: (np.ones((g.height,g.width), np.float32), {"mock": True}))
    cfg = {
        "sst": {"dataset_id":"temp", "variable":"thetao", "depth_target_m":0.0},
        "chl": {"dataset_id":"chlor", "variable":"chl", "depth_target_m":0.0},
        "ssh": {"dataset_id":"height", "variable":"zos"},
        "currents": {"dataset_id":"current", "variables":["uo","vo"], "depth_target_m":0.0},
        "waves": {"dataset_id":"wave", "variable":"VHM0"},
    }
    grid=GridSpec(lon_min=53,lon_max=55,lat_min=12,lat_max=14,width=9,height=7)
    out, status=pipeline._try_copernicus_layers(grid,(53,12,55,14),"2026-11-01T00:00:00Z",cfg)
    assert status["ok"], status["errors"]
    assert out["sst_c"].shape == (7,9)
    # Three depth-bearing datasets: first try zero, retry with valid depth.
    assert len([c for c in calls if c[1] == 0.0]) == 3
    assert len([c for c in calls if c[1] == depth_min]) == 3
    assert len([c for c in calls if c[1] is None]) == 2
    records=[json.loads(line) for line in (tmp_path/"logs"/"download_manifest.jsonl").read_text().splitlines()]
    assert len(records) == 5
    assert all(r["ok"] for r in records)
    assert all(r["depth_retry"]["corrected_depth_m"] == depth_min
               for r in records if r["layer"] in ("sst","chl","currents"))
