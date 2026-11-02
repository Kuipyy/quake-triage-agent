"""Unit tests for per-building zonal statistics."""
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from quaketriage.buildings.zonal import ZonalInput, zonal_building_stats

CRS = "EPSG:32637"
X0, Y0, RES = 500000.0, 4000000.0, 1.0


def _write(path, arr):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs=CRS, transform=from_origin(X0, Y0, RES, RES),
                       nodata=float("nan")) as dst:
        dst.write(arr.astype("float32"), 1)


def _bld(*boxes):
    # boxes in pixel coords (col0, row0, col1, row1)
    geoms = [box(X0 + c0, Y0 - r1, X0 + c1, Y0 - r0) for c0, r0, c1, r1 in boxes]
    return gpd.GeoDataFrame({"id": range(len(geoms))}, geometry=geoms, crs=CRS)


def test_mean_fraction_and_flag(tmp_path):
    prob = np.zeros((20, 20))
    prob[2:6, 2:6] = 0.8          # building A fully damaged
    prob[10:14, 10:14] = 0.1      # building B undamaged
    unc = np.full((20, 20), 0.05)
    _write(tmp_path / "p.tif", prob)
    _write(tmp_path / "u.tif", unc)
    b = _bld((2, 2, 6, 6), (10, 10, 14, 14))
    r = zonal_building_stats(b, tmp_path / "p.tif", tmp_path / "u.tif", threshold=0.5)
    assert r.damage_prob.tolist() == pytest.approx([0.8, 0.1])
    assert r.damage_frac.tolist() == [1.0, 0.0]
    assert r.damaged.tolist() == [True, False]
    assert r.uncertainty.tolist() == pytest.approx([0.05, 0.05])
    assert not r.no_valid_pixels.any()


def test_nan_pixels_ignored_and_flagged(tmp_path):
    prob = np.full((20, 20), 0.6)
    prob[2:6, 2:4] = np.nan                  # half of building A is NaN (guard-rejected)
    prob[10:14, 10:14] = np.nan              # building B entirely NaN
    _write(tmp_path / "p.tif", prob)
    b = _bld((2, 2, 6, 6), (10, 10, 14, 14))
    r = zonal_building_stats(b, tmp_path / "p.tif", threshold=0.5, min_valid_fraction=0.5)
    assert r.valid_fraction.tolist() == pytest.approx([0.5, 0.0])
    assert r.damage_prob[0] == pytest.approx(0.6) and r.damaged[0]
    assert np.isnan(r.damage_prob[1]) and r.no_valid_pixels[1] and not r.damaged[1]
    strict = zonal_building_stats(b, tmp_path / "p.tif", threshold=0.5, min_valid_fraction=0.75)
    assert strict.no_valid_pixels.tolist() == [True, True]


def test_reprojects_and_keeps_order_and_crs(tmp_path):
    prob = np.zeros((20, 20))
    prob[2:6, 2:6] = 0.9
    _write(tmp_path / "p.tif", prob)
    b = _bld((10, 10, 14, 14), (2, 2, 6, 6)).to_crs("EPSG:4326")
    r = zonal_building_stats(b, tmp_path / "p.tif", threshold=0.5)
    assert r.crs == b.crs
    assert r.damaged.tolist() == [False, True]


def test_building_outside_raster_is_flagged(tmp_path):
    _write(tmp_path / "p.tif", np.zeros((10, 10)))
    b = _bld((100, 100, 104, 104))
    assert zonal_building_stats(b, tmp_path / "p.tif").no_valid_pixels.tolist() == [True]


def test_uncertainty_grid_mismatch_raises(tmp_path):
    _write(tmp_path / "p.tif", np.zeros((10, 10)))
    _write(tmp_path / "u.tif", np.zeros((12, 12)))
    with pytest.raises(ValueError):
        zonal_building_stats(_bld((1, 1, 3, 3)), tmp_path / "p.tif", tmp_path / "u.tif")


def test_input_schema_defaults():
    inp = ZonalInput(buildings_path="a.gpkg", prob_raster="p.tif")
    assert 0 <= inp.threshold <= 1 and inp.uncertainty_raster is None
