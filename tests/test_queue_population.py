"""Unit tests for the verification queue and population distribution."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from quaketriage.buildings.population import estimate_floors, estimate_population
from quaketriage.buildings.queue import flag_verification


def test_queue_rules_and_reasons():
    g = gpd.GeoDataFrame({
        "uncertainty": [0.05, 0.5, np.nan, np.nan, 0.05],
        "no_valid_pixels": [False, False, False, True, False],
        "guard_warning": [False, False, False, False, True],
    }, geometry=[box(0, 0, 1, 1)] * 5, crs="EPSG:32637")
    r = flag_verification(g, tau=0.2)
    assert r.needs_verification.tolist() == [False, True, True, True, True]
    assert r.verification_reason.tolist() == [
        "", "uncertainty > 0.2", "uncertainty unknown", "no valid imagery",
        "image-quality warning"]


def test_queue_without_optional_columns_flags_everything_unknown():
    g = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:32637")
    assert flag_verification(g).needs_verification.tolist() == [True]


def test_floors_levels_then_type_then_default():
    g = pd.DataFrame({"building": ["apartments", "house", "weird", "yes"],
                      "building:levels": ["7", None, None, "abc"]})
    assert estimate_floors(g).tolist() == [7.0, 2.0, 3.0, 3.0]
    assert estimate_floors(pd.DataFrame({"building": ["apartments"]})).tolist() == [5.0]


def _worldpop(path, pop_a, pop_b):
    """Two 0.001-degree cells side by side near Antakya (lon 36.20, lat 36.20)."""
    arr = np.array([[pop_a, pop_b]], dtype="float32")
    with rasterio.open(path, "w", driver="GTiff", height=1, width=2, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(36.200, 36.2010, 0.001, 0.001),
                       nodata=-99999.0) as dst:
        dst.write(arr, 1)


def _blds():
    # three buildings: two in the left cell (areas 1:3), one in the right cell
    def ll(lon, lat, d=0.00005):
        return box(lon - d, lat - d, lon + d, lat + d)

    geoms = [ll(36.2003, 36.2005), ll(36.2006, 36.2005, 0.0000866), ll(36.2015, 36.2005)]
    return gpd.GeoDataFrame({"building": ["residential", "residential", "school"]},
                            geometry=geoms, crs="EPSG:4326")


def test_population_is_conserved_and_split_by_floor_area(tmp_path):
    _worldpop(tmp_path / "wp.tif", 100.0, 40.0)
    r = estimate_population(_blds(), tmp_path / "wp.tif")
    left, right = r.residents_day.iloc[:2], r.residents_day.iloc[2]
    assert left.sum() == pytest.approx(100.0)       # whole cell assigned
    assert right == pytest.approx(40.0)
    ratio = r.floor_area_m2.iloc[1] / r.floor_area_m2.iloc[0]
    assert left.iloc[1] / left.iloc[0] == pytest.approx(ratio)
    # night factors: residential 0.9, school 0.1
    assert r.estimated_persons.iloc[0] == pytest.approx(r.residents_day.iloc[0] * 0.9)
    assert r.estimated_persons.iloc[2] == pytest.approx(40.0 * 0.1)


def test_building_outside_raster_gets_zero(tmp_path):
    _worldpop(tmp_path / "wp.tif", 100.0, 40.0)
    far = gpd.GeoDataFrame({"building": ["yes"]}, geometry=[box(36.5, 36.5, 36.5001, 36.5001)],
                           crs="EPSG:4326")
    both = pd.concat([_blds(), far], ignore_index=True)
    r = estimate_population(gpd.GeoDataFrame(both, crs="EPSG:4326"), tmp_path / "wp.tif")
    assert r.residents_day.iloc[-1] == 0.0
