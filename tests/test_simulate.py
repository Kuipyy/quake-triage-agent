"""Unit tests for HOTOSM labelling and the simulated detector."""
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from quaketriage.buildings.simulate import (
    SimulatorConfig,
    building_metrics,
    label_destroyed,
    simulate_detector,
)
from quaketriage.config import get

CRS = "EPSG:32637"


def _grid(n=400, size=10.0, gap=5.0):
    geoms = [box(500000 + i * (size + gap), 4000000, 500000 + i * (size + gap) + size,
                 4000000 + size) for i in range(n)]
    return gpd.GeoDataFrame({"building": ["residential"] * n}, geometry=geoms, crs=CRS)


def test_label_destroyed_by_overlap():
    b = _grid(3)
    h = gpd.GeoDataFrame(geometry=[
        b.geometry.iloc[0].buffer(1),                                   # covers building 0
        box(500015 + 5, 4000000, 500015 + 10, 4000010),                 # half of building 1
        box(500030, 4000000, 500030 + 2, 4000010),                      # 20 % of building 2
    ], crs=CRS)
    assert label_destroyed(b, h, 0.5).tolist() == [True, True, False]
    assert label_destroyed(b, h, 0.1).tolist() == [True, True, True]


def test_label_destroyed_with_non_default_indices():
    b = _grid(3)
    b.index = [10, 20, 30]
    h = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), b.geometry.iloc[2].buffer(1)], crs=CRS)
    h.index = [7, 99]                      # e.g. after a bbox filter
    assert label_destroyed(b, h).tolist() == [False, False, True]


def test_simulator_rates_and_determinism():
    b = _grid(4000)
    truth = np.arange(len(b)) < 800                      # 20 % destroyed
    cfg = SimulatorConfig(miss_rate=0.25, false_alarm_rate=0.05, seed=1)
    s1, s2 = simulate_detector(b, truth, cfg), simulate_detector(b, truth, cfg)
    assert s1.damage_prob.equals(s2.damage_prob)
    m = building_metrics(s1)
    assert m["recall"] == pytest.approx(0.75, abs=0.05)
    assert m["fp"] / (len(b) - 800) == pytest.approx(0.05, abs=0.02)
    assert s1.uncertainty.between(0, 1).all()
    thr = get("thresholds.damage_prob")
    assert (s1.damage_prob[s1.damaged] > thr).all() and (s1.damage_prob[~s1.damaged] < thr).all()


def test_perfect_detector():
    b = _grid(500)
    truth = np.arange(len(b)) % 5 == 0
    s = simulate_detector(b, truth, SimulatorConfig(miss_rate=0, false_alarm_rate=0, seed=2))
    m = building_metrics(s)
    assert m["recall"] == 1.0 and m["precision"] == 1.0 and m["fp"] == 0


def test_size_dependent_misses_hurt_person_recall():
    rng = np.random.default_rng(0)
    sizes = np.where(np.arange(2000) % 2 == 0, 6.0, 20.0)         # half small, half large
    geoms = [box(500000 + i * 30, 4000000, 500000 + i * 30 + s, 4000000 + s)
             for i, s in enumerate(sizes)]
    b = gpd.GeoDataFrame({"building": ["residential"] * len(geoms)}, geometry=geoms, crs=CRS)
    b["estimated_persons"] = (sizes ** 2) / 10.0                   # people ~ floor area
    truth = rng.random(len(b)) < 0.3
    flat = building_metrics(simulate_detector(b, truth, SimulatorConfig(miss_rate=0.3, seed=3)))
    sized = building_metrics(simulate_detector(
        b, truth, SimulatorConfig(miss_rate=0.3, size_exponent=1.0, seed=3)))
    assert sized["person_recall"] > sized["recall"] > 0     # large (populous) buildings found more
    assert flat["person_recall"] == pytest.approx(flat["recall"], abs=0.08)


def test_metrics_without_persons_column():
    b = _grid(50)
    s = simulate_detector(b, np.zeros(50, bool), SimulatorConfig(false_alarm_rate=0.0))
    m = building_metrics(s)
    assert "person_recall" not in m and m["n_truth"] == 0 and np.isnan(m["recall"])


def test_assemble_demo_buildings():
    from quaketriage.buildings.simulate import assemble_demo_buildings

    osm = _grid(6)                                   # 10 m squares every 15 m along x
    osm.loc[0, "geometry"] = osm.geometry.iloc[0]     # overlaps a HOTOSM polygon (duplicate)
    hot = gpd.GeoDataFrame(
        geometry=[osm.geometry.iloc[0].buffer(0.5), box(500500, 4000000, 500510, 4000010)], crs=CRS)
    out, destroyed = assemble_demo_buildings(osm, hot, zone_buffer_m=20.0)
    # building 0 dropped as duplicate; building 1 (5 m away) kept; far buildings dropped;
    # the 2nd HOTOSM polygon is added as destroyed
    assert destroyed.sum() == 2 and len(out) == len(destroyed)
    assert out.crs.to_string() == CRS
    assert (~destroyed).sum() >= 1 and (out.geometry.iloc[~destroyed].centroid.x < 500040).all()
