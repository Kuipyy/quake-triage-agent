import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from quaketriage.planning.topsis import (
    TopsisInput,
    ahp_weights,
    prioritise,
    rank_table,
    topsis_scores,
)


def test_topsis_dominant_alternative_wins():
    m = np.array([[1.0, 1.0], [5.0, 5.0], [3.0, 3.0]])
    s = topsis_scores(m, np.array([0.5, 0.5]), np.array([True, True]))
    assert s.argmax() == 1 and s.argmin() == 0
    assert np.isclose(s[1], 1.0) and np.isclose(s[0], 0.0)


def test_topsis_cost_criterion_reverses():
    m = np.array([[1.0], [5.0]])
    s = topsis_scores(m, np.array([1.0]), np.array([False]))
    assert s[0] > s[1]


def test_topsis_constant_column_is_neutral():
    s = topsis_scores(np.ones((3, 2)), np.array([1, 1]), np.array([True, True]))
    assert np.allclose(s, 0.5)


def test_topsis_rejects_bad_input():
    with pytest.raises(ValueError):
        topsis_scores(np.ones((2, 2)), np.array([1.0]), np.array([True]))
    with pytest.raises(ValueError):
        topsis_scores(np.array([[np.nan, 1.0]]), np.array([1, 1]), np.array([True, True]))
    with pytest.raises(ValueError):
        topsis_scores(np.ones((2, 1)), np.array([0.0]), np.array([True]))


def test_ahp_consistent_matrix():
    # criterion 0 is 3x criterion 1 and 9x criterion 2 -> perfectly consistent
    a = [[1, 3, 9], [1 / 3, 1, 3], [1 / 9, 1 / 3, 1]]
    w, cr = ahp_weights(a)
    assert np.isclose(w.sum(), 1) and np.allclose(w, [9 / 13, 3 / 13, 1 / 13], atol=1e-6)
    assert cr < 1e-6


def test_ahp_rejects_non_reciprocal_and_flags_inconsistent():
    with pytest.raises(ValueError):
        ahp_weights([[1, 2], [2, 1]])
    _, cr = ahp_weights([[1, 9, 1 / 9], [1 / 9, 1, 9], [9, 1 / 9, 1]])  # cyclic preferences
    assert cr > 0.1


def _buildings():
    # ~UTM-agnostic: points near the demo area; B is nearest to the depot
    pts = [Point(36.160, 36.200), Point(36.161, 36.200), Point(36.190, 36.200)]
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3], "damage_prob": [0.9, 0.9, 0.9], "estimated_persons": [10, 10, 10],
         "floors": [4, 4, 4]}, geometry=pts, crs=4326)


def test_prioritise_prefers_closer_building_when_otherwise_equal():
    out = prioritise(_buildings(), TopsisInput(depots=[(36.1605, 36.200)]))
    assert out["priority"].iloc[2] < out["priority"].iloc[0]
    assert (out["distance_to_depot_m"] > 0).all()
    assert rank_table(out)["id"].iloc[-1] == 3


def test_prioritise_weights_override_and_ahp():
    b = _buildings()
    b["estimated_persons"] = [1, 1, 100]
    only_people = {"damage_prob": 0, "estimated_persons": 1, "floor_count": 0,
                   "distance_to_depot": 0}
    out = prioritise(b, TopsisInput(depots=[(36.16, 36.2)], weights=only_people))
    assert out["priority"].idxmax() == 2
    with pytest.raises(ValueError):
        prioritise(b, TopsisInput(depots=[(36.16, 36.2)], weights={"bogus": 1}))
    with pytest.raises(ValueError):
        prioritise(b, TopsisInput(depots=[(36.16, 36.2)], ahp_pairwise=[[1, 2], [0.5, 1]]))


def test_depot_validation():
    with pytest.raises(ValueError):
        TopsisInput(depots=[(95.0, 200.0)])


def _crowded_intact_building():
    pts = [Point(36.160, 36.200), Point(36.161, 36.200), Point(36.162, 36.200)]
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3], "damage_prob": [0.9, 0.7, 0.05],
         "estimated_persons": [10, 10, 500], "floors": [4, 4, 4]}, geometry=pts, crs=4326)


def test_damage_gate_keeps_intact_crowded_building_out_of_the_ranking():
    out = prioritise(_crowded_intact_building(), TopsisInput(depots=[(36.160, 36.200)]))
    assert out["candidate"].tolist() == [True, True, False]
    assert out.loc[2, "priority"] == 0.0
    # TOPSIS runs among the two candidates only: the dominant one gets 1.0, the other 0.0
    assert out.loc[0, "priority"] == 1.0 and out.loc[1, "priority"] == 0.0


def test_gate_can_be_disabled_and_then_crowd_wins():
    out = prioritise(_crowded_intact_building(),
                     TopsisInput(depots=[(36.160, 36.200)], damage_gate=None))
    assert out["candidate"].all()
    assert out["priority"].idxmax() == 2  # the crowded intact building tops the list again


def test_gate_with_no_candidates_gives_all_zero_priorities():
    b = _crowded_intact_building()
    b["damage_prob"] = [0.01, float("nan"), 0.05]
    out = prioritise(b, TopsisInput(depots=[(36.160, 36.200)]))
    assert not out["candidate"].any() and (out["priority"] == 0.0).all()
