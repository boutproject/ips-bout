import sys
import types

import numpy as np


def _install_fake_hypnotoad(monkeypatch, *, intersections_plan):
    """
    Install a minimal fake `hypnotoad` module tree.

    `intersections_plan` is a callable:
        (wall_rz, p_start, p_end) -> None or ndarray[(n, 2)]
    """

    class Point2D:
        def __init__(self, r, z):
            self.r = float(r)
            self.z = float(z)

    def calc_distance(p_a, p_b):
        return float(np.hypot(p_a.r - p_b.r, p_a.z - p_b.z))

    def find_intersections(wall_rz, p_start, p_end):
        return intersections_plan(wall_rz, p_start, p_end)

    hypnotoad = types.ModuleType("hypnotoad")
    hypnotoad.Point2D = Point2D

    hypnotoad_core = types.ModuleType("hypnotoad.core")
    hypnotoad_core_equilibrium = types.ModuleType("hypnotoad.core.equilibrium")
    hypnotoad_core_equilibrium.find_intersections = find_intersections
    hypnotoad_core_equilibrium.calc_distance = calc_distance

    monkeypatch.setitem(sys.modules, "hypnotoad", hypnotoad)
    monkeypatch.setitem(sys.modules, "hypnotoad.core", hypnotoad_core)
    monkeypatch.setitem(
        sys.modules, "hypnotoad.core.equilibrium", hypnotoad_core_equilibrium
    )


def test_imas_coils_maps_supply_to_coils_and_applies_turns():
    from ipsbout.linear_mesh_generator import imas_coils

    pf_active = {
        "supply": [{"current": {"data": 10.0}}, {"current": {"data": -3.0}}],
        "coil": [
            {
                "element": [
                    {
                        "geometry": {"rectangle": {"r": 0.5, "z": 1.0}},
                        "turns_with_sign": 2.0,
                    }
                ]
            },
            {
                "element": [
                    {
                        "geometry": {"rectangle": {"r": 0.6, "z": 1.2}},
                        "turns_with_sign": -4.0,
                    }
                ]
            },
        ],
        "circuit": [
            {"supply_index": 0, "coil_indices": [0]},
            {"supply_index": 1, "coil_indices": [1]},
        ],
    }

    coils = imas_coils(pf_active)

    np.testing.assert_allclose(coils["r"], np.array([0.5, 0.6]))
    np.testing.assert_allclose(coils["z"], np.array([1.0, 1.2]))
    # current = supply_current * turns_with_sign
    np.testing.assert_allclose(coils["current"], np.array([20.0, 12.0]))


def test_calc_penalty_mask_all_inside_is_zero(monkeypatch):
    from ipsbout.linear_mesh_generator import calc_penalty_mask

    def intersections_plan(_wall_rz, _p_start, _p_end):
        return None

    _install_fake_hypnotoad(monkeypatch, intersections_plan=intersections_plan)

    grid_data = {
        "Rxy": np.zeros((1, 1)),
        "Rxy_ylow": np.array([[0.0, 0.0]]),
        "Zxy_ylow": np.array([[0.0, 1.0]]),
    }
    wall_rz = np.array([[0.0, 0.0], [1.0, 0.0]])

    mask = calc_penalty_mask(grid_data, wall_rz)
    np.testing.assert_allclose(mask, np.array([[0.0]]))


def test_calc_penalty_mask_all_outside_is_one(monkeypatch):
    from ipsbout.linear_mesh_generator import calc_penalty_mask

    def intersections_plan(_wall_rz, p_start, p_end):
        # Ray-casting test uses p0->p1 and p0->p2. Return an odd number
        # of intersections for both to classify the vertex as outside.
        if (p_start.r, p_start.z) == (0.01, 1.0):
            return np.array([[p_end.r, p_end.z]])
        return None

    _install_fake_hypnotoad(monkeypatch, intersections_plan=intersections_plan)

    grid_data = {
        "Rxy": np.zeros((1, 1)),
        "Rxy_ylow": np.array([[0.0, 0.0]]),
        "Zxy_ylow": np.array([[0.0, 1.0]]),
    }
    wall_rz = np.array([[0.0, 0.0], [1.0, 0.0]])

    mask = calc_penalty_mask(grid_data, wall_rz)
    np.testing.assert_allclose(mask, np.array([[1.0]]))


def test_calc_penalty_mask_crossing_wall_is_fraction(monkeypatch):
    from ipsbout.linear_mesh_generator import calc_penalty_mask

    p0_key = (0.01, 1.0)

    def intersections_plan(_wall_rz, p_start, p_end):
        if (p_start.r, p_start.z) == p0_key:
            # Classify p1 as outside (odd count) and p2 as inside (None)
            if (p_end.r, p_end.z) == (0.0, 0.0):
                return np.array([[0.0, 0.0]])
            if (p_end.r, p_end.z) == (0.0, 2.0):
                return None
        # For the p1->p2 segment, return a single intersection at the midpoint.
        if (p_start.r, p_start.z) == (0.0, 0.0) and (p_end.r, p_end.z) == (0.0, 2.0):
            return np.array([[0.0, 1.0]])
        if (p_start.r, p_start.z) == (0.0, 2.0) and (p_end.r, p_end.z) == (0.0, 0.0):
            return np.array([[0.0, 1.0]])
        return None

    _install_fake_hypnotoad(monkeypatch, intersections_plan=intersections_plan)

    grid_data = {
        "Rxy": np.zeros((1, 1)),
        "Rxy_ylow": np.array([[0.0, 0.0]]),
        "Zxy_ylow": np.array([[0.0, 2.0]]),
    }
    wall_rz = np.array([[0.0, 0.0], [1.0, 0.0]])

    mask = calc_penalty_mask(grid_data, wall_rz)
    np.testing.assert_allclose(mask, np.array([[0.5]]))
