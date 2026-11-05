import numpy as np
import pytest

import main


def make(mod, **params):
    sim = mod.GuidanceSimulation()
    for name, value in params.items():
        setattr(sim, name, value)
    return sim


# Базовый случай лабораторной (вариант 1): ПУ летит прямолинейно, цель неподвижна.
BASE = dict(z_0_0=100.0, x_0_0=500.0, V_0_0=250.0, V_1_0=500.0, z_2_0=5100.0, x_2_0=0.0, V_2_0=0.0,
            K_m1=2000.0, K_m2=10.0, q31=1000.0, q32=1000.0, q33=1000.0)


@pytest.mark.parametrize("method", [1, 2, 3])
@pytest.mark.parametrize("inerc", [False, True])
def test_base_case_identical_to_original(orig, method, inerc):
    """Прежние результаты базового случая должны сохраниться бит в бит."""
    params = dict(BASE, sw_method=method, sw_inerc=inerc, T_j=5.0)
    r_old = make(orig, **params).run_simulation()
    r_new = make(main, **params).run_simulation()
    for key, old in r_old.items():
        if isinstance(old, np.ndarray):
            assert np.array_equal(old, r_new[key]), key
    assert r_old["K_stop"] == r_new["K_stop"]
    assert r_old["flag_break"] == r_new["flag_break"]


def test_default_settings_identical_to_original(orig):
    """Без явного выбора режимов модель ведёт себя как раньше."""
    r_old = make(orig).run_simulation()
    r_new = make(main).run_simulation()
    assert np.array_equal(r_old["z_1"], r_new["z_1"])
    assert np.array_equal(r_old["wv_2"], r_new["wv_2"])


def _numeric_rate(beta, T):
    return np.gradient(np.unwrap(beta), T)


@pytest.mark.parametrize("v0, pu_mode, v2, psi2, target_mode", [
    (0.0, 0, 200.0, 0.0, 1),      # ПУ стоит, цель движется прямолинейно
    (250.0, 1, 200.0, 180.0, 1),  # оба движутся прямолинейно
    (250.0, 1, 0.0, 90.0, 0),     # ПУ движется, цель стоит
    (250.0, 2, 150.0, 45.0, 2),   # оба маневрируют
])
def test_wv2_matches_numeric_derivative(v0, pu_mode, v2, psi2, target_mode):
    sim = make(main, **dict(BASE, V_0_0=v0, V_2_0=v2, psi_2_deg_0=psi2, sw_method=1,
                            sw_dvizh_pu=pu_mode, sw_dvizh_celi=target_mode,
                            j_0ppr_m1=-20.0, j_2ppr_m1=-20.0))
    r = sim.run_simulation()
    k = int(r["K_stop"]) - 1
    num = _numeric_rate(r["beta_2"][:k], sim.T)
    idx = slice(5, k - 5)
    # Допуск 1e-3 рад/с (0.06 град/с) - погрешность численного дифференцирования;
    # ошибка старой формулы в этих случаях - порядка 0.04 рад/с.
    assert np.allclose(r["wv_2"][idx], num[idx], atol=1e-3)


def test_pu_stationary_stays_in_place():
    sim = make(main, **dict(BASE, sw_method=1, sw_dvizh_pu=0, sw_dvizh_celi=1, V_2_0=200.0, psi_2_deg_0=0.0))
    r = sim.run_simulation()
    assert np.all(r["z_0"] == BASE["z_0_0"])
    assert np.all(r["x_0"] == BASE["x_0_0"])
    assert np.all(r["V_0"] == 0.0)


@pytest.mark.parametrize("pu_mode", [2, 3])
def test_pu_stationary_ignores_maneuver_when_v0_zero(pu_mode):
    sim = make(main, **dict(BASE, sw_method=1, sw_dvizh_pu=pu_mode, V_0_0=0.0))
    r = sim.run_simulation()
    assert np.all(np.isfinite(r["z_0"])) and np.all(r["z_0"] == BASE["z_0_0"])


def test_target_stationary_forces_zero_speed():
    sim = make(main, **dict(BASE, sw_method=1, sw_dvizh_celi=0, V_2_0=300.0))
    r = sim.run_simulation()
    assert np.all(r["z_2"] == BASE["z_2_0"])
    assert np.all(r["x_2"] == BASE["x_2_0"])


def _circle_radius(p1, p2, p3):
    a = np.linalg.norm(p2 - p3); b = np.linalg.norm(p1 - p3); c = np.linalg.norm(p1 - p2)
    area2 = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    return a * b * c / (2 * area2)


def test_pu_constant_turn_radius():
    j = -20.0
    sim = make(main, **dict(BASE, sw_method=1, sw_dvizh_pu=2, j_0ppr_m1=j, sw_stop=False, tk=30.0))
    r = sim.run_simulation()
    pts = np.column_stack([r["z_0"], r["x_0"]])
    n = len(pts) - 1
    radius = _circle_radius(pts[0], pts[n // 2], pts[n])
    assert radius == pytest.approx(BASE["V_0_0"] ** 2 / abs(j), rel=0.01)


def test_pu_sinusoidal_matches_target_law():
    """Синусоидальный манёвр ПУ рассчитывается по тому же закону, что у цели."""
    sim = make(main, **dict(BASE, sw_method=1, sw_dvizh_pu=3, sw_dvizh_celi=3, V_2_0=200.0))
    r = sim.run_simulation()
    assert np.allclose(r["j_0ppr"][1:] / sim.j_0ppr_m2, r["j_2ppr"][1:] / sim.j_2ppr_m2)


@pytest.mark.parametrize("method", [1, 2, 3])
@pytest.mark.parametrize("v0, v2", [(0.0, 50.0), (250.0, 50.0), (100.0, 300.0)])
def test_small_range_runs(method, v0, v2):
    sim = make(main, **dict(BASE, z_2_0=600.0, x_2_0=0.0, V_0_0=v0, V_2_0=v2, psi_2_deg_0=270.0,
                            sw_method=method, sw_dvizh_pu=0 if v0 == 0 else 1, sw_dvizh_celi=1))
    r = sim.run_simulation()
    assert r["flag_stop"]
    assert int(r["K_stop"]) < len(r["t"])
    assert np.all(np.isfinite(r["hv"][: int(r["K_stop"])]))
