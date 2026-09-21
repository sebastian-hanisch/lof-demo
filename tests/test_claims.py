"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Datensätze belegt (Mittel; Toleranz ±0.02 = Rundung auf zwei Stellen plus Luft).
Positive UND negative Aussagen: wo LOF gegen den Isolation Forest oder die Wurzel verliert, steht das hier ebenso als Test wie dort, wo er gewinnt."""

from functools import lru_cache

import numpy as np
import pytest

import lof_constants as C
import lof_evaluation as ev

TOL = 0.02
STANDARD = ev.Settings()


@lru_cache(maxsize=None)
def _runs(items, settings):
    return tuple(ev._analyse_seed(s, settings, dict(items)) for s in C.SWEEP_SEEDS)


def runs(settings=STANDARD, **kw):
    return _runs(tuple(sorted(kw.items())), settings)


def m(det, key, settings=STANDARD, **kw):
    return float(np.nanmean([a.scores[det][key] for a in runs(settings, **kw)]))


def oracle(det, settings=STANDARD, **kw):
    return float(np.mean([a.oracle_f1[det] for a in runs(settings, **kw)]))


def normal_lof_median(settings=STANDARD, **kw):
    return float(np.mean([np.median(a.values["lof"][~a.ds.anomaly]) for a in runs(settings, **kw)]))


def near(value, expected, tol=TOL):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


def kn(n):
    """k folgt der Tourenzahl (höchstens n / 2), wie im Sweep und in der Dimensionstabelle."""
    return ev.Settings(k=min(20, ev.k_max(n)))


# --- Seitenleiste: Touren und Merkmale ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,lof_f1,if_f1,lof_fa,if_fa", [(20, 0.96, 0.60, 0.011, 0.156), (30, 0.94, 0.67, 0.015, 0.119), (50, 0.88, 0.78, 0.027, 0.062), (100, 0.86, 0.89, 0.036, 0.027), (200, 0.91, 0.96, 0.020, 0.009)])
def test_tours_sweep_lof_is_calibrated_at_small_samples_the_isolation_forest_is_not(n, lof_f1, if_f1, lof_fa, if_fa):
    s = kn(n)
    near(m("lof", "f1", s, n=n), lof_f1)
    near(m("iforest", "f1", s, n=n), if_f1)
    near(m("lof", "false_alarm", s, n=n), lof_fa, 0.01)
    near(m("iforest", "false_alarm", s, n=n), if_fa, 0.01)
    near(m("lof", "auc", s, n=n), 1.00, 0.01)
    near(m("iforest", "auc", s, n=n), 1.00, 0.01)


@pytest.mark.parametrize("p,lof_f1,if_f1", [(2, 0.82, 0.92), (5, 0.90, 0.92), (8, 0.92, 0.94), (12, 0.94, 0.96), (20, 0.95, 0.96), (30, 0.94, 0.95)])
def test_feature_count_sweep(p, lof_f1, if_f1):
    near(m("lof", "auc", p=p), 1.00, 0.01)
    near(m("iforest", "auc", p=p), 1.00, 0.01)
    near(m("lof", "f1", p=p), lof_f1)
    near(m("iforest", "f1", p=p), if_f1)


@pytest.mark.parametrize("nn,lof_recall,if_recall,lof_f1,if_f1,rob_auc", [(0, 1.00, 0.99, 0.94, 0.96, 1.00), (10, 0.71, 0.91, 0.82, 0.94, 0.99), (20, 0.18, 0.71, 0.30, 0.82, 0.96), (30, 0.03, 0.58, 0.05, 0.72, 0.91),
                                                                          (40, 0.00, 0.39, 0.00, 0.56, 0.88)])
def test_noise_features_keep_the_ranking_but_break_the_lof_threshold(nn, lof_recall, if_recall, lof_f1, if_f1, rob_auc):
    near(m("lof", "recall", n_noise=nn), lof_recall)
    near(m("iforest", "recall", n_noise=nn), if_recall)
    near(m("lof", "f1", n_noise=nn), lof_f1)
    near(m("iforest", "f1", n_noise=nn), if_f1)
    near(m("robust", "auc", n_noise=nn), rob_auc, 0.015)
    for det in ("lof", "iforest"):
        near(m(det, "auc", n_noise=nn), 1.00 if nn < 30 else 0.99, 0.012)
    if nn == 40:
        near(oracle("lof", n_noise=40), 0.87)                                                             # mit bekanntem Anteil trägt die Rangfolge
        near(oracle("iforest", n_noise=40), 0.84)


@pytest.mark.parametrize("modes,lof_f1,if_f1,rob_f1,rob_auc", [(1, 0.94, 0.96, 0.84, 1.00), (2, 0.94, 0.96, 0.74, 0.95), (3, 0.93, 0.97, 0.36, 0.85)])
def test_modes_sweep(modes, lof_f1, if_f1, rob_f1, rob_auc):
    near(m("lof", "auc", n_modes=modes), 1.00, 0.01)
    near(m("lof", "f1", n_modes=modes), lof_f1)
    near(m("iforest", "f1", n_modes=modes), if_f1)
    near(m("robust", "f1", n_modes=modes), rob_f1)
    near(m("robust", "auc", n_modes=modes), rob_auc)


@pytest.mark.parametrize("curv,lof_f1,if_f1,rob_f1,lof_fa", [(0.0, 0.94, 0.96, 0.84, 0.014), (0.25, 0.90, 0.98, 0.49, 0.026), (0.5, 0.83, 0.97, 0.41, 0.045), (0.75, 0.78, 0.97, 0.39, 0.063), (1.0, 0.74, 0.95, 0.38, 0.078)])
def test_curvature_sweep_the_ranking_holds_but_the_lof_false_alarms_grow(curv, lof_f1, if_f1, rob_f1, lof_fa):
    near(m("lof", "f1", curvature=curv), lof_f1)
    near(m("iforest", "f1", curvature=curv), if_f1)
    near(m("robust", "f1", curvature=curv), rob_f1)
    near(m("lof", "false_alarm", curvature=curv), lof_fa, 0.01)
    near(m("lof", "auc", curvature=curv), 1.00, 0.01)


@pytest.mark.parametrize("noise,lof_f1,if_f1", [(0.0, 0.84, 0.94), (0.25, 0.94, 0.96), (0.5, 0.97, 0.96), (1.0, 0.92, 0.96)])
def test_noise_sweep(noise, lof_f1, if_f1):
    near(m("lof", "f1", noise=noise), lof_f1)
    near(m("iforest", "f1", noise=noise), if_f1)
    near(m("lof", "auc", noise=noise), 1.00, 0.01)
    if noise == 0.0:
        near(m("lof", "false_alarm", noise=noise), 0.042, 0.01)


# --- Seitenleiste: Anomalien --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("c,lof_auc,lof_f1,if_f1", [(2, 1.00, 0.75, 0.42), (5, 1.00, 0.88, 0.81), (10, 1.00, 0.94, 0.96), (20, 0.99, 0.90, 0.97), (30, 0.90, 0.43, 0.93), (40, 0.72, 0.15, 0.87), (45, 0.64, 0.09, 0.81)])
def test_contamination_sweep_lof_wins_at_few_anomalies_and_collapses_at_many_scattered_ones(c, lof_auc, lof_f1, if_f1):
    near(m("lof", "auc", contamination=c), lof_auc, 0.015)
    near(m("iforest", "auc", contamination=c), 1.00, 0.01)
    near(m("lof", "f1", contamination=c), lof_f1)
    near(m("iforest", "f1", contamination=c), if_f1)


def test_kinds_dense_group_and_gap_at_the_default_and_a_matching_k():
    near(m("lof", "auc", kind="cluster"), 0.35)
    near(m("iforest", "auc", kind="cluster"), 0.95)
    near(m("robust", "auc", kind="cluster"), 1.00, 0.01)
    near(m("lof", "auc", ev.Settings(k=40), kind="cluster"), 0.97)
    near(m("lof", "auc", n_modes=2, kind="gap"), 0.53)
    near(m("iforest", "auc", n_modes=2, kind="gap"), 0.54)
    near(m("robust", "auc", n_modes=2, kind="gap"), 0.40)
    near(m("lof", "auc", ev.Settings(k=40), n_modes=2, kind="gap"), 0.71)


@pytest.mark.parametrize("strength,auc,lof_f1,if_f1", [(3.0, 0.96, 0.48, 0.66), (4.0, 0.99, 0.88, 0.85), (6.0, 1.00, 0.94, 0.96), (9.0, 1.00, 0.94, 0.99), (12.0, 1.00, 0.94, 1.00)])
def test_strength_sweep(strength, auc, lof_f1, if_f1):
    near(m("lof", "auc", strength=strength), auc, 0.012)
    near(m("iforest", "auc", strength=strength), auc, 0.012)
    near(m("lof", "f1", strength=strength), lof_f1)
    near(m("iforest", "f1", strength=strength), if_f1)
    if strength == 3.0:
        near(m("lof", "recall", strength=strength), 0.33)


# --- LOF: k und Schwellen ------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k,auc,f1,fa,median", [(5, 0.86, 0.54, 0.009, 1.02), (10, 1.00, 0.91, 0.010, 1.02), (20, 1.00, 0.94, 0.014, 1.02), (40, 1.00, 0.89, 0.027, 1.03), (80, 1.00, 0.89, 0.029, 1.04)])
def test_k_sweep_in_the_default_case(k, auc, f1, fa, median):
    s = ev.Settings(k=k)
    near(m("lof", "auc", s), auc, 0.02)
    near(m("lof", "f1", s), f1)
    near(m("lof", "false_alarm", s), fa, 0.01)
    near(normal_lof_median(s), median, 0.01)
    near(m("iforest", "f1", s), 0.96)                                                                    # der Isolation Forest hängt nicht an k


@pytest.mark.parametrize("cut,f1,recall,fa", [(1.1, 0.52, 1.00, 0.205), (1.2, 0.69, 1.00, 0.099), (1.3, 0.82, 1.00, 0.050), (1.5, 0.94, 1.00, 0.014), (2.0, 0.96, 0.95, 0.003), (2.5, 0.85, 0.75, 0.000)])
def test_cutoff_sweep_the_window_is_narrow(cut, f1, recall, fa):
    s = ev.Settings(cutoff=cut)
    near(m("lof", "f1", s), f1)
    near(m("lof", "recall", s), recall)
    near(m("lof", "false_alarm", s), fa, 0.01)
    near(m("iforest", "f1", s), 0.96)                                                                    # die LOF-Schwelle ändert den Isolation Forest nicht


@pytest.mark.parametrize("q,rob_f1,cla_f1", [(0.9, 0.67, 0.69), (0.95, 0.77, 0.66), (0.975, 0.84, 0.57), (0.99, 0.89, 0.44), (0.999, 0.90, 0.15)])
def test_chi2_quantile_sweep_for_the_root_detectors(q, rob_f1, cla_f1):
    s = ev.Settings(quantile=q)
    near(m("robust", "f1", s), rob_f1)
    near(m("classical", "f1", s), cla_f1)
    near(m("lof", "f1", s), m("lof", "f1"), 1e-12)                                                       # LOF hängt nicht am chi²-Quantil
    near(m("iforest", "f1", s), m("iforest", "f1"), 1e-12)


@pytest.mark.parametrize("share,lof_f1,rob_f1,cla_f1", [(2, 0.33, 0.33, 0.32), (5, 0.67, 0.67, 0.56), (10, 0.97, 0.90, 0.69), (20, 0.67, 0.66, 0.58), (40, 0.40, 0.40, 0.39)])
def test_assumed_share_sweep_a_wrong_share_costs_all_detectors_the_same(share, lof_f1, rob_f1, cla_f1):
    s = ev.Settings(threshold_kind="share", share=share)
    near(m("lof", "f1", s), lof_f1)
    near(m("iforest", "f1", s), lof_f1)
    near(m("robust", "f1", s), rob_f1)
    near(m("classical", "f1", s), cla_f1)
    near(m("lof", "auc", s), 1.00, 0.01)


# --- Experimente: Tabellen ---------------------------------------------------------------------------------------------------------------


def test_group_table_lof_sees_the_group_only_when_k_exceeds_it():
    g = ev.group_table()
    by = {(c["contamination"], c["k"]): c["lof_auc"] for c in g["cells"]}
    expected = {5: {10: 0.34, 20: 0.99, 40: 1.00, 80: 1.00, 100: 1.00}, 10: {10: 0.45, 20: 0.35, 40: 0.97, 80: 1.00, 100: 1.00}, 20: {10: 0.51, 20: 0.45, 40: 0.37, 80: 0.95, 100: 0.99},
                30: {10: 0.52, 20: 0.47, 40: 0.42, 80: 0.37, 100: 0.71}}
    for c, row in expected.items():
        for k, v in row.items():
            near(by[(c, k)], v, 0.03)
    for b, (if_auc, rob_auc) in zip(g["baselines"], ((0.99, 1.00), (0.95, 1.00), (0.85, 1.00), (0.70, 0.49))):
        near(b["iforest_auc"], if_auc)
        near(b["robust_auc"], rob_auc, 0.03)
    assert all(by[(c, k)] < 0.55 for c, k in ((5, 10), (10, 20), (20, 40), (30, 80)))                     # k kleiner als die Gruppe: LOF sieht sie nicht (unter oder um Raten)
    assert all(by[(c, k)] > 0.9 for c, k in ((5, 40), (10, 80), (20, 100), (5, 100)))


def test_gap_table_the_window_and_the_wins():
    rows = {(r["n_modes"], r["contamination"]): r for r in ev.gap_table()}
    for key, expected in {(2, 10): {20: 0.53, 30: 0.66, 40: 0.71, 50: 0.69, 60: 0.65, 70: 0.58, 90: 0.37}, (3, 10): {20: 0.36, 30: 0.41, 40: 0.60, 50: 0.76, 60: 0.84, 70: 0.86, 90: 0.14},
                          (2, 5): {20: 0.84, 30: 0.86, 40: 0.84, 50: 0.81, 60: 0.75, 70: 0.68, 90: 0.49}}.items():
        for k, v in expected.items():
            near(rows[key]["by_k"][k], v, 0.03)
    near(rows[(2, 10)]["iforest_auc"], 0.54)
    near(rows[(2, 10)]["robust_auc"], 0.40)
    near(rows[(3, 10)]["iforest_auc"], 0.27, 0.03)
    near(rows[(3, 10)]["robust_auc"], 0.38, 0.03)
    near(rows[(2, 5)]["iforest_auc"], 0.71)
    assert max(rows[(3, 10)]["by_k"].values()) > 0.8 > max(rows[(3, 10)]["iforest_auc"], rows[(3, 10)]["robust_auc"], rows[(3, 10)]["classical_auc"]) and rows[(3, 10)]["by_k"][90] < 0.2


def test_modes_table_at_the_default_k():
    by = {(r["n_modes"], r["kind"]): r for r in ev.modes_table()}
    assert len(by) == 8 and (1, "gap") not in by
    for modes in (1, 2, 3):
        near(by[(modes, "scattered")]["lof_auc"], 1.00, 0.01)
    for modes, lof_auc, if_auc in ((1, 0.35, 0.95), (2, 0.38, 0.98), (3, 0.39, 0.95)):
        near(by[(modes, "cluster")]["lof_auc"], lof_auc, 0.03)
        near(by[(modes, "cluster")]["iforest_auc"], if_auc)
    near(by[(2, "gap")]["lof_auc"], 0.53)
    near(by[(2, "gap")]["iforest_auc"], 0.54)
    near(by[(3, "gap")]["lof_auc"], 0.36, 0.03)
    near(by[(3, "gap")]["iforest_auc"], 0.27, 0.03)


def test_masking_table_with_the_default_and_a_matching_k():
    rows = {r["x"]: r for r in ev.masking_table()}
    assert sorted(rows) == list(ev.MASKING_CONTAMINATION)
    for c, lof_auc in ((2, 1.00), (5, 0.99), (10, 0.35), (15, 0.42), (20, 0.45), (25, 0.46), (30, 0.47), (35, 0.48), (40, 0.49), (45, 0.50)):
        near(rows[c]["lof_auc"], lof_auc, 0.03)
    for c, matched in ((5, 0.99), (10, 0.99), (15, 0.98), (20, 0.98), (25, 0.94), (30, 0.71)):
        near(rows[c]["lof_matched_auc"], matched, 0.03)
    for c, if_auc in ((5, 0.99), (10, 0.95), (15, 0.91), (20, 0.85), (25, 0.79), (30, 0.70)):
        near(rows[c]["iforest_auc"], if_auc)
    for c in (5, 10, 15, 20, 25):
        near(rows[c]["robust_auc"], 1.00, 0.02)
    near(rows[30]["robust_auc"], 0.49, 0.03)
    assert all(rows[c]["lof_matched_auc"] > rows[c]["iforest_auc"] for c in (10, 15, 20, 25)) and rows[35]["lof_matched_auc"] < 0.6 and rows[35]["robust_auc"] < 0.6


def test_k_table_k_near_n_inverts_the_ranking():
    scattered, cluster = ev.k_table()
    for k in (5, 10, 20, 30, 50, 70, 90):
        near(scattered["by_k"][k], 1.00, 0.02)
    assert scattered["by_k"][99] < 0.03
    for k, v in {5: 0.43, 10: 0.39, 20: 0.58, 30: 0.99, 50: 1.00, 70: 1.00, 90: 0.38, 99: 0.05}.items():
        near(cluster["by_k"][k], v, 0.04)


def test_dimension_table_the_lof_is_calibrated_at_small_samples_and_both_keep_the_ranking():
    cells = {(c["n"], c["p"]): c for c in ev.dimension_table()}
    assert min(c[f"{d}_auc"] for c in cells.values() for d in ("lof", "iforest")) >= 0.985
    for n in (20, 30):
        for p in (2, 5, 12, 20, 30):
            assert 0.88 <= cells[(n, p)]["lof_f1"] + 0.005 and cells[(n, p)]["lof_f1"] <= 0.965, (n, p)
            assert 0.53 <= cells[(n, p)]["iforest_f1"] <= 0.69, (n, p)
    near(cells[(20, 12)]["lof_f1"], 0.96, 0.03)
    near(cells[(20, 12)]["iforest_f1"], 0.60, 0.03)
    for p in (2, 5, 12, 20, 30):
        assert 0.025 <= cells[(100, p)]["lof_false_alarm"] <= 0.065, p


def test_threshold_table_normal_lof_is_about_one_for_every_tour_count():
    t = ev.threshold_table()
    for r, expected in zip(t["normal_scores"], (1.007, 1.024, 1.030, 1.018, 1.016)):
        near(r["median"], expected, 0.012)
    assert [r["n"] for r in t["normal_scores"]] == [20, 50, 100, 300, 600] and all(1.0 <= r["median"] <= 1.04 for r in t["normal_scores"])
    assert [(r["factor"], r["x"]) for r in t["wrong_share"]] == [(0.5, 5), (1.0, 10), (2.0, 20)]
    f = {r["factor"]: r for r in t["wrong_share"]}
    near(f[1.0]["lof_f1"], 0.97)
    near(f[0.5]["lof_f1"], 0.67)
    near(f[2.0]["lof_f1"], 0.67)
    assert f[1.0]["lof_f1"] > f[1.0]["robust_f1"] > f[1.0]["classical_f1"]


def test_costs_lof_is_faster_than_the_forest_and_needs_standardised_units():
    t = ev.cost_table()
    assert [r["n"] for r in t["times"]] == [100, 300, 600]
    for r in t["times"]:
        assert r["lof"] < r["iforest"] / 3, r                                                            # Zeitangaben sind rechnerabhängig: nur das Verhältnis wird behauptet
    raw = [r for r in t["units"] if not r["standardize"]][0]
    std = [r for r in t["units"] if r["standardize"]][0]
    near(std["lof_auc"], 1.00, 0.01)
    near(std["lof_f1"], 0.94)
    near(raw["lof_auc"], 0.87, 0.03)
    near(raw["lof_f1"], 0.68, 0.05)
    assert raw["iforest_auc"] == std["iforest_auc"]                                                    # der Isolation Forest ist skaleninvariant: derselbe Wald, dieselbe AUC


# --- Presets ----------------------------------------------------------------------------------------------------------------------------


def test_preset_help_numbers_standard_and_dense_groups():
    near(m("lof", "f1"), 0.94)
    near(m("iforest", "f1"), 0.96)
    near(m("lof", "false_alarm"), 0.014, 0.006)
    near(normal_lof_median(), 1.02, 0.01)
    small = ev.Settings(k=20)
    near(m("lof", "auc", small, kind="cluster"), 0.35)
    near(m("lof", "recall", small, kind="cluster"), 0.01, 0.02)
    near(m("iforest", "auc", small, kind="cluster"), 0.95)
    near(m("robust", "auc", small, kind="cluster"), 1.00, 0.01)
    good = ev.Settings(k=40)
    near(m("lof", "auc", good, kind="cluster"), 0.97)
    near(m("lof", "f1", good, kind="cluster"), 0.62, 0.03)
    near(m("lof", "recall", good, kind="cluster"), 0.67, 0.03)
    near(m("iforest", "f1", good, kind="cluster"), 0.66, 0.03)
    near(m("iforest", "false_alarm", good, kind="cluster"), 0.12, 0.02)


def test_preset_help_numbers_gap_noise_few_tours_and_many_anomalies():
    g = ev.Settings(k=60)
    near(m("lof", "auc", g, n_modes=3, kind="gap"), 0.84)
    near(m("iforest", "auc", g, n_modes=3, kind="gap"), 0.27, 0.03)
    near(m("robust", "auc", g, n_modes=3, kind="gap"), 0.38, 0.03)
    near(m("lof", "f1", g, n_modes=3, kind="gap"), 0.00, 0.02)
    near(oracle("lof", g, n_modes=3, kind="gap"), 0.02, 0.03)
    near(m("lof", "auc", n_noise=40), 0.99, 0.01)
    near(m("lof", "recall", n_noise=40), 0.00, 0.02)
    near(m("iforest", "f1", n_noise=40), 0.56)
    p3 = dict(n=20, p=30)
    s3 = ev.Settings(k=10)
    near(m("lof", "auc", s3, **p3), 1.00, 0.01)
    near(m("iforest", "auc", s3, **p3), 1.00, 0.01)
    for det in ("classical", "robust"):                                                                # n < p: singuläre Kovarianz, der Wert hängt an der Plattform-Rundung
        assert 0.3 <= m(det, "auc", s3, **p3) <= 0.75, det
    near(m("lof", "f1", s3, **p3), 0.92, 0.03)
    near(m("lof", "false_alarm", s3, **p3), 0.02, 0.015)
    near(m("iforest", "f1", s3, **p3), 0.61, 0.03)
    near(m("iforest", "false_alarm", s3, **p3), 0.144, 0.02)
    near(m("lof", "auc", contamination=40), 0.72)
    near(m("lof", "f1", contamination=40), 0.15)
    near(m("lof", "recall", contamination=40), 0.08, 0.03)
    near(m("iforest", "auc", contamination=40), 1.00, 0.01)
    near(m("iforest", "f1", contamination=40), 0.87)
    near(m("robust", "auc", contamination=40), 0.97)


def test_analysis_time_stays_small():
    a = ev.analyse(ev.make_dataset(n=600, p=30, n_noise=40, contamination=45))
    assert a.seconds["lof"] < 1.0 and a.seconds["iforest"] < 4.0 and a.seconds["robust"] < 10.0
