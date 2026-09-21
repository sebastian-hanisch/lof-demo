"""Szenario (identisch zu den Vorgänger-Demos), Kennzahlen mit Handinstanzen, Analyse und Schwellen für vier Detektoren, k-Grenze, Score-Karten, Urteil."""

import numpy as np
import pytest

import lof_algorithm as lof
import lof_constants as C
import lof_ee_algorithm as alg
import lof_evaluation as ev
import lof_isolation_forest as isf

# Zeilensummen der ersten acht Zeilen der PCA-Demo (dieselben normalen Zeilen wie in den Vorgänger-Demos): permutationsinvariant, eingefroren
PCA_ROW_SUMS = [39967.27507413389, 42083.657538741994, 40172.02304566072, 44766.540605465496, 45034.44402326185, 55757.45917619934, 50727.55911151846, 43923.05872324106]


# --- Szenario ------------------------------------------------------------------------------------------------------------------------


def test_normal_rows_equal_the_pca_and_the_predecessor_rows():
    ds = ev.make_dataset(contamination=1)
    assert not ds.anomaly[:8].any() and ds.X.shape == (300, 12)
    assert np.allclose(ds.X[:8].sum(axis=1), PCA_ROW_SUMS, rtol=1e-12)


def test_default_dataset_is_frozen_at_the_predecessor_values():
    ds = ev.make_dataset()
    assert float(ds.X.sum()) == pytest.approx(13396385.215115668, rel=1e-12) and int(ds.anomaly.sum()) == 30 and ds.X[0, 0] == pytest.approx(19701.54827513291, rel=1e-12)


def test_noise_features_are_appended_and_change_nothing_else():
    base = ev.make_dataset()
    for n_noise in (1, 10, 40):
        ds = ev.make_dataset(n_noise=n_noise)
        assert ds.X.shape == (300, 12 + n_noise) and np.array_equal(ds.X[:, :12], base.X) and np.array_equal(ds.anomaly, base.anomaly)
        assert len(ds.names) == 12 + n_noise and ds.names[-1] == f"Rauschmerkmal {n_noise}" and ds.n_noise == n_noise
    ds = ev.make_dataset(n_noise=40)
    extra = ds.X[:, 12:]
    assert abs(extra.mean() - C.EXTRA_MEAN) < 1.0 and abs(extra.std() - C.EXTRA_SCALE) < 1.0
    off = np.abs(np.corrcoef(extra.T))
    np.fill_diagonal(off, 0.0)
    assert off.max() < 0.3 and abs(np.corrcoef(np.c_[extra[:, 0], ds.X[:, 0]].T)[0, 1]) < 0.3
    assert np.array_equal(ev.make_dataset(p=30, n_noise=5).X[:, :30], ev.make_dataset(p=30).X)


@pytest.mark.parametrize("n,pct,expected", [(300, 10, 30), (300, 1, 3), (20, 1, 1), (30, 10, 3), (600, 45, 270)])
def test_contamination_is_exact_with_at_least_one_anomaly(n, pct, expected):
    assert int(ev.make_dataset(n=n, contamination=pct).anomaly.sum()) == expected


def test_gap_needs_two_modes_and_kinds_have_their_geometry():
    assert ev.make_dataset(kind="gap").kind == "scattered"
    gap = ev.make_dataset(n_modes=2, kind="gap")
    assert np.abs(gap.z[gap.anomaly]).max() < 1.2
    clu = ev.make_dataset(kind="cluster", contamination=20)
    zc = clu.z[clu.anomaly]
    assert np.linalg.norm(zc.mean(axis=0) - 6.0 * np.array([np.cos(C.CLUSTER_ANGLE), np.sin(C.CLUSTER_ANGLE)])) < 0.15


def test_dataset_is_deterministic():
    assert np.array_equal(ev.make_dataset(seed=3, n_noise=4).X, ev.make_dataset(seed=3, n_noise=4).X)


# --- Kennzahlen: Handinstanzen (aus der Wurzel übernommen) ----------------------------------------------------------------------------


def test_roc_auc_and_average_precision_hand_instances():
    assert ev.roc_auc(np.array([1, 2, 3, 4.0]), np.array([0, 0, 1, 1], bool)) == 1.0
    assert ev.roc_auc(np.array([1, 1, 1, 1.0]), np.array([0, 0, 1, 1], bool)) == 0.5
    assert ev.roc_auc(np.array([1, 4, 2, 3.0]), np.array([0, 1, 1, 0], bool)) == pytest.approx(0.75)
    assert np.isnan(ev.roc_auc(np.array([1.0, 2.0]), np.array([False, False])))
    rng = np.random.default_rng(0)
    s, y = rng.standard_normal(200), rng.random(200) < 0.3
    assert ev.roc_auc(s, y) == pytest.approx(np.mean([(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]]))
    assert ev.average_precision(np.array([4, 3, 2, 1.0]), np.array([1, 0, 1, 0], bool)) == pytest.approx((1 + 2 / 3) / 2)


def test_roc_curve_and_flag_metrics():
    rng = np.random.default_rng(1)
    s, y = rng.standard_normal(300), rng.random(300) < 0.2
    fpr, tpr = ev.roc_curve(s, y)
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0.0, 0.0, 1.0, 1.0) and np.trapezoid(tpr, fpr) == pytest.approx(ev.roc_auc(s, y), abs=1e-9)
    m = ev.flag_metrics(np.array([1, 1, 0, 0, 1, 0], bool), np.array([1, 0, 1, 0, 0, 0], bool))
    assert m["precision"] == pytest.approx(1 / 3) and m["recall"] == 0.5 and m["false_alarm"] == pytest.approx(0.5) and m["f1"] == pytest.approx(0.4)


def test_k_max_is_half_the_tours_between_k_min_plus_one_and_k_max():
    assert [ev.k_max(n) for n in (8, 20, 100, 300, 600)] == [C.K_MIN + 1, 10, 50, 100, 100]


# --- Analyse und Schwellen ----------------------------------------------------------------------------------------------------------------


def _params(**kw):
    p = {**ev.DEFAULT_DATA, **kw}
    return (p["n"], p["p"], p["n_noise"], p["n_modes"], p["curvature"], p["noise"], p["contamination"], p["kind"], p["strength"], 7)


def alg_chi(dof):
    return alg.chi2_ppf(C.DEFAULT_QUANTILE, dof)


def test_analysis_fields_and_consistency():
    a = ev.analyse_for(_params())
    assert set(a.scores) == set(ev.DETECTORS) == set(a.flags) == set(a.values) == {"lof", "iforest", "classical", "robust"}
    assert a.k == 20 and a.lof_fit.neighbors.shape == (300, 20) and a.paths_if.shape == (100, 300) and a.forest_if.psi == 256 and a.robust.h == 156 and a.classical.dof == 12
    assert np.allclose(a.values["lof"], lof.fit_lof(ev.standardise(a.ds.X), 20).lof) and np.allclose(a.values["iforest"], isf.score_from_paths(a.paths_if, a.forest_if.psi))
    for d in ev.DETECTORS:
        assert a.scores[d]["n_flagged"] == int(a.flags[d].sum())
    assert a.scores["lof"]["threshold"] == 1.5 and (a.flags["lof"] == (a.values["lof"] > 1.5)).all()
    assert a.scores["iforest"]["threshold"] == 0.5 and (a.flags["iforest"] == (a.values["iforest"] > 0.5)).all()
    assert a.scores["classical"]["threshold"] == pytest.approx(alg_chi(12)) and (a.flags["robust"] == (a.values["robust"] > alg_chi(12))).all()
    assert set(a.oracle_f1) == {"lof", "iforest"} and all(0 <= v <= 1 for v in a.oracle_f1.values()) and a.seconds["lof"] > 0 and a.seconds["iforest"] > 0


def test_share_threshold_flags_exactly_the_assumed_share_for_all_four_detectors():
    for share in (2, 10, 40):
        a = ev.analyse_for(_params(), ev.Settings(threshold_kind="share", share=share))
        for d in ev.DETECTORS:
            assert int(a.flags[d].sum()) == max(1, round(share / 100 * 300))
            assert a.flags[d][np.argsort(-a.values[d])[:3]].all()
    a = ev.analyse_for(_params(), ev.Settings(threshold_kind="share", share=10))
    assert a.scores["lof"]["threshold"] > 1.0 and a.scores["lof"]["f1"] == pytest.approx(a.oracle_f1["lof"]) and a.scores["iforest"]["f1"] == pytest.approx(a.oracle_f1["iforest"])         # 10 % = wahrer Anteil


def test_lof_cutoff_and_quantile_move_only_their_own_detectors():
    base = ev.analyse_for(_params())
    cut = ev.analyse_for(_params(), ev.Settings(cutoff=2.5))
    assert cut.scores["lof"]["recall"] < base.scores["lof"]["recall"] and cut.scores["lof"]["false_alarm"] <= base.scores["lof"]["false_alarm"]
    assert cut.scores["iforest"] == base.scores["iforest"] and cut.scores["classical"] == base.scores["classical"] and cut.scores["robust"] == base.scores["robust"]
    q = ev.analyse_for(_params(), ev.Settings(quantile=0.999))
    assert q.scores["lof"] == base.scores["lof"] and q.scores["iforest"] == base.scores["iforest"] and q.scores["robust"]["false_alarm"] <= base.scores["robust"]["false_alarm"]
    assert q.scores["classical"]["auc"] == base.scores["classical"]["auc"]                       # nur die Schwelle, nicht die Rangfolge


def test_k_changes_only_lof_and_is_clamped_to_n_minus_one():
    a = ev.analyse_for(_params(), ev.Settings(k=5))
    b = ev.analyse_for(_params(), ev.Settings(k=80))
    assert a.k == 5 and b.k == 80 and not np.allclose(a.values["lof"], b.values["lof"])
    assert np.array_equal(a.values["iforest"], b.values["iforest"]) and a.values["robust"].tolist() == b.values["robust"].tolist()
    assert ev.analyse_for(_params(n=20), ev.Settings(k=10_000)).k == 19


def test_standardisation_matters_only_for_lof():
    a = ev.analyse_for(_params())
    b = ev.analyse_for(_params(), ev.Settings(standardize=False))
    assert np.array_equal(a.values["iforest"], b.values["iforest"]) and a.values["robust"].tolist() == b.values["robust"].tolist() and not np.allclose(a.values["lof"], b.values["lof"])
    assert b.scores["lof"]["auc"] < a.scores["lof"]["auc"] - 0.05                                    # auf Rohdaten dominiert das Merkmal mit der größten Einheit


def test_forest_seed_is_separate_from_the_data_seed_and_lof_is_deterministic():
    a, b = ev.analyse_for(_params()), ev.analyse_for(_params(), ev.Settings(start=5))
    assert np.array_equal(a.ds.X, b.ds.X) and a.values["iforest"].tolist() != b.values["iforest"].tolist() and abs(a.scores["iforest"]["auc"] - b.scores["iforest"]["auc"]) < 0.02
    assert np.array_equal(a.values["lof"], b.values["lof"])                                           # LOF hat keinen Zufall


def test_noise_features_are_used_and_few_tours_many_features_work():
    b = ev.analyse_for(_params(n_noise=10))
    assert b.ds.X.shape[1] == 22 and b.classical.dof == 22
    c = ev.analyse_for(_params(n=20, p=30), ev.Settings(k=10))
    assert c.k == 10 and c.scores["lof"]["auc"] > 0.95 and c.forest_if.psi == 20


def test_sweep_rows_labels_and_k_following_the_tour_count():
    rows = ev.sweep("contamination", values=(5, 10))
    assert [r["x"] for r in rows] == [5, 10]
    for r in rows:
        for d in ev.DETECTORS:
            assert r[f"{d}_auc_min"] <= r[f"{d}_auc"] <= r[f"{d}_auc_max"] and r[f"{d}_auc_std"] >= 0
        assert "lof_oracle_f1" in r and "iforest_oracle_f1" in r and "lof_seconds" in r and "lof_normal_median" in r
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    s = ev.sweep("cutoff", values=(1.2, 2.5))
    assert s[0]["lof_recall"] > s[1]["lof_recall"] and s[0]["iforest_auc"] == s[1]["iforest_auc"] and s[0]["classical_auc"] == s[1]["classical_auc"] and s[0]["robust_recall"] == s[1]["robust_recall"]
    k = ev.sweep("k", values=(5, 40))
    assert k[0]["iforest_auc"] == k[1]["iforest_auc"] and k[0]["robust_auc"] == k[1]["robust_auc"] and k[0]["lof_false_alarm"] < k[1]["lof_false_alarm"]
    n = ev.sweep("n", values=(20,))
    assert n[0]["lof_auc"] > 0.9                                                                   # k = 20 würde bei n = 20 klemmen (AUC 0.1); der Sweep begrenzt k auf n / 2


def test_settings_defaults_agree_with_the_constants():
    s = ev.Settings()
    assert (s.k, s.threshold_kind, s.cutoff, s.quantile, s.share) == (C.DEFAULT_K, C.DEFAULT_THRESHOLD_KIND, C.DEFAULT_CUTOFF, C.DEFAULT_QUANTILE, C.DEFAULT_SHARE) and s.standardize is True
    assert set(ev.DEFAULT_DATA) == set(ev.DATA_KEYS) and C.SWEEP_SEEDS == tuple(range(100000, 100005))


# --- Experimente: Form der Tabellen ----------------------------------------------------------------------------------------------------------


def test_group_gap_and_k_tables_have_their_documented_shape():
    g = ev.group_table()
    assert [(c["contamination"], c["k"]) for c in g["cells"]] == [(c, k) for c in ev.GROUP_CONTAMINATION for k in ev.GROUP_K]
    assert [b["contamination"] for b in g["baselines"]] == list(ev.GROUP_CONTAMINATION) and [round(b["group"]) for b in g["baselines"]] == [15, 30, 60, 90]
    gp = ev.gap_table()
    assert [(r["n_modes"], r["contamination"]) for r in gp] == [(2, 10), (3, 10), (2, 5)] and all(sorted(r["by_k"]) == list(ev.GAP_K) for r in gp)
    kt = ev.k_table()
    assert [r["kind"] for r in kt] == ["scattered", "cluster"] and sorted(kt[0]["by_k"]) == [5, 10, 20, 30, 50, 70, 90, 99]


def test_lof_only_matches_the_full_analysis():
    auc, m = ev._lof_only(100000, 20, {})
    a = ev._analyse_seed(100000, ev.Settings(), {})
    assert auc == pytest.approx(a.scores["lof"]["auc"]) and m == int(a.ds.anomaly.sum())


def test_score_maps_shape_and_lof_rises_away_from_the_data():
    ds = ev.make_dataset(p=2)
    X2 = ds.X[~ds.anomaly]
    xs, ys, S_lof, S_if = ev.score_maps(X2, grid=40)
    assert S_lof.shape == S_if.shape == (40, 40) and len(xs) == 40 and len(ys) == 40
    center = (np.argmin(np.abs(ys - X2[:, 1].mean())), np.argmin(np.abs(xs - X2[:, 0].mean())))
    assert S_lof[center] < 1.6 and S_lof[0, 0] > 3.0 and S_lof[-1, -1] > 3.0 and S_if[0, 0] > S_if[center] + 0.1


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------


def _verdict(**kw):
    settings = ev.Settings(**{k: kw.pop(k) for k in ("k", "threshold_kind", "share", "cutoff") if k in kw})
    return ev.verdict(ev.analyse_for(_params(**kw), settings))


def test_verdict_codes_for_the_presets_and_edge_cases():
    assert _verdict()[:2] == ("success", "comparable")
    assert _verdict(kind="cluster", k=20)[:2] == ("warning", "k_window")
    assert _verdict(kind="cluster", k=40)[1] in ("lof_wins", "comparable")
    assert _verdict(n_modes=3, kind="gap", k=60)[:2] == ("success", "lof_wins")
    assert _verdict(n_modes=2, kind="gap", k=20)[1] == "gap"
    assert _verdict(n_noise=40)[:2] == ("warning", "threshold_off")
    assert _verdict(n=20, p=30, k=10)[1] in ("lof_wins", "threshold_off")
    assert _verdict(contamination=40)[:2] == ("warning", "others_win")
    assert _verdict(n=100, kind="cluster", contamination=20, k=90)[:2] == ("warning", "k_window")                     # k größer als n - Gruppe
    assert _verdict(curvature=1.0)[1] in ("threshold_off", "iforest_f1", "comparable")


def test_verdict_data_carries_the_numbers_the_messages_use():
    kind, code, data = _verdict(kind="cluster", k=20)
    assert code == "k_window"
    for key in ("lof_auc", "iforest_auc", "classical_auc", "robust_auc", "best_other_auc", "lof_recall", "lof_false_alarm", "lof_f1", "lof_threshold", "iforest_f1", "iforest_false_alarm", "classical_f1", "robust_f1",
                "oracle_lof", "oracle_if", "contamination", "n_anomalies", "n", "p", "k", "kind"):
        assert key in data
    assert data["p"] == 12 and data["n_noise"] == 0 and data["k"] == 20 and data["n_anomalies"] == 30 and data["contamination"] == pytest.approx(10.0)


def test_analysis_time_stays_small():
    a = ev.analyse(ev.make_dataset(n=600, p=30, n_noise=40, contamination=45))
    assert a.seconds["lof"] < 1.0 and a.seconds["iforest"] < 4.0 and a.seconds["robust"] < 10.0
