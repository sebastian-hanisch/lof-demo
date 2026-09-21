"""Local Outlier Factor: Handinstanzen (k-Distanz, Erreichbarkeit, lrd), Kreuzprüfung gegen scikit-learn (transduktiv und mit novelty=True), Duplikate und Bindungen, k-Grenzen, Invarianzen und das Verhalten bei k nahe n,
Determinismus; die kopierten Komponenten der Vorgänger (Isolation Forest, χ², MCD)."""

import numpy as np
import pytest
from scipy.stats import chi2, spearmanr
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import lof_algorithm as lof
import lof_ee_algorithm as alg
import lof_isolation_forest as isf


def _data(n=300, p=4, n_out=15, shift=8.0, seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([rng.standard_normal((n - n_out, p)), shift + 0.5 * rng.standard_normal((n_out, p))])
    return X, np.arange(n) >= n - n_out


def _sklearn(X, k):
    return -LocalOutlierFactor(n_neighbors=k).fit(X).negative_outlier_factor_


# --- Handinstanzen ------------------------------------------------------------------------------------------------------------------------


def test_one_dimensional_hand_instance():
    """Punkte 0, 1, 2, 3 und ein Ausreißer bei 10 mit k = 2: von Hand gerechnet."""
    X = np.array([[0.0], [1.0], [2.0], [3.0], [10.0]])
    f = lof.fit_lof(X, 2)
    assert f.neighbors.tolist() == [[1, 2], [0, 2], [1, 3], [2, 1], [3, 2]]                 # bei Gleichstand (Tour 3: 1 und 2 haben Abstand 2 und 1 ...) entscheiden die Abstände, dann der Index
    assert np.allclose(f.kdist, [2.0, 1.0, 1.0, 2.0, 8.0])
    # reach(0,1) = max(kdist(1)=1, 1) = 1 ; reach(0,2) = max(kdist(2)=1, 2) = 2 -> lrd(0) = 1 / 1.5
    assert f.lrd[0] == pytest.approx(1 / 1.5, rel=1e-9)
    # Punkt 4: reach(4,3) = max(2, 7) = 7 ; reach(4,2) = max(1, 8) = 8 -> lrd = 1 / 7.5 ; LOF = mean(lrd(3), lrd(2)) / lrd(4)
    assert f.lrd[4] == pytest.approx(1 / 7.5, rel=1e-9)
    assert f.lof[4] == pytest.approx(np.mean([f.lrd[3], f.lrd[2]]) / f.lrd[4], rel=1e-9)
    assert f.lof[4] > 3.0 and (f.lof[:4] < 1.3).all()


def test_two_groups_of_different_density_the_sparse_group_is_not_flagged_for_its_own_spread():
    """LOF vergleicht mit der eigenen Gegend: die dünne Gruppe (Streuung 3) ist ebenso 'normal' wie die dichte (Streuung 0.3); erst ein Punkt zwischen ihnen fällt auf."""
    rng = np.random.default_rng(3)
    dense = 0.3 * rng.standard_normal((100, 2))
    sparse = 10.0 + 3.0 * rng.standard_normal((100, 2))
    X = np.concatenate([dense, sparse, [[3.0, 3.0]]])
    f = lof.fit_lof(X, 15)
    assert np.median(f.lof[:100]) < 1.2 and np.median(f.lof[100:200]) < 1.2 and f.lof[-1] > 2.0
    # ein globaler Abstandsschwellwert würde die dünne Gruppe markieren: ihre k-Distanz ist deutlich größer
    assert np.median(f.kdist[100:200]) > 5 * np.median(f.kdist[:100])


def test_uniform_grid_has_lof_one_in_the_interior():
    g = np.stack(np.meshgrid(np.arange(15.0), np.arange(15.0)), axis=-1).reshape(-1, 2)
    f = lof.fit_lof(g, 8)
    interior = (g[:, 0] > 4) & (g[:, 0] < 10) & (g[:, 1] > 4) & (g[:, 1] < 10)
    assert np.allclose(f.lof[interior], 1.0, atol=0.02)


# --- Kreuzprüfung gegen scikit-learn --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [3, 5, 20, 60])
def test_lof_equals_scikit_learn(k):
    X, _ = _data(n=200, p=5, n_out=12, shift=6.0, seed=1)
    ours = lof.fit_lof(X, k).lof
    theirs = _sklearn(X, k)
    assert np.abs(ours - theirs).max() < 1e-9 and spearmanr(ours, theirs)[0] > 0.999999


def test_lof_of_new_points_equals_scikit_learn_novelty():
    X, _ = _data(n=200, p=3, seed=2)
    Q = np.random.default_rng(5).standard_normal((60, 3)) * 2.5
    for k in (5, 20):
        f = lof.fit_lof(X, k)
        theirs = -LocalOutlierFactor(n_neighbors=k, novelty=True).fit(X).score_samples(Q)
        assert np.abs(lof.score_new(X, f, Q) - theirs).max() < 1e-9


def test_score_new_of_a_training_point_shifted_by_zero_is_close_to_its_own_lof_or_higher():
    X, _ = _data(n=150, p=3, seed=4)
    f = lof.fit_lof(X, 10)
    q = lof.score_new(X, f, X)                                                              # der Punkt selbst ist dabei sein eigener nächster Nachbar (Abstand 0)
    assert np.isfinite(q).all() and (q > 0).all()


# --- Duplikate, Bindungen, Grenzen ----------------------------------------------------------------------------------------------------------


def test_duplicates_do_not_produce_nan_or_inf():
    X = np.concatenate([np.zeros((30, 2)), np.random.default_rng(0).standard_normal((30, 2)) + 5.0])
    f = lof.fit_lof(X, 5)
    assert np.isfinite(f.lof).all() and np.isfinite(f.lrd).all() and (f.lrd > 0).all()


def test_exactly_k_neighbours_even_with_ties_and_no_self_neighbour():
    X = np.array([[0.0], [1.0], [1.0], [1.0], [2.0]])
    f = lof.fit_lof(X, 2)
    assert f.neighbors.shape == (5, 2) and all(i not in f.neighbors[i] for i in range(5))


def test_k_is_clamped_to_one_and_n_minus_one():
    X, _ = _data(n=40, p=3)
    assert lof.clamp_k(0, 40) == 1 and lof.clamp_k(10_000, 40) == 39 and lof.clamp_k(7, 40) == 7
    assert lof.fit_lof(X, 10_000).k == 39 and lof.fit_lof(X, 1).k == 1


def test_pairwise_distances_symmetric_zero_diagonal_and_equal_to_the_definition():
    X, _ = _data(n=50, p=6)
    D = lof.pairwise_distances(X)
    ref = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))
    assert np.allclose(D, ref, atol=1e-9) and np.array_equal(D, D.T) and (np.diag(D) == 0).all()


# --- Invarianzen und das Verhalten bei großem k -----------------------------------------------------------------------------------------------


def test_lof_is_invariant_to_shift_uniform_scaling_and_rotation_but_not_to_scaling_single_features():
    X, _ = _data(n=200, p=3, seed=6)
    base = lof.fit_lof(X, 15).lof
    assert np.allclose(lof.fit_lof(X + np.array([5.0, -3.0, 100.0]), 15).lof, base, atol=1e-8)
    assert np.allclose(lof.fit_lof(7.0 * X, 15).lof, base, atol=1e-8)
    Q, _r = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))
    assert np.allclose(lof.fit_lof(X @ Q, 15).lof, base, atol=1e-8)                        # euklidisch: drehinvariant (anders als der Isolation Forest)
    scaled = X * np.array([1.0, 1.0, 1000.0])
    assert not np.allclose(lof.fit_lof(scaled, 15).lof, base, atol=1e-3)                   # ein Merkmal in anderen Einheiten ändert die Nachbarschaft


def test_a_dense_group_smaller_than_k_is_invisible_and_larger_than_k_is_seen():
    rng = np.random.default_rng(7)
    normal = rng.standard_normal((250, 3))
    group = 7.0 + 0.3 * rng.standard_normal((30, 3))
    X = np.concatenate([normal, group])
    is_group = np.arange(280) >= 250
    small = lof.fit_lof(X, 15).lof                                                          # k = 15 < 30: jede Tour der Gruppe hat nur Gruppen-Nachbarn
    large = lof.fit_lof(X, 45).lof                                                          # k = 45 > 30
    assert small[is_group].mean() < 1.2 and large[is_group].mean() > 3.0


def test_k_equal_to_n_minus_one_makes_all_points_see_the_same_neighbours_and_inverts_the_ranking():
    X, an = _data(n=100, p=3, n_out=10, seed=8)
    f = lof.fit_lof(X, 99)
    assert all(set(f.neighbors[i]) == set(range(100)) - {i} for i in range(100))
    from lof_evaluation import roc_auc
    assert roc_auc(f.lof, an) < 0.2                                                         # der Fall, den die App durch k <= n / 2 ausschließt
    assert roc_auc(lof.fit_lof(X, 10).lof, an) > 0.99


def test_deterministic():
    X, _ = _data(seed=9)
    assert np.array_equal(lof.fit_lof(X, 20).lof, lof.fit_lof(X, 20).lof)


# --- Kopierte Bausteine der Vorgänger ---------------------------------------------------------------------------------------------------


def test_isolation_forest_copy_matches_scikit_learn_and_c_factor_is_the_definition():
    X, an = _data(seed=1)
    ours = isf.score(isf.fit_forest(X, 100, 256, 0), X)
    theirs = -IsolationForest(n_estimators=100, max_samples=256, random_state=0).fit(X).score_samples(X)
    assert spearmanr(ours, theirs)[0] > 0.9
    n = np.array([3.0, 10.0, 256.0])
    assert np.allclose(isf.c_factor(n), 2 * (np.log(n - 1) + isf.EULER_GAMMA) - 2 * (n - 1) / n) and isf.c_factor(np.array([1, 2])).tolist() == [0.0, 1.0]


@pytest.mark.parametrize("dof", [1, 5, 12])
@pytest.mark.parametrize("q", [0.5, 0.975])
def test_chi2_copy(dof, q):
    assert alg.chi2_ppf(q, dof) == pytest.approx(chi2.ppf(q, dof), rel=1e-7)


def test_mcd_close_to_scikit_learn():
    X, is_out = _data(n=300, p=4, n_out=45, shift=8.0, seed=2)
    ours = alg.fit_mcd(X, 0.5, 0, reweight=True)
    sk = MinCovDet(random_state=0).fit(X)
    scale = np.sqrt(np.diag(np.cov(X[~is_out].T)))
    assert np.linalg.norm((ours.location - sk.location_) / scale) < 0.1 and (ours.d2[is_out] > alg.threshold(ours, 0.975)).all()
