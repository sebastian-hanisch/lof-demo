"""Unabhängiges Orakel für den LOF: Schleifen-Rechnung direkt nach der Definition (k-Distanz, Erreichbarkeitsdistanz, lrd, LOF) und scikit-learn
(`_lrd`, k-Distanz, LOF, novelty) auf vielen Zufallsinstanzen, auch mit Duplikaten und ganzzahligen Gitterpunkten (Abstandsgleichstände); k wird auf 1 ... n - 1 begrenzt."""

import math

import numpy as np
import pytest

import lof_algorithm as lof

neighbors = pytest.importorskip("sklearn.neighbors")


def _loop_lof(X, k):
    """Definition mit Python-Schleifen; Nachbarn = die k kleinsten Abstände (Gleichstand: kleinster Index, wie stabiles Sortieren)."""
    n = len(X)
    d = [[math.dist(X[i], X[j]) for j in range(n)] for i in range(n)]
    nb, kd = [], []
    for i in range(n):
        cand = sorted((d[i][j], j) for j in range(n) if j != i)[:k]
        nb.append([j for _, j in cand])
        kd.append(cand[-1][0])
    lrd = [1.0 / (sum(max(kd[j], d[i][j]) for j in nb[i]) / k + 1e-10) for i in range(n)]
    lof = [sum(lrd[j] for j in nb[i]) / k / lrd[i] for i in range(n)]
    return np.array(lof), np.array(lrd), np.array(kd)


def _instances(count=60, seed=12345):
    rng = np.random.default_rng(seed)
    for it in range(count):
        n, p = int(rng.integers(3, 35)), int(rng.integers(1, 5))
        kind = it % 4
        if kind == 0:
            X = rng.standard_normal((n, p))
        elif kind == 1:
            X = rng.integers(0, 4, (n, p)).astype(float)                       # Duplikate und Abstandsgleichstände
        elif kind == 2:
            X = np.concatenate([rng.standard_normal((n // 2 + 1, p)) * 0.2, rng.standard_normal((n - n // 2 - 1, p)) * 3 + 5])
        else:
            X = rng.standard_normal((n, p)) * 10 ** rng.uniform(-3, 3)
        yield rng, X, int(rng.integers(1, n + 3))                               # k bis über n hinaus: Begrenzung


def test_lof_equals_the_loop_definition_also_with_duplicates_and_ties():
    for _, X, k in _instances():
        kk = min(max(k, 1), len(X) - 1)
        f = lof.fit_lof(X, k)
        assert f.k == kk
        ref_lof, ref_lrd, ref_kd = _loop_lof(X, kk)
        assert np.allclose(f.lof, ref_lof, rtol=1e-7, atol=1e-7)
        assert np.allclose(f.lrd, ref_lrd, rtol=1e-7, atol=1e-7)
        assert np.allclose(f.kdist, ref_kd, rtol=1e-9, atol=1e-9)


def test_lof_lrd_kdist_and_novelty_equal_scikit_learn_without_a_tie_at_the_k_th_neighbour():
    checked = 0
    for rng, X, k in _instances():
        n = len(X)
        kk = min(max(k, 1), n - 1)
        D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)) + np.diag(np.full(n, np.inf))
        Ds = np.sort(D, axis=1)
        if kk < n - 1 and np.any(np.isclose(Ds[:, kk - 1], Ds[:, kk])):
            continue                                                            # welcher gleich weite Nachbar zählt, entscheiden die Bibliotheken verschieden
        f = lof.fit_lof(X, k)
        sk = neighbors.LocalOutlierFactor(n_neighbors=kk).fit(X)
        assert np.allclose(f.lof, -sk.negative_outlier_factor_, rtol=1e-6, atol=1e-6)
        assert np.allclose(f.lrd, sk._lrd, rtol=1e-6, atol=1e-6)
        assert np.allclose(f.kdist, sk._distances_fit_X_[:, -1], atol=1e-9)
        assert np.array_equal(np.sort(f.neighbors, axis=1), np.sort(sk.kneighbors(return_distance=False), axis=1))       # dieselben k Nachbarn je Punkt
        Q = rng.standard_normal((6, X.shape[1])) * 2
        Dq = np.sort(np.sqrt(((Q[:, None] - X[None]) ** 2).sum(-1)), axis=1)
        if kk < n and np.any(np.isclose(Dq[:, kk - 1], Dq[:, kk])):
            continue
        theirs = -neighbors.LocalOutlierFactor(n_neighbors=kk, novelty=True).fit(X).score_samples(Q)
        assert np.allclose(lof.score_new(X, f, Q), theirs, rtol=1e-6, atol=1e-6)
        checked += 1
    assert checked >= 25                                                        # das Orakel prüft wirklich viele Instanzen
