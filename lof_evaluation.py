"""Auswertung der LOF-Demo: Kennzahlen der Anomalie-Erkennung (AUC, mittlere Präzision, Precision/Recall/F1, Fehlalarmrate; aus der Wurzel-Demo übernommen), Analyse einer Aufnahme für den Local Outlier Factor,
den Isolation Forest und die beiden Schätzer der Wurzel (klassisch, robust), Sweeps, Experimente auf Abruf, Score-Anisotropie und Urteil."""

import time
from dataclasses import dataclass

import numpy as np

import lof_algorithm as lof
import lof_isolation_forest as isf
import lof_ee_algorithm as alg
import lof_constants as C
import lof_scenario as sc


# --- Kennzahlen -----------------------------------------------------------------------------------------------------------------


def roc_auc(score, positive):
    """Fläche unter der ROC-Kurve über die Rangsumme (Mann-Whitney), Bindungen zählen halb. NaN, wenn eine Klasse fehlt."""
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = np.asarray(score)[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(score, positive):
    """Mittlere Präzision (Fläche unter der Precision-Recall-Kurve als Summe über die Treffer). NaN ohne Anomalien."""
    positive = np.asarray(positive, dtype=bool)
    if not positive.any():
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = positive[order]
    precision_at = np.cumsum(hits) / (np.arange(len(hits)) + 1.0)
    return float(precision_at[hits].sum() / positive.sum())


def flag_metrics(flagged, positive):
    """Precision, Recall, F1 und Fehlalarmrate (Anteil der Normalen, die markiert werden). Ohne Anomalien: Recall/F1 NaN; ohne Markierung: Precision 1 (nichts falsch)."""
    flagged, positive = np.asarray(flagged, bool), np.asarray(positive, bool)
    tp = int((flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    recall = tp / n_pos if n_pos else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if n_pos == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if n_pos and (precision + recall) > 0 else (float("nan") if not n_pos else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "false_alarm": fp / n_neg if n_neg else float("nan"), "n_flagged": int(flagged.sum())}


def roc_curve(score, positive):
    """ROC-Kurve: (Fehlalarmrate, Trefferquote) für alle Schwellen, von (0, 0) bis (1, 1)."""
    positive = np.asarray(positive, bool)
    order = np.argsort(-np.asarray(score), kind="mergesort")
    hits = positive[order]
    tpr = np.concatenate([[0.0], np.cumsum(hits) / max(hits.sum(), 1)])
    fpr = np.concatenate([[0.0], np.cumsum(~hits) / max((~hits).sum(), 1)])
    return fpr, tpr


# --- Analyse einer Aufnahme ---------------------------------------------------------------------------------------------------------


def standardise(X):
    """Kennzahlen auf Mittelwert 0 und Streuung 1: LOF rechnet mit euklidischen Abständen, ihre Einheiten (Meter gegen Prozent) würden sonst die Nachbarschaft bestimmen."""
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)


def k_max(n):
    """Größte Nachbarzahl, die die App anbietet: höchstens n / 2 (bei k nahe n sehen alle Touren dieselben Nachbarn; LOF wird unbrauchbar) und höchstens K_MAX."""
    return int(max(C.K_MIN + 1, min(C.K_MAX, n // 2)))


@dataclass(frozen=True)
class Settings:
    k: int = C.DEFAULT_K                                # Nachbarn des LOF
    threshold_kind: str = C.DEFAULT_THRESHOLD_KIND      # "standard": LOF-Schwelle, Score 0.5 (Isolation Forest), chi²-Quantil (klassisch, robust); "share": der erwartete Anteil für alle vier
    cutoff: float = C.DEFAULT_CUTOFF                    # LOF-Schwelle
    quantile: float = C.DEFAULT_QUANTILE
    share: int = C.DEFAULT_SHARE                        # erwarteter Anteil der Anomalien [%]
    start: int = 0                                      # Seed des Isolation Forest und der MCD-Starts (entkoppelt vom Seed der Aufnahme)
    standardize: bool = True                            # LOF arbeitet auf standardisierten Kennzahlen (der Isolation Forest ist skaleninvariant, LOF nicht)


DATA_KEYS = ("n", "p", "n_noise", "n_modes", "curvature", "noise", "contamination", "kind", "strength")
DEFAULT_DATA = dict(n=C.DEFAULT_N_TOURS, p=C.DEFAULT_P, n_noise=C.DEFAULT_N_NOISE, n_modes=C.DEFAULT_N_MODES, curvature=C.DEFAULT_CURVATURE, noise=C.DEFAULT_NOISE,
                    contamination=C.DEFAULT_CONTAMINATION, kind=C.DEFAULT_KIND, strength=C.DEFAULT_STRENGTH)
DETECTORS = ("lof", "iforest", "classical", "robust")
DETECTOR_NAMES = {"lof": "LOF", "iforest": "Isolation Forest", "classical": "klassisch", "robust": "robust (MCD)"}
METRICS = ("auc", "ap", "precision", "recall", "f1", "false_alarm")
IF_TREES, IF_PSI, IF_CUTOFF = C.DEFAULT_TREES, C.DEFAULT_PSI, C.DEFAULT_CUTOFF_IF


def make_dataset(n=C.DEFAULT_N_TOURS, p=C.DEFAULT_P, n_noise=C.DEFAULT_N_NOISE, n_modes=C.DEFAULT_N_MODES, curvature=C.DEFAULT_CURVATURE, noise=C.DEFAULT_NOISE,
                 contamination=C.DEFAULT_CONTAMINATION, kind=C.DEFAULT_KIND, strength=C.DEFAULT_STRENGTH, seed=C.DEFAULT_SEED):
    return sc.generate_dataset(n, p, n_modes, curvature, noise, contamination, kind, strength, seed, n_noise)


@dataclass(frozen=True)
class Analysis:
    ds: sc.Dataset
    settings: Settings
    params: tuple                 # (n, p, n_noise, n_modes, curvature, noise, contamination, kind, strength, seed)
    k: int                        # verwendete Nachbarzahl (auf 1 ... n - 1 begrenzt)
    lof_fit: lof.Fit
    forest_if: isf.Forest
    paths_if: np.ndarray          # (Bäume, n) Pfadlängen
    values: dict                  # Detektor -> Anomalie-Wert je Tour (LOF: LOF, Isolation Forest: Score, klassisch/robust: quadrierter Mahalanobis-Abstand)
    classical: alg.Fit
    robust: alg.Fit
    scores: dict                  # Detektor -> Kennzahlen (auc, ap, precision, recall, f1, false_alarm, n_flagged, threshold)
    flags: dict                   # Detektor -> markierte Touren bei der gewählten Schwelle
    oracle_f1: dict               # Detektor (lof, iforest) -> F1, wenn der wahre Anteil bekannt wäre (die k größten Werte)
    seconds: dict


_ROOT_CACHE = {}


def _root_fits(X, key, start):
    """Klassische und robuste Schätzung der Wurzel (unabhängig von den LOF-Reglern: werden je Aufnahme nur einmal gerechnet)."""
    cache_key = (key, start)
    if key is not None and cache_key in _ROOT_CACHE:
        return _ROOT_CACHE[cache_key]
    t0 = time.perf_counter()
    classical = alg.fit_classical(X)
    t1 = time.perf_counter()
    robust = alg.fit_mcd(X, C.DEFAULT_SUPPORT, start, reweight=C.DEFAULT_REWEIGHT)
    out = (classical, robust, t1 - t0, time.perf_counter() - t1)
    if key is not None:
        if len(_ROOT_CACHE) > 600:
            _ROOT_CACHE.clear()
        _ROOT_CACHE[cache_key] = out
    return out


def _thresholds(values, settings, classical, robust):
    """Markierung und Schwellenwert je Detektor: Standard = LOF-Schwelle, Score 0.5 bzw. chi²-Quantil, sonst die k größten Werte mit dem angenommenen Anteil."""
    flags, thr = {}, {}
    if settings.threshold_kind == "share":
        for d in DETECTORS:
            flags[d] = isf.flag_top(values[d], settings.share / 100.0)
            thr[d] = float(np.sort(values[d])[::-1][int(flags[d].sum()) - 1])
    else:
        thr["lof"] = settings.cutoff
        thr["iforest"] = IF_CUTOFF
        thr["classical"] = alg.threshold(classical, settings.quantile)
        thr["robust"] = alg.threshold(robust, settings.quantile)
        for d in DETECTORS:
            flags[d] = values[d] > thr[d]
    return flags, thr


def analyse(ds, settings=Settings(), params=None, root_key=None):
    secs = {}
    X = ds.X
    k = lof.clamp_k(settings.k, len(X))
    Xl = standardise(X) if settings.standardize else X
    t0 = time.perf_counter()
    lof_fit = lof.fit_lof(Xl, k)
    secs["lof"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    forest_if = isf.fit_forest(X, IF_TREES, IF_PSI, settings.start)
    paths_if = isf.path_lengths(forest_if, X)
    secs["iforest"] = time.perf_counter() - t0
    classical, robust, secs["classical"], secs["robust"] = _root_fits(X, root_key, settings.start)
    values = {"lof": lof_fit.lof, "iforest": isf.score_from_paths(paths_if, forest_if.psi), "classical": classical.d2, "robust": robust.d2}
    flags, thr = _thresholds(values, settings, classical, robust)
    scores = {}
    for d in DETECTORS:
        m = flag_metrics(flags[d], ds.anomaly)
        m.update(auc=roc_auc(values[d], ds.anomaly), ap=average_precision(values[d], ds.anomaly), threshold=thr[d])
        scores[d] = m
    oracle = {d: flag_metrics(isf.flag_top(values[d], ds.anomaly.mean()), ds.anomaly)["f1"] for d in ("lof", "iforest")}
    return Analysis(ds, settings, params, k, lof_fit, forest_if, paths_if, values, classical, robust, scores, flags, oracle, secs)


def analyse_for(params, settings=Settings()):
    """`params` = (n, p, n_noise, n_modes, curvature, noise, contamination, kind, strength, seed)."""
    return analyse(make_dataset(*params), settings, params, root_key=params)


# --- Score-Karten -----------------------------------------------------------------------------------------------------------------


def _map_scorers(X2, settings):
    """Bewerter für neue Punkte auf zwei Merkmalen ohne Anomalien: (LOF mit standardisierten Merkmalen, Isolation Forest auf den Rohmerkmalen)."""
    mu, sd = X2.mean(axis=0), X2.std(axis=0)
    fit = lof.fit_lof((X2 - mu) / sd, lof.clamp_k(settings.k, len(X2)))
    forest = isf.fit_forest(X2, IF_TREES, min(IF_PSI, len(X2)), settings.start)
    return (lambda G: lof.score_new((X2 - mu) / sd, fit, (G - mu) / sd)), (lambda G: isf.score(forest, G))


def score_maps(X2, settings=Settings(), grid=80, pad=0.5):
    """Score-Karten beider Detektoren über zwei Merkmale (nur auf diesen zwei Merkmalen): (x-Achse, y-Achse, LOF [y, x], Score des Isolation Forest [y, x])."""
    lo, hi = X2.min(axis=0), X2.max(axis=0)
    span = hi - lo
    xs = np.linspace(lo[0] - pad * span[0], hi[0] + pad * span[0], grid)
    ys = np.linspace(lo[1] - pad * span[1], hi[1] + pad * span[1], grid)
    G = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    f_lof, f_if = _map_scorers(X2, settings)
    return xs, ys, f_lof(G).reshape(grid, grid), f_if(G).reshape(grid, grid)


# --- Sweeps und Experimente -----------------------------------------------------------------------------------------------------------

SWEEP_VALUES = {
    "k": (5, 10, 20, 40, 80),
    "n": (20, 30, 50, 100, 200, 400, 600),
    "p": (2, 5, 8, 12, 20, 30),
    "n_noise": (0, 5, 10, 20, 30, 40),
    "n_modes": (1, 2, 3),
    "curvature": (0.0, 0.25, 0.5, 0.75, 1.0),
    "noise": (0.0, 0.25, 0.5, 0.75, 1.0),
    "contamination": (2, 5, 10, 20, 30, 40, 45),
    "strength": (3.0, 4.0, 6.0, 9.0, 12.0),
    "cutoff": (1.1, 1.2, 1.3, 1.5, 2.0, 2.5),
    "quantile": (0.9, 0.95, 0.975, 0.99, 0.999),
    "share": (2, 5, 10, 20, 40),
}
SWEEP_LABELS = {"k": "Nachbarn k", "n": "Anzahl Touren", "p": "Anzahl Merkmale", "n_noise": "Anzahl Rauschmerkmale", "n_modes": "Anzahl Betriebsarten", "curvature": "Krümmung des Normalbereichs",
                "noise": "Rauschen", "contamination": "Anteil der Anomalien [%]", "strength": "Abstand der Anomalien (Faktor-σ)", "cutoff": "LOF-Schwelle",
                "quantile": "chi²-Quantil der Schwelle (klassisch, robust)", "share": "angenommener Anteil der Anomalien [%]"}
SETTING_PARAMETERS = ("k", "cutoff", "quantile", "share")


def _record(a):
    out = {f"{d}_{k}": a.scores[d][k] for d in DETECTORS for k in METRICS}
    out["n_anomalies"] = float(a.ds.anomaly.sum())
    out["lof_oracle_f1"], out["iforest_oracle_f1"] = a.oracle_f1["lof"], a.oracle_f1["iforest"]
    out["lof_seconds"], out["iforest_seconds"] = a.seconds["lof"], a.seconds["iforest"]
    out["lof_normal_median"] = float(np.median(a.values["lof"][~a.ds.anomaly]))
    return out


def _summarise(x, per_seed):
    row = {"x": x}
    for key in per_seed[0]:
        arr = np.array([r[key] for r in per_seed], dtype=float)
        ok = not np.isnan(arr).all()
        row[key] = float(np.nanmean(arr)) if ok else float("nan")
        row[key + "_std"] = float(np.nanstd(arr)) if ok else float("nan")
        row[key + "_min"] = float(np.nanmin(arr)) if ok else float("nan")
        row[key + "_max"] = float(np.nanmax(arr)) if ok else float("nan")
    return row


def _analyse_seed(seed, settings, kw):
    data = {**DEFAULT_DATA, **kw}
    ds = make_dataset(seed=seed, **data)
    return analyse(ds, settings, None, root_key=(tuple(sorted(data.items())), seed))


def _mean_over_seeds(settings=Settings(), seeds=C.SWEEP_SEEDS, **kw):
    """Mittel (mit Streuung und Spanne) aller Kennzahlen über die festen Sweep-Datensätze für eine Datenkonfiguration."""
    return _summarise(None, [_record(_analyse_seed(s, settings, kw)) for s in seeds])


def sweep(parameter, values=None, settings=Settings(), **base):
    """Mittel, Streuung und Spanne der Kennzahlen der vier Detektoren über die festen Sweep-Datensätze in Abhängigkeit von einem Regler (alle anderen wie in `base`)."""
    values = SWEEP_VALUES[parameter] if values is None else values
    rows = []
    for x in values:
        if parameter in SETTING_PARAMETERS:
            row = _mean_over_seeds(Settings(**{**settings.__dict__, parameter: x}), **base)
        elif parameter == "n":                                                   # k folgt der Tourenzahl (höchstens n / 2), sonst wäre LOF bei n = 20 mit k = 20 unbrauchbar
            row = _mean_over_seeds(Settings(**{**settings.__dict__, "k": min(settings.k, k_max(x))}), **{**base, "n": x})
        else:
            row = _mean_over_seeds(settings, **{**base, parameter: x})
        row["x"] = x
        rows.append(row)
    return rows


def _lof_only(seed, k, kw, standardize=True):
    """LOF allein (ohne Wald und Wurzel) auf einer Aufnahme: (AUC, Gruppengröße)."""
    ds = make_dataset(seed=seed, **{**DEFAULT_DATA, **kw})
    X = standardise(ds.X) if standardize else ds.X
    return roc_auc(lof.fit_lof(X, k).lof, ds.anomaly), int(ds.anomaly.sum())


GROUP_CONTAMINATION = (5, 10, 20, 30)
GROUP_K = (10, 20, 40, 80, 100)


def group_table(settings=Settings(), **base):
    """LOF gegen die Größe der Anomaliegruppe: AUC über Anteil der dichten Gruppe (Zeilen) und Nachbarzahl k (Spalten); dazu Isolation Forest und robuste Schätzung je Anteil. LOF sieht die Gruppe nur, wenn k größer als ihre Größe ist."""
    cells, baselines = [], []
    for c in GROUP_CONTAMINATION:
        kw = {**base, "kind": "cluster", "n_modes": 1, "contamination": c}
        for k in GROUP_K:
            res = [_lof_only(s, k, kw) for s in C.SWEEP_SEEDS]
            cells.append({"contamination": c, "k": k, "lof_auc": float(np.mean([r[0] for r in res])), "group": float(np.mean([r[1] for r in res]))})
        row = _mean_over_seeds(settings, **kw)
        baselines.append({"contamination": c, "iforest_auc": row["iforest_auc"], "robust_auc": row["robust_auc"], "group": row["n_anomalies"]})
    return {"cells": cells, "baselines": baselines}


GAP_K = (20, 30, 40, 50, 60, 70, 90)


def gap_table(settings=Settings(), **base):
    """Anomalien in der Lücke: AUC des LOF über k für zwei und drei Betriebsarten (10 %) und zwei (5 %), dazu Isolation Forest, klassisch und robust."""
    rows = []
    for n_modes, c in ((2, 10), (3, 10), (2, 5)):
        kw = {**base, "kind": "gap", "n_modes": n_modes, "contamination": c}
        by_k = {k: float(np.mean([_lof_only(s, k, kw)[0] for s in C.SWEEP_SEEDS])) for k in GAP_K}
        row = _mean_over_seeds(settings, **kw)
        rows.append({"n_modes": n_modes, "contamination": c, "by_k": by_k, "iforest_auc": row["iforest_auc"], "classical_auc": row["classical_auc"], "robust_auc": row["robust_auc"], "group": row["n_anomalies"]})
    return rows


def modes_table(settings=Settings(), **base):
    """Betriebsarten × Art der Anomalien: AUC der vier Detektoren (die Art 'Lücke' gibt es erst ab zwei Betriebsarten)."""
    rows = []
    for n_modes in (1, 2, 3):
        for kind in C.KINDS:
            if kind == "gap" and n_modes == 1:
                continue
            row = _mean_over_seeds(settings, **{**base, "n_modes": n_modes, "kind": kind})
            row.update(n_modes=n_modes, kind=kind)
            rows.append(row)
    return rows


MASKING_CONTAMINATION = (2, 5, 10, 15, 20, 25, 30, 35, 40, 45)


def masking_table(settings=Settings(), **base):
    """Dichte Gruppe abseits: AUC der vier Detektoren über den Anteil (LOF mit dem Standard-k), dazu LOF mit einem k passend zur Gruppengröße (1.5-fache Gruppengröße, höchstens n / 2)."""
    rows = []
    for c in MASKING_CONTAMINATION:
        kw = {**base, "kind": "cluster", "contamination": c, "n_modes": 1}
        row = _mean_over_seeds(settings, **kw)
        n = kw.get("n", C.DEFAULT_N_TOURS)
        matched = [_lof_only(s, min(k_max(n), int(round(1.5 * make_dataset(seed=s, **{**DEFAULT_DATA, **kw}).anomaly.sum()))), kw)[0] for s in C.SWEEP_SEEDS]
        row["x"] = c
        row["lof_matched_auc"] = float(np.mean(matched))
        rows.append(row)
    return rows


def k_table(settings=Settings(), **base):
    """k gegen die Tourenzahl: AUC des LOF über k bis n - 1 bei n = 100, verstreute Anomalien und eine dichte Gruppe von 20 %. Bei k nahe n sehen alle Touren fast dieselben Nachbarn."""
    n = 100
    ks = (5, 10, 20, 30, 50, 70, 90, 99)
    rows = []
    for kind, c in (("scattered", 10), ("cluster", 20)):
        kw = {**base, "n": n, "kind": kind, "contamination": c, "n_modes": 1}
        rows.append({"kind": kind, "contamination": c, "by_k": {k: float(np.mean([_lof_only(s, k, kw)[0] for s in C.SWEEP_SEEDS])) for k in ks}})
    return rows


DIMENSION_N = (20, 30, 50, 100, 200, 400)
DIMENSION_P = (2, 5, 12, 20, 30)


def dimension_table(settings=Settings(), **base):
    """Hohe Dimension: F1 der vier Detektoren bei der Standardschwelle über Tourenzahl n (Zeilen) und Merkmalszahl p (Spalten); k folgt der Tourenzahl (höchstens n / 2)."""
    cells = []
    for n in DIMENSION_N:
        for p in DIMENSION_P:
            row = _mean_over_seeds(Settings(**{**settings.__dict__, "k": min(settings.k, k_max(n))}), **{**base, "n": n, "p": p})
            row.update(n=n, p=p)
            cells.append(row)
    return cells


def threshold_table(settings=Settings(), **base):
    """Schwelle: (1) Kennzahlen über die LOF-Schwelle; (2) F1 aller vier Detektoren bei ½-, 1- und 2-fach angenommenem Anteil; (3) der mittlere LOF der normalen Touren je Tourenzahl (k = Standard, höchstens n / 2)."""
    cut = sweep("cutoff", settings=Settings(**{**settings.__dict__, "threshold_kind": "standard"}), **base)
    true_share = base.get("contamination", C.DEFAULT_CONTAMINATION)
    wrong = []
    for factor in (0.5, 1.0, 2.0):
        row = _mean_over_seeds(Settings(**{**settings.__dict__, "threshold_kind": "share", "share": int(round(true_share * factor))}), **base)
        row["x"], row["factor"] = int(round(true_share * factor)), factor
        wrong.append(row)
    normal = []
    for n in (20, 50, 100, 300, 600):
        s_n = Settings(**{**settings.__dict__, "k": min(settings.k, k_max(n))})
        meds = [np.median(_analyse_seed(seed, s_n, {**base, "n": n}).values["lof"][~make_dataset(seed=seed, **{**DEFAULT_DATA, **base, "n": n}).anomaly]) for seed in C.SWEEP_SEEDS]
        normal.append({"n": n, "median": float(np.mean(meds))})
    return {"cutoff": cut, "wrong_share": wrong, "normal_scores": normal}


def cost_table(settings=Settings(), **base):
    """Rechenzeit (LOF gegen Isolation Forest) über die Tourenzahl, und LOF auf Rohdaten gegen standardisierte Kennzahlen (AUC und F1 bei der Schwelle)."""
    times = []
    for n in (100, 300, 600):
        rows = [_analyse_seed(seed, Settings(**{**settings.__dict__, "k": min(settings.k, k_max(n))}), {**base, "n": n}) for seed in C.SWEEP_SEEDS[:3]]
        times.append({"n": n, "lof": float(np.mean([a.seconds["lof"] for a in rows])), "iforest": float(np.mean([a.seconds["iforest"] for a in rows]))})
    units = []
    for standardize in (True, False):
        row = _mean_over_seeds(Settings(**{**settings.__dict__, "standardize": standardize}), **base)
        row["standardize"] = standardize
        units.append(row)
    return {"times": times, "units": units}


# --- Urteil ------------------------------------------------------------------------------------------------------------------------

WIN_MARGIN = 0.05             # AUC-Abstand, ab dem ein Detektor als besser gilt
GAP_AUC = 0.8
THRESHOLD_F1_DROP = 0.15
F1_MARGIN = 0.10              # F1-Abstand bei der Schwelle zum Isolation Forest bei gleicher Rangfolge


def verdict(a):
    """(Art, Code, Kennzahlen): Lücke (keiner findet sie), k-Fenster (LOF scheitert an k gegen Gruppengröße), andere besser, LOF besser, falsche Schwelle bei guter Rangfolge, sonst gleichauf."""
    ds = a.ds
    l = a.scores["lof"]
    others = {d: a.scores[d]["auc"] for d in ("iforest", "classical", "robust")}
    best_other = max(others.values())
    m = int(ds.anomaly.sum())
    data = {"n": ds.n, "p": ds.p + ds.n_noise, "n_noise": ds.n_noise, "n_modes": ds.n_modes, "kind": ds.kind, "contamination": 100.0 * ds.anomaly.mean(), "n_anomalies": m, "k": a.k,
            "oracle_lof": a.oracle_f1["lof"], "oracle_if": a.oracle_f1["iforest"], "best_other_auc": best_other, **{f"{d}_{k}": v for d in DETECTORS for k, v in a.scores[d].items()}}
    if ds.kind == "gap" and max(best_other, l["auc"]) < GAP_AUC:
        return "warning", "gap", data
    if ds.kind in ("cluster", "gap") and best_other - l["auc"] >= WIN_MARGIN and (a.k <= m or a.k >= ds.n - m):
        return "warning", "k_window", data
    if best_other - l["auc"] >= WIN_MARGIN:
        return "warning", "others_win", data
    if l["auc"] - best_other >= WIN_MARGIN:
        return "success", "lof_wins", data
    if l["auc"] >= 0.95 and a.oracle_f1["lof"] - l["f1"] > THRESHOLD_F1_DROP:
        return "warning", "threshold_off", data
    d_f1 = l["f1"] - a.scores["iforest"]["f1"]
    if d_f1 >= F1_MARGIN:
        return "success", "lof_wins", data
    if -d_f1 >= F1_MARGIN:
        return "warning", "iforest_f1", data
    return "success", "comparable", data
