"""Plotly-Visualisierungen der LOF-Demo: Touren in der Ebene der größten Streuung, k-Nachbarschaft, lokale Dichte und LOF-Werte, Score-Karten (LOF gegen Isolation Forest), ROC-Kurven, Kennzahlen-Balken, Sweeps
und die Experimente (k gegen Gruppengröße, Lücke, Masking, Schwelle, Dimension, Kosten). Alle Figuren laufen durch `lock_axes`."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import lof_ee_algorithm as alg

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, TEAL, PINK = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf", "#00838f", "#c2185b"
DETECTOR_COLORS = {"lof": PINK, "iforest": PURPLE, "classical": ORANGE, "robust": TEAL}
DETECTOR_NAMES = {"lof": "LOF", "iforest": "Isolation Forest", "classical": "klassisch", "robust": "robust (MCD)"}
KIND_NAMES = {"scattered": "verstreute Ausreißer", "cluster": "dichte Gruppe abseits", "gap": "in der Lücke"}
DETECTORS = ("lof", "iforest", "classical", "robust")


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


# --- Projektion -----------------------------------------------------------------------------------------------------------------


def standardise(X):
    m, s = X.mean(axis=0), X.std(axis=0)
    s = np.where(s < 1e-12, 1.0, s)
    return (X - m) / s


def projection(X, robust):
    """Die Touren in der Ebene der zwei größten Streuungsrichtungen der robusten Kovarianz (standardisierte Kennzahlen): (Punkte n x 2, Achsen)."""
    s = np.where(X.std(axis=0) < 1e-12, 1.0, X.std(axis=0))
    axes = alg.projection_axes(robust.covariance / np.outer(s, s))
    return standardise(X) @ axes, axes


def _points(P, anomaly, flagged=None):
    normal = ~anomaly
    traces = [go.Scatter(x=P[normal, 0], y=P[normal, 1], mode="markers", marker=dict(size=6, color=BLUE, opacity=0.55), hoverinfo="skip", name="normale Touren"),
              go.Scatter(x=P[anomaly, 0], y=P[anomaly, 1], mode="markers", marker=dict(size=8, color=RED, symbol="diamond"), hoverinfo="skip", name="Sonderfahrten (Wahrheit)")]
    if flagged is not None and flagged.any():
        traces.append(go.Scatter(x=P[flagged, 0], y=P[flagged, 1], mode="markers", marker=dict(size=13, color="black", line=dict(width=2), symbol="circle-open"), hoverinfo="skip", name="als Anomalie markiert"))
    return traces


def build_scatter(P, anomaly, flagged=None, height=380):
    """Touren in der Projektionsebene (blau = normal, rote Rauten = Sonderfahrten, Kreise = als Anomalie markiert)."""
    fig = go.Figure(_points(P, anomaly, flagged))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Hauptrichtung 1 (standardisiert)", zeroline=False), yaxis=dict(title="Hauptrichtung 2", zeroline=False),
                      legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_features(X, anomaly, names, i=0, j=1):
    """Zwei Rohmerkmale gegeneinander (Einheiten wie gemessen)."""
    j = min(j, X.shape[1] - 1)
    fig = go.Figure(_points(np.stack([X[:, i], X[:, j]], axis=1), anomaly))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title=names[i], zeroline=False), yaxis=dict(title=names[j], zeroline=False), showlegend=False)
    return lock_axes(fig)


# --- Schritte: Nachbarschaft, Dichte, LOF ---------------------------------------------------------------------------------------------


def build_neighborhood(P, anomaly, picks, neighbors, height=400):
    """Die k Nachbarn zweier Touren (gemessen in ALLEN Merkmalen, gezeigt in der Ebene der zwei Hauptrichtungen): Linien von der Tour zu ihren Nachbarn, Nachbarn hervorgehoben.
    `picks` = [(Index, Name, Farbe)]; `neighbors` = [n, k] Indizes."""
    fig = go.Figure(_points(P, anomaly))
    for idx, name, color in picks:
        nb = neighbors[idx]
        xs, ys = [], []
        for j in nb:
            xs += [P[idx, 0], P[j, 0], None]
            ys += [P[idx, 1], P[j, 1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1), opacity=0.45, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=P[nb, 0], y=P[nb, 1], mode="markers", marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(color=color, width=2)), hoverinfo="skip", name=f"Nachbarn: {name}"))
        fig.add_trace(go.Scatter(x=[P[idx, 0]], y=[P[idx, 1]], mode="markers", marker=dict(size=16, color=color, line=dict(width=3), symbol="circle-open"), hoverinfo="skip", name=name))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Hauptrichtung 1", zeroline=False, range=[P[:, 0].min() - 0.5, P[:, 0].max() + 0.5]),
                      yaxis=dict(title="Hauptrichtung 2", zeroline=False, range=[P[:, 1].min() - 0.5, P[:, 1].max() + 0.5]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_density_plane(P, lrd, anomaly, height=360):
    """Lokale Erreichbarkeitsdichte je Tour (Farbe, logarithmisch; dunkel = dünn besetzt) in der Ebene der zwei Hauptrichtungen; Sonderfahrten als Rauten mit Rand."""
    z = np.log10(lrd)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=P[~anomaly, 0], y=P[~anomaly, 1], mode="markers", marker=dict(size=7, color=z[~anomaly], colorscale="Viridis", cmin=float(z.min()), cmax=float(z.max()), showscale=True,
                                                                                            colorbar=dict(title="log10 lrd", thickness=12)), hoverinfo="skip", name="normale Touren"))
    fig.add_trace(go.Scatter(x=P[anomaly, 0], y=P[anomaly, 1], mode="markers", marker=dict(size=9, color=z[anomaly], colorscale="Viridis", cmin=float(z.min()), cmax=float(z.max()), symbol="diamond",
                                                                                          line=dict(color=RED, width=2)), hoverinfo="skip", name="Sonderfahrten"))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Hauptrichtung 1", zeroline=False), yaxis=dict(title="Hauptrichtung 2", zeroline=False), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_lrd_bars(entries):
    """Zwei Touren: eigene lokale Dichte gegen die mittlere Dichte ihrer Nachbarn (LOF = Verhältnis). `entries` = [(Name, Farbe, lrd, mittlere lrd der Nachbarn, LOF)]."""
    fig = go.Figure()
    labels = [e[0] for e in entries]
    fig.add_trace(go.Bar(x=labels, y=[e[2] for e in entries], name="eigene Dichte lrd", marker_color=[e[1] for e in entries], text=[f"{e[2]:.2f}" for e in entries], textposition="outside", hoverinfo="skip"))
    fig.add_trace(go.Bar(x=labels, y=[e[3] for e in entries], name="mittlere Dichte der Nachbarn", marker_color=GRAY, text=[f"{e[3]:.2f}" for e in entries], textposition="outside", hoverinfo="skip"))
    for i, e in enumerate(entries):
        fig.add_annotation(x=labels[i], y=max(e[2], e[3]) * 1.25, text=f"LOF = {e[4]:.2f}", showarrow=False, font=dict(size=12))
    top = max(max(e[2], e[3]) for e in entries) * 1.5
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, top], title="Dichte (1 / Erreichbarkeitsdistanz)"), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_lof_hist(values, anomaly, thr):
    """Histogramm des LOF (normale Touren blass, Sonderfahrten kräftig); senkrechte Linie = Schwelle. Normale Touren liegen bei ~1."""
    hi = float(min(max(values.max(), thr * 1.05), 8.0))
    lo = float(min(values.min(), 0.8))
    bins = dict(start=lo, end=hi, size=(hi - lo) / 50.0)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values[~anomaly], xbins=bins, marker_color=BLUE, opacity=0.55, name="normale Touren", hoverinfo="skip"))
    if anomaly.any():
        fig.add_trace(go.Histogram(x=values[anomaly], xbins=bins, marker_color=RED, opacity=0.9, name="Sonderfahrten", hoverinfo="skip"))
    fig.add_vline(x=float(thr), line=dict(color="black", dash="dash"), annotation_text="Schwelle", annotation_position="top")
    fig.add_vline(x=1.0, line=dict(color=GRAY, dash="dot"))
    fig.update_layout(height=320, barmode="overlay", margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(title=f"LOF (Werte über {hi:.1f} abgeschnitten)" if values.max() > hi else "LOF (1 = so dicht wie die Nachbarn)", range=[lo, hi]),
                      yaxis=dict(title="Touren"), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_roc(curves):
    """ROC-Kurven (Fehlalarmrate gegen Trefferquote) der vier Detektoren; `curves` = {Detektor: (fpr, tpr, auc)}."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"), hoverinfo="skip", showlegend=False))
    for name, (fpr, tpr, auc) in curves.items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=DETECTOR_COLORS[name], width=3), name=f"{DETECTOR_NAMES[name]} (AUC {auc:.2f})", hoverinfo="skip"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Fehlalarmrate", range=[0, 1]), yaxis=dict(title="Trefferquote", range=[0, 1.02]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_method_bars(scores):
    """Kennzahlen der vier Detektoren nebeneinander: AUC, Recall, Precision, Fehlalarmrate (bei der gewählten Schwelle)."""
    keys = (("auc", "AUC"), ("recall", "Recall"), ("precision", "Precision"), ("false_alarm", "Fehlalarmrate"))
    fig = go.Figure()
    for det in DETECTORS:
        y = [scores[det][k] for k, _ in keys]
        fig.add_trace(go.Bar(x=[lab for _, lab in keys], y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=9), hoverinfo="skip"))
    fig.update_layout(height=340, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.15]), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


# --- Sweeps und Experimente ----------------------------------------------------------------------------------------------------------


def _band(fig, xs, rows, key, color, col):
    y, sd = np.array([r[key] for r in rows]), np.array([r[key + "_std"] for r in rows])
    fig.add_trace(go.Scatter(x=list(xs) + list(xs)[::-1], y=list(np.nan_to_num(y + sd)) + list(np.nan_to_num(y - sd))[::-1], fill="toself", fillcolor=color, opacity=0.13, line=dict(width=0), hoverinfo="skip",
                             showlegend=False), row=1, col=col)


def build_sweep(rows, xlabel, current=None):
    """Links AUC der vier Detektoren (mit Streuung über die Sweep-Datensätze), rechts F1 (durchgezogen) und Fehlalarmrate (gestrichelt) bei der gewählten Schwelle."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("AUC (Rangfolge)", "F1 und Fehlalarmrate bei der Schwelle"), horizontal_spacing=0.12)
    xs = [r["x"] for r in rows]
    for det in DETECTORS:
        color = DETECTOR_COLORS[det]
        _band(fig, xs, rows, f"{det}_auc", color, 1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_auc"] for r in rows], mode="lines+markers", name=DETECTOR_NAMES[det], line=dict(color=color, width=3), hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_f1"] for r in rows], mode="lines+markers", line=dict(color=color, width=3), hoverinfo="skip", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_false_alarm"] for r in rows], mode="lines+markers", line=dict(color=color, width=2, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=2)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(range=[0, 1.05])
    if current is not None:
        for col in (1, 2):
            fig.add_vline(x=current, line=dict(color=RED, dash="dash"), row=1, col=col)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_score_maps(xs, ys, S_lof, S_if, X2, names, ellipse=None):
    """Score-Karten beider Detektoren über zwei Merkmale nebeneinander (je eigene Farbskala; dunkler = auffälliger) mit den Touren. Beim Isolation Forest (rechts) liegen Bänder entlang der Achsen
    (Geister-Regionen), beim LOF nicht. Optional die Ellipse der robusten Schätzung (gestrichelt)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("LOF (Nachbarn, euklidisch)", "Isolation Forest (achsenparallel)"), horizontal_spacing=0.10, shared_yaxes=True)
    cap = float(min(S_lof.max(), 6.0))
    for col, S, lo, hi, title in ((1, np.minimum(S_lof, cap), 1.0, cap, "LOF"), (2, S_if, float(S_if.min()), float(S_if.max()), "Score")):
        fig.add_trace(go.Heatmap(x=xs, y=ys, z=S, colorscale="Viridis", reversescale=True, zmin=lo, zmax=hi, showscale=True, colorbar=dict(title=title, thickness=10, x=0.455 if col == 1 else 1.0), hoverinfo="skip"),
                      row=1, col=col)
        fig.add_trace(go.Scatter(x=X2[:, 0], y=X2[:, 1], mode="markers", marker=dict(size=3, color="white", line=dict(color="black", width=0.4)), hoverinfo="skip", showlegend=False), row=1, col=col)
        if ellipse is not None:
            fig.add_trace(go.Scatter(x=ellipse[:, 0], y=ellipse[:, 1], mode="lines", line=dict(color=TEAL, width=2.5, dash="dash"), hoverinfo="skip", showlegend=False), row=1, col=col)
    fig.update_xaxes(title=names[0])
    fig.update_yaxes(title=names[1], col=1)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_group(table):
    """LOF gegen die Größe der Anomaliegruppe: AUC über k (eine Linie je Gruppengröße; senkrecht gepunktet: die Gruppengröße), waagerechte gestrichelte Linien: Isolation Forest und robuste Schätzung."""
    fig = go.Figure()
    palette = (BLUE, GREEN, ORANGE, RED)
    contamination = sorted({c["contamination"] for c in table["cells"]})
    for color, c in zip(palette, contamination):
        cells = [x for x in table["cells"] if x["contamination"] == c]
        base = [b for b in table["baselines"] if b["contamination"] == c][0]
        m = int(round(base["group"]))
        fig.add_trace(go.Scatter(x=[x["k"] for x in cells], y=[x["lof_auc"] for x in cells], mode="lines+markers", line=dict(color=color, width=3), name=f"Gruppe {m} Touren ({c} %)", hoverinfo="skip"))
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Nachbarn k"), yaxis=dict(title="AUC des LOF", range=[0, 1.05]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_gap(rows):
    """Anomalien in der Lücke: AUC des LOF über k (kräftig), dazu Isolation Forest und robuste Schätzung (gestrichelt, konstant) je Konfiguration."""
    fig = go.Figure()
    for color, r in zip((BLUE, GREEN, ORANGE), rows):
        name = f"{r['n_modes']} Betriebsarten, {int(round(r['group']))} Anomalien"
        ks = sorted(r["by_k"])
        fig.add_trace(go.Scatter(x=ks, y=[r["by_k"][k] for k in ks], mode="lines+markers", line=dict(color=color, width=3), name=f"LOF: {name}", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[ks[0], ks[-1]], y=[r["iforest_auc"]] * 2, mode="lines", line=dict(color=color, width=2, dash="dash"), name="Isolation Forest", hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[ks[0], ks[-1]], y=[r["robust_auc"]] * 2, mode="lines", line=dict(color=color, width=2, dash="dot"), name="robust", hoverinfo="skip", showlegend=False))
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Nachbarn k"), yaxis=dict(title="AUC (gestrichelt: Isolation Forest, gepunktet: robust)", range=[0, 1.05]), legend=dict(orientation="h", y=-0.4))
    return lock_axes(fig)


def build_k_table(rows):
    """k gegen die Tourenzahl (n = 100): AUC des LOF über k bis n - 1 für verstreute Anomalien und eine dichte Gruppe von 20 %."""
    fig = go.Figure()
    for color, r in zip((BLUE, RED), rows):
        ks = sorted(r["by_k"])
        fig.add_trace(go.Scatter(x=ks, y=[r["by_k"][k] for k in ks], mode="lines+markers", line=dict(color=color, width=3), name=f"{KIND_NAMES[r['kind']]} ({r['contamination']} %)", hoverinfo="skip"))
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Nachbarn k (n = 100)"), yaxis=dict(title="AUC des LOF", range=[0, 1.05]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_modes(rows):
    """AUC der vier Detektoren je Anzahl Betriebsarten und Art der Anomalien (die Linie bei 0.5 ist Raten)."""
    labels = [f"{r['n_modes']} Betriebsart{'en' if r['n_modes'] > 1 else ''}<br>{KIND_NAMES[r['kind']]}" for r in rows]
    fig = go.Figure()
    for det in DETECTORS:
        y = [r[f"{det}_auc"] for r in rows]
        fig.add_trace(go.Bar(x=labels, y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=7), hoverinfo="skip"))
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"))
    fig.update_layout(height=420, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="AUC", range=[0, 1.15]), legend=dict(orientation="h", y=-0.55))
    return lock_axes(fig)


def build_masking(rows):
    """Dichte Gruppe abseits: AUC der vier Detektoren über den Anteil der Gruppe, dazu der LOF mit einem k passend zur Gruppengröße (gestrichelt)."""
    fig = go.Figure()
    xs = [r["x"] for r in rows]
    for det in DETECTORS:
        fig.add_trace(go.Scatter(x=xs, y=[r[f"{det}_auc"] for r in rows], mode="lines+markers", name=DETECTOR_NAMES[det] + (" (k = 20)" if det == "lof" else ""), line=dict(color=DETECTOR_COLORS[det], width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["lof_matched_auc"] for r in rows], mode="lines+markers", name="LOF, k = 1.5 × Gruppengröße", line=dict(color=PINK, width=2, dash="dash"), hoverinfo="skip"))
    fig.add_hline(y=0.5, line=dict(color=GRAY, dash="dot"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Anteil der Gruppe [%]"), yaxis=dict(title="AUC", range=[0, 1.05]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_cutoff(rows):
    """LOF: F1 (durchgezogen), Recall (gepunktet) und Fehlalarmrate (gestrichelt) über die LOF-Schwelle."""
    xs = [str(r["x"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["lof_f1"] for r in rows], mode="lines+markers", name="F1", line=dict(color=PINK, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["lof_recall"] for r in rows], mode="lines+markers", name="Recall", line=dict(color=PINK, width=2, dash="dot"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["lof_false_alarm"] for r in rows], mode="lines+markers", name="Fehlalarmrate", line=dict(color=PINK, width=2, dash="dash"), hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="LOF-Schwelle", type="category"), yaxis=dict(range=[0, 1.05]), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_wrong_share(rows):
    """F1 der vier Detektoren, wenn der angenommene Anteil das ½-, 1- und 2-fache des wahren ist."""
    xs = [f"{r['x']} % ({r['factor']:g}×)" for r in rows]
    fig = go.Figure()
    for det in DETECTORS:
        y = [r[f"{det}_f1"] for r in rows]
        fig.add_trace(go.Bar(x=xs, y=y, name=DETECTOR_NAMES[det], marker_color=DETECTOR_COLORS[det], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=9), hoverinfo="skip"))
    fig.update_layout(height=320, barmode="group", margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="angenommener Anteil (Vielfaches des wahren)"), yaxis=dict(title="F1", range=[0, 1.15]),
                      legend=dict(orientation="h", y=-0.4))
    return lock_axes(fig)


def build_dimension(cells):
    """F1 bei der Standardschwelle von LOF (links) und Isolation Forest (rechts) über Tourenzahl n (Zeilen) und Merkmalszahl p (Spalten)."""
    ns = sorted({c["n"] for c in cells})
    ps = sorted({c["p"] for c in cells})
    grid = {(c["n"], c["p"]): c for c in cells}
    fig = make_subplots(rows=1, cols=2, subplot_titles=("F1 LOF", "F1 Isolation Forest"), horizontal_spacing=0.14)
    for col, key, scale in ((1, "lof_f1", "Reds"), (2, "iforest_f1", "Purples")):
        z = [[grid[(n, p)][key] for p in ps] for n in ns]
        fig.add_trace(go.Heatmap(z=z, x=[f"p = {p}" for p in ps], y=[f"n = {n}" for n in ns], colorscale=scale, zmin=0.0, zmax=1, text=[[f"{v:.2f}" for v in row] for row in z], texttemplate="%{text}", showscale=False,
                                 hoverinfo="skip"), row=1, col=col)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_costs(times):
    """Rechenzeit (Modell bauen und alle Touren bewerten) von LOF und Isolation Forest über die Tourenzahl."""
    fig = go.Figure()
    for key, color, name in (("lof", PINK, "LOF"), ("iforest", PURPLE, "Isolation Forest")):
        y = [t[key] for t in times]
        fig.add_trace(go.Bar(x=[f"n = {t['n']}" for t in times], y=y, name=name, marker_color=color, text=[f"{v:.3f} s" for v in y], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=280, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Sekunden"), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)
