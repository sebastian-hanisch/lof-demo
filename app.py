"""Local Outlier Factor - Anomalien haben eine dünnere Nachbarschaft als ihre Nachbarn - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - den Local Outlier Factor - und lässt stattdessen das Beispiel wachsen.
Viertes Stück der Anomalie-Erkennung-Linie der "Konzepte"-Reihe: ein eigener Ast direkt nach der Wurzel (Elliptic Envelope), neben Isolation Forest und Extended IF. LOF setzt an den dort gemessenen Schwächen an
(dichte Anomaliegruppen, Lücke zwischen den Betriebsarten) - und hat eigene. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import lof_constants as C
import lof_ee_algorithm as alg
from lof_evaluation import (
    SWEEP_LABELS,
    SWEEP_VALUES,
    Settings,
    analyse_for,
    cost_table,
    dimension_table,
    gap_table,
    group_table,
    k_max,
    k_table,
    masking_table,
    modes_table,
    roc_curve,
    score_maps,
    sweep,
    threshold_table,
    verdict,
)
from lof_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    k_cap,
    kind_options,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from lof_visualization import (
    build_cutoff,
    build_density_plane,
    build_dimension,
    build_features,
    build_gap,
    build_group,
    build_k_table,
    build_lof_hist,
    build_lrd_bars,
    build_masking,
    build_method_bars,
    build_modes,
    build_neighborhood,
    build_roc,
    build_scatter,
    build_score_maps,
    build_sweep,
    build_wrong_share,
    build_costs,
    projection,
)

st.set_page_config(page_title="LOF – Sebastian Hanisch", layout="wide")
BLUE_TXT, RED_TXT = "#1f77b4", "#d62728"


def _pct(x):
    return "–" if x is None or np.isnan(x) else f"{x:.0%}"


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse_for(data_params, settings)


@st.cache_data(show_spinner=False)
def _sweep(parameter, base, settings, values):
    return sweep(parameter, values=values, settings=settings, **dict(base))


@st.cache_data(show_spinner=False)
def _maps(X2, settings):
    xs, ys, S_lof, S_if = score_maps(X2, settings)
    robust = alg.fit_mcd(X2, C.DEFAULT_SUPPORT, 0)
    ellipse = alg.ellipse_points(robust.location, robust.covariance, alg.chi2_ppf(0.975, 2), np.eye(2))
    return xs, ys, S_lof, S_if, ellipse


@st.cache_data(show_spinner=False)
def _group(base, settings):
    return group_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _gap(base, settings):
    return gap_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _modes(base, settings):
    return modes_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _masking(base, settings):
    return masking_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _k_table(base, settings):
    return k_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _dimension(base, settings):
    return dimension_table(settings, **dict(base)), sweep("n_noise", settings=settings, **{k: v for k, v in dict(base).items() if k != "n_noise"})


@st.cache_data(show_spinner=False)
def _threshold(base, settings):
    return threshold_table(settings, **dict(base))


@st.cache_data(show_spinner=False)
def _costs(base, settings):
    return cost_table(settings, **dict(base))


st.title("🏘️ Local Outlier Factor – Anomalien haben eine dünnere Nachbarschaft")
st.markdown(
    """
Die Wurzel dieser Linie beschreibt das Normale als **eine** Wolke, der **Isolation Forest** fragt, wie schnell sich eine Tour vom Rest abtrennen lässt. Der **Local Outlier Factor (LOF)** fragt etwas Lokaleres:
**Ist die Umgebung dieser Tour dünner besetzt als die Umgebung ihrer Nachbarn?** Er vergleicht die **lokale Dichte** einer Tour mit der ihrer k nächsten Nachbarn: LOF ≈ 1 heißt "so dicht wie die Nachbarn", deutlich darüber "abgelegen im Vergleich zur eigenen Gegend".
Damit setzt er genau an den Schwächen an, die in den Vorgängern gemessen wurden - **dichte Anomaliegruppen** und die **Lücke zwischen den Betriebsarten** - und hat eigene: er braucht ein **k**, das zur Gruppengröße passt, und er leidet unter **vielen irrelevanten Merkmalen**.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - viertes Stück der Anomalie-Erkennung-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel. "
    "Szenario, Isolation Forest und die Schätzer der Wurzel (klassisch, robust per MCD) sind wortgleich aus den Vorgänger-Demos übernommen, damit der Vergleich denselben Boden hat. Die Linie hat keinen Konvergenzpunkt; "
    "LOF ist ein eigener Ast, seine Fortsetzung (Feature Bagging gegen den Dimensionsfluch) ist noch nicht gebaut."
)

with st.expander("So funktioniert der Local Outlier Factor", expanded=True):
    st.markdown(
        """
1. **k Nachbarn.** Für jede Tour werden ihre $k$ nächsten Nachbarn gesucht (euklidischer Abstand über die **standardisierten** Kennzahlen - LOF ist nicht skaleninvariant). Die **k-Distanz** ist der Abstand zum $k$-ten Nachbarn.
2. **Erreichbarkeit.** $\\text{reach}(a, b) = \\max(\\text{kdist}(b), d(a, b))$: liegt $a$ mitten im Umfeld von $b$, zählt die k-Distanz von $b$ - das glättet Zufallsschwankungen.
3. **Lokale Dichte.** $\\text{lrd}(a) = 1 / \\text{Mittel}(\\text{reach}(a, b))$ über die Nachbarn $b$ - je kürzer die Wege zu den Nachbarn, desto dichter.
4. **LOF.** $\\text{LOF}(a) = \\text{Mittel}(\\text{lrd}(b)) / \\text{lrd}(a)$: das Verhältnis der Dichte der Nachbarn zur eigenen. Normale Touren liegen bei ≈ 1, eine Tour in dünnerer Umgebung darüber.
5. **Schwelle.** Anders als der Isolation-Forest-Score hat LOF eine Bedeutung: 1 = so dicht wie die Nachbarn. Die Faustregel hier ist **LOF > 1.5**; alternativ ein **erwarteter Anteil** (die größten Werte).

Was **nicht** vorausgesetzt wird: eine Verteilungsform, ein Normalbereich, Kovarianz. Was LOF **braucht**: ein $k$ (größer als jede Anomaliegruppe, kleiner als die Betriebsarten), standardisierte Merkmale und wenige irrelevante Merkmale.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_tours = st.slider(
        "Touren", *bounds("n_tours_slider"), key="n_tours_slider", step=10,
        help="Anzahl der Touren. Bei der Schwelle hat LOF (k folgt der Tourenzahl, höchstens n / 2) F1 0.96 / 0.94 / 0.88 / 0.86 / 0.91 bei 20 / 30 / 50 / 100 / 200 Touren, der Isolation Forest 0.60 / 0.67 / 0.78 / 0.89 / 0.96: "
             "die normalen Touren liegen bei LOF ≈ 1, unabhängig von der Tourenzahl, der Isolation Forest hat bei kleinen Stichproben viele Fehlalarme. Die AUC ist bei beiden überall 1.00.",
    )
    p_features = st.slider(
        "Merkmale", *bounds("p_slider"), key="p_slider",
        help="Anzahl der Kennzahlen je Tour (ab 13 zusätzliche Mischungen der versteckten Faktoren). F1 bei 2 / 5 / 8 / 12 / 20 / 30 Merkmalen: LOF 0.82 / 0.90 / 0.92 / 0.94 / 0.95 / 0.94, Isolation Forest 0.92 / 0.92 / 0.94 / 0.96 / 0.96 / 0.95; "
             "die AUC ist bei beiden 1.00.",
    )
    n_noise = st.slider(
        "Rauschmerkmale", *bounds("n_noise_slider"), key="n_noise_slider", step=5,
        help="Zusätzliche unabhängige Spalten ohne Zusammenhang mit den Faktoren. Bei 0 / 10 / 20 / 30 / 40: Recall bei der Schwelle bei LOF 1.00 / 0.71 / 0.18 / 0.03 / 0.00, beim Isolation Forest 0.99 / 0.91 / 0.71 / 0.58 / 0.39 "
             "(F1 0.94 / 0.82 / 0.30 / 0.05 / 0.00 gegen 0.96 / 0.94 / 0.82 / 0.72 / 0.56). Die AUC bleibt bei beiden 0.99-1.00: die Rangfolge trägt, aber die LOF-Werte rücken an 1 heran (Dimensionsfluch), die Schwelle 1.5 passt nicht mehr.",
    )
    n_modes = st.slider(
        "Betriebsarten", *bounds("n_modes_slider"), key="n_modes_slider",
        help="Aus wie vielen Gruppen (Stadt, Land, Fernverkehr) die normalen Touren stammen. F1 bei 1 / 2 / 3 Betriebsarten: LOF 0.94 / 0.94 / 0.93, Isolation Forest 0.96 / 0.96 / 0.97, robuste Schätzung der Wurzel 0.84 / 0.74 / 0.36 "
             "(AUC 1.00 / 0.95 / 0.85): LOF ist wie der Isolation Forest unempfindlich gegen mehrere Betriebsarten (AUC 1.00).",
    )
    curvature = st.slider(
        "Krümmung des Normalbereichs", *bounds("curvature_slider"), key="curvature_slider", step=0.25,
        help="Biegt die normale Fläche (nicht mehr konvex). Bei 0 / 0.25 / 0.5 / 0.75 / 1 bleibt die AUC bei 1.00, aber der F1 bei der Schwelle fällt bei LOF 0.94 / 0.90 / 0.83 / 0.78 / 0.74 (Fehlalarmrate 1.4 % → 7.8 %: "
             "am Rand der gebogenen Fläche ist die Umgebung dünner), beim Isolation Forest 0.96 / 0.98 / 0.97 / 0.97 / 0.95; die robuste Schätzung mit χ²-Schwelle fällt auf 0.84 / 0.49 / 0.41 / 0.39 / 0.38.",
    )
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Messrauschen der Kennzahlen. Bei 0 / 0.25 / 0.5 / 1.0: F1 von LOF 0.84 / 0.94 / 0.97 / 0.92, des Isolation Forest 0.94 / 0.96 / 0.96 / 0.96 (AUC beide 1.00; ohne Rauschen hat LOF mehr Fehlalarme, 4.2 %).",
    )
    contamination = st.slider(
        "Anteil der Anomalien [%]", *bounds("contamination_slider"), key="contamination_slider",
        help="Wie viele Touren Sonderfahrten sind. AUC von LOF bei 2 / 5 / 10 / 20 / 30 / 40 / 45 %: 1.00 / 1.00 / 1.00 / 0.99 / 0.90 / 0.72 / 0.64 (Isolation Forest überall 1.00), F1 0.75 / 0.88 / 0.94 / 0.90 / 0.43 / 0.15 / 0.09 "
             "gegen 0.42 / 0.81 / 0.96 / 0.97 / 0.93 / 0.87 / 0.81: bei wenigen Anomalien ist LOF an der Schwelle besser (weniger Fehlalarme), bei vielen bricht er ein - verstreute Anomalien sind einander Nachbarn.",
    )
    kind = st.selectbox(
        "Art der Anomalien", kind_options(n_modes), key="kind_select", format_func=lambda k: C.KIND_LABELS[k],
        help="Verstreut: jede Anomalie in einer anderen Richtung. Dichte Gruppe: alle beieinander - bei 10 % (30 Touren) AUC LOF 0.35 mit k = 20, 0.97 mit k = 40 (Isolation Forest 0.95, robust 1.00): LOF sieht die Gruppe nur, wenn k größer als die Gruppe ist. "
             "In der Lücke (ab zwei Betriebsarten): LOF 0.53 mit k = 20 (0.71 mit k = 40), Isolation Forest 0.54, robust 0.40.",
    )
    if kind != "gap":
        seed_widget("strength_slider")
        strength = st.slider(
            "Abstand der Anomalien (Faktor-σ)", *bounds("strength_slider"), key="strength_slider", step=0.5,
            help="Wie weit die Anomalien im Faktorraum vom Normalen entfernt sind. Bei 3 / 4 / 6 / 9 / 12: AUC von LOF 0.96 / 0.99 / 1.00 / 1.00 / 1.00; F1 von LOF 0.48 / 0.88 / 0.94 / 0.94 / 0.94, des Isolation Forest "
                 "0.66 / 0.85 / 0.96 / 0.99 / 1.00 (bei kleinem Abstand liegen die Anomalien dicht am Rand: Recall von LOF nur 0.33 bei 3 σ).",
        )
        st.session_state["_strength_kept"] = strength
    else:
        strength = float(st.session_state.get("_strength_kept", C.DEFAULT_STRENGTH))

    st.markdown("**Local Outlier Factor**")
    k_neighbors = st.slider(
        "Nachbarn k", C.K_MIN, k_cap(int(n_tours)), key="k_slider",
        help="Wie viele Nachbarn die lokale Dichte bestimmen (höchstens n / 2: bei k nahe n sehen alle Touren dieselben Nachbarn). Im Standardfall bei k = 5 / 10 / 20 / 40 / 80: AUC 0.86 / 1.00 / 1.00 / 1.00 / 1.00, F1 0.54 / 0.91 / 0.94 / 0.89 / 0.89 "
             "(Fehlalarmrate 0.9 % / 1.0 % / 1.4 % / 2.7 % / 2.9 %). Bei dichten Gruppen und in der Lücke entscheidet k über alles (siehe Experimente unten).",
    )
    threshold_kind = st.selectbox(
        "Schwelle", C.THRESHOLD_KINDS, key="threshold_kind_select", format_func=lambda k: C.THRESHOLD_LABELS[k],
        help="Standard: LOF über der LOF-Schwelle, Isolation Forest über Score 0.5, klassisch und robust über dem χ²-Quantil. Erwarteter Anteil: bei allen vieren werden die größten Werte markiert - "
             "dann entscheidet nur die Rangfolge, aber der Anteil muss bekannt sein.",
    )
    if threshold_kind == "standard":
        seed_widget("cutoff_slider")
        cutoff = st.slider(
            "LOF-Schwelle", *bounds("cutoff_slider"), key="cutoff_slider", step=0.05,
            help="Ab welchem LOF eine Tour markiert wird (1 = so dicht wie die Nachbarn). Bei 1.1 / 1.2 / 1.3 / 1.5 / 2.0 / 2.5: F1 0.52 / 0.69 / 0.82 / 0.94 / 0.96 / 0.85, Recall 1.00 / 1.00 / 1.00 / 1.00 / 0.95 / 0.75, "
                 "Fehlalarmrate 20.5 % / 9.9 % / 5.0 % / 1.4 % / 0.3 % / 0.0 %. Die normalen Touren haben im Mittel LOF ≈ 1.02, daher ist die Schwelle deutbarer als ein Isolation-Forest-Score.",
        )
        seed_widget("quantile_slider")
        quantile = st.slider(
            "Schwelle: χ²-Quantil (klassisch, robust)", *bounds("quantile_slider"), key="quantile_slider", step=0.001, format="%.3f",
            help="Ab welchem Anteil der χ²-Verteilung eine Tour bei den Schätzern der Wurzel als Anomalie gilt. Bei 0.9 / 0.95 / 0.975 / 0.99 / 0.999: F1 der robusten Schätzung 0.67 / 0.77 / 0.84 / 0.89 / 0.90, "
                 "der klassischen 0.69 / 0.66 / 0.57 / 0.44 / 0.15.",
        )
        st.session_state["_cutoff_kept"] = cutoff
        st.session_state["_quantile_kept"] = quantile
        share = int(st.session_state.get("_share_kept", C.DEFAULT_SHARE))
    else:
        seed_widget("share_slider")
        share = st.slider(
            "Angenommener Anteil der Anomalien [%]", *bounds("share_slider"), key="share_slider",
            help="Wie viele Touren als Anomalie markiert werden (die größten Werte, für alle vier Detektoren). Beim wahren Anteil 10 % ist der F1 bei angenommenen 2 / 5 / 10 / 20 / 40 % bei LOF und Isolation Forest "
                 "0.33 / 0.67 / 0.97 / 0.67 / 0.40, für die robuste Schätzung 0.33 / 0.67 / 0.90 / 0.66 / 0.40, für die klassische 0.32 / 0.56 / 0.69 / 0.58 / 0.39.",
        )
        st.session_state["_share_kept"] = share
        cutoff = float(st.session_state.get("_cutoff_kept", C.DEFAULT_CUTOFF))
        quantile = float(st.session_state.get("_quantile_kept", C.DEFAULT_QUANTILE))
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Aufnahme generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für die Touren und die Anomalien.")

sync_query_params({
    "n_tours_slider": int(n_tours), "p_slider": int(p_features), "n_noise_slider": int(n_noise), "n_modes_slider": int(n_modes), "curvature_slider": float(curvature), "noise_slider": float(noise),
    "contamination_slider": int(contamination), "kind_select": kind, "strength_slider": float(strength), "k_slider": int(k_neighbors),
    "threshold_kind_select": threshold_kind, "cutoff_slider": float(cutoff), "quantile_slider": float(quantile), "share_slider": int(share), "seed_input": int(seed),
})

data_params = (int(n_tours), int(p_features), int(n_noise), int(n_modes), float(curvature), float(round(noise, 2)), int(contamination), kind, float(strength), int(seed))
settings = Settings(k=int(k_neighbors), threshold_kind=threshold_kind, cutoff=float(round(cutoff, 2)), quantile=float(quantile), share=int(share))
with st.spinner("Rechne die Nachbarschaften..."):
    a = _analysis(data_params, settings)
level, code, vd = verdict(a)
ds = a.ds
ls, ifs, cs, rs = (a.scores[d] for d in ("lof", "iforest", "classical", "robust"))
n_anom = int(ds.anomaly.sum())
n_total = ds.p + ds.n_noise
base_data = tuple(sorted({"n": int(n_tours), "p": int(p_features), "n_noise": int(n_noise), "n_modes": int(n_modes), "curvature": float(curvature), "noise": float(round(noise, 2)),
                          "contamination": int(contamination), "kind": kind, "strength": float(strength)}.items()))
data_key = data_params + (settings,)

# --- LOF in Aktion --------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 LOF in Aktion")
STEP_LABELS = {1: "1 · Touren", 2: "2 · Nachbarschaft", 3: "3 · Dichte und LOF", 4: "4 · LOF-Karte", 5: "5 · Ergebnis"}
if "lof_step" not in st.session_state or st.session_state.get("lof_step_owner") != data_key:
    st.session_state["lof_step"] = 1
    st.session_state["lof_step_owner"] = data_key
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="lof_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()

P, axes = projection(ds.X, a.robust)
fit = a.lof_fit
lof_values = a.values["lof"]
normal_idx = np.flatnonzero(~ds.anomaly)
anom_idx = np.flatnonzero(ds.anomaly)
i_anom = int(anom_idx[np.argsort(lof_values[anom_idx])[len(anom_idx) // 2]])                     # eine typische Sonderfahrt: die mit dem mittleren LOF
i_norm = int(normal_idx[np.argmin(np.abs(P[normal_idx]).sum(axis=1))])                            # eine normale Tour nahe der Mitte
picks = [(i_norm, "normale Tour", BLUE_TXT), (i_anom, "Sonderfahrt", RED_TXT)]


def _render(current_step):
    with view_slot.container():
        if current_step == 1:
            c1, c2 = st.columns(2)
            c1.markdown("**Die Touren in der Ebene ihrer größten Streuung** (rote Rauten = Sonderfahrten)")
            c1.plotly_chart(build_scatter(P, ds.anomaly), width="stretch", key="step_scatter")
            c2.markdown(f"**Zwei Rohmerkmale: {ds.names[0]} gegen {ds.names[min(1, ds.p - 1)]}** (Einheiten wie gemessen)")
            c2.plotly_chart(build_features(ds.X, ds.anomaly, ds.names, 0, 1), width="stretch", key="step_features")
        elif current_step == 2:
            st.markdown(f"**Die {a.k} nächsten Nachbarn zweier Touren** (gemessen in allen {n_total} standardisierten Merkmalen, gezeigt in der Ebene der zwei Hauptrichtungen)")
            st.plotly_chart(build_neighborhood(P, ds.anomaly, picks, fit.neighbors), width="stretch", key="step_neighborhood")
        elif current_step == 3:
            c1, c2 = st.columns(2)
            c1.markdown("**Lokale Erreichbarkeitsdichte je Tour** (dunkel = dünn besetzt)")
            c1.plotly_chart(build_density_plane(P, fit.lrd, ds.anomaly), width="stretch", key="step_density")
            entries = [("normale Tour", BLUE_TXT, float(fit.lrd[i_norm]), float(fit.lrd[fit.neighbors[i_norm]].mean()), float(lof_values[i_norm])),
                       ("Sonderfahrt", RED_TXT, float(fit.lrd[i_anom]), float(fit.lrd[fit.neighbors[i_anom]].mean()), float(lof_values[i_anom]))]
            c2.markdown("**Eigene Dichte gegen die ihrer Nachbarn** (LOF = Verhältnis)")
            c2.plotly_chart(build_lrd_bars(entries), width="stretch", key="step_lrd_bars")
            st.markdown("**LOF je Tour mit der Schwelle**")
            st.plotly_chart(build_lof_hist(lof_values, ds.anomaly, ls["threshold"]), width="stretch", key="step_lof_hist")
        elif current_step == 4:
            X2 = ds.X[~ds.anomaly][:, :2]
            xs, ys, S_lof, S_if, ell = _maps(X2, settings)
            st.markdown(f"**Score-Karten über die ersten zwei Merkmale ({ds.names[0]}, {ds.names[1]}), ohne Anomalien** (beide Detektoren nur auf diesen zwei Merkmalen; dunkler = auffälliger; gestrichelt: robuste Ellipse)")
            st.plotly_chart(build_score_maps(xs, ys, S_lof, S_if, X2, ds.names[:2], ell), width="stretch", key="step_maps")
        else:
            c1, c2 = st.columns(2)
            c1.markdown("**Kennzahlen bei der gewählten Schwelle**")
            c1.plotly_chart(build_method_bars(a.scores), width="stretch", key="step_bars")
            c2.markdown("**ROC-Kurven** (unabhängig von der Schwelle)")
            curves = {d: (*roc_curve(a.values[d], ds.anomaly), a.scores[d]["auc"]) for d in ("lof", "iforest", "classical", "robust")}
            c2.plotly_chart(build_roc(curves), width="stretch", key="step_roc")


if auto_play:
    for s in STEP_LABELS:
        _render(s)
        time.sleep(1.2)
    step = 5
else:
    _render(step)

if step == 1:
    st.caption(f"{ds.n} Touren mit {n_total} Kennzahlen" + (f" ({ds.n_noise} davon reines Rauschen)" if ds.n_noise else "") + f", davon {n_anom} Sonderfahrten ({n_anom / ds.n:.0%}; {C.KIND_LABELS[ds.kind]}), "
               f"{ds.n_modes} Betriebsart{'en' if ds.n_modes > 1 else ''}. Die Ebene ist die der zwei größten Streuungsrichtungen der robusten Schätzung in standardisierten Kennzahlen; LOF rechnet mit allen Kennzahlen.")
elif step == 2:
    n_anom_nb = int(ds.anomaly[fit.neighbors[i_anom]].sum())
    st.caption(f"Die normale Tour hat die k-Distanz {fit.kdist[i_norm]:.2f}, die Sonderfahrt {fit.kdist[i_anom]:.2f} (standardisierte Einheiten). Von den {a.k} Nachbarn der Sonderfahrt sind {n_anom_nb} selbst Sonderfahrten"
               + (" - bei einer dichten Gruppe von mindestens k Touren sind die Nachbarn der Gruppe die Gruppe selbst, und LOF sieht sie nicht." if n_anom_nb >= a.k - 1 else "."))
elif step == 3:
    st.caption(f"Die normale Tour hat LOF {lof_values[i_norm]:.2f}, die Sonderfahrt {lof_values[i_anom]:.2f}. Mittlerer LOF der normalen Touren {lof_values[~ds.anomaly].mean():.2f} (Median {np.median(lof_values[~ds.anomaly]):.2f}), der Sonderfahrten "
               f"{lof_values[ds.anomaly].mean():.2f}; bei der Schwelle {ls['threshold']:.2f} werden {ls['n_flagged']} Touren markiert.")
elif step == 4:
    st.caption("Links LOF (auf zwei standardisierten Merkmalen): der Wert steigt gleichmäßig mit dem Abstand zu den Daten, die Höhenlinien sind rund - keine Bänder entlang der Achsen, denn LOF misst euklidische Abstände. Rechts der Isolation Forest, "
               "dessen achsenparallele Schnitte Bänder entlang der Achsen erzeugen (die Geister-Regionen des Vorgängers). Die Ellipse ist die der Wurzel. Beide Karten sind über nur zwei Merkmale gerechnet.")
else:
    st.caption("Die ROC-Kurve zeigt die Rangfolge (AUC), die Balken die Wirkung der Schwelle: eine perfekte Rangfolge kann trotzdem viele Fehlalarme oder verpasste Anomalien haben, wenn die Schwelle nicht passt.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Detektoren gefunden haben – LOF gegen Isolation Forest gegen Wurzel")
st.caption(
    "Anomalie = Tour über der Schwelle (Standard: LOF über der LOF-Schwelle, Isolation Forest über Score 0.5, χ²-Quantil bei klassisch und robust). **AUC**: Wahrscheinlichkeit, dass eine zufällige Sonderfahrt einen größeren Wert hat als eine "
    "zufällige normale Tour (1 = perfekte Rangfolge, 0.5 = Raten, darunter: die Anomalien wirken normaler als die Normalen). **Recall**: Anteil der gefundenen Sonderfahrten. **Fehlalarmrate**: Anteil der normalen Touren, die markiert werden."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("AUC (LOF)", f"{ls['auc']:.2f}", delta=f"Isolation Forest {ifs['auc']:.2f} · robust {rs['auc']:.2f}", delta_color="off", help="Rangfolge der Anomalie-Werte; darunter die anderen Detektoren.")
m2.metric("Recall (LOF)", _pct(ls["recall"]), delta=f"Isolation Forest {_pct(ifs['recall'])} · robust {_pct(rs['recall'])}", delta_color="off", help=f"Anteil der {n_anom} Sonderfahrten, die bei der Schwelle markiert werden.")
m3.metric("Fehlalarmrate (LOF)", f"{ls['false_alarm']:.1%}", delta=f"Isolation Forest {ifs['false_alarm']:.1%} · robust {rs['false_alarm']:.1%}", delta_color="off", help="Anteil der normalen Touren, die als Anomalie markiert werden.")
m4.metric("F1 (LOF)", f"{ls['f1']:.2f}", delta=f"Isolation Forest {ifs['f1']:.2f} · mit bekanntem Anteil {a.oracle_f1['lof']:.2f}", delta_color="off",
          help="Harmonisches Mittel aus Precision und Recall bei der Schwelle; darunter der F1 des Isolation Forest und der F1 von LOF, wenn der wahre Anteil bekannt wäre.")

_t = vd
if code == "lof_wins":
    if _t["kind"] == "gap":
        st.success(f"✅ LOF findet, was die anderen nicht finden: AUC {_t['lof_auc']:.2f} gegen {_t['iforest_auc']:.2f} (Isolation Forest), {_t['robust_auc']:.2f} (robust), {_t['classical_auc']:.2f} (klassisch) - die Anomalien in der Lücke sind für LOF "
                   f"dünner besetzt als die Betriebsarten ringsum. An der Schwelle: F1 {_t['lof_f1']:.2f} (mit bekanntem Anteil {_t['oracle_lof']:.2f}) - die Lücken-Touren liegen im Rang weit oben, aber nicht ganz oben.")
    elif _t["lof_auc"] - _t["best_other_auc"] >= 0.05:
        st.success(f"✅ LOF ist besser: AUC {_t['lof_auc']:.2f} gegen {_t['best_other_auc']:.2f} beim besten anderen Detektor; F1 {_t['lof_f1']:.2f} (Recall {_pct(_t['lof_recall'])}) gegen {_t['iforest_f1']:.2f} (Isolation Forest).")
    else:
        st.success(f"✅ Gleiche Rangfolge (AUC LOF {_t['lof_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}), aber LOF hat an der Schwelle das bessere F1: {_t['lof_f1']:.2f} gegen {_t['iforest_f1']:.2f} "
                   f"(Fehlalarmrate {_t['lof_false_alarm']:.1%} gegen {_t['iforest_false_alarm']:.1%}). Sein Wert für normale Touren liegt bei ≈ 1, unabhängig von der Stichprobengröße - der Isolation-Forest-Score verschiebt sich dort.")
elif code == "k_window":
    m = _t["n_anomalies"]
    reason = (f"k = {_t['k']} ist nicht größer als die Gruppe ({m} Touren): jede Tour der Gruppe hat nur Gruppen-Nachbarn und ist \"lokal dicht\" (LOF ≈ 1)" if _t["k"] <= m
              else f"k = {_t['k']} reicht bis über die Grenzen der Gruppe hinaus (n − Gruppe = {_t['n'] - m}): alle Touren haben fast dieselben Nachbarn")
    st.warning(f"⚠️ k passt nicht zur Anomaliegruppe: {reason}. AUC LOF {_t['lof_auc']:.2f} gegen {_t['best_other_auc']:.2f} beim besten anderen Detektor. "
               f"Mit k zwischen der Gruppengröße und n − Gruppe (im Experiment unten) sieht LOF die Gruppe - aber dafür muss die Gruppengröße bekannt sein.")
elif code == "others_win":
    st.warning(f"⚠️ Ein anderer Detektor ist besser: AUC LOF {_t['lof_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}, klassisch {_t['classical_auc']:.2f}. "
               "Bei vielen verstreuten Anomalien sind diese einander Nachbarn und wirken lokal nicht dünner besetzt; der Isolation Forest hat dieses Problem nicht.")
elif code == "gap":
    st.warning(f"⚠️ Die Anomalien liegen in der Lücke zwischen den Betriebsarten: AUC {_t['lof_auc']:.2f} bei LOF (k = {_t['k']}), {_t['iforest_auc']:.2f} beim Isolation Forest, {_t['robust_auc']:.2f} robust, {_t['classical_auc']:.2f} klassisch. "
               f"Mit k passend zur Gruppe ({_t['n_anomalies']} Touren) - größer als sie, kleiner als eine Betriebsart - trennt LOF sie besser (bis 0.86 bei drei Betriebsarten; siehe Experiment Lücke).")
elif code == "threshold_off":
    st.warning(f"⚠️ Die Schwelle passt nicht: die Rangfolge ist gut (AUC {_t['lof_auc']:.2f}), aber bei der LOF-Schwelle {_t['lof_threshold']:.2f} ist F1 {_t['lof_f1']:.2f} (Recall {_pct(_t['lof_recall'])}, Fehlalarmrate {_t['lof_false_alarm']:.1%}); "
               f"mit dem wahren Anteil wären es {_t['oracle_lof']:.2f}. Bei vielen Rauschmerkmalen rücken die LOF-Werte an 1 heran, auf gebogenen Flächen wächst die Fehlalarmrate. Der Isolation Forest erreicht {_t['iforest_f1']:.2f}.")
elif code == "iforest_f1":
    st.warning(f"⚠️ Gleiche Rangfolge (AUC LOF {_t['lof_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}), aber der Isolation Forest hat bei seiner Schwelle das bessere F1: {_t['iforest_f1']:.2f} gegen {_t['lof_f1']:.2f} bei LOF "
               f"(Fehlalarmrate {_t['iforest_false_alarm']:.1%} gegen {_t['lof_false_alarm']:.1%}).")
elif code == "comparable":
    st.success(f"✅ Alle ranken gleich gut (AUC LOF {_t['lof_auc']:.2f}, Isolation Forest {_t['iforest_auc']:.2f}, robust {_t['robust_auc']:.2f}); bei der Schwelle: F1 {_t['lof_f1']:.2f} (LOF), {_t['iforest_f1']:.2f} (Isolation Forest), "
               f"{_t['robust_f1']:.2f} (robust), {_t['classical_f1']:.2f} (klassisch). Im Standardfall unterscheidet LOF sich nicht - seine Stärken und Schwächen zeigen sich erst bei dichten Gruppen, in der Lücke und bei vielen Merkmalen.")

d1, d2c = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    rows = [("AUC", "auc", "{:.2f}"), ("mittlere Präzision (AP)", "ap", "{:.2f}"), ("Precision", "precision", "{:.2f}"), ("Recall", "recall", "{:.2f}"), ("F1", "f1", "{:.2f}"),
            ("Fehlalarmrate", "false_alarm", "{:.3f}"), ("Schwelle", "threshold", "{:.2f}")]
    st.table({"Kennzahl": [r[0] for r in rows], "LOF": [r[2].format(ls[r[1]]) for r in rows], "Isolation Forest": [r[2].format(ifs[r[1]]) for r in rows],
              "klassisch": [r[2].format(cs[r[1]]) for r in rows], "robust (MCD)": [r[2].format(rs[r[1]]) for r in rows]})
with d2c:
    st.markdown("**Was gerechnet wurde**")
    st.table({"": ["Rechenzeit", "Parameter", "Bewertung", "mittlerer Wert (normal / Anomalie)"],
              "LOF": [f"{a.seconds['lof'] * 1000:.1f} ms", f"k = {a.k}", "Dichte gegenüber den Nachbarn", f"{lof_values[~ds.anomaly].mean():.2f} / {lof_values[ds.anomaly].mean():.2f}"],
              "Isolation Forest": [f"{a.seconds['iforest'] * 1000:.0f} ms", f"{len(a.forest_if.trees)} Bäume × ψ = {a.forest_if.psi}", "Pfadlänge in zufälligen Bäumen", f"{a.values['iforest'][~ds.anomaly].mean():.2f} / {a.values['iforest'][ds.anomaly].mean():.2f}"],
              "robust (MCD)": [f"{a.seconds['robust'] * 1000:.0f} ms", f"h = {a.robust.h}", "Kovarianz (alle Merkmale)", f"d² {a.values['robust'][~ds.anomaly].mean():.1f} / {a.values['robust'][ds.anomaly].mean():.1f}"]})
    st.caption("LOF ist nicht skaleninvariant: er rechnet auf standardisierten Kennzahlen (auf Rohdaten fiele seine AUC im Standardfall von 1.00 auf 0.87, siehe Experiment Kosten und Einheiten).")

st.markdown("---")

# --- Sweeps ----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von k, Anomalien und Daten ab?")
sweep_options = [k for k in SWEEP_LABELS if not ((kind == "gap" and k in ("strength", "n_modes")) or (threshold_kind == "share" and k in ("cutoff", "quantile")) or (threshold_kind == "standard" and k == "share"))]
if st.session_state.get("sweep_select") not in sweep_options:
    st.session_state["sweep_select"] = sweep_options[0]
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", sweep_options, format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
current = {"k": int(k_neighbors), "n": int(n_tours), "p": int(p_features), "n_noise": int(n_noise), "n_modes": int(n_modes), "curvature": float(curvature), "noise": float(noise),
           "contamination": int(contamination), "strength": float(strength), "cutoff": float(cutoff), "quantile": float(quantile), "share": int(share)}[sweep_param]
sweep_values = SWEEP_VALUES[sweep_param]
if sweep_param == "k":
    allowed = tuple(v for v in sweep_values if v <= k_max(int(n_tours)))
    sweep_values = allowed if len(allowed) >= 2 else (C.K_MIN, k_max(int(n_tours)))
if st.button("Sweep über 5 feste Datensätze berechnen (dauert einige Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, data_key)}
if (sweep_param, data_key) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Datensätze..."):
        rows_sweep = _sweep(sweep_param, tuple(kv for kv in base_data if kv[0] != sweep_param), settings, sweep_values)
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param], current=current), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band) über 5 feste Sweep-Datensätze (getrennt vom Seed oben); alle anderen Regler wie in der Seitenleiste. Links die Rangfolge (AUC), rechts F1 (durchgezogen) und Fehlalarmrate (gestrichelt) bei der gewählten Schwelle. "
               "Bei der Tourenzahl folgt k der Tourenzahl (höchstens n / 2); bei den Reglern von LOF (k, Schwelle) ändern sich nicht die Linien der anderen Detektoren.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 k gegen die Größe der Anomaliegruppe")
if st.button("k von 10 bis 100 gegen Gruppen von 5–30 % vergleichen (dauert etwa 20 Sekunden)", key="group_start"):
    st.session_state["group_on"] = True
if st.session_state.get("group_on"):
    with st.spinner("Rechne 4 Gruppengrößen × 5 Werte von k × 5 Datensätze..."):
        gt = _group(tuple(kv for kv in base_data if kv[0] not in ("kind", "contamination", "n_modes")), settings)
    st.plotly_chart(build_group(gt), width="stretch", key="group_chart")
    st.caption("AUC des LOF (eine Linie je Gruppengröße, Mittel über 5 feste Datensätze, eine Betriebsart, dichte Gruppe abseits): **LOF sieht die Gruppe nur, wenn k größer ist als sie** - bei 15 Touren reicht k = 20 (AUC 0.99), bei 30 Touren erst k = 40 (0.97), "
               "bei 60 Touren k = 80 (0.95), bei 90 Touren nicht einmal k = 100 (0.71). Ist k zu klein, ist die Gruppe selbst lokal dicht und LOF liegt unter Raten (0.34-0.52). Zum Vergleich: der Isolation Forest hat 0.99 / 0.95 / 0.85 / 0.70 "
               "bei 5 / 10 / 20 / 30 %, die robuste Schätzung 1.00 / 1.00 / 1.00 / 0.49. Das passende k setzt voraus, dass man die Gruppengröße kennt.")

st.markdown("---")

st.subheader("🔬 Anomalien in der Lücke zwischen den Betriebsarten")
if st.button("k gegen Lücken-Anomalien vergleichen (dauert etwa 20 Sekunden)", key="gap_start"):
    st.session_state["gap_on"] = True
if st.session_state.get("gap_on"):
    with st.spinner("Rechne 3 Konfigurationen × 7 Werte von k × 5 Datensätze..."):
        gp = _gap(tuple(kv for kv in base_data if kv[0] not in ("kind", "contamination", "n_modes")), settings)
    st.plotly_chart(build_gap(gp), width="stretch", key="gap_chart")
    st.caption("AUC des LOF über k (Mittel über 5 feste Datensätze); gestrichelt der Isolation Forest, gepunktet die robuste Schätzung je Konfiguration. Bei drei Betriebsarten und 30 Lücken-Touren erreicht LOF 0.84-0.86 mit k = 60-70 "
               "(Isolation Forest 0.27, robust 0.38), bei zwei Betriebsarten nur 0.71 (k = 40; Isolation Forest 0.54, robust 0.40), bei zwei Betriebsarten und 15 Touren 0.86 (k = 30; Isolation Forest 0.71). "
               "Die Wirkung hat ein **Fenster**: k größer als die Lücken-Gruppe und kleiner als eine Betriebsart - bei k = 90 fällt die AUC auf 0.37 (zwei Betriebsarten) und 0.14 (drei).")

st.markdown("---")

st.subheader("🔬 Betriebsarten × Art der Anomalien")
if st.button("Betriebsarten × Art der Anomalien vergleichen (dauert etwa 15 Sekunden)", key="modes_start"):
    st.session_state["modes_on"] = True
if st.session_state.get("modes_on"):
    with st.spinner("Vergleiche 1-3 Betriebsarten × 3 Arten × 5 Datensätze..."):
        mt = _modes(tuple(kv for kv in base_data if kv[0] not in ("kind", "n_modes")), settings)
    st.plotly_chart(build_modes(mt), width="stretch", key="modes_chart")
    st.caption(f"AUC der vier Detektoren mit dem gewählten k = {int(k_neighbors)} (Mittel über 5 feste Datensätze; die Zahlen im Text gelten für k = 20). Verstreute Anomalien findet LOF bei jeder Zahl von Betriebsarten (AUC 1.00). "
               "Die dichte Gruppe abseits (30 Touren) übersieht LOF mit k = 20 (AUC 0.35 / 0.38 / 0.39, unter Raten), der Isolation Forest sieht sie (0.95 / 0.98 / 0.95). In der Lücke: LOF 0.53 (zwei Betriebsarten, gleich dem Isolation Forest) und 0.36 (drei; Isolation Forest 0.27).")

st.markdown("---")

st.subheader("🔬 Masking: die Gruppe wächst")
if st.button("Anteil der dichten Gruppe von 2 bis 45 % durchfahren (dauert etwa 30 Sekunden)", key="masking_start"):
    st.session_state["masking_on"] = True
if st.session_state.get("masking_on"):
    with st.spinner("Rechne 10 Anteile × 5 Datensätze, LOF auch mit passendem k..."):
        mk = _masking(tuple(kv for kv in base_data if kv[0] not in ("kind", "contamination", "n_modes")), settings)
    st.plotly_chart(build_masking(mk), width="stretch", key="masking_chart")
    st.caption("Mittel über 5 feste Datensätze, eine Betriebsart. Mit k = 20 sieht LOF ab 10 % keine Gruppe mehr (AUC 0.35-0.50), obwohl er bei 2 und 5 % perfekt ist (1.00 / 0.99). **Mit einem k von 1.5 × Gruppengröße** (höchstens n / 2, gestrichelt) "
               "trennt LOF die Gruppe bis 25 % (0.99 / 0.99 / 0.98 / 0.98 / 0.94 bei 5 / 10 / 15 / 20 / 25 %) - besser als der Isolation Forest (0.99 / 0.95 / 0.91 / 0.85 / 0.79) und wie die robuste Schätzung (1.00). Ab 30 % kippt auch die Wurzel (0.49) und LOF bleibt "
               "bei 0.71, der Isolation Forest 0.70. Ab 35 % ist keiner mehr deutlich über Raten - die Gruppe *ist* der Normalbereich.")

st.markdown("---")

st.subheader("🔬 k nahe an der Tourenzahl")
if st.button("k gegen die Tourenzahl vergleichen (dauert etwa 10 Sekunden)", key="k_start"):
    st.session_state["k_on"] = True
if st.session_state.get("k_on"):
    with st.spinner("Rechne k von 5 bis 99 bei 100 Touren × 5 Datensätze..."):
        kt = _k_table(tuple(kv for kv in base_data if kv[0] not in ("n", "kind", "contamination", "n_modes")), settings)
    st.plotly_chart(build_k_table(kt), width="stretch", key="k_chart")
    st.caption("100 Touren, AUC des LOF über k bis n − 1 (Mittel über 5 feste Datensätze). Verstreute Anomalien (10 %): AUC 1.00 bis k = 90, bei **k = 99 = n − 1: 0.00** - alle Touren haben dieselben Nachbarn, die Reihenfolge kehrt sich um. "
               "Dichte Gruppe von 20 Touren: unter Raten für k ≤ 20 (0.43 / 0.39 / 0.58), 0.99-1.00 für k = 30-70, bei k = 90 wieder 0.38 (0.05 bei k = 99). Deshalb bietet der Regler höchstens k = n / 2 an.")

st.markdown("---")

st.subheader("🔬 Rauschmerkmale und wenige Touren bei vielen Merkmalen")
if st.button("Rauschmerkmale und Tourenzahl × Merkmalszahl durchfahren (dauert etwa 70 Sekunden)", key="dimension_start"):
    st.session_state["dimension_on"] = True
if st.session_state.get("dimension_on"):
    with st.spinner("Rechne 6 Tourenzahlen × 5 Merkmalszahlen × 5 Datensätze und die Rauschmerkmale..."):
        dt, nt = _dimension(tuple(kv for kv in base_data if kv[0] not in ("n", "p")), settings)
    st.markdown("**Rauschmerkmale 0 bis 40**")
    st.plotly_chart(build_sweep(nt, SWEEP_LABELS["n_noise"]), width="stretch", key="noise_chart")
    st.markdown("**F1 bei der Standardschwelle: Tourenzahl × Merkmalszahl** (k folgt der Tourenzahl, höchstens n / 2)")
    st.plotly_chart(build_dimension(dt), width="stretch", key="dimension_chart")
    st.caption("Mittel über 5 feste Datensätze. Die AUC beider bleibt bei jeder Tourenzahl und Merkmalszahl bei 1.00 und bei 40 Rauschmerkmalen bei 0.99. Aber bei Rauschmerkmalen brechen die LOF-Werte der Anomalien Richtung 1 ein: Recall 1.00 / 0.71 / 0.18 / 0.03 / 0.00 bei 0 / 10 / 20 / 30 / 40, "
               "F1 0.94 → 0.00 (Isolation Forest 0.96 → 0.56); mit bekanntem Anteil bleiben 0.87 (Isolation Forest 0.84). Der **Dimensionsfluch** wirkt hier auf die Schwelle, nicht auf die Rangfolge. "
               "Bei wenigen Touren gewinnt LOF: F1 0.88-0.96 bei n ≤ 30 (Isolation Forest 0.54-0.68), weil der LOF der Normalen bei ≈ 1 liegt; bei n = 100 haben beide etwa gleich viele Fehlalarme (LOF 3-6 %, Isolation Forest 3 %).")

st.markdown("---")

st.subheader("🔬 Die Schwelle: LOF hat einen Wert mit Bedeutung")
if st.button("Schwellen vergleichen (dauert etwa 25 Sekunden)", key="threshold_start"):
    st.session_state["threshold_on"] = True
if st.session_state.get("threshold_on"):
    with st.spinner("Rechne 6 Schwellen × 5 Datensätze und drei angenommene Anteile..."):
        tt = _threshold(base_data, settings)
    c1, c2 = st.columns(2)
    c1.markdown("**LOF: F1, Recall und Fehlalarmrate je LOF-Schwelle**")
    c1.plotly_chart(build_cutoff(tt["cutoff"]), width="stretch", key="cutoff_chart")
    c2.markdown("**F1 bei falsch angenommenem Anteil** (½×, 1×, 2× des wahren)")
    c2.plotly_chart(build_wrong_share(tt["wrong_share"]), width="stretch", key="wrong_share_chart")
    st.table({"Tourenzahl": [r["n"] for r in tt["normal_scores"]], "mittlerer LOF der normalen Touren (Median)": [f"{r['median']:.3f}" for r in tt["normal_scores"]]})
    st.caption("Links: der LOF der Normalen liegt bei 1.01-1.03 - für 20 bis 600 Touren (Tabelle; k = 20, bei n = 20 nur k = 10) - deshalb hat die Schwelle eine Bedeutung. Das Fenster ist trotzdem eng: F1 0.52 / 0.69 / 0.82 / 0.94 / 0.96 / 0.85 bei 1.1 / 1.2 / 1.3 / 1.5 / 2.0 / 2.5 "
               "(Fehlalarmrate 20.5 % bei 1.1, Recall 0.75 bei 2.5). Rechts: ein falsch angenommener Anteil kostet alle vier Detektoren gleich viel, denn dann entscheidet nur die Rangfolge (0.97 → 0.67 bei LOF).")

st.markdown("---")

st.subheader("🔬 Kosten und Einheiten")
if st.button("Rechenzeit und Rohdaten-Vergleich berechnen (dauert etwa 10 Sekunden)", key="cost_start"):
    st.session_state["cost_on"] = True
if st.session_state.get("cost_on"):
    with st.spinner("Messe die Rechenzeit..."):
        ct = _costs(base_data, settings)
    c1, c2 = st.columns(2)
    c1.markdown("**Rechenzeit: Modell bauen und alle Touren bewerten**")
    c1.plotly_chart(build_costs(ct["times"]), width="stretch", key="cost_chart")
    c2.markdown("**LOF auf standardisierten Kennzahlen gegen Rohdaten**")
    c2.table({"": ["standardisiert", "Rohdaten (Meter, Minuten, ...)"], "AUC": [f"{ct['units'][0]['lof_auc']:.2f}", f"{ct['units'][1]['lof_auc']:.2f}"], "F1 bei der Schwelle": [f"{ct['units'][0]['lof_f1']:.2f}", f"{ct['units'][1]['lof_f1']:.2f}"],
              "Recall": [f"{ct['units'][0]['lof_recall']:.2f}", f"{ct['units'][1]['lof_recall']:.2f}"]})
    st.caption("Mittel über 3 bzw. 5 feste Datensätze. LOF rechnet alle paarweisen Abstände (O(n²)), ist aber vektorisiert und bei n ≤ 600 schneller als der Wald des Isolation Forest (rechnerabhängig: auf diesem Rechner 0.015 s gegen 0.14 s bei 600 Touren). "
               "Die Einheiten spielen für den Isolation Forest keine Rolle, für LOF schon: auf Rohdaten dominiert das Merkmal in Metern die Nachbarschaft (AUC 0.87, F1 0.68 statt 1.00 und 0.94).")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **k größer als jede Anomaliegruppe** | Ist k kleiner als die Gruppe, ist sie lokal dicht: AUC 0.35 bei 30 Touren und k = 20 (Isolation Forest 0.95, robust 1.00). Mit k = 40 0.97. Zu großes k schadet auch: bei n = 100 und einer Gruppe von 20 Touren fällt die AUC bei k = 90 auf 0.38, bei k = n − 1 auf 0.05 (verstreute Anomalien: 0.00). Das passende k setzt Wissen über die Gruppengröße voraus. | (ein k pro Anwendung; Ensembles über mehrere k) |
| **Wenige, verstreute Anomalien** | Bei vielen verstreuten Anomalien sind sie einander Nachbarn: AUC 0.90 / 0.72 / 0.64 bei 30 / 40 / 45 % (Isolation Forest 1.00), F1 0.43 / 0.15 / 0.09. | Isolation Forest (schnitt-basiert, ohne Nachbarschaft) |
| **Wenige irrelevante Merkmale** | Bei 40 Rauschmerkmalen bleibt die AUC bei 0.99, aber der Recall an der Schwelle 1.5 fällt auf 0.00 (F1 0.00; Isolation Forest 0.56): die Abstände verwischen, LOF → 1. Mit bekanntem Anteil 0.87. | Ensembles über Merkmalsteilmengen (**Feature Bagging**) |
| **Die Umgebung ist überall gleich dicht** | Auf gebogenem Normalbereich steigt die Fehlalarmrate von 1.4 % auf 7.8 % (F1 0.94 → 0.74), die AUC bleibt 1.00. Am Rand einer Fläche ist die Umgebung dünner. | Isolation Forest (F1 0.95 bei Krümmung 1) |
| **Einheiten spielen keine Rolle** | Für LOF nicht wahr: auf Rohdaten AUC 0.87 und F1 0.68 statt 1.00 und 0.94 - die Kennzahlen müssen standardisiert werden. | (Vorverarbeitung) |
| **Die Schwelle 1.5 passt überall** | Die Normalen liegen bei ≈ 1.02, aber das Fenster ist eng (F1 0.52 bei 1.1, 0.85 bei 2.5, bei 1.5-2.0 0.94-0.96); bei Rauschmerkmalen, gebogener Fläche und kleinem Abstand (Recall 0.33 bei 3 σ) passt sie nicht. | Kalibrierung; parameterfreie Verfahren (**ECOD**) |
"""
)
st.caption(
    "Die Nachbarn der Anomalie-Erkennung-Linie: die Wurzel Elliptic Envelope, Isolation Forest und Extended IF (gebaut), Feature Bagging (die Fortsetzung von LOF), One-Class SVM und Deep SVDD, ECOD und ein Autoencoder (noch nicht gebaut). "
    "Keiner ist überlegen: LOF gewinnt bei dichten Gruppen und in der Lücke (mit passendem k), bei kleinen Stichproben an der Schwelle und ist schnell; er verliert bei vielen Anomalien, vielen Rauschmerkmalen und gebogenen Flächen."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**k-Nachbarschaft.** Für eine Menge $X$ von $n$ Punkten und $a \in X$: $N_k(a)$ = die $k$ nächsten Punkte (euklidisch, ohne $a$; bei Abstandsgleichheit entscheidet die Reihenfolge der Indizes, wie in scikit-learn), $\text{kdist}(a) = \max_{b \in N_k(a)} d(a, b)$.

**Erreichbarkeit und Dichte.** $\text{reach}_k(a, b) = \max\{\text{kdist}(b),\, d(a, b)\}$, $\ \text{lrd}_k(a) = \Big(\tfrac{1}{k}\sum_{b \in N_k(a)} \text{reach}_k(a, b)\Big)^{-1}$ (mit einem kleinen $\varepsilon$ im Nenner gegen Duplikate).

**LOF.** $\text{LOF}_k(a) = \dfrac{\tfrac{1}{k}\sum_{b \in N_k(a)} \text{lrd}_k(b)}{\text{lrd}_k(a)}$. Ein Punkt in einer Umgebung von gleichmäßiger Dichte hat LOF $\approx 1$, ein Punkt in dünnerer Umgebung als seine Nachbarn LOF $> 1$.

**Standardisieren.** LOF rechnet auf $z_j = (x_j - \bar x_j)/s_j$, damit kein Merkmal die Abstände dominiert; der Isolation Forest ist gegen Skalierung unempfindlich, LOF nicht.

**Neue Punkte** (Score-Karten): der LOF eines Punkts $q$ gegen die Trainingsmenge mit $N_k(q)$ und $\text{reach}_k(q, b) = \max\{\text{kdist}(b), d(q, b)\}$ (wie `novelty=True` in scikit-learn).

**Schwelle und Vergleich.** Standard: $\text{LOF} > 1{,}5$ (Faustregel), Isolation Forest $s > 0{,}5$, $\chi^2$-Quantil bei klassisch und robust; alternativ die $\lceil \alpha n \rceil$ größten Werte bei angenommenem Anteil $\alpha$. Kennzahlen: AUC (Rangsumme, Bindungen halb), mittlere Präzision, Precision, Recall, F1, Fehlalarmrate.

**Grenzen.** (1) $k$ muss größer als jede Anomaliegruppe und kleiner als $n$ minus Gruppe sein. (2) Viele verstreute Anomalien werden einander Nachbarn. (3) Viele irrelevante Merkmale ziehen die Abstände zusammen (Dimensionsfluch). (4) Nicht skaleninvariant, nicht deutbar auf gebogenen Flächen.

Implementiert in `lof_algorithm.py` (Abstände, k-Nachbarn, Erreichbarkeit, lrd, LOF, LOF neuer Punkte), `lof_isolation_forest.py` und `lof_ee_algorithm.py` (Isolation Forest, klassisch und MCD, wortgleich aus den Vorgängern),
`lof_scenario.py` (Touren mit Betriebsarten, Krümmung, Anomalien, Rauschmerkmalen), `lof_evaluation.py` (Kennzahlen, Sweeps, Experimente, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
