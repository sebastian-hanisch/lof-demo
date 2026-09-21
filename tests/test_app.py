"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Randgrößen, Schritt-Zustand, ausgeblendete Regler, k-Grenze, Experimente auf Abruf, Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import lof_constants as C
from lof_evaluation import SWEEP_LABELS, k_max
from lof_presets import PRESET_KEYS, k_cap

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
EXPECTED_KIND = {"Standardfall": "success", "Dichte Gruppe, k zu klein (20)": "warning", "Dichte Gruppe, k passend (40)": "success", "Lücke, 3 Betriebsarten (k = 60)": "success",
                 "Viele Rauschmerkmale (40)": "warning", "Wenige Touren, viele Merkmale": "success", "Viele Anomalien (40 %)": "warning"}


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    """Erst Tourenzahl und Betriebsarten setzen und laufen lassen (die Grenze von k und die Art "Lücke" hängen davon ab; AppTest prüft gegen die Optionen des vorigen Laufs), dann alles andere."""
    at.session_state["n_tours_slider"] = p["n"]
    at.session_state["n_modes_slider"] = p["n_modes"]
    at.run()
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox)}


def _slider(at, prefix):
    return [s for s in at.sidebar.slider if s.label.startswith(prefix)][0]


def test_default_renders_without_exception():
    at = _run()
    assert any("LOF in Aktion" in m.value for m in at.markdown)
    assert not at.error and len(at.success) == 1 and not at.warning


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdict_kind(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    kind = EXPECTED_KIND[name]
    assert (len(at.success) == 1 and not at.warning) if kind == "success" else (len(at.warning) == 1 and not at.success)


def test_extreme_settings_render():
    def small(at):
        at.session_state["n_tours_slider"] = C.N_TOURS_MIN
        at.session_state["p_slider"] = C.P_MAX
        at.session_state["n_noise_slider"] = C.N_NOISE_MAX
        at.session_state["contamination_slider"] = C.CONTAMINATION_MIN
        at.session_state["k_slider"] = C.K_MAX
    _run(small)

    def large(at):
        at.session_state["n_tours_slider"] = C.N_TOURS_MAX
        at.session_state["p_slider"] = C.P_MIN
        at.session_state["n_modes_slider"] = C.N_MODES_MAX
        at.session_state["curvature_slider"] = C.CURVATURE_MAX
        at.session_state["noise_slider"] = C.NOISE_MAX
        at.session_state["contamination_slider"] = C.CONTAMINATION_MAX
        at.session_state["kind_select"] = "cluster"
        at.session_state["strength_slider"] = C.STRENGTH_MIN
        at.session_state["k_slider"] = C.K_MAX
    _run(large)


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("p", [2, 12, 30])
def test_every_step_renders(step, p):
    def setup(at):
        at.session_state["p_slider"] = p
        at.session_state["lof_step"] = step
    _run(setup)


def test_every_step_renders_with_few_tours_many_features_noise_and_the_share_threshold():
    def setup(at):
        at.session_state["n_tours_slider"] = 20
        at.session_state["p_slider"] = 30
        at.session_state["n_noise_slider"] = 40
    for step in (1, 2, 3, 4, 5):
        at = _run(setup)
        at.session_state["lof_step"] = step
        at.run()
        assert not at.exception
    at = _run(lambda a: a.session_state.__setitem__("threshold_kind_select", "share"))
    for step in (3, 4, 5):
        at.session_state["lof_step"] = step
        at.run()
        assert not at.exception


def test_k_slider_is_capped_by_half_the_number_of_tours_and_its_state_is_clamped():
    at = _run()
    k_slider = _slider(at, "Nachbarn k")
    assert k_slider.max == k_cap(300) == k_max(300) == 100 and k_slider.value == C.DEFAULT_K
    at.session_state["k_slider"] = 100
    at.run()
    at.session_state["n_tours_slider"] = 100
    at.run()
    assert not at.exception
    k_slider = _slider(at, "Nachbarn k")
    assert k_slider.max == 50 and k_slider.value == 50                                         # der Zustand wurde auf n / 2 begrenzt
    at.session_state["n_tours_slider"] = 20
    at.run()
    assert not at.exception and _slider(at, "Nachbarn k").max == 10 and _slider(at, "Nachbarn k").value == 10


def test_threshold_controls_are_hidden_per_kind_and_their_values_are_kept():
    at = _run(lambda a: (a.session_state.__setitem__("cutoff_slider", 1.75), a.session_state.__setitem__("quantile_slider", 0.99)))
    assert {"LOF-Schwelle", "Schwelle: χ²-Quantil (klassisch, robust)"} <= _labels(at) and "Angenommener Anteil der Anomalien [%]" not in _labels(at)
    at.session_state["threshold_kind_select"] = "share"
    at.run()
    assert not at.exception and "Angenommener Anteil der Anomalien [%]" in _labels(at) and not ({"LOF-Schwelle", "Schwelle: χ²-Quantil (klassisch, robust)"} & _labels(at))
    at.session_state["share_slider"] = 25
    at.run()
    at.session_state["threshold_kind_select"] = "standard"
    at.run()
    assert not at.exception
    assert _slider(at, "LOF-Schwelle").value == 1.75 and _slider(at, "Schwelle: χ²").value == 0.99
    at.session_state["threshold_kind_select"] = "share"
    at.run()
    assert _slider(at, "Angenommener").value == 25


def test_sweep_options_follow_the_threshold_kind_and_the_gap_kind():
    at = _run()
    options = [s for s in at.selectbox if s.key == "sweep_select"][0].options
    assert SWEEP_LABELS["cutoff"] in options and SWEEP_LABELS["quantile"] in options and SWEEP_LABELS["share"] not in options and SWEEP_LABELS["k"] in options
    at.session_state["threshold_kind_select"] = "share"
    at.run()
    options = [s for s in at.selectbox if s.key == "sweep_select"][0].options
    assert SWEEP_LABELS["share"] in options and SWEEP_LABELS["cutoff"] not in options and SWEEP_LABELS["quantile"] not in options
    at = _run(lambda a: a.session_state.__setitem__("n_modes_slider", 2))
    at.session_state["kind_select"] = "gap"
    at.run()
    options = [s for s in at.selectbox if s.key == "sweep_select"][0].options
    assert not ({SWEEP_LABELS["strength"], SWEEP_LABELS["n_modes"]} & set(options)) and SWEEP_LABELS["contamination"] in options


def test_strength_slider_is_hidden_for_gap_anomalies_and_its_value_is_kept():
    at = _run(lambda a: (a.session_state.__setitem__("strength_slider", 9.0)))
    assert "Abstand der Anomalien (Faktor-σ)" in _labels(at)
    at.session_state["n_modes_slider"] = 2
    at.run()
    at.session_state["kind_select"] = "gap"
    at.run()
    assert not at.exception and "Abstand der Anomalien (Faktor-σ)" not in _labels(at)
    at.session_state["kind_select"] = "scattered"
    at.run()
    assert not at.exception and _slider(at, "Abstand").value == 9.0


def test_gap_kind_needs_two_modes_and_falls_back_when_modes_drop():
    at = _run(lambda a: a.session_state.__setitem__("n_modes_slider", 2))
    at.session_state["kind_select"] = "gap"
    at.run()
    assert not at.exception and C.KIND_LABELS["gap"] in [s for s in at.sidebar.selectbox if s.label == "Art der Anomalien"][0].options
    at.session_state["n_modes_slider"] = 1
    at.run()
    assert not at.exception and C.KIND_LABELS["gap"] not in [s for s in at.sidebar.selectbox if s.label == "Art der Anomalien"][0].options
    assert at.session_state["kind_select"] == "scattered"


def test_step_state_resets_when_the_data_or_settings_change_and_survives_reruns():
    at = _run()
    at.session_state["lof_step"] = 4
    at.run()
    assert not at.exception and at.session_state["lof_step"] == 4
    at.session_state["contamination_slider"] = 20
    at.run()
    assert not at.exception and at.session_state["lof_step"] == 1


def test_experiments_run_on_demand_and_sweeps_run():
    at = _run()
    for parameter in ("k", "n_noise", "cutoff", "contamination", "n"):
        [s for s in at.selectbox if s.key == "sweep_select"][0].select(parameter)
        at.run()
        [b for b in at.button if b.key == "sweep_start"][0].click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    keys = ("group", "gap", "modes", "masking", "k", "dimension", "threshold", "cost")
    for key in keys:
        [b for b in at.button if b.key == f"{key}_start"][0].click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    assert all(at.session_state[f"{k}_on"] for k in keys)


def test_every_figure_of_the_visualisation_module_is_axis_locked():
    source = (ROOT / "lof_visualization.py").read_text(encoding="utf-8")
    assert len(re.findall(r"return lock_axes\(fig\)", source)) == len(re.findall(r"^def build_", source, flags=re.M))
    assert len(re.findall(r"^\s+return fig$", source, flags=re.M)) == 1


def test_every_plotly_chart_has_an_explicit_unique_key():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    parts = source.split("plotly_chart(")[1:]
    keys = [re.search(r'key="([a-z_]+)"', part.split("plotly_chart(")[0]) for part in parts]
    assert len(parts) >= 20 and all(keys)
    names = [k.group(1) for k in keys]
    assert len(set(names)) == len(names)


def test_app_has_no_dead_file_links_and_the_verbatim_footer():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert not re.findall(r"\]\([a-z_]+\.py\)", source)
    assert "https://sebastianhanisch.net/kontakt.html" in source and "Interesse an einer maßgeschneiderten Lösung für" in source
