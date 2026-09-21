"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert, bewusst weit)."""

import pytest

import lof_constants as C
from lof_evaluation import Settings, analyse_for, verdict
from lof_presets import k_cap


def _settings(p):
    return Settings(k=p["k"], threshold_kind=p["threshold_kind"], cutoff=p["cutoff"], quantile=p["quantile"], share=p["share"])


def _measure(p):
    params = (p["n"], p["p"], p["n_noise"], p["n_modes"], p["curvature"], p["noise"], p["contamination"], p["kind"], p["strength"], p["seed"])
    a = analyse_for(params, _settings(p))
    out = {"verdict": verdict(a)[1]}
    for det in ("lof", "iforest", "classical", "robust"):
        for key, value in a.scores[det].items():
            out[f"{det}_{key}"] = value
    return out


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 7


def test_wins_and_losses_are_both_shown():
    allowed = {c for b in C.PRESET_EXPECTED_BANDS.values() for c in b["verdict"]}
    assert {"comparable", "k_window", "lof_wins", "threshold_off", "others_win"} <= allowed
    assert sum("lof_wins" in b["verdict"] for b in C.PRESET_EXPECTED_BANDS.values()) == 3 and sum(b["verdict"] == ("others_win",) for b in C.PRESET_EXPECTED_BANDS.values()) == 1


def test_the_first_preset_is_the_default_setting():
    assert C.PRESETS["Standardfall"] == C._preset()


def test_preset_settings_are_within_slider_bounds():
    for p in C.PRESETS.values():
        assert C.N_TOURS_MIN <= p["n"] <= C.N_TOURS_MAX and p["n"] % 10 == 0 and C.P_MIN <= p["p"] <= C.P_MAX and C.N_NOISE_MIN <= p["n_noise"] <= C.N_NOISE_MAX and p["n_noise"] % 5 == 0
        assert C.N_MODES_MIN <= p["n_modes"] <= C.N_MODES_MAX and C.CURVATURE_MIN <= p["curvature"] <= C.CURVATURE_MAX and abs(p["curvature"] * 4 - round(p["curvature"] * 4)) < 1e-9
        assert C.NOISE_MIN <= p["noise"] <= C.NOISE_MAX and C.CONTAMINATION_MIN <= p["contamination"] <= C.CONTAMINATION_MAX
        assert p["kind"] in C.KINDS and (p["kind"] != "gap" or p["n_modes"] >= 2)
        assert C.STRENGTH_MIN <= p["strength"] <= C.STRENGTH_MAX and abs(p["strength"] * 2 - round(p["strength"] * 2)) < 1e-9
        assert C.K_MIN <= p["k"] <= k_cap(p["n"])                                                                                                   # k nicht über n / 2
        assert p["threshold_kind"] in C.THRESHOLD_KINDS and C.CUTOFF_MIN <= p["cutoff"] <= C.CUTOFF_MAX and abs(p["cutoff"] * 20 - round(p["cutoff"] * 20)) < 1e-9 and C.QUANTILE_MIN <= p["quantile"] <= C.QUANTILE_MAX and C.SHARE_MIN <= p["share"] <= C.SHARE_MAX


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if key == "verdict":
            assert value in expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"
