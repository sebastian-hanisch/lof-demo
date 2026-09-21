"""Defaults, Regler-Grenzen und Presets für die LOF-Demo (Anomalie-Erkennung an Lieferrouten-Kennzahlen; Szenario, Isolation Forest und Vergleichsschätzer aus den Vorgänger-Demos)."""

# --- Merkmale: die 12 Kennzahlen der PCA-Demo (Name, Einheit, Mittelwert, typische Streuung), dazu Zusatzmerkmale für den Fall n < p --------------------
FEATURES = (
    ("Distanz", "m", 45000.0, 15000.0),
    ("Stopps", "Anzahl", 60.0, 20.0),
    ("Ladegewicht", "kg", 1200.0, 400.0),
    ("Zeitfenster-Enge", "min", 90.0, 30.0),
    ("Verspätung", "min", 12.0, 8.0),
    ("Überstunden", "min", 25.0, 15.0),
    ("Fahrzeit je km", "s", 90.0, 25.0),
    ("Stop-and-go-Anteil", "%", 22.0, 10.0),
    ("Parkzeit", "min", 35.0, 12.0),
    ("Retourenquote", "Anteil", 0.06, 0.02),
    ("Sonderwünsche", "Anzahl", 4.0, 2.0),
    ("Zustellversuche", "Anzahl", 1.3, 0.5),
)
N_BASE_FEATURES = len(FEATURES)
GROUP_OF_FEATURE = tuple(i // 3 for i in range(N_BASE_FEATURES))
# Reihenfolge, in der die ersten p Merkmale gewählt werden: reihum durch die vier Gruppen, damit schon p = 2 beide latenten Faktoren sieht (Distanz und Zeitfenster-Enge)
FEATURE_ORDER = (0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11)
EXTRA_MEAN, EXTRA_SCALE = 50.0, 10.0                   # Zusatzmerkmale 13 ... p (zufällige Mischungen der latenten Faktoren plus eigenes Rauschen)

# --- Regler ------------------------------------------------------------------------------------------------------------
DEFAULT_N_TOURS = 300
N_TOURS_MIN, N_TOURS_MAX = 20, 600
DEFAULT_P = 12
DEFAULT_N_NOISE = 0
N_NOISE_MIN, N_NOISE_MAX = 0, 40
P_MIN, P_MAX = 2, 30
N_MODES_MIN, N_MODES_MAX = 1, 3
DEFAULT_N_MODES = 1
DEFAULT_CURVATURE = 0.0
CURVATURE_MIN, CURVATURE_MAX = 0.0, 1.0
DEFAULT_NOISE = 0.25
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_CONTAMINATION = 10                             # Prozent
CONTAMINATION_MIN, CONTAMINATION_MAX = 1, 45
KINDS = ("scattered", "cluster", "gap")
KIND_LABELS = {"scattered": "verstreute Ausreißer", "cluster": "dichte Gruppe abseits", "gap": "in der Lücke zwischen den Betriebsarten"}
DEFAULT_KIND = "scattered"
DEFAULT_STRENGTH = 6.0
STRENGTH_MIN, STRENGTH_MAX = 3.0, 12.0
DEFAULT_SUPPORT = 0.5                                   # Stützanteil h/n (0.5 = größter Bruchpunkt); 1.0 = alle Punkte = klassische Schätzung
SUPPORT_MIN, SUPPORT_MAX = 0.5, 1.0
DEFAULT_QUANTILE = 0.975                                # chi^2-Quantil der Schwelle
QUANTILE_MIN, QUANTILE_MAX = 0.90, 0.999
DEFAULT_REWEIGHT = True
DEFAULT_SEED = 7

# --- Isolation Forest (Vergleich) ------------------------------------------------------------------------------------
DEFAULT_TREES = 100
DEFAULT_PSI = 256                                       # Unterstichprobe je Baum (höchstens die Tourenzahl)
DEFAULT_CUTOFF_IF = 0.5                                 # nominelle Score-Schwelle des Isolation Forest

# --- LOF ------------------------------------------------------------------------------------------------------------------
DEFAULT_K = 20                                          # Nachbarn (Standard in der Literatur/sklearn)
K_MIN, K_MAX = 3, 100
THRESHOLD_KINDS = ("standard", "share")
THRESHOLD_LABELS = {"standard": "Standard (LOF-Schwelle bzw. Score 0.5 bzw. χ²-Quantil)", "share": "erwarteter Anteil (für alle vier)"}
DEFAULT_THRESHOLD_KIND = "standard"
DEFAULT_CUTOFF = 1.5                                    # LOF-Schwelle (Faustregel; zu messen)
CUTOFF_MIN, CUTOFF_MAX = 1.0, 3.0
DEFAULT_SHARE = 10                                      # angenommener Anteil der Anomalien [%] (= der wahre im Standardfall)
SHARE_MIN, SHARE_MAX = 1, 45

# --- Erzeugung ---------------------------------------------------------------------------------------------------------
Q = 2                                                   # latente Faktoren (fest; die PCA-Demo variiert sie, hier geht es um Anomalien)
CROSS_LOADING = 0.15
WITHIN_LOADINGS = (0.95, 0.9, 0.85)
CURVATURE_FREQUENCY = 1.6
CURVATURE_AMPLITUDE = 2.0
LAYOUT_SEED = 20240915                                  # dieselben festen Matrizen wie in der PCA-Demo
EXTRA_LAYOUT_SEED = LAYOUT_SEED + 2
MODE_RADIUS = 2.2                                       # Betriebsarten liegen auf einem Kreis dieses Radius im Faktorraum
MODE_SD = 0.6                                           # Streuung innerhalb einer Betriebsart (bei nur einer Betriebsart 1, wie in der PCA-Demo)
CLUSTER_SD = 0.3                                        # Streuung der dichten Anomalie-Gruppe
GAP_SD = 0.3                                            # Streuung der Anomalien in der Lücke
CLUSTER_ANGLE = 0.6                                     # Richtung der dichten Gruppe im Faktorraum (Bogenmaß)

# --- Auswertung --------------------------------------------------------------------------------------------------------
MCD_STARTS = 500                                        # zufällige Startmengen (je zwei C-Schritte)
MCD_KEEP = 10                                           # die besten davon laufen bis zur Konvergenz
MCD_INITIAL_STEPS = 2
MCD_MAX_STEPS = 50
RIDGE = 1e-9                                            # relative Regularisierung der Kovarianz (n < p)
SWEEP_SEEDS = tuple(100_000 + i for i in range(5))

# --- Presets (werden nach der Messung festgelegt) ------------------------------------------------------------------------


def _preset(**kw):
    base = dict(n=DEFAULT_N_TOURS, p=DEFAULT_P, n_noise=DEFAULT_N_NOISE, n_modes=DEFAULT_N_MODES, curvature=DEFAULT_CURVATURE, noise=DEFAULT_NOISE, contamination=DEFAULT_CONTAMINATION,
                kind=DEFAULT_KIND, strength=DEFAULT_STRENGTH, k=DEFAULT_K, threshold_kind=DEFAULT_THRESHOLD_KIND, cutoff=DEFAULT_CUTOFF, quantile=DEFAULT_QUANTILE, share=DEFAULT_SHARE, seed=DEFAULT_SEED)
    base.update(kw)
    return base
PRESETS = {
    "Standardfall": _preset(),
    "Dichte Gruppe, k zu klein (20)": _preset(kind="cluster", k=20),
    "Dichte Gruppe, k passend (40)": _preset(kind="cluster", k=40),
    "Lücke, 3 Betriebsarten (k = 60)": _preset(n_modes=3, kind="gap", k=60),
    "Viele Rauschmerkmale (40)": _preset(n_noise=40),
    "Wenige Touren, viele Merkmale": _preset(n=20, p=30, k=10),
    "Viele Anomalien (40 %)": _preset(contamination=40),
}
PRESET_HELP = {
    "Standardfall": "Im Mittel über fünf Aufnahmen: 300 Touren, 12 Merkmale, 10 % verstreute Anomalien, k = 20. LOF und Isolation Forest haben AUC 1.00; bei der Schwelle (LOF 1.5, Isolation Forest 0.5) F1 0.94 (Fehlalarmrate 1.4 %) "
                    "und 0.96. Die normalen Touren liegen bei LOF ≈ 1.02.",
    "Dichte Gruppe, k zu klein (20)": "Im Mittel über fünf Aufnahmen: 10 % der Touren (30) bilden eine dichte Gruppe abseits, aber k = 20 ist kleiner als die Gruppe: jede Tour der Gruppe hat nur Gruppen-Nachbarn und ist \"lokal dicht\" (LOF ≈ 1). "
                                      "AUC 0.35 (unter Raten), Recall 1 %; Isolation Forest 0.95, robuste Schätzung 1.00.",
    "Dichte Gruppe, k passend (40)": "Im Mittel über fünf Aufnahmen: dieselbe Gruppe (30 Touren), aber k = 40 > 30: die Nachbarn der Gruppe liegen teils bei den Normalen, die Gruppe fällt auf. AUC 0.97 (Isolation Forest 0.95, robust 1.00); "
                                     "F1 0.62 bei Recall 0.67 (Isolation Forest 0.66 bei 12 % Fehlalarmen). Das passende k setzt Wissen über die Gruppengröße voraus.",
    "Lücke, 3 Betriebsarten (k = 60)": "Im Mittel über fünf Aufnahmen: drei Betriebsarten, 10 % Anomalien (30 Touren) in der Lücke dazwischen, k = 60. LOF hat AUC 0.84, der Isolation Forest 0.27, die Wurzel 0.38 - hier ist LOF der einzige Detektor "
                                       "über Raten. An der Schwelle 1.5 findet er trotzdem nichts (F1 0.00): die Lücken-Touren liegen im Rang oben, aber nicht ganz oben.",
    "Viele Rauschmerkmale (40)": "Im Mittel über fünf Aufnahmen: 40 unabhängige Rauschmerkmale. Die Rangfolge bleibt gut (AUC 0.99, wie beim Isolation Forest), aber die LOF-Werte rücken an 1 heran: bei der Schwelle 1.5 Recall 0, F1 0.00 "
                                 "(Isolation Forest 0.56); mit bekanntem Anteil 0.87. Dimensionsfluch - der Ansatzpunkt für Feature Bagging.",
    "Wenige Touren, viele Merkmale": "Im Mittel über fünf Aufnahmen: 20 Touren, 30 Merkmale, k = 10. Beide Verfahren haben AUC 1.00 (klassisch 0.60, robust 0.55); LOF F1 0.92 bei 2 % Fehlalarmen, der Isolation Forest 0.61 bei 14 % - "
                                     "der LOF der Normalen liegt bei ≈ 1, unabhängig von der Tourenzahl.",
    "Viele Anomalien (40 %)": "Im Mittel über fünf Aufnahmen: 40 % verstreute Anomalien (120 Touren). LOF hat AUC 0.72 und F1 0.15 (Recall 8 %), der Isolation Forest 1.00 und 0.87, die robuste Schätzung 0.97: bei so vielen Anomalien "
                              "sind sie einander Nachbarn und wirken lokal nicht dünner besetzt.",
}
# Bänder (Seed des Presets; mit dem ausgelieferten Code kalibriert, bewusst weit): Kennzahlen der Detektoren (lof_*, iforest_*, classical_*, robust_*) und erlaubte Urteile (verdict)
PRESET_EXPECTED_BANDS = {
    "Standardfall": {"lof_auc": (0.97, 1.0), "iforest_auc": (0.97, 1.0), "lof_f1": (0.85, 1.0), "iforest_f1": (0.85, 1.0), "verdict": ("comparable",)},
    "Dichte Gruppe, k zu klein (20)": {"lof_auc": (0.0, 0.6), "iforest_auc": (0.8, 1.0), "robust_auc": (0.95, 1.0), "lof_recall": (0.0, 0.1), "verdict": ("k_window",)},
    "Dichte Gruppe, k passend (40)": {"lof_auc": (0.9, 1.0), "iforest_auc": (0.85, 0.99), "lof_recall": (0.6, 1.0), "iforest_false_alarm": (0.05, 0.3), "verdict": ("lof_wins", "comparable")},
    "Lücke, 3 Betriebsarten (k = 60)": {"lof_auc": (0.7, 1.0), "iforest_auc": (0.0, 0.5), "robust_auc": (0.0, 0.6), "lof_recall": (0.0, 0.1), "verdict": ("lof_wins",)},
    "Viele Rauschmerkmale (40)": {"lof_auc": (0.9, 1.0), "iforest_auc": (0.9, 1.0), "lof_recall": (0.0, 0.2), "verdict": ("threshold_off",)},
    "Wenige Touren, viele Merkmale": {"lof_auc": (0.95, 1.0), "iforest_auc": (0.95, 1.0), "lof_f1": (0.8, 1.0), "iforest_f1": (0.4, 0.8), "robust_recall": (0.0, 0.2), "verdict": ("lof_wins", "threshold_off")},
    "Viele Anomalien (40 %)": {"lof_auc": (0.5, 0.85), "iforest_auc": (0.9, 1.0), "robust_auc": (0.9, 1.0), "lof_recall": (0.0, 0.3), "verdict": ("others_win",)},
}
