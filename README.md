# Local Outlier Factor – Anomalien haben eine dünnere Nachbarschaft – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-lof-demo.streamlit.app/)**

Viertes Stück der **Anomalie-Erkennung-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – den **Local Outlier Factor (LOF)** – an einem wachsenden Beispiel, gegen den [isolation-forest-demo](../isolation-forest-demo)
und die beiden Schätzer der Wurzel ([elliptic-envelope-demo](../elliptic-envelope-demo): klassisch und robust per MCD). Vehikel: dieselben **Lieferrouten-Kennzahlen** wie in den Vorgängern und der [pca-demo](../pca-demo)
(Szenario, Isolation Forest und Wurzel-Schätzer wortgleich übernommen, per Test gegen eingefrorene Werte geprüft).

**Einordnung in die Reihe (die Kanten des Graphen):** LOF ist ein **eigener Ast direkt nach der Wurzel**, neben Isolation Forest und Extended Isolation Forest. Die Wurzel beschreibt das Normale als eine Wolke, der Isolation Forest fragt, wie schnell sich eine Tour abtrennen lässt.
LOF fragt Lokaleres: **ist die Umgebung dieser Tour dünner besetzt als die Umgebung ihrer k Nachbarn?** (LOF ≈ 1 = so dicht wie die Nachbarn.) Er setzt an den in den Vorgängern gemessenen Schwächen an – **dichte Anomaliegruppen (Masking)** und die **Lücke zwischen den Betriebsarten** –
und hat eigene: er braucht ein **k, das zur Gruppengröße passt**, und leidet unter **irrelevanten Merkmalen** (der Ansatzpunkt für [Feature Bagging](../feature-bagging-demo): mit den Teilmengengrößen des Originals hilft das Ensemble kaum, mit winzigen Teilmengen deutlich, aber mit großer Streuung). Die Linie hat **keinen Konvergenzpunkt**.
```
elliptic-envelope-demo (Wurzel: robuste Ellipse)
  ├─ ECOD                       (Kontrast: verteilungsfrei)                              [nicht gebaut]
  ├─ lof-demo → feature-bagging-demo (lokale Dichte; Ensembles gegen viele Merkmale)      [dieses Stück → Nachfolger gebaut]
  ├─ One-Class SVM → Deep SVDD  (gelernte Grenze)                                        [nicht gebaut]
  ├─ isolation-forest-demo → extended-isolation-forest-demo (Zufallsbäume)               [gebaut]
  └─ autoencoder-anomalie-demo  (Rekonstruktionsfehler)                                  [gebaut]
```

| Frage | Ergebnis (300 Touren, 12 Merkmale, 10 % verstreute Anomalien im Abstand 6 Faktor-σ, ein Normalbereich, Rauschen 0.25; k = 20, LOF-Schwelle 1.5, Isolation Forest 100 Bäume × ψ = 256, Score-Schwelle 0.5, χ²-Quantil 0.975; Mittel über 5 feste Datensätze, Seeds 100000–100004) |
|---|---|
| Standardfall | ✅ Rangfolge bei allen vieren gut (AUC LOF 1.00, Isolation Forest 1.00, robust 1.00, klassisch 0.95); F1 an der Schwelle **0.94** (LOF, Fehlalarmrate 1.4 %) gegen 0.96 (Isolation Forest), 0.84 (robust), 0.57 (klassisch). Der LOF der Normalen liegt bei **1.02** |
| Dichte Gruppe abseits | ✅/❌ **k gegen Gruppengröße**: LOF sieht die Gruppe nur, wenn k größer ist als sie. 30 Touren: AUC **0.35** mit k = 20 (unter Raten; Isolation Forest 0.95, robust 1.00), **0.97** mit k = 40. Gruppen von 15 / 30 / 60 / 90 Touren brauchen k = 20 / 40 / 80 / >100 (AUC 0.99 / 0.97 / 0.95 / 0.71). Mit k = 1.5 × Gruppengröße bis 25 % besser als der Isolation Forest (0.99 / 0.99 / 0.98 / 0.98 / 0.94 gegen 0.99 / 0.95 / 0.91 / 0.85 / 0.79 bei 5 / 10 / 15 / 20 / 25 %) – aber das setzt voraus, dass man die Gruppengröße kennt |
| Anomalien in der Lücke | ✅ **LOF ist der einzige Detektor über Raten**: drei Betriebsarten, 30 Lücken-Touren, k = 60–70: AUC **0.84–0.86** (Isolation Forest 0.27, robust 0.38, klassisch 0.38); zwei Betriebsarten k = 40: 0.71 (Isolation Forest 0.54, robust 0.40); 15 Lücken-Touren, zwei Betriebsarten: 0.86 (Isolation Forest 0.71). ❌ Mit k = 20 nur 0.53 – und ein **Fenster**: bei k = 90 fällt die AUC auf 0.37 / 0.14. An der Schwelle 1.5 findet LOF die Lücken-Touren trotzdem nicht (F1 0.00): sie liegen im Rang weit oben, aber nicht ganz oben |
| Kleine Stichproben | ✅ LOF ist kalibriert: der LOF der Normalen liegt bei **1.007 / 1.024 / 1.030 / 1.018 / 1.016** bei 20 / 50 / 100 / 300 / 600 Touren. F1 an der Schwelle bei 20 / 30 / 50 / 100 Touren **0.96 / 0.94 / 0.88 / 0.86** (k = min(20, n / 2)) gegen 0.60 / 0.67 / 0.78 / 0.89 beim Isolation Forest (Fehlalarmrate 15.6 % bei 20 Touren gegen 1.1 %); 20 Touren × 30 Merkmale: 0.92 gegen 0.61 |
| Rauschmerkmale | ❌ Rangfolge bleibt (AUC **0.99** bei 40 Rauschmerkmalen, wie beim Isolation Forest; robust 0.88), aber die LOF-Werte rücken an 1 heran: Recall an der Schwelle 1.00 / 0.71 / 0.18 / 0.03 / **0.00** bei 0 / 10 / 20 / 30 / 40 (Isolation Forest 0.99 / 0.91 / 0.71 / 0.58 / 0.39), F1 0.94 → 0.00 (0.96 → 0.56); mit bekanntem Anteil 0.87 (0.84). **Dimensionsfluch**, der auf die Schwelle wirkt, nicht auf die Rangfolge |
| Viele Anomalien | ❌ Verstreute Anomalien werden einander Nachbarn: AUC **0.90 / 0.72 / 0.64** bei 30 / 40 / 45 % (Isolation Forest überall 1.00), F1 0.43 / 0.15 / 0.09 (0.93 / 0.87 / 0.81); bei wenigen Anomalien dagegen ✅ besser an der Schwelle (2 %: F1 0.75 gegen 0.42) |
| Gebogener Normalbereich | ❌ AUC bleibt 1.00, aber die Fehlalarmrate steigt mit der Krümmung von 1.4 % auf **7.8 %** (F1 0.94 → 0.74; Isolation Forest 0.96 → 0.95, robust mit χ²-Schwelle 0.84 → 0.38): am Rand der Fläche ist die Umgebung dünner |
| Betriebsarten | ✅ LOF ist wie der Isolation Forest unempfindlich gegen mehrere Betriebsarten (AUC 1.00 bei 1 / 2 / 3; robust 1.00 / 0.95 / 0.85, F1 0.84 / 0.74 / 0.36) |
| Schwelle | ⚠️ LOF-Schwelle 1.1 / 1.2 / 1.3 / 1.5 / 2.0 / 2.5: F1 0.52 / 0.69 / 0.82 / **0.94 / 0.96** / 0.85 (Fehlalarmrate 20.5 % bis 0.0 %, Recall 1.00 bis 0.75): der Wert hat eine Bedeutung, das Fenster ist trotzdem eng. Ein falsch angenommener Anteil (½× / 2×) senkt F1 bei allen Detektoren gleich (0.97 → 0.67) |
| k nahe n | ❌ Bei 100 Touren: verstreute Anomalien AUC 1.00 bis k = 90, bei **k = 99 = n − 1: 0.00** (alle Touren haben dieselben Nachbarn, die Reihenfolge kehrt sich um); dichte Gruppe von 20 Touren: 0.99–1.00 für k = 30–70, 0.38 bei k = 90. Die App bietet deshalb höchstens k = n / 2 an |
| Einheiten | ⚠️ LOF ist **nicht skaleninvariant**: auf Rohdaten AUC 0.87, F1 0.68 statt 1.00 und 0.94 – die Kennzahlen müssen standardisiert werden (der Isolation Forest ist skaleninvariant) |
| Rechenzeit | ✅ LOF ist hier **schneller** als der Wald: 0.0004 / 0.0035 / 0.015 s gegen 0.06 / 0.13 / 0.14 s bei 100 / 300 / 600 Touren (rechnerabhängig; O(n²) vektorisiert, n ≤ 600) |

## Was die Demo zeigt

1. **LOF in Aktion** (Schritt-Slider + Abspielen): **Touren** → **Nachbarschaft** (die k Nachbarn einer normalen Tour und einer Sonderfahrt, gemessen in allen Merkmalen, gezeigt in der Ebene der zwei größten Streuungsrichtungen) → **Dichte und LOF**
   (lokale Erreichbarkeitsdichte je Tour, eigene Dichte gegen die der Nachbarn, LOF-Histogramm mit Schwelle) → **LOF-Karte** (Score-Karte über zwei Merkmale gegen die des Isolation Forest: runde Höhenlinien gegen Bänder entlang der Achsen) → **Ergebnis** (Kennzahlen und ROC-Kurven der vier Detektoren).
2. **Was die Detektoren gefunden haben – gegen die Wurzel:** AUC, Recall, Fehlalarmrate, F1 (dazu der F1 mit bekanntem Anteil), Urteil (Codes: Lücke → k passt nicht zur Gruppe → anderer Detektor besser → LOF besser → falsche Schwelle bei guter Rangfolge → Isolation Forest hat das bessere F1 → gleichauf), Detailtabelle mit Rechenzeiten.
3. **📐 Sweeps** über k, Touren, Merkmale, Rauschmerkmale, Betriebsarten, Krümmung, Rauschen, Anteil und Abstand der Anomalien, LOF-Schwelle, χ²-Quantil und angenommenen Anteil (feste Datensätze ab 100000, Streuung).
4. **🔬 Experimente auf Abruf:** k gegen die Größe der Anomaliegruppe, Anomalien in der Lücke, Betriebsarten × Art der Anomalien, Masking (Gruppe 2–45 %, dazu LOF mit passendem k), k nahe n, Rauschmerkmale und Tourenzahl × Merkmalszahl, Schwelle (LOF-Schwelle, falscher Anteil, mittlerer LOF je Tourenzahl), Kosten und Einheiten.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (k gegen Gruppengröße, viele verstreute Anomalien, Rauschmerkmale, gebogene Flächen, Einheiten, Schwelle).

Regler: Touren (20–600), Merkmale (2–30), Rauschmerkmale (0–40), Betriebsarten (1–3), Krümmung, Rauschen, Anteil der Anomalien (1–45 %), Art (verstreut / dichte Gruppe / in der Lücke – ab zwei Betriebsarten), Abstand (bei "Lücke" ausgeblendet, Wert bleibt erhalten),
**Nachbarn k** (3 bis höchstens n / 2, höchstens 100), **Schwelle** (Standard: LOF-Schwelle und χ²-Quantil, ausgeblendet beim erwarteten Anteil; sonst der angenommene Anteil für alle vier).

## Messwerte der Presets (Seed 7; sie prüfen sich mit weiten Bändern selbst)

| Preset | AUC LOF | F1 LOF | AUC IF | F1 IF | AUC robust | Urteil |
|---|---|---|---|---|---|---|
| Standardfall | 1.00 | 0.97 | 1.00 | 0.98 | 1.00 | gleich gut |
| Dichte Gruppe, k zu klein (20) | 0.31 | 0.00 | 0.97 | 0.63 | 1.00 | k passt nicht zur Gruppe |
| Dichte Gruppe, k passend (40) | 0.99 | 0.90 | 0.97 | 0.63 | 1.00 | LOF besser |
| Lücke, 3 Betriebsarten (k = 60) | 0.88 | 0.00 | 0.32 | 0.00 | 0.31 | LOF besser |
| Viele Rauschmerkmale (40) | 0.98 | 0.06 | 0.99 | 0.42 | 0.90 | Schwelle passt nicht |
| Wenige Touren, viele Merkmale (20 × 30, k = 10) | 1.00 | 1.00 | 1.00 | 0.57 | 0.64 | LOF besser (an der Schwelle) |
| Viele Anomalien (40 %) | 0.72 | 0.19 | 1.00 | 0.88 | 0.96 | anderer Detektor besser |

## Modell und Verfahren

- **Szenario** (`lof_scenario.py`): wortgleich aus den Vorgängern (zwei versteckte Faktoren, 12 Kennzahlen der PCA-Demo, Betriebsarten, Krümmung, Anomalien verstreut / dichte Gruppe / in der Lücke mit exaktem Anteil, Zusatzmerkmale bis p = 30, Rauschmerkmale); die Drehung des Extended-IF-Stücks entfällt (LOF ist rotationsinvariant).
- **LOF** (`lof_algorithm.py`, numpy von Grund auf): euklidische Abstände über die Gram-Matrix, genau k Nachbarn (bei Abstandsgleichheit entscheidet der Index, wie in scikit-learn), k-Distanz, Erreichbarkeitsdistanz reach(a, b) = max(kdist(b), d(a, b)), lrd(a) = 1 / Mittel(reach), LOF(a) = Mittel(lrd der Nachbarn) / lrd(a);
  ein kleines ε im Nenner fängt Duplikate ab; LOF neuer Punkte für die Score-Karten. Auf **standardisierten** Kennzahlen. **Schwelle:** Standard LOF > 1.5 (Faustregel) oder der **angenommene Anteil** (die größten Werte, für alle vier Detektoren).
- **Isolation Forest** (`lof_isolation_forest.py`) und **Wurzel** (`lof_ee_algorithm.py`: χ² ohne scipy, klassisch, FastMCD): wortgleich aus den Vorgängern.
- **Auswertung** (`lof_evaluation.py`): AUC, mittlere Präzision, Precision, Recall, F1, Fehlalarmrate für vier Detektoren; **F1 mit bekanntem Anteil** als Referenz für die Schwelle; Sweeps, Experiment-Tabellen, Urteil. Bei den Sweeps über die Tourenzahl folgt k der Tourenzahl (höchstens n / 2).

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutungen, die nicht (ganz) stimmten:** (1) "LOF trennt dichte Gruppen und die Lücke deutlich besser" – nur mit **passendem k**, und das passende k setzt Wissen über die Gruppengröße voraus; mit dem Standard-k 20 ist LOF bei einer Gruppe von 30 Touren **schlechter als Raten** (0.35). (2) "LOF gewinnt auf gebogenem Normalbereich" – die Rangfolge ist für alle gleich (AUC 1.00),
  an der Schwelle verliert LOF (F1 0.74, Fehlalarmrate 7.8 %). (3) "Rauschmerkmale schaden LOF stärker als dem Isolation Forest" – für die **Rangfolge** nicht (AUC 0.99 bei beiden), für die **Schwelle** ja (Recall 0.00 gegen 0.39). (4) "O(n²) macht LOF spürbar langsamer" – bei n ≤ 600 ist er vektorisiert
  **schneller** als der Wald (0.015 gegen 0.14 s). Nicht vorhergesehen: (5) **viele verstreute Anomalien** (AUC 0.64 bei 45 %, Isolation Forest 1.00), (6) **k = n − 1** kehrt die Rangfolge um (AUC 0.00), (7) der LOF der Normalen ist zwar überall ≈ 1 (kalibriert), das nutzbare Schwellenfenster (1.5–2.0) aber eng.
- **Ein Sweep-Artefakt, das auffiel:** beim Sweep über die Tourenzahl gab k = 20 bei 20 Touren AUC 0.11 – k = n − 1 (bei k = 20 auf 19 begrenzt) ist der degenerierte Fall, kein Messwert für LOF. Der Sweep begrenzt k jetzt auf n / 2, der Regler ebenfalls.
- **Der Vergleich der Geister-Regionen** (Score-Anisotropie wie im Vorgänger) ist bewusst nur eine **Karte**: LOF-Werte (1 und größer) und Isolation-Forest-Scores (0 bis 1) haben verschiedene Einheiten, ein Zahlenvergleich wäre irreführend. Die Karte zeigt runde Höhenlinien beim LOF, Bänder entlang der Achsen beim Isolation Forest.
- **Das passende k ist Vorwissen:** die Demo kennt die Gruppengröße (die Tabelle "Masking" nutzt k = 1.5 × Gruppengröße als Referenz), ein Anwender nicht. Ensembles über mehrere k sind ein möglicher Ausweg (nicht gebaut).
- **Synthetische Daten:** zwei Faktoren, lineare Mischung, weißes Gauß'sches Rauschen, feste Betriebsarten-Geometrie; die Anomalien liegen im Faktorraum weit draußen (Abstand 3–12 σ), was allen Verfahren entgegenkommt. Literatur nur mit Namen: Breunig, Kriegel, Ng und Sander (LOF).

## Verifikation

- LOF: Handinstanz in einer Dimension (k-Distanz, Erreichbarkeit, lrd, LOF von Hand), zwei Gruppen verschiedener Dichte, Gitter (LOF = 1 im Inneren); **gegen `sklearn.neighbors.LocalOutlierFactor`** (Abweichung < 1e-9, transduktiv und mit `novelty=True`); Duplikate, Bindungen, k-Grenzen; Invarianz gegen Verschiebung, gleichmäßige Skalierung und Drehung, nicht gegen die Skalierung einzelner Merkmale;
  eine Gruppe kleiner als k bleibt unsichtbar, größer als k wird gesehen; k = n − 1 kehrt die Rangfolge um.
- Übernommene Bausteine: Isolation Forest (sklearn-Kreuzprüfung, c(n)) und Wurzel-Schätzer (χ² gegen scipy, MCD gegen `sklearn.covariance.MinCovDet`). Szenario: normale Zeilen wie in der PCA-Demo (eingefrorene Zeilensummen), eingefrorener Standardfall, `n_noise` ändert nur angehängte Spalten, exakter Anomalie-Anteil, Geometrie der Arten.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, k-gegen-Gruppengröße-, Lücken-, Betriebsarten-, Masking-, k-nahe-n-, Dimensions-, Schwellen- und Kostentabellen; jeweils Mittel über die festen Sweep-Datensätze, positive **und** negative Aussagen; Rechenzeiten nur als Verhältnis);
  alle 7 Presets in Bändern; AppTest-Rauchtests (Default, jedes Preset, jeder Schritt bei 2, 12 und 30 Merkmalen, wenige Touren mit vielen Merkmalen und Rauschmerkmalen, k-Grenze folgt der Tourenzahl, ausgeblendete Regler behalten ihre Werte, Sweep-Optionen folgen der Schwellenart, Art "Lücke" nur ab zwei Betriebsarten, Experimente auf Abruf),
  Achsensperre und explizite eindeutige Schlüssel aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (k gegen Gruppe, Lücke, Betriebsarten, Masking, k nahe n, Dimension, Schwelle, Kosten), 🚧 Grenzen, Mathe |
| `lof_algorithm.py` | Abstände, k-Nachbarn, Erreichbarkeit, lrd, LOF, LOF neuer Punkte |
| `lof_isolation_forest.py`, `lof_ee_algorithm.py` | Isolation Forest, χ²-Verteilung, klassische Schätzung, FastMCD (wortgleich aus den Vorgängern) |
| `lof_scenario.py`, `lof_constants.py` | Touren mit Betriebsarten, Krümmung, Anomalien und Rauschmerkmalen; Konstanten, Presets |
| `lof_evaluation.py` | Kennzahlen, Analyse, Schwellen, Sweeps, Experimente, Score-Karten, Urteil |
| `lof_presets.py`, `lof_visualization.py` | Permalink/Presets (ausgeblendete Regler, k-Grenze), Plotly-Figuren (achsengesperrt) |
| `tests/` | LOF (Handinstanzen, sklearn-Kreuzprüfung, Invarianzen), Szenario und Kennzahlen, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
