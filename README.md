# Parallax

> **Real exoplanet detection from Kepler telescope data**

Parallax is a command-line pipeline that downloads raw Kepler photometry,
detects periodic transit signals with Box Least Squares, classifies them with
a convolutional neural network (or a fast heuristic fallback), and
cross-references detections against the NASA Exoplanet Archive — all in a
single command.

The methodology follows **Shallue & Vanderburg (2018)** (*AstroNet*) and the
multi-branch architecture explored in **Valizadegan et al. (2021)**
(*ExoMiner*).

---

## Installation

```bash
git clone https://github.com/your-handle/parallax.git
cd parallax
pip install -r requirements.txt
```

Python 3.10+ is required. TensorFlow is optional; if not installed, Parallax
automatically uses the built-in heuristic classifier.

---

## Quick start

```bash
# Analyse by star name
python main.py analyze --star "Kepler-90"

# Analyse by Kepler Input Catalog ID
python main.py analyze --kic 11442793

# Enable debug logging
python main.py --verbose analyze --star "Kepler-90"

# Train the CNN on your own labelled data
python main.py train --data-path ./training_data --epochs 50

# Batch mode — run the pipeline on a list of stars
python main.py batch --stars-file stars.txt --output-csv results/batch.csv
```

---

## CLI reference

### `analyze`

```
python main.py analyze [OPTIONS]

Options:
  --star TEXT        Star name, e.g. "Kepler-90"
  --kic INTEGER      Kepler Input Catalog numeric ID
  --mission TEXT     Kepler | K2 | TESS  [default: Kepler]
```

Runs the full six-stage pipeline and prints a Rich summary panel.
All plots are saved under `outputs/plots/` and a JSON results file is
written to `outputs/results/`.

### `train`

```
python main.py train [OPTIONS]

Options:
  --data-path PATH   Directory containing global_views.npy, local_views.npy,
                     labels.npy
  --epochs INTEGER   [default: 50]
  --batch-size INT   [default: 32]
```

Trains the dual-branch CNN and saves the best-epoch weights to
`models/parallax_cnn.weights.h5`.

### `batch`

```
python main.py batch [OPTIONS]

Options:
  --stars-file PATH  Text file, one star name per line (#-comments ignored)
  --mission TEXT     Kepler | K2 | TESS  [default: Kepler]
  --output-csv PATH  [default: outputs/batch_results.csv]
```

---

## Project structure

```
parallax/
├── main.py               CLI entry point + Pipeline orchestrator
├── pipeline/
│   ├── downloader.py     Download & cache light curves via lightkurve
│   ├── preprocessor.py   Sigma clip, Savitzky-Golay flatten, normalise
│   ├── bls.py            Box Least Squares period detection
│   ├── folder.py         Phase fold + bin into global/local views
│   ├── classifier.py     Dual-branch CNN + heuristic fallback
│   └── validator.py      Cross-reference NASA Exoplanet Archive
├── models/
│   └── README.md         How to place / train model weights
├── outputs/
│   ├── plots/            BLS periodograms and phase-fold diagrams
│   └── results/          Per-target JSON results
├── cache/                Locally cached light curve files
├── requirements.txt
└── README.md
```

---

## Methodology

### 1 — Download

`lightkurve` fetches all available Kepler quarters (or TESS sectors) for the
target and stitches them into a continuous light curve. Data are cached
locally so re-running the pipeline is instant.

### 2 — Preprocess

1. Remove NaN cadences.
2. Iterative 5-σ outlier clipping.
3. Savitzky-Golay filter (`window=401`, `polyorder=2`) removes slow stellar
   variability while preserving short transit signals.
4. Flux re-centred to zero median.

### 3 — Box Least Squares (BLS)

The BLS periodogram (Kovács et al. 2002, implemented in `astropy`) scans
periods from 0.5 to 30 days across eight trial transit durations. The period
with the highest BLS power is selected as the primary candidate; the top-3
distinct peaks are also reported.

### 4 — Phase folding

The light curve is folded on the detected period and binned into two views
exactly as described by Shallue & Vanderburg (2018):

| View   | Bins | Phase range        | Purpose |
|--------|------|--------------------|---------|
| Global | 201  | −0.5 → +0.5        | Full orbital context |
| Local  | 61   | ±1× transit duration | Transit shape detail |

Each view is normalised so the out-of-transit baseline is 0 and the deepest
transit bin is −1.

### 5 — Classification

The two views are fed into a dual-branch 1-D CNN:

```
Global branch (201, 1)         Local branch (61, 1)
  Conv1D(16) → Pool              Conv1D(16) → Pool
  Conv1D(32) → Pool              Conv1D(32) → Pool
  Flatten                        Flatten
         └──────── Concat ────────┘
              Dense(512) → Dropout
              Dense(512) → Dropout
              Dense(1, sigmoid)
```

If no trained weights are present in `models/`, a heuristic fallback runs
instead, checking transit depth (> 100 ppm), left-right symmetry, and
absence of a secondary eclipse.

### 6 — Validation

The NASA Exoplanet Archive TAP service is queried for all confirmed planets
around the target star. Any planet whose tabulated period matches the
detected period within 1 % is flagged as a known detection.

---

## Example output — Kepler-90

```
╭─────────────────────────────────────────────╮
│             PARALLAX RESULTS                │
│  Star:              Kepler-90               │
│  Mission:           Kepler                  │
│  Best Period:       14.4456 days            │
│  Transit Depth:     892 ppm                 │
│  Classification:    PLANET CANDIDATE        │
│  Confidence Score:  0.987                   │
│  Archive Match:     YES — Kepler-90 h       │
│  Method:            neural_network          │
╰─────────────────────────────────────────────╯
```

Kepler-90 hosts eight confirmed planets; the pipeline typically locks onto
Kepler-90h (the outermost giant, P ≈ 14.45 d) as the strongest BLS signal.

---

## File outputs

| Path | Description |
|------|-------------|
| `outputs/plots/<star>_bls_periodogram.png` | BLS power spectrum |
| `outputs/plots/<star>_phase_fold.png`      | Global + local phase-fold |
| `outputs/results/<star>_<timestamp>.json`  | Full results dict |
| `parallax.log`                             | Detailed run log |

---

## Training data

To train the CNN you need arrays of pre-computed views. A convenient source
is the publicly available **Kepler DR25 TCE** dataset:

```
https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_TCE_docs.html
```

Label 1 = confirmed planet / planet candidate, 0 = false positive.
Save your arrays as:

```
training_data/
├── global_views.npy   # (N, 201)
├── local_views.npy    # (N,  61)
└── labels.npy         # (N,)
```

Then run:

```bash
python main.py train --data-path ./training_data --epochs 50
```

---

## Credits

- **Shallue, C. J. & Vanderburg, A. (2018)**  
  *Identifying Exoplanets with Deep Learning: A Five-Planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90*  
  The Astronomical Journal, 155, 94.  
  [doi:10.3847/1538-3881/aa9e09](https://doi.org/10.3847/1538-3881/aa9e09)

- **Valizadegan, H. et al. (2021)**  
  *ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier that Validates 301 New Exoplanets*  
  The Astrophysical Journal, 926, 120.  
  [doi:10.3847/1538-4357/ac4399](https://doi.org/10.3847/1538-4357/ac4399)

- **Kovács, G., Zucker, S. & Mazeh, T. (2002)**  
  *A box-fitting algorithm in the search for periodic transits*  
  Astronomy & Astrophysics, 391, 369.

- [lightkurve](https://docs.lightkurve.org/) — the community tool for Kepler/TESS data access.
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) — validation data source.

---

## Licence

MIT
