# Model Weights

Place trained CNN weight files in this directory.

## Expected filename

```
parallax_cnn.weights.h5
```

Parallax looks for this file automatically when you run `analyze`. If it is
missing the pipeline falls back to the built-in heuristic classifier.

## Training your own model

1. Prepare labelled data in NumPy format:

   | File | Shape | Description |
   |------|-------|-------------|
   | `global_views.npy` | `(N, 201)` | Global phase-fold views |
   | `local_views.npy`  | `(N, 61)`  | Local (transit-centred) views |
   | `labels.npy`       | `(N,)`     | Binary labels — `1` = planet, `0` = FP |

2. Run:

   ```bash
   python main.py train --data-path ./training_data --epochs 50
   ```

   The best-epoch weights are written here automatically.

## Pre-trained weights (community)

The original AstroNet weights from Shallue & Vanderburg (2018) are available
at the [Google Research GitHub repository](https://github.com/google-research/exoplanet-ml).
Note that the architecture used there differs slightly; you may need to adapt
the weight file or retrain using the `train` command above.

## Architecture summary

```
Input (global) (201, 1)          Input (local) (61, 1)
      │                                  │
Conv1D(16, 5, relu)               Conv1D(16, 5, relu)
MaxPool1D(5, stride=2)            MaxPool1D(7, stride=2)
Conv1D(32, 5, relu)               Conv1D(32, 5, relu)
MaxPool1D(5, stride=2)            MaxPool1D(7, stride=2)
Flatten → 1440 units              Flatten → 256 units
              └──── Concat (1696) ────┘
                Dense(512, relu)
                Dropout(0.5)
                Dense(512, relu)
                Dropout(0.5)
                Dense(1, sigmoid)   ← planet probability
```
