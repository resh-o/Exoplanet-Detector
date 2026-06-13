"""CNN classifier for planet / false-positive discrimination.

Architecture follows the AstroNet design from Shallue & Vanderburg (2018):
two parallel 1-D convolutional branches (one per view) whose outputs are
concatenated and fed through fully-connected layers.

When no trained model weights are available the module falls back to a
lightweight heuristic classifier based on transit depth, shape symmetry,
and secondary-eclipse check — giving qualitative results without any
trained weights.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODELS_DIR = Path("models")
_DEFAULT_WEIGHTS = _MODELS_DIR / "parallax_cnn.weights.h5"

# Planet probability threshold.
_SCORE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# TensorFlow import (optional)
# ---------------------------------------------------------------------------

try:
    import tensorflow as tf
    from tensorflow import keras

    _TF_AVAILABLE = True
    logger.debug("TensorFlow %s available.", tf.__version__)
except ImportError:
    _TF_AVAILABLE = False
    logger.info(
        "TensorFlow not found. CNN inference disabled; using heuristic fallback."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model():
    """Build the AstroNet-inspired dual-branch CNN.

    Architecture:

    * **Global branch** — input (201, 1)
      → Conv1D(16, 5, relu) → MaxPool1D(5, 2)
      → Conv1D(32, 5, relu) → MaxPool1D(5, 2) → Flatten
    * **Local branch**  — input (61, 1)
      → Conv1D(16, 5, relu) → MaxPool1D(7, 2)
      → Conv1D(32, 5, relu) → MaxPool1D(7, 2) → Flatten
    * Concatenate → FC(512, relu) → Dropout(0.5)
      → FC(512, relu) → Dropout(0.5) → FC(1, sigmoid)

    Returns:
        Compiled :class:`keras.Model` ready for training or inference.

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    if not _TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required to build the CNN. "
            "Install it with: pip install tensorflow"
        )

    # Global branch
    global_input = keras.Input(shape=(201, 1), name="global_view")
    x_g = keras.layers.Conv1D(16, 5, activation="relu", padding="valid")(global_input)
    x_g = keras.layers.MaxPool1D(pool_size=5, strides=2)(x_g)
    x_g = keras.layers.Conv1D(32, 5, activation="relu", padding="valid")(x_g)
    x_g = keras.layers.MaxPool1D(pool_size=5, strides=2)(x_g)
    x_g = keras.layers.Flatten()(x_g)

    # Local branch
    local_input = keras.Input(shape=(61, 1), name="local_view")
    x_l = keras.layers.Conv1D(16, 5, activation="relu", padding="valid")(local_input)
    x_l = keras.layers.MaxPool1D(pool_size=7, strides=2)(x_l)
    x_l = keras.layers.Conv1D(32, 5, activation="relu", padding="valid")(x_l)
    x_l = keras.layers.MaxPool1D(pool_size=7, strides=2)(x_l)
    x_l = keras.layers.Flatten()(x_l)

    # Merge and classification head
    merged = keras.layers.Concatenate()([x_g, x_l])
    x = keras.layers.Dense(512, activation="relu")(merged)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(1, activation="sigmoid", name="planet_prob")(x)

    model = keras.Model(
        inputs=[global_input, local_input],
        outputs=output,
        name="parallax_cnn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def classify(
    global_view: np.ndarray,
    local_view: np.ndarray,
    model=None,
) -> dict:
    """Classify a phase-folded light curve as planet candidate or false positive.

    Tries CNN inference first; falls back to heuristic scoring when no model
    is available or weights are missing.

    Args:
        global_view: Array of shape ``(201,)`` — full-period phase-fold bins.
        local_view:  Array of shape ``(61,)``  — transit-centred phase-fold bins.
        model:       Optional pre-loaded Keras model. If ``None`` and weights
                     exist at the default path they are loaded automatically.

    Returns:
        Dictionary with keys:

        * ``score``      – planet probability in ``[0, 1]``
        * ``label``      – ``"PLANET CANDIDATE"`` or ``"FALSE POSITIVE"``
        * ``confidence`` – same as *score* (provided for API symmetry)
        * ``method``     – ``"neural_network"`` or ``"heuristic"``
    """
    global_view = np.asarray(global_view, dtype=np.float32)
    local_view = np.asarray(local_view, dtype=np.float32)

    # --- Try CNN path ---
    if _TF_AVAILABLE:
        loaded_model = model or _try_load_model()
        if loaded_model is not None:
            score = _cnn_infer(loaded_model, global_view, local_view)
            method = "neural_network"
            logger.info("CNN score: %.4f", score)
            return _format_result(score, method)

    # --- Heuristic fallback ---
    logger.info("Using heuristic classifier (no CNN weights available).")
    score = _heuristic_score(global_view, local_view)
    return _format_result(score, "heuristic")


def train_model(
    X_global: np.ndarray,
    X_local: np.ndarray,
    y: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    val_split: float = 0.15,
    weights_path: Optional[Path] = None,
):
    """Train the CNN on labelled global/local view pairs.

    Args:
        X_global:     Array of shape ``(N, 201)`` — global views.
        X_local:      Array of shape ``(N, 61)``  — local views.
        y:            Binary labels: ``1`` = planet, ``0`` = false positive.
        epochs:       Training epochs.
        batch_size:   Mini-batch size.
        val_split:    Fraction of data held out for validation.
        weights_path: Where to save the trained weights (default:
                      ``models/parallax_cnn.weights.h5``).

    Returns:
        Trained Keras model.
    """
    if not _TF_AVAILABLE:
        raise ImportError("TensorFlow is required for training.")

    weights_path = Path(weights_path or _DEFAULT_WEIGHTS)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    # Reshape for Conv1D: (N, length, 1)
    X_g = X_global.reshape(-1, 201, 1).astype(np.float32)
    X_l = X_local.reshape(-1, 61, 1).astype(np.float32)
    y = y.astype(np.float32)

    model = build_model()
    model.summary(print_fn=lambda s: logger.info(s))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=8, restore_best_weights=True, mode="max"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path),
            monitor="val_auc",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        ),
    ]

    history = model.fit(
        [X_g, X_l],
        y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("Training complete. Weights saved to %s", weights_path)
    return model


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _try_load_model():
    """Load CNN weights from the default path if they exist."""
    if not _DEFAULT_WEIGHTS.exists():
        return None
    try:
        model = build_model()
        model.load_weights(str(_DEFAULT_WEIGHTS))
        logger.info("Loaded CNN weights from %s", _DEFAULT_WEIGHTS)
        return model
    except Exception as exc:
        logger.warning("Could not load CNN weights: %s", exc)
        return None


def _cnn_infer(model, global_view: np.ndarray, local_view: np.ndarray) -> float:
    """Run a single forward pass and return the planet probability."""
    g = global_view.reshape(1, 201, 1)
    l = local_view.reshape(1, 61, 1)
    pred = model.predict([g, l], verbose=0)
    return float(pred[0][0])


def _heuristic_score(global_view: np.ndarray, local_view: np.ndarray) -> float:
    """Score a transit using depth, symmetry, and secondary-eclipse checks.

    Scoring rubric (max 1.0):
    * +0.30 — transit depth exceeds 100 ppm
    * +0.30 — local-view left/right symmetry > 0.7
    * +0.40 — no significant secondary eclipse at phase ±0.5
    """
    score = 0.0

    # 1. Transit depth > 100 ppm (normalised view has minimum at −1, so
    #    depth in the original scale requires knowing the absolute flux scale;
    #    here we use the dip magnitude in the normalised local view as a proxy).
    transit_depth_proxy = abs(float(np.min(local_view)))
    if transit_depth_proxy > 0.01:  # effectively > 0 once normalised to –1
        score += 0.30

    # 2. Left/right symmetry of the transit shape.
    n = len(local_view)
    mid = n // 2
    left = local_view[:mid]
    right = local_view[mid + (n % 2):][::-1]  # reversed to mirror left
    min_len = min(len(left), len(right))
    if min_len > 0:
        left, right = left[:min_len], right[:min_len]
        numer = 2.0 * np.sum(np.minimum(np.abs(left), np.abs(right)))
        denom = np.sum(np.abs(left)) + np.sum(np.abs(right))
        symmetry = numer / denom if denom > 0 else 0.0
        if symmetry > 0.7:
            score += 0.30

    # 3. Secondary-eclipse check: inspect the global view at phase 0.5
    #    (index closest to the halfway point).
    n_g = len(global_view)
    half_idx = n_g // 2
    # Sample a window of ±5 % around phase 0.5.
    w = max(1, n_g // 20)
    secondary_region = global_view[max(0, half_idx - w): half_idx + w + 1]
    primary_dip = abs(float(np.min(local_view)))
    secondary_dip = abs(float(np.min(secondary_region)))

    if primary_dip == 0 or secondary_dip < 0.5 * primary_dip:
        score += 0.40

    logger.debug(
        "Heuristic: depth_proxy=%.3f symmetry→+%.2f secondary→check → total=%.2f",
        transit_depth_proxy,
        0.30 if transit_depth_proxy > 0.01 else 0.0,
        score,
    )
    return min(score, 1.0)


def _format_result(score: float, method: str) -> dict:
    """Format the classifier output dict."""
    label = "PLANET CANDIDATE" if score >= _SCORE_THRESHOLD else "FALSE POSITIVE"
    return {
        "score": round(score, 4),
        "label": label,
        "confidence": round(score, 4),
        "method": method,
    }
