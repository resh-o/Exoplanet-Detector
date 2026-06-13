"""Phase folding and binning into global + local views.

Implements the two-view representation from Shallue & Vanderburg (2018):

* **Global view** — 201 evenly-spaced phase bins spanning the full orbital
  period ``[-0.5, +0.5]``.
* **Local view**  — 61 evenly-spaced bins spanning ``2×`` the transit duration,
  centred on the transit dip.

Both views are normalised so that the out-of-transit baseline is 0 and the
deepest point of the transit dip is −1.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lightkurve import LightCurve

logger = logging.getLogger(__name__)

N_GLOBAL = 201
N_LOCAL = 61


def phase_fold(
    lc: LightCurve,
    period: float,
    t0: float,
    transit_duration: float = 0.1,
    star_name: str = "star",
    output_dir: Path = Path("outputs/plots"),
) -> dict:
    """Phase-fold a light curve and return normalised global and local views.

    Args:
        lc:               Preprocessed, zero-median light curve.
        period:           Orbital period in days (from BLS).
        t0:               Transit epoch in the same time units as ``lc.time``.
        transit_duration: Transit duration in days (from BLS).
        star_name:        Target identifier used in the output filename.
        output_dir:       Directory for the saved phase-fold plots.

    Returns:
        Dictionary with keys:

        * ``global_view`` – :class:`numpy.ndarray` of shape ``(201,)``
        * ``local_view``  – :class:`numpy.ndarray` of shape ``(61,)``
        * ``phase``       – sorted phase array ``[-0.5, +0.5]``
        * ``flux``        – flux array sorted by phase
        * ``folded_lc``   – dict with ``phase`` and ``flux`` keys for the
          full phase-folded light curve (before binning)
    """
    time = np.asarray(lc.time.value, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)

    # Remove any NaN/Inf cadences.
    valid = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[valid], flux[valid]

    # Compute phase in [-0.5, +0.5] with the transit centred at 0.
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0

    sort_idx = np.argsort(phase)
    phase_s = phase[sort_idx]
    flux_s = flux[sort_idx]

    # --- Global view: 201 bins over [-0.5, +0.5] ---
    global_bins = np.linspace(-0.5, 0.5, N_GLOBAL + 1)
    global_view = _bin_fold(phase_s, flux_s, global_bins)
    global_view = _normalise_view(global_view)

    # --- Local view: 61 bins over ±(transit_duration / period) ---
    half_width = (transit_duration / period)
    # Guard: make sure the local view is at least 2 % of the period.
    half_width = max(half_width, 0.01)
    local_bins = np.linspace(-half_width, half_width, N_LOCAL + 1)
    local_view = _bin_fold(phase_s, flux_s, local_bins)
    local_view = _normalise_view(local_view)

    logger.info(
        "Phase fold: period=%.4f d | global %d bins | local %d bins (±%.4f)",
        period,
        N_GLOBAL,
        N_LOCAL,
        half_width,
    )

    _plot_views(
        phase_s,
        flux_s,
        global_view,
        local_view,
        half_width=half_width,
        period=period,
        star_name=star_name,
        output_dir=output_dir,
    )

    return {
        "global_view": global_view,
        "local_view": local_view,
        "phase": phase_s,
        "flux": flux_s,
        "folded_lc": {"phase": phase_s, "flux": flux_s},
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _bin_fold(
    phase: np.ndarray,
    flux: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """Bin a phase-sorted light curve into len(bins)-1 bins using the mean."""
    n_bins = len(bins) - 1
    binned = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        if mask.any():
            binned[i] = np.mean(flux[mask])

    # Linearly interpolate any empty bins so the CNN always gets a dense array.
    nan_mask = np.isnan(binned)
    if nan_mask.any() and not nan_mask.all():
        idx = np.arange(n_bins)
        binned[nan_mask] = np.interp(
            idx[nan_mask], idx[~nan_mask], binned[~nan_mask]
        )
    elif nan_mask.all():
        binned[:] = 0.0

    return binned


def _normalise_view(view: np.ndarray) -> np.ndarray:
    """Subtract the median and scale so the deepest point is −1."""
    v = view - np.median(view)
    abs_min = np.abs(np.min(v))
    if abs_min > 0:
        v = v / abs_min
    return v


def _plot_views(
    phase: np.ndarray,
    flux: np.ndarray,
    global_view: np.ndarray,
    local_view: np.ndarray,
    half_width: float,
    period: float,
    star_name: str,
    output_dir: Path,
) -> None:
    """Save a two-panel phase-fold plot (scatter + global view, local view)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_phase_centers = np.linspace(-0.5, 0.5, N_GLOBAL)
    local_phase_centers = np.linspace(-half_width, half_width, N_LOCAL)
    # Convert local phase axis to hours for readability.
    local_hours = local_phase_centers * period * 24.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Global view
    ax = axes[0]
    ax.scatter(phase, flux, s=0.3, alpha=0.3, color="steelblue", rasterized=True)
    ax.plot(global_phase_centers, global_view, color="crimson", lw=1.5, label="Binned")
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalised Flux")
    ax.set_title(f"Global View — {star_name}\n(P = {period:.4f} d)", fontsize=10)
    ax.legend(fontsize=8)

    # Local view (zoomed in on transit)
    ax = axes[1]
    # Scatter only the in-transit points.
    local_mask = (phase >= -half_width) & (phase <= half_width)
    ax.scatter(
        phase[local_mask] * period * 24.0,
        flux[local_mask],
        s=1.5,
        alpha=0.5,
        color="steelblue",
        rasterized=True,
    )
    ax.plot(local_hours, local_view, color="crimson", lw=1.8, label="Binned")
    ax.set_xlabel("Time from mid-transit (hours)")
    ax.set_ylabel("Normalised Flux")
    ax.set_title(f"Local View — {star_name}\n(±{half_width * period * 24:.2f} h)", fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    safe = star_name.replace(" ", "_")
    out_path = output_dir / f"{safe}_phase_fold.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Phase-fold plot saved to %s", out_path)
