"""Box Least Squares (BLS) periodogram for transit detection.

Scans periods from 0.5 to 30 days and returns the best-fit transit parameters
along with the top-3 candidate periods. The periodogram is saved as a PNG.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import BoxLeastSquares
from lightkurve import LightCurve

logger = logging.getLogger(__name__)

# Minimum BLS power below which no significant signal is declared.
BLS_POWER_THRESHOLD = 10.0

# Period search bounds (days).
_P_MIN = 0.5
_P_MAX = 30.0

# Trial transit durations (days).
_TRIAL_DURATIONS = np.array([0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])

# Minimum number of transits required in the time series.
_MIN_TRANSITS = 3


def find_best_period(
    lc: LightCurve,
    star_name: str = "star",
    output_dir: Path = Path("outputs/plots"),
) -> dict:
    """Run a BLS periodogram and return the best transit parameters.

    The BLS is evaluated over an automatically-determined period grid spanning
    ``[0.5, 30]`` days, testing each of several trial durations. The period
    with the highest BLS power is returned as the best candidate, together with
    the next two runner-up peaks.

    A plot of the full periodogram (with the top-3 peaks annotated) is written
    to *output_dir*.

    Args:
        lc:         Preprocessed, zero-median light curve.
        star_name:  Target identifier used in the output filename.
        output_dir: Directory for the saved periodogram plot.

    Returns:
        Dictionary with keys:

        * ``best_period``   – best-fit orbital period (days)
        * ``best_t0``       – transit epoch (same time units as ``lc.time``)
        * ``transit_duration`` – best-fit transit duration (days)
        * ``depth``         – fractional transit depth
        * ``power``         – BLS power at the best period
        * ``top_candidates``– list of the top-3 candidate dicts (same fields)
        * ``significant``   – ``True`` if power > :data:`BLS_POWER_THRESHOLD`
    """
    time = np.asarray(lc.time.value, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)

    # Remove any surviving NaNs to protect the BLS solver.
    valid = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[valid], flux[valid]

    logger.info(
        "Running BLS on %d cadences (P = %.1f – %.1f d).", len(time), _P_MIN, _P_MAX
    )

    bls = BoxLeastSquares(time, flux)

    try:
        result = bls.autopower(
            _TRIAL_DURATIONS,
            minimum_period=_P_MIN,
            maximum_period=_P_MAX,
            minimum_n_transit=_MIN_TRANSITS,
            frequency_factor=1.0,
        )
    except Exception as exc:
        logger.error("BLS failed: %s", exc)
        raise RuntimeError(f"BLS periodogram failed: {exc}") from exc

    periods = np.asarray(result.period)
    powers = np.asarray(result.power)

    # --- Best peak ---
    best_idx = int(np.argmax(powers))
    best_period = float(periods[best_idx])
    best_power = float(powers[best_idx])

    # Retrieve per-period stats from the BLS model.
    best_stats = bls.compute_stats(
        best_period,
        float(result.duration[best_idx]),
        float(result.transit_time[best_idx]),
    )

    best_t0 = float(result.transit_time[best_idx])
    best_duration = float(result.duration[best_idx])
    best_depth = float(result.depth[best_idx])

    # --- Top-3 peaks (exclude regions within 10 % of the best period) ---
    top_candidates = _extract_top_candidates(periods, powers, result, n=3)

    significant = best_power > BLS_POWER_THRESHOLD
    if not significant:
        logger.warning(
            "BLS power %.2f is below threshold %.2f — no significant signal.",
            best_power,
            BLS_POWER_THRESHOLD,
        )
    else:
        logger.info(
            "Best period: %.4f d | depth: %.0f ppm | power: %.1f",
            best_period,
            best_depth * 1e6,
            best_power,
        )

    # --- Save periodogram plot ---
    _plot_periodogram(
        periods,
        powers,
        top_candidates,
        star_name=star_name,
        output_dir=output_dir,
    )

    return {
        "best_period": best_period,
        "best_t0": best_t0,
        "transit_duration": best_duration,
        "depth": best_depth,
        "power": best_power,
        "top_candidates": top_candidates,
        "significant": significant,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_top_candidates(
    periods: np.ndarray,
    powers: np.ndarray,
    result,
    n: int = 3,
) -> list:
    """Return the top-*n* distinct BLS peaks as a list of dicts."""
    candidates = []
    used_periods = []
    sorted_idx = np.argsort(powers)[::-1]

    for idx in sorted_idx:
        p = float(periods[idx])
        # Skip harmonics / aliases of already-accepted periods.
        if any(
            abs(p - up) / up < 0.10 or abs(p - 2 * up) / (2 * up) < 0.10
            for up in used_periods
        ):
            continue
        candidates.append(
            {
                "period": p,
                "t0": float(result.transit_time[idx]),
                "transit_duration": float(result.duration[idx]),
                "depth": float(result.depth[idx]),
                "power": float(powers[idx]),
            }
        )
        used_periods.append(p)
        if len(candidates) >= n:
            break

    return candidates


def _plot_periodogram(
    periods: np.ndarray,
    powers: np.ndarray,
    top_candidates: list,
    star_name: str,
    output_dir: Path,
) -> None:
    """Save a BLS periodogram PNG to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(periods, powers, color="steelblue", lw=0.6, alpha=0.9)

    colours = ["crimson", "darkorange", "forestgreen"]
    for i, cand in enumerate(top_candidates):
        colour = colours[i % len(colours)]
        ax.axvline(
            cand["period"],
            color=colour,
            ls="--",
            lw=1.2,
            label=f"#{i+1}: {cand['period']:.4f} d  (SNR {cand['power']:.1f})",
        )

    ax.set_xlabel("Period (days)", fontsize=11)
    ax.set_ylabel("BLS Power", fontsize=11)
    ax.set_title(f"BLS Periodogram — {star_name}", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(_P_MIN, _P_MAX)
    fig.tight_layout()

    safe = star_name.replace(" ", "_")
    out_path = output_dir / f"{safe}_bls_periodogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("BLS periodogram saved to %s", out_path)
