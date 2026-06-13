"""Light curve preprocessing: NaN removal, sigma clipping, flattening, normalisation.

The pipeline follows the data-cleaning approach described in Shallue & Vanderburg
(2018): remove bad cadences, remove stellar variability with a Savitzky-Golay
filter, then re-centre the flux on zero.
"""

import logging

import numpy as np
from lightkurve import LightCurve

logger = logging.getLogger(__name__)

# Sigma threshold for outlier rejection.
_SIGMA_CLIP = 5.0

# Savitzky-Golay window length (must be odd; will be auto-adjusted if the
# light curve is shorter than this value).
_SG_WINDOW = 401
_SG_POLYORDER = 2


def preprocess(lc: LightCurve) -> LightCurve:
    """Clean, flatten, and normalise a Kepler/TESS light curve.

    Processing steps applied in order:

    1. Remove NaN cadences.
    2. Remove 5-sigma outliers via iterative sigma-clipping.
    3. Flatten stellar variability with a Savitzky-Golay filter
       (``window_length=401``, ``polyorder=2``).
    4. Shift the median flux to zero.

    Args:
        lc: Raw :class:`~lightkurve.LightCurve` as returned by
            :func:`~pipeline.downloader.download_lightcurve`.

    Returns:
        Cleaned, zero-median :class:`~lightkurve.LightCurve`.
    """
    logger.info("Preprocessing: %d raw cadences.", len(lc))

    # 1. Drop NaN flux values.
    lc = lc.remove_nans()
    logger.debug("After NaN removal: %d cadences.", len(lc))

    # 2. Remove 5-sigma outliers.
    lc = lc.remove_outliers(sigma=_SIGMA_CLIP)
    logger.debug("After sigma clipping (%.0fσ): %d cadences.", _SIGMA_CLIP, len(lc))

    # 3. Flatten with Savitzky-Golay filter.
    lc = _flatten(lc)
    logger.debug("Flattened with Savitzky-Golay (window=%d).", _SG_WINDOW)

    # 4. Normalise flux to zero median.
    lc = _zero_median(lc)

    logger.info("Preprocessing complete: %d cadences remaining.", len(lc))
    return lc


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _flatten(lc: LightCurve) -> LightCurve:
    """Apply Savitzky-Golay detrending, auto-adjusting window for short curves."""
    n = len(lc)
    window = _SG_WINDOW

    # window must be odd and strictly less than the array length.
    window = min(window, n - 1 if n % 2 == 0 else n - 2)
    if window % 2 == 0:
        window -= 1
    window = max(window, _SG_POLYORDER + 2)  # minimum viable window

    try:
        flat_lc = lc.flatten(window_length=window, polyorder=_SG_POLYORDER)
    except Exception as exc:
        logger.warning(
            "SG flatten failed (%s); returning unflattened light curve.", exc
        )
        return lc

    return flat_lc


def _zero_median(lc: LightCurve) -> LightCurve:
    """Re-centre the flux so that its median is exactly zero."""
    flux = np.asarray(lc.flux, dtype=np.float64)
    flux -= np.nanmedian(flux)

    # Build a clean LightCurve with plain float arrays (no astropy units).
    return LightCurve(time=lc.time, flux=flux)
