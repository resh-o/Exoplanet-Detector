"""Light curve downloader for Kepler/TESS data via lightkurve.

Downloads all available quarters/sectors for a target, stitches them into a
single continuous light curve, and caches the result locally to avoid
redundant network requests on subsequent runs.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import lightkurve as lk
from lightkurve import LightCurve

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")


def download_lightcurve(
    star_name: Optional[str] = None,
    kic_id: Optional[int] = None,
    mission: str = "Kepler",
) -> LightCurve:
    """Download and stitch all available light curve data for a star.

    Searches the specified mission archive for the given target, downloads every
    available quarter or sector, and stitches them into a single LightCurve.
    The result is pickled to a local cache directory so subsequent calls are
    instant.

    Args:
        star_name: Human-readable star identifier (e.g. ``"Kepler-90"``).
        kic_id:    Kepler Input Catalog numeric ID (e.g. ``11442793``).
                   Takes precedence over *star_name* when both are supplied.
        mission:   Archive to query — ``"Kepler"``, ``"K2"``, or ``"TESS"``.

    Returns:
        Stitched :class:`lightkurve.LightCurve` containing every available
        observation for the target.

    Raises:
        ValueError:  If neither *star_name* nor *kic_id* is provided.
        RuntimeError: If the archive search returns no results or the
                      download fails.
    """
    if star_name is None and kic_id is None:
        raise ValueError("Provide either star_name or kic_id.")

    # Build a canonical search string and a filesystem-safe cache key.
    if kic_id is not None:
        search_id = f"KIC {kic_id}"
        cache_key = f"kic_{kic_id}_{mission.lower()}"
    else:
        search_id = star_name
        safe = star_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        cache_key = f"{safe}_{mission.lower()}"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    # --- Cache hit ---
    if cache_file.exists():
        logger.info("Loading cached light curve from %s", cache_file)
        try:
            with open(cache_file, "rb") as fh:
                lc: LightCurve = pickle.load(fh)
            logger.info("Loaded %d data points from cache.", len(lc))
            return lc
        except Exception as exc:
            logger.warning("Cache read failed (%s) — re-downloading.", exc)
            cache_file.unlink(missing_ok=True)

    # --- Search archive ---
    logger.info("Searching %s archive for '%s'…", mission, search_id)
    try:
        search_result = lk.search_lightcurve(search_id, mission=mission)
    except Exception as exc:
        raise RuntimeError(
            f"Archive search failed for '{search_id}': {exc}"
        ) from exc

    if len(search_result) == 0:
        raise RuntimeError(
            f"No {mission} light curves found for '{search_id}'.\n"
            "  • Verify the star name matches the catalog exactly (e.g. 'Kepler-90').\n"
            "  • Try searching by KIC ID with --kic <number>.\n"
            "  • Browse https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html"
        )

    logger.info("Found %d file(s). Downloading all quarters…", len(search_result))

    try:
        lc_collection = search_result.download_all(quality_bitmask="default")
    except Exception as exc:
        raise RuntimeError(
            f"Download failed for '{search_id}': {exc}"
        ) from exc

    if lc_collection is None or len(lc_collection) == 0:
        raise RuntimeError(
            f"Download returned an empty collection for '{search_id}'."
        )

    # --- Stitch all quarters into one light curve ---
    logger.info("Stitching %d segment(s)…", len(lc_collection))
    try:
        stitched: LightCurve = lc_collection.stitch()
    except Exception as exc:
        logger.warning("Stitch failed (%s) — using first segment only.", exc)
        stitched = lc_collection[0]

    import numpy as np
    t_min = float(np.nanmin(stitched.time.value))
    t_max = float(np.nanmax(stitched.time.value))
    logger.info(
        "Stitched light curve: %d points spanning %.1f days.",
        len(stitched),
        t_max - t_min,
    )

    # --- Cache to disk ---
    try:
        with open(cache_file, "wb") as fh:
            pickle.dump(stitched, fh)
        logger.info("Cached to %s", cache_file)
    except Exception as exc:
        logger.warning("Could not write cache: %s", exc)

    return stitched
