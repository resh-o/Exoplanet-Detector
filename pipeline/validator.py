"""Cross-reference detected transit periods against the NASA Exoplanet Archive.

Uses the IPAC TAP service (ADQL) to look up known confirmed planets around a
target star and checks whether any of them match the BLS-detected period within
a user-specified fractional tolerance.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
_TAP_TABLE = "ps"          # Planetary Systems composite table
_REQUEST_TIMEOUT = 15      # seconds


def validate_against_archive(
    star_name: str,
    period: float,
    tolerance: float = 0.01,
) -> dict:
    """Check whether a detected period matches any known planet in the NASA archive.

    Queries the NASA Exoplanet Archive TAP endpoint for all confirmed planets
    around *star_name* and returns information about the closest period match
    (if any is found within *tolerance*).

    Args:
        star_name:  Host star name exactly as it appears in the archive
                    (e.g. ``"Kepler-90"``).
        period:     BLS-detected orbital period in days.
        tolerance:  Maximum fractional period deviation for a match:
                    ``|P_known - P_detected| / P_known ≤ tolerance``.

    Returns:
        Dictionary with keys:

        * ``match_found``   – ``True`` if a matching planet was found
        * ``planet_name``   – name of the matching planet (or ``None``)
        * ``known_period``  – tabulated orbital period in days (or ``None``)
        * ``known_radius``  – planet radius in Earth radii (or ``None``)
        * ``source``        – ``"NASA Exoplanet Archive"``
        * ``skipped``       – ``True`` if the archive could not be reached
    """
    logger.info(
        "Querying NASA Exoplanet Archive for '%s' (P=%.4f d, tol=%.1f%%)…",
        star_name,
        period,
        tolerance * 100,
    )

    try:
        planets = _fetch_planets(star_name)
    except requests.exceptions.ConnectionError:
        logger.warning("NASA archive unreachable — skipping validation.")
        return _skipped_result()
    except requests.exceptions.Timeout:
        logger.warning("NASA archive request timed out — skipping validation.")
        return _skipped_result()
    except Exception as exc:
        logger.warning("Archive query failed (%s) — skipping validation.", exc)
        return _skipped_result()

    if not planets:
        logger.info("No confirmed planets found for '%s' in the archive.", star_name)
        return _no_match_result()

    logger.info("Archive returned %d known planet(s).", len(planets))

    # Find the closest period match.
    best: Optional[dict] = None
    best_deviation = float("inf")

    for planet in planets:
        p_known = planet.get("pl_orbper")
        if p_known is None:
            continue
        try:
            p_known = float(p_known)
        except (TypeError, ValueError):
            continue

        if p_known <= 0:
            continue

        deviation = abs(p_known - period) / p_known
        if deviation < best_deviation:
            best_deviation = deviation
            best = planet
            best["_deviation"] = deviation

    if best is None or best_deviation > tolerance:
        logger.info(
            "No period match within %.1f%% tolerance (closest: %.2f%%).",
            tolerance * 100,
            best_deviation * 100 if best else float("nan"),
        )
        return _no_match_result()

    planet_name = best.get("pl_name", "Unknown")
    known_period = float(best.get("pl_orbper", 0.0))
    known_radius_str = best.get("pl_rade")
    known_radius = float(known_radius_str) if known_radius_str else None

    logger.info(
        "Archive match: %s | P=%.4f d | deviation=%.3f%%",
        planet_name,
        known_period,
        best_deviation * 100,
    )

    return {
        "match_found": True,
        "planet_name": planet_name,
        "known_period": known_period,
        "known_radius": known_radius,
        "period_deviation_pct": round(best_deviation * 100, 3),
        "source": "NASA Exoplanet Archive",
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fetch_planets(star_name: str) -> list:
    """Return a list of planet dicts for *star_name* from the TAP service."""
    # Use SQL LIKE with a case-insensitive workaround — the archive uses
    # lower-case ADQL, so we match the original and a title-cased variant.
    adql = (
        f"SELECT pl_name, hostname, pl_orbper, pl_rade "
        f"FROM {_TAP_TABLE} "
        f"WHERE LOWER(hostname) LIKE LOWER('{_sanitise(star_name)}')"
    )

    params = {"query": adql, "format": "json"}
    resp = requests.get(_TAP_URL, params=params, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()

    data = resp.json()

    # The TAP JSON format is {"columnNames": [...], "rows": [[...], ...]}.
    if isinstance(data, dict) and "columnNames" in data:
        columns = [c.lower() for c in data["columnNames"]]
        rows = data.get("rows", [])
        return [dict(zip(columns, row)) for row in rows]

    # Some IPAC TAP responses return a list of dicts directly.
    if isinstance(data, list):
        return data

    return []


def _sanitise(name: str) -> str:
    """Escape single quotes in a star name for safe SQL interpolation."""
    return name.replace("'", "''")


def _no_match_result() -> dict:
    return {
        "match_found": False,
        "planet_name": None,
        "known_period": None,
        "known_radius": None,
        "period_deviation_pct": None,
        "source": "NASA Exoplanet Archive",
        "skipped": False,
    }


def _skipped_result() -> dict:
    return {
        "match_found": False,
        "planet_name": None,
        "known_period": None,
        "known_radius": None,
        "period_deviation_pct": None,
        "source": "NASA Exoplanet Archive",
        "skipped": True,
    }
