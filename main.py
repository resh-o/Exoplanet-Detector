"""Parallax — CLI entry point.

Usage
-----
    python main.py analyze --star "Kepler-90"
    python main.py analyze --kic 11442793
    python main.py train  --data-path ./training_data
    python main.py batch  --stars-file stars.txt
"""

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Pipeline imports are deferred to inside command functions so that `--help`
# works even when optional heavy dependencies (lightkurve, tensorflow) are
# not yet installed.
_BLS_POWER_THRESHOLD = 10.0  # mirrors pipeline/bls.py default

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_LOG_FILE = "parallax.log"
_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

console = Console()


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=[
            logging.FileHandler(_LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Silence very noisy third-party loggers at DEBUG level.
    for noisy in ("lightkurve", "astropy", "matplotlib", "urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


logger = logging.getLogger("parallax")


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class Pipeline:
    """Orchestrates the full Parallax exoplanet detection pipeline.

    The pipeline runs six stages in sequence:

    1. **Download** — fetch all Kepler/TESS quarters via :mod:`~pipeline.downloader`.
    2. **Preprocess** — clean, flatten, and normalise via :mod:`~pipeline.preprocessor`.
    3. **BLS** — detect candidate periods via :mod:`~pipeline.bls`.
    4. **Fold** — build global + local views via :mod:`~pipeline.folder`.
    5. **Classify** — score with CNN or heuristic via :mod:`~pipeline.classifier`.
    6. **Validate** — cross-reference NASA archive via :mod:`~pipeline.validator`.
    """

    def __init__(
        self,
        output_dir: Path = Path("outputs"),
        model=None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.results_dir = self.output_dir / "results"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model = model  # optional pre-loaded Keras model

    def run(
        self,
        star_name: Optional[str] = None,
        kic_id: Optional[int] = None,
        mission: str = "Kepler",
    ) -> dict:
        """Execute the full pipeline and return a results dictionary.

        Args:
            star_name: Human-readable star name (e.g. ``"Kepler-90"``).
            kic_id:    Kepler Input Catalog numeric ID.
            mission:   Data archive (``"Kepler"``, ``"K2"``, ``"TESS"``).

        Returns:
            Results dict containing all pipeline outputs (BLS params,
            classification score, archive match, etc.).

        Raises:
            RuntimeError: Propagated from any failing pipeline stage.
        """
        display_name = star_name or f"KIC {kic_id}"
        started = datetime.utcnow()

        console.rule(f"[bold cyan]Parallax[/bold cyan] — {display_name}")

        # Lazy imports so the CLI works even when heavy deps are absent.
        import numpy as np
        from pipeline.downloader import download_lightcurve
        from pipeline.preprocessor import preprocess
        from pipeline.bls import find_best_period
        from pipeline.folder import phase_fold
        from pipeline.classifier import classify
        from pipeline.validator import validate_against_archive

        # ── Stage 1: Download ────────────────────────────────────────────────
        _status("Downloading light curve…")
        lc = download_lightcurve(
            star_name=star_name, kic_id=kic_id, mission=mission
        )
        n_points = len(lc)
        time_span = float(
            np.nanmax(lc.time.value) - np.nanmin(lc.time.value)
        )
        _ok(f"{n_points:,} cadences spanning {time_span:.1f} days")

        # ── Stage 2: Preprocess ──────────────────────────────────────────────
        _status("Preprocessing…")
        lc_clean = preprocess(lc)
        _ok(f"{len(lc_clean):,} cadences after cleaning")

        # ── Stage 3: BLS ─────────────────────────────────────────────────────
        _status("Running BLS periodogram…")
        bls = find_best_period(
            lc_clean,
            star_name=display_name,
            output_dir=self.plots_dir,
        )

        if not bls["significant"]:
            _warn("No significant transit signal found (BLS power too low).")
            results = _build_results(
                display_name, mission, bls, None, None, n_points, time_span, started
            )
            self._save_results(results, display_name)
            return results

        _ok(
            f"Best period: {bls['best_period']:.4f} d | "
            f"depth: {bls['depth'] * 1e6:.0f} ppm | "
            f"power: {bls['power']:.1f}"
        )

        # ── Stage 4: Phase fold ──────────────────────────────────────────────
        _status("Phase folding…")
        fold = phase_fold(
            lc_clean,
            period=bls["best_period"],
            t0=bls["best_t0"],
            transit_duration=bls["transit_duration"],
            star_name=display_name,
            output_dir=self.plots_dir,
        )
        _ok("Global view (201 bins) and local view (61 bins) generated")

        # ── Stage 5: Classify ────────────────────────────────────────────────
        _status("Classifying…")
        clf = classify(
            fold["global_view"], fold["local_view"], model=self.model
        )
        _ok(
            f"{clf['label']}  (score={clf['score']:.3f}, "
            f"method={clf['method']})"
        )

        # ── Stage 6: Validate ─────────────────────────────────────────────────
        _status("Querying NASA Exoplanet Archive…")
        val = validate_against_archive(display_name, bls["best_period"])
        if val["skipped"]:
            _warn("Archive unreachable — validation skipped.")
        elif val["match_found"]:
            _ok(
                f"Archive match: {val['planet_name']} "
                f"(P={val['known_period']:.4f} d)"
            )
        else:
            _ok("No matching confirmed planet found in archive.")

        # ── Assemble and save results ─────────────────────────────────────────
        results = _build_results(
            display_name, mission, bls, clf, val, n_points, time_span, started
        )
        self._save_results(results, display_name)
        return results

    def _save_results(self, results: dict, star_name: str) -> None:
        """Write results to a timestamped JSON file."""
        ts = results["timestamp"].replace(":", "-").replace(" ", "_")
        safe = star_name.replace(" ", "_")
        out = self.results_dir / f"{safe}_{ts}.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=_json_default)
        logger.info("Results saved to %s", out)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable DEBUG logging."
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Parallax — real exoplanet detection from Kepler telescope data."""
    _configure_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.option("--star", "star_name", default=None, help="Star name (e.g. 'Kepler-90').")
@click.option("--kic", "kic_id", default=None, type=int, help="KIC numeric ID.")
@click.option(
    "--mission",
    default="Kepler",
    show_default=True,
    type=click.Choice(["Kepler", "K2", "TESS"], case_sensitive=False),
    help="Data mission.",
)
def analyze(star_name: Optional[str], kic_id: Optional[int], mission: str) -> None:
    """Run the full detection pipeline on a single target.

    Provide either --star or --kic (not both). Results and plots are written
    to outputs/.
    """
    if star_name is None and kic_id is None:
        raise click.UsageError("Provide --star <name> or --kic <id>.")

    try:
        pipeline = Pipeline()
        results = pipeline.run(
            star_name=star_name, kic_id=kic_id, mission=mission
        )
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1)

    _print_summary_panel(results)


@cli.command()
@click.option(
    "--data-path",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing training data.",
)
@click.option("--epochs", default=50, show_default=True, type=int)
@click.option("--batch-size", default=32, show_default=True, type=int)
def train(data_path: str, epochs: int, batch_size: int) -> None:
    """Train the CNN classifier on labelled global/local view pairs.

    The training directory must contain:

    \b
      global_views.npy  — shape (N, 201)
      local_views.npy   — shape (N, 61)
      labels.npy        — shape (N,)  with values 0 or 1
    """
    import numpy as np

    data = Path(data_path)
    try:
        X_global = np.load(data / "global_views.npy")
        X_local = np.load(data / "local_views.npy")
        y = np.load(data / "labels.npy")
    except FileNotFoundError as exc:
        raise click.BadParameter(
            f"Missing training file: {exc.filename}. "
            "Expected global_views.npy, local_views.npy, labels.npy."
        )

    console.print(
        f"[cyan]Training on {len(y):,} examples "
        f"({int(y.sum())} positives, {int(len(y) - y.sum())} negatives)[/cyan]"
    )

    from pipeline.classifier import train_model as _train_model
    model = _train_model(
        X_global, X_local, y, epochs=epochs, batch_size=batch_size
    )
    console.print("[green]Training complete.[/green]")


@cli.command()
@click.option(
    "--stars-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Text file with one star name per line.",
)
@click.option(
    "--mission",
    default="Kepler",
    show_default=True,
    type=click.Choice(["Kepler", "K2", "TESS"], case_sensitive=False),
)
@click.option(
    "--output-csv",
    default="outputs/batch_results.csv",
    show_default=True,
    help="Path for the summary CSV.",
)
def batch(stars_file: str, mission: str, output_csv: str) -> None:
    """Run the pipeline on every star listed in a text file.

    Each line in STARS_FILE should be a star name (e.g. ``Kepler-90``).
    Lines beginning with ``#`` are treated as comments and skipped.
    Results are written to a CSV and to individual JSON files under outputs/.
    """
    star_names = _read_stars_file(stars_file)
    if not star_names:
        raise click.UsageError("No star names found in the file.")

    console.print(f"[cyan]Batch mode: {len(star_names)} target(s)[/cyan]")

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline()
    rows: list = []

    for i, name in enumerate(star_names, 1):
        console.rule(f"[{i}/{len(star_names)}] {name}")
        try:
            res = pipeline.run(star_name=name, mission=mission)
            rows.append(_result_to_csv_row(res))
        except Exception as exc:
            console.print(f"[red]FAILED:[/red] {exc}")
            rows.append({
                "star": name,
                "best_period": "",
                "transit_depth_ppm": "",
                "classification": "ERROR",
                "confidence": "",
                "archive_match": "",
                "error": str(exc),
            })

    _write_csv(rows, csv_path)
    console.print(f"\n[green]Batch complete.[/green] CSV → {csv_path}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_summary_panel(results: dict) -> None:
    """Render the Rich summary panel to stdout."""
    bls = results.get("bls") or {}
    clf = results.get("classification") or {}
    val = results.get("validation") or {}

    period_str = (
        f"{bls['best_period']:.4f} days" if bls.get("best_period") else "N/A"
    )
    depth_str = (
        f"{bls['depth'] * 1e6:.0f} ppm" if bls.get("depth") else "N/A"
    )

    if not clf:
        label_str = "N/A — no significant signal"
        score_str = "—"
    else:
        label_str = clf.get("label", "N/A")
        score_str = f"{clf.get('score', 0):.3f}"

    if val and val.get("match_found"):
        archive_str = f"YES — {val['planet_name']}"
    elif val and val.get("skipped"):
        archive_str = "Archive unreachable"
    else:
        archive_str = "No match found"

    colour = (
        "green"
        if clf and clf.get("label") == "PLANET CANDIDATE"
        else "yellow"
    )

    content = (
        f"[bold]Star:[/bold]             {results.get('star', 'N/A')}\n"
        f"[bold]Mission:[/bold]          {results.get('mission', 'N/A')}\n"
        f"[bold]Best Period:[/bold]      {period_str}\n"
        f"[bold]Transit Depth:[/bold]    {depth_str}\n"
        f"[bold]Classification:[/bold]   [{colour}]{label_str}[/{colour}]\n"
        f"[bold]Confidence Score:[/bold] {score_str}\n"
        f"[bold]Archive Match:[/bold]    {archive_str}\n"
        f"[bold]Method:[/bold]           {clf.get('method', 'N/A')}"
    )

    console.print()
    console.print(
        Panel(
            content,
            title="[bold white]PARALLAX RESULTS[/bold white]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def _status(msg: str) -> None:
    console.print(f"  [dim]→[/dim] {msg}", end=" ")


def _ok(msg: str) -> None:
    console.print(f"[green]✓[/green]  {msg}")


def _warn(msg: str) -> None:
    console.print(f"[yellow]⚠[/yellow]  {msg}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _build_results(
    star_name: str,
    mission: str,
    bls: dict,
    clf: Optional[dict],
    val: Optional[dict],
    n_points: int,
    time_span: float,
    started: datetime,
) -> dict:
    return {
        "star": star_name,
        "mission": mission,
        "timestamp": started.strftime("%Y-%m-%d %H:%M:%S"),
        "bls": bls,
        "classification": clf,
        "validation": val,
        "data": {
            "n_points": n_points,
            "time_span_days": round(time_span, 2),
        },
    }


def _read_stars_file(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return [
            line.strip()
            for line in fh
            if line.strip() and not line.strip().startswith("#")
        ]


def _result_to_csv_row(res: dict) -> dict:
    bls = res.get("bls") or {}
    clf = res.get("classification") or {}
    val = res.get("validation") or {}
    return {
        "star": res.get("star", ""),
        "best_period": bls.get("best_period", ""),
        "transit_depth_ppm": round(bls.get("depth", 0) * 1e6, 1) if bls.get("depth") else "",
        "classification": clf.get("label", "NO SIGNAL"),
        "confidence": clf.get("score", ""),
        "archive_match": val.get("planet_name", "") if val and val.get("match_found") else "None",
        "error": "",
    }


def _write_csv(rows: list, path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(obj):
    """JSON serialiser for numpy scalars and other non-standard types."""
    import numpy as np
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
