"""Collect every number, table row and figure reference used in the report.

The functions in this module read the experiment outputs in ``experiments/`` and return plain
Python values (numbers, formatted strings, lists of row dictionaries). They contain no HTML or
LaTeX: presentation lives in ``templates/``. Both report builders (``build_report.py`` for
HTML/PDF and ``build_latex.py`` for LaTeX) call :func:`build_context`.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------

#: Gallery size: 145.32 crore Aadhaar holders x 10 fingers (UIDAI enrolment dashboard).
GALLERY_SIZE = 14.532e9
#: HERS packing: templates per ciphertext (paper setting, polynomial degree n = 4,096).
HERS_SLOTS = 4096
#: Seconds in a year, for converting long run times.
SECONDS_PER_YEAR = 3.156e7
#: Millimetres per pixel at 500 dpi.
MM_PER_PIXEL = 0.0508

DATABASES = ["2002_DB1", "2002_DB2", "2002_DB3", "2002_DB4", "2004_DB1", "2004_DB2", "2004_DB3", "2004_DB4"]
SYNTHETIC = ["2002_DB4", "2004_DB4"]
DATABASE_NAMES = {
    "2002_DB1": ("FVC2002 DB1", "optical"),
    "2002_DB2": ("FVC2002 DB2", "optical"),
    "2002_DB3": ("FVC2002 DB3", "capacitive"),
    "2002_DB4": ("FVC2002 DB4", "synthetic"),
    "2004_DB1": ("FVC2004 DB1", "optical"),
    "2004_DB2": ("FVC2004 DB2", "optical"),
    "2004_DB3": ("FVC2004 DB3", "thermal"),
    "2004_DB4": ("FVC2004 DB4", "synthetic"),
}
DATABASE_NOTES = {
    "2004_DB1": "JIPNet trained on it",
    "2004_DB2": "JIPNet trained on it",
    "2002_DB1": "paper’s test set",
    "2002_DB3": "paper’s test set",
}
#: EER / TAR at FAR 0.1% reported in the JIPNet paper (Table V, 160x160 patches).
PAPER_RESULTS = {
    "2002_DB1": {"jipnet": "1% / 93%", "afrnet": "29% / 3%"},
    "2002_DB3": {"jipnet": "3% / 67%", "afrnet": "35% / 1%"},
}
#: Supporting figures shown in the appendix: (file name, caption).
APPENDIX_FIGURES = [
    (
        "b_hills_JIPNet.png",
        "JIPNet score distributions per database: same-finger (green) and different-finger (red) pairs.",
    ),
    (
        "b_afr_vs_jip_hills.png",
        "Score distributions on FVC2002 DB1/DB3: AFR-Net-style baseline (top) and JIPNet (bottom).",
    ),
    ("b_pose.png", "JIPNet: share of genuine pairs within a given rotation or translation error."),
    ("b_pose_overlap.png", "JIPNet: pose error by overlap between the two patches."),
    ("b_coverage.png", "Effect of fingerprint coverage within the patch on EER and TAR at FAR 0.1%."),
    (
        "b_fail_JIPNet_2002_DB1.png",
        "JIPNet lowest-scoring genuine pairs, highest-scoring impostor pairs and largest rotation errors, FVC2002 DB1.",
    ),
    ("b_fail_JIPNet_2002_DB4.png", "As Figure A6, synthetic FVC2002 DB4."),
    (
        "b_fail_AFR_2002_DB1.png",
        "AFR-Net-style baseline lowest-scoring genuine and highest-scoring impostor pairs, FVC2002 DB1.",
    ),
]


# ---------------------------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------------------------


def fmt(value: float, spec: str) -> str:
    """Format a number with a Python format specification.

    Args:
        value: Number to format.
        spec: Format specification, e.g. ``".1%"`` or ``".0f"``.

    Returns:
        The formatted string.
    """
    return format(value, spec)


def fmt_range(low: float, high: float, spec: str) -> str:
    """Format a numeric range as ``"low–high"`` (en dash).

    Args:
        low: Lower bound.
        high: Upper bound.
        spec: Format specification applied to both bounds.

    Returns:
        The formatted range, e.g. ``"1.6%–3.2%"``.
    """
    return f"{format(low, spec)}–{format(high, spec)}"


def sci_parts(value: float, digits: int = 1) -> tuple[str, int]:
    """Split a number into mantissa and power of ten for scientific notation.

    Args:
        value: Number to split.
        digits: Decimal places of the mantissa.

    Returns:
        ``(mantissa, exponent)``, e.g. ``("5.0", 20)`` for 5.0e20.
    """
    exponent = int(f"{value:e}".split("e")[1])
    return f"{value / 10 ** exponent:.{digits}f}", exponent


def sci_text(value: float, digits: int = 1) -> str:
    """Scientific notation with Unicode superscripts, for HTML (e.g. ``"5.0 × 10²⁰"``).

    Args:
        value: Number to format.
        digits: Decimal places of the mantissa.

    Returns:
        The formatted string.
    """
    mantissa, exponent = sci_parts(value, digits)
    superscript = str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mantissa} × 10{superscript}"


def sci_latex(value: float, digits: int = 1) -> str:
    """Scientific notation as a LaTeX math expression (e.g. ``"$5.0 \\times 10^{20}$"``).

    Args:
        value: Number to format.
        digits: Decimal places of the mantissa.

    Returns:
        The LaTeX string.
    """
    mantissa, exponent = sci_parts(value, digits)
    return f"${mantissa} \\times 10^{{{exponent}}}$"


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------


def load_json(name: str) -> dict[str, Any]:
    """Load a JSON file from ``experiments/``.

    Args:
        name: File name, e.g. ``"flops.json"``.

    Returns:
        The parsed JSON content.
    """
    with open(os.path.join(ROOT, "experiments", name)) as handle:
        return json.load(handle)


def load_jipnet_summary() -> pd.DataFrame:
    """Load the per-database JIPNet metrics written by ``report_figs.py``.

    Returns:
        DataFrame indexed by database key (e.g. ``"2002_DB1"``) with EER, TAR and pose columns.
    """
    summary = pd.read_csv(os.path.join(ROOT, "experiments", "summary.csv"))
    jipnet = summary[(summary.model == "JIPNet") & ~summary.db.str.startswith("ALL")]
    return jipnet.set_index("db")


# ---------------------------------------------------------------------------------------------
# Table rows
# ---------------------------------------------------------------------------------------------


def jipnet_rows(jipnet: pd.DataFrame) -> list[dict[str, str]]:
    """Rows of the JIPNet EER / TAR table (Part b.1).

    Args:
        jipnet: Per-database JIPNet metrics from :func:`load_jipnet_summary`.

    Returns:
        One dictionary per database with database name, sensor, note, EER and TAR (with ranges).
    """
    rows = []
    for key in DATABASES:
        r = jipnet.loc[key]
        name, sensor = DATABASE_NAMES[key]
        rows.append(
            {
                "name": name,
                "sensor": sensor,
                "note": DATABASE_NOTES.get(key, ""),
                "eer": fmt(r.EER, ".1%"),
                "eer_range": fmt_range(r.EER_lo, r.EER_hi, ".1%"),
                "tar": fmt(r.TAR_0p1, ".0%"),
                "tar_range": fmt_range(r.TAR_lo, r.TAR_hi, ".0%"),
            }
        )
    return rows


def pose_rows(jipnet: pd.DataFrame) -> list[dict[str, str]]:
    """Rows of the JIPNet pose-error table (Part b.3).

    Args:
        jipnet: Per-database JIPNet metrics from :func:`load_jipnet_summary`.

    Returns:
        One dictionary per database with typical and worst-10% rotation and translation errors.
    """
    rows = []
    for key in DATABASES:
        r = jipnet.loc[key]
        name, sensor = DATABASE_NAMES[key]
        rows.append(
            {
                "name": name,
                "sensor": sensor,
                "rot_typ": fmt(r.turn_med, ".1f"),
                "rot_p90": fmt(r.turn_p90, ".0f"),
                "shift_typ": fmt(r.shift_med, ".1f"),
                "shift_typ_mm": fmt(r.shift_med * MM_PER_PIXEL, ".2f"),
                "shift_p90": fmt(r.shift_p90, ".0f"),
            }
        )
    return rows


def comparison_rows(afr: dict[str, dict[str, float]]) -> list[dict[str, str]]:
    """Rows of the JIPNet vs AFR-Net-style comparison table (Part b.2).

    Args:
        afr: AFR-Net-style results per database, from ``experiments/afr_summary.json``.

    Returns:
        One dictionary per database with measured and paper values for both models.
    """
    rows = []
    for key in ["2002_DB1", "2002_DB3"]:
        a = afr[key]
        name, sensor = DATABASE_NAMES[key]
        rows.append(
            {
                "name": name,
                "sensor": sensor,
                "jip_eer": fmt(a["jip_same400_EER"], ".1%"),
                "jip_tar": fmt(a["jip_same400_TAR"], ".0%"),
                "jip_paper": PAPER_RESULTS[key]["jipnet"],
                "afr_eer": fmt(a["afr_EER"], ".1%"),
                "afr_tar": fmt(a["afr_TAR"], ".0%"),
                "afr_paper": PAPER_RESULTS[key]["afrnet"],
            }
        )
    return rows


# ---------------------------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------------------------


def build_context() -> dict[str, Any]:
    """Assemble all values needed by the report templates.

    Returns:
        Dictionary of template variables: summary ranges, timing and operation-count figures,
        table rows, rotation analysis values and the appendix figure list.
    """
    jipnet = load_jipnet_summary()
    real = jipnet.drop(SYNTHETIC)
    synthetic = jipnet.loc[SYNTHETIC]
    afr = load_json("afr_summary.json")
    rotation = load_json("afr_rotation.json")["2002_DB1"]
    flops = load_json("flops.json")
    bench = load_json("bench.json")

    n = GALLERY_SIZE
    jip_sec = float(jipnet.sec_med.median())
    afr_eer = [v["afr_EER"] for v in afr.values()]
    jip_same = [v["jip_same400_EER"] for v in afr.values()]
    hers_mults = 32 * -(-n // HERS_SLOTS)
    jip_full_flop = n * flops["JIPNet_pair_GFLOP"] * 1e9

    return {
        # Executive summary and failure analysis
        "real_eer_range": fmt_range(real.EER.min(), real.EER.max(), ".1%"),
        "real_tar_range": fmt_range(real.TAR_0p1.min(), real.TAR_0p1.max(), ".0%"),
        "syn_eer_range": fmt_range(synthetic.EER.min(), synthetic.EER.max(), ".1%"),
        "rot_range": fmt_range(jipnet.turn_med.min(), jipnet.turn_med.max(), ".0f"),
        "shift_range": fmt_range(jipnet.shift_med.min(), jipnet.shift_med.max(), ".0f"),
        "shift_mm_range": fmt_range(
            jipnet.shift_med.min() * MM_PER_PIXEL, jipnet.shift_med.max() * MM_PER_PIXEL, ".1f"
        ),
        "jip_same_range": fmt_range(min(jip_same), max(jip_same), ".1%"),
        "afr_range": fmt_range(min(afr_eer), max(afr_eer), ".1%"),
        "afr_eer_db1": fmt(afr["2002_DB1"]["afr_EER"], ".1%"),
        # Operation counts (machine-independent)
        "afr_gflop": fmt(flops["AFRNet_template_GFLOP"], ".1f"),
        "jip_gflop": fmt(flops["JIPNet_pair_GFLOP"], ".1f"),
        "search_tflop_768": fmt(2 * n * 768 / 1e12, ".0f"),
        "search_tflop_32": fmt(2 * n * 32 / 1e12, ".1f"),
        "indexed_gflop_32": fmt(2 * n * 32 * 0.01 / 1e9, ".0f"),
        "jip_k1000_tflop": fmt(1000 * flops["JIPNet_pair_GFLOP"] / 1e3, ".1f"),
        "hers_mults_text": sci_text(hers_mults),
        "hers_mults_latex": sci_latex(hers_mults),
        "jip_full_text": sci_text(jip_full_flop),
        "jip_full_latex": sci_latex(jip_full_flop),
        # Wall-clock timings (test machine)
        "afr_template_sec": fmt(min(v["sec_per_patch"] for v in afr.values()), ".1f"),
        "compare_ns_768": fmt(bench["768"] * 1e9, ".0f"),
        "scan_minutes_768": fmt(bench["768"] * n / 60, ".0f"),
        "jip_sec": fmt(jip_sec, ".2f"),
        "jip_1000_minutes": fmt(jip_sec * 1000 / 60, ".0f"),
        "jip_full_years": fmt(n * jip_sec / SECONDS_PER_YEAR, ",.0f"),
        # Rotation analysis (AFR-Net-style baseline, FVC2002 DB1)
        "rot_score_low": fmt(rotation[0]["afr_med"], ".2f"),
        "rot_score_high": fmt(rotation[-1]["afr_med"], ".2f"),
        # Tables and figures
        "jipnet_rows": jipnet_rows(jipnet),
        "pose_rows": pose_rows(jipnet),
        "comparison_rows": comparison_rows(afr),
        "appendix_figures": APPENDIX_FIGURES,
    }


if __name__ == "__main__":
    for key, value in build_context().items():
        if not isinstance(value, list):
            print(f"{key:20s} {value}")
