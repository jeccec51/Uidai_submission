"""Render the LaTeX report from ``templates/report.tex.j2`` and optionally compile it.

Usage (from the repository root)::

    python src/build_latex.py              # write reports/latex/UIDAI_review.tex
    python src/build_latex.py --compile    # also build the PDF with latexmk

The figures used by the LaTeX report are copied to ``reports/latex/figures/``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from jinja2 import Environment, FileSystemLoader, StrictUndefined

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402
from report_data import build_context  # noqa: E402

TEMPLATE_DIR = os.path.join(ROOT, "templates")
FIGURE_DIR = os.path.join(ROOT, "reports", "figures")
LATEX_DIR = os.path.join(ROOT, "reports", "latex")
TEX_OUT = os.path.join(LATEX_DIR, "UIDAI_review.tex")

#: Characters that need translating from report text into LaTeX.
LATEX_REPLACEMENTS = [
    ("\\", r"\textbackslash{}"),
    ("%", r"\%"),
    ("&", r"\&"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("–", "--"),
    ("’", "'"),
    ("°", r"$^\circ$"),
    ("×", r"$\times$"),
    ("≈", r"$\approx$"),
]


def tex_escape(value: object) -> str:
    """Convert a report value to LaTeX-safe text.

    Args:
        value: Number or string from the report context.

    Returns:
        The text with LaTeX special characters escaped and typographic symbols translated.
    """
    text = str(value)
    for old, new in LATEX_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def latex_environment() -> Environment:
    """Create a Jinja environment whose delimiters do not clash with LaTeX syntax.

    Returns:
        Environment using ``\\VAR{...}`` for values, ``\\BLOCK{...}`` for statements and
        ``\\#{...}`` for comments, with a ``tex`` filter for escaping.
    """
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        block_start_string=r"\BLOCK{",
        block_end_string="}",
        variable_start_string=r"\VAR{",
        variable_end_string="}",
        comment_start_string=r"\#{",
        comment_end_string="}",
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    env.filters["tex"] = tex_escape
    return env


def copy_figures() -> None:
    """Copy the result figures and the diagram images into ``reports/latex/figures/``.

    Result figures come from ``reports/figures/``; diagram images (``templates/diagrams/*.png``)
    are stored as ``diagram_<name>.png``.
    """
    target = os.path.join(LATEX_DIR, "figures")
    os.makedirs(target, exist_ok=True)
    for name in os.listdir(FIGURE_DIR):
        if name.endswith(".png"):
            shutil.copy2(os.path.join(FIGURE_DIR, name), target)
    diagram_dir = os.path.join(TEMPLATE_DIR, "diagrams")
    for name in os.listdir(diagram_dir):
        if name.endswith(".png"):
            shutil.copy2(os.path.join(diagram_dir, name), os.path.join(target, f"diagram_{name}"))


def compile_pdf() -> None:
    """Compile ``UIDAI_review.tex`` with latexmk (pdflatex) and remove auxiliary files."""
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", os.path.basename(TEX_OUT)],
        cwd=LATEX_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(["latexmk", "-c"], cwd=LATEX_DIR, check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    """Render the LaTeX source and, with ``--compile``, the PDF."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compile", action="store_true", help="compile the PDF with latexmk")
    args = parser.parse_args()

    tex = latex_environment().get_template("report.tex.j2").render(**build_context())
    os.makedirs(LATEX_DIR, exist_ok=True)
    with open(TEX_OUT, "w") as handle:
        handle.write(tex)
    copy_figures()
    print(f"wrote {TEX_OUT}")
    if args.compile:
        compile_pdf()
        print(f"wrote {TEX_OUT[:-4]}.pdf")


if __name__ == "__main__":
    main()
