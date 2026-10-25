"""Render the HTML report from ``templates/report.html.j2`` and print it to an A4 PDF.

Usage (from the repository root)::

    python src/build_report.py            # HTML and PDF
    python src/build_report.py --no-pdf   # HTML only

Outputs: ``reports/UIDAI_review_submission.html`` and ``reports/UIDAI_review_submission.pdf``.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from functools import lru_cache

from jinja2 import Environment, FileSystemLoader, StrictUndefined

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import ROOT  # noqa: E402
from report_data import build_context  # noqa: E402

TEMPLATE_DIR = os.path.join(ROOT, "templates")
FIGURE_DIR = os.path.join(ROOT, "reports", "figures")
HTML_OUT = os.path.join(ROOT, "reports", "UIDAI_review_submission.html")
PDF_OUT = os.path.join(ROOT, "reports", "UIDAI_review_submission.pdf")
FOOTER = (
    '<div style="font-size:8px;width:100%;text-align:center;color:#666">Sreejith Markkassery · '
    "Review of AI/ML research for fingerprint deduplication · page "
    '<span class="pageNumber"></span> of <span class="totalPages"></span></div>'
)


@lru_cache(maxsize=None)
def image_data_uri(file_name: str) -> str:
    """Embed a figure as a base64 data URI so the HTML report is a single self-contained file.

    Args:
        file_name: PNG file name inside ``reports/figures/``.

    Returns:
        A ``data:image/png;base64,...`` string, or an empty string if the file is missing.
    """
    path = os.path.join(FIGURE_DIR, file_name)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as handle:
        return "data:image/png;base64," + base64.b64encode(handle.read()).decode()


def render_html(context: dict) -> str:
    """Fill the HTML template with report values.

    Args:
        context: Template variables from :func:`report_data.build_context`.

    Returns:
        The complete HTML document.
    """
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    env.globals["image"] = image_data_uri
    return env.get_template("report.html.j2").render(**context)


def write_pdf(html_path: str, pdf_path: str) -> None:
    """Print an HTML file to an A4 PDF with a page footer, using headless Chromium.

    Args:
        html_path: Path of the rendered HTML report.
        pdf_path: Path of the PDF to write.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.emulate_media(media="print", color_scheme="light")
        page.goto(f"file://{html_path}", wait_until="load")
        page.pdf(
            path=pdf_path,
            format="A4",
            print_background=True,
            margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
            display_header_footer=True,
            header_template="<span></span>",
            footer_template=FOOTER,
        )
        browser.close()


def main() -> None:
    """Build the HTML report and, unless ``--no-pdf`` is given, the PDF."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-pdf", action="store_true", help="write the HTML report only")
    args = parser.parse_args()

    html = render_html(build_context())
    with open(HTML_OUT, "w") as handle:
        handle.write(html)
    print(f"wrote {HTML_OUT}")
    if not args.no_pdf:
        write_pdf(HTML_OUT, PDF_OUT)
        print(f"wrote {PDF_OUT}")


if __name__ == "__main__":
    main()
