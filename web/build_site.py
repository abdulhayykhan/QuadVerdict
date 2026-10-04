"""
build_site.py — Injects benchmark results into the standalone web template.

Design specification (TRD §7, PRD §6.5):
1. Reads web/index.html and ensures the placeholder /*__RESULTS_JSON__*/ appears exactly once.
2. Reads the results JSON (real or --mock), validates it against results/schema.json.
3. Computes SHA-256 hash of the results payload.
4. Minifies JSON with separators=(",", ":") and escapes '</' -> '<\\/' to prevent premature script tag closure.
5. Injects the minified JSON into the template, prepending a build comment with timestamp and results SHA-256.
6. Writes output to dist/index.html (or specified --out path) and reports final file size.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.export import validate  # noqa: E402

PLACEHOLDER_TOKEN = "/*__RESULTS_JSON__*/"
MAX_SIZE_BYTES = 1024 * 1024  # 1 MB warning threshold


def escape_json_for_script(json_str: str) -> str:
    """Escape '</' sequence in JSON strings to prevent HTML script tag breakout.

    In HTML documents, any sequence matching '</' inside a <script> block can prematurely
    close the script tag. In JSON, '\\/' is a valid escape sequence for '/', so '</' can
    be safely escaped as '<\\/'.
    """
    return json_str.replace("</", "<\\/")


def build_site(
    template_path: Path | str,
    results_path: Path | str,
    out_path: Path | str,
    validate_schema: bool = True,
) -> Path:
    """Inject benchmark results JSON into HTML template and write to output path.

    Parameters
    ----------
    template_path : Path or str
        Path to web/index.html source template.
    results_path : Path or str
        Path to results.json or mock_results.json.
    out_path : Path or str
        Destination path (e.g. dist/index.html).
    validate_schema : bool, default=True
        Whether to validate results against results/schema.json before injection.

    Returns
    -------
    Path
        Resolved destination path of the built file.

    Raises
    ------
    FileNotFoundError
        If template_path or results_path does not exist.
    ValueError
        If the placeholder token is missing or appears more than once in the template.
    SchemaValidationError
        If schema validation fails.
    """
    template_path = Path(template_path).resolve()
    results_path = Path(results_path).resolve()
    out_path = Path(out_path).resolve()

    if not template_path.is_file():
        raise FileNotFoundError(f"Template file not found at: {template_path}")
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Results file not found at: {results_path}. "
            "Run the benchmark first or pass --mock to build with mock data."
        )

    # 1. Read template and enforce single placeholder invariant
    template_text = template_path.read_text(encoding="utf-8")
    token_count = template_text.count(PLACEHOLDER_TOKEN)
    if token_count == 0:
        raise ValueError(
            f"Placeholder token {PLACEHOLDER_TOKEN} was not found in template {template_path}."
        )
    if token_count > 1:
        raise ValueError(
            f"Placeholder token {PLACEHOLDER_TOKEN} appeared {token_count} times in template "
            f"(must appear exactly once)."
        )

    # 2. Read and parse results JSON
    raw_results_bytes = results_path.read_bytes()
    sha256_hex = hashlib.sha256(raw_results_bytes).hexdigest()

    try:
        results_data = json.loads(raw_results_bytes.decode("utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"Failed to parse JSON from {results_path}: {err}") from err

    # 3. Validate against results/schema.json
    if validate_schema:
        validate(results_data)

    # 4. Minify and escape JSON string
    minified_json = json.dumps(results_data, separators=(",", ":"))
    escaped_json = escape_json_for_script(minified_json)

    # Sanity check: Ensure '</script>' is not in the escaped JSON
    if "</script>" in escaped_json.lower():
        raise ValueError("Escaped JSON still contains unescaped '</script>' tag.")

    # 5. Build output with header comment
    built_iso_utc = datetime.datetime.now(datetime.UTC).isoformat()
    header_comment = f"<!-- built: {built_iso_utc} results-sha256: {sha256_hex} -->\n"

    injected_html = template_text.replace(PLACEHOLDER_TOKEN, escaped_json, 1)
    final_html = header_comment + injected_html

    # 6. Ensure destination directory exists and write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(final_html, encoding="utf-8")

    # 7. Check file size
    final_size_bytes = len(final_html.encode("utf-8"))
    final_size_kb = final_size_bytes / 1024

    print(f"[build_site] Template: {template_path.name}")
    print(f"[build_site] Results:  {results_path.name} (SHA-256: {sha256_hex[:16]}...)")
    print(f"[build_site] Output:   {out_path} ({final_size_kb:.1f} KB)")

    if final_size_bytes > MAX_SIZE_BYTES:
        print(
            f"[build_site] WARNING: Built site size ({final_size_kb:.1f} KB) exceeds 1 MB limit!"
        )

    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inject QuadVerdict benchmark results into the standalone web template."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=config.RESULTS_DIR / "results.json",
        help="Path to results.json (default: results/results.json)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use web/mock_results.json instead of results/results.json",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "web" / "index.html",
        help="Path to HTML template (default: web/index.html)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "dist" / "index.html",
        help="Destination path for built site (default: dist/index.html)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip JSON schema and invariant validation (not recommended)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for web/build_site.py."""
    args = parse_args(argv)

    if args.mock:
        results_file = ROOT / "web" / "mock_results.json"
    else:
        results_file = args.results

    try:
        build_site(
            template_path=args.template,
            results_path=results_file,
            out_path=args.out,
            validate_schema=not args.no_validate,
        )
        return 0
    except Exception as err:
        print(f"[build_site] ERROR: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
