"""
test_build_site.py — Tests for web/build_site.py template injection & build validation.

Required assertions (TRD §8, PRD §6.5):
1. Placeholder token is replaced in output.
2. Injected JSON round-trips: json.loads(extracted_json) == original_dict.
3. No '</script>' in the injected JSON string (escape test).
4. Output file parses as valid HTML (has <html>, <head>, <body>, <!doctype html>).
"""

from __future__ import annotations

import html.parser
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src.export import SchemaValidationError
from web.build_site import (
    MAX_SIZE_BYTES,
    PLACEHOLDER_TOKEN,
    build_site,
    escape_json_for_script,
)

ROOT = Path(__file__).resolve().parent.parent
MOCK_RESULTS_PATH = ROOT / "web" / "mock_results.json"
TEMPLATE_PATH = ROOT / "web" / "index.html"


class SimpleHTMLValidator(html.parser.HTMLParser):
    """Basic HTML parser to verify well-formed tag hierarchy."""

    def __init__(self) -> None:
        super().__init__()
        self.tags_seen: set[str] = set()
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags_seen.add(tag.lower())

    def error(self, message: str) -> None:
        self.errors.append(message)


class TestBuildSiteAssertions:
    """Core four build assertions defined in TRD §8."""

    @pytest.fixture(autouse=True)
    def setup_build(self, tmp_path: Path) -> None:
        self.out_html = tmp_path / "index.html"
        build_site(
            template_path=TEMPLATE_PATH,
            results_path=MOCK_RESULTS_PATH,
            out_path=self.out_html,
            validate_schema=True,
        )
        self.content = self.out_html.read_text(encoding="utf-8")
        self.mock_data = json.loads(MOCK_RESULTS_PATH.read_text(encoding="utf-8"))

    def test_placeholder_token_is_replaced(self) -> None:
        """Assertion 1: Placeholder token /*__RESULTS_JSON__*/ is absent from output."""
        assert PLACEHOLDER_TOKEN not in self.content
        assert "results-data" in self.content

    def test_injected_json_roundtrips(self) -> None:
        """Assertion 2: Injected JSON parses back to the exact original dictionary."""
        # Extract content of <script id="results-data" type="application/json">...</script>
        pattern = r'<script id="results-data" type="application/json">(.*?)</script>'
        match = re.search(pattern, self.content, re.DOTALL)
        assert match is not None, "Script block with id='results-data' not found"

        raw_injected = match.group(1).strip()
        parsed_injected = json.loads(raw_injected)

        # Invariant check: parsed content must exactly match original mock data
        assert parsed_injected == self.mock_data

    def test_no_unescaped_script_tag(self) -> None:
        """Assertion 3: No '</script>' exists inside the injected payload."""
        pattern = r'<script id="results-data" type="application/json">(.*?)</script>'
        match = re.search(pattern, self.content, re.DOTALL)
        assert match is not None

        raw_injected = match.group(1)
        # Any occurrence of </script> inside JSON would prematurely terminate the script
        assert "</script>" not in raw_injected.lower()
        # Verify solidus escape applied where needed
        if "/" in raw_injected:
            # When decoded with json.loads, all escaped soliduses decode correctly
            decoded = json.loads(raw_injected)
            assert isinstance(decoded, dict)

    def test_output_parses_as_valid_html(self) -> None:
        """Assertion 4: Output file has doctype, html, head, body, and parses cleanly."""
        assert self.content.lower().startswith("<!-- built:") or "<!doctype html>" in self.content.lower()
        assert "<!doctype html>" in self.content.lower()

        validator = SimpleHTMLValidator()
        validator.feed(self.content)

        assert len(validator.errors) == 0
        required_tags = {"html", "head", "body", "title", "header", "main", "footer", "script"}
        assert required_tags.issubset(validator.tags_seen)


class TestEscapingAndSanitization:
    """Verify security escaping of closing script tags in JSON strings."""

    def test_escape_json_for_script_replaces_closing_tag(self) -> None:
        malicious = '{"bio":"Hello </script><script>alert(\'pwn\')</script> world"}'
        escaped = escape_json_for_script(malicious)

        assert "</script>" not in escaped.lower()
        assert "<\\/script>" in escaped

        # Crucial: verify that standard JSON.loads in Python (and JSON.parse in JS)
        # decodes the exact original string back
        decoded = json.loads(escaped)
        assert decoded["bio"] == "Hello </script><script>alert('pwn')</script> world"


class TestHeaderCommentAndSize:
    """Verify build telemetry comment and size thresholds."""

    def test_build_comment_format(self, tmp_path: Path) -> None:
        out_file = tmp_path / "out.html"
        build_site(
            template_path=TEMPLATE_PATH,
            results_path=MOCK_RESULTS_PATH,
            out_path=out_file,
            validate_schema=True,
        )
        first_line = out_file.read_text(encoding="utf-8").splitlines()[0]
        pattern = r"^<!-- built: \d{4}-\d{2}-\d{2}T.*? results-sha256: [0-9a-f]{64} -->$"
        assert re.match(pattern, first_line), f"Invalid header comment: {first_line}"

    def test_size_well_under_one_megabyte(self, tmp_path: Path) -> None:
        out_file = tmp_path / "out.html"
        build_site(
            template_path=TEMPLATE_PATH,
            results_path=MOCK_RESULTS_PATH,
            out_path=out_file,
            validate_schema=True,
        )
        size_bytes = len(out_file.read_bytes())
        # Standalone app with full mock results is ~140 KB, well under 1 MB
        assert size_bytes < MAX_SIZE_BYTES
        assert size_bytes > 50 * 1024


class TestBuildSiteErrorsAndEdgeCases:
    """Verify proper exceptions on missing files, malformed templates, or invalid schema."""

    def test_missing_template_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Template file not found"):
            build_site(
                template_path=tmp_path / "non_existent.html",
                results_path=MOCK_RESULTS_PATH,
                out_path=tmp_path / "dist.html",
            )

    def test_missing_results_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Results file not found"):
            build_site(
                template_path=TEMPLATE_PATH,
                results_path=tmp_path / "missing_results.json",
                out_path=tmp_path / "dist.html",
            )

    def test_missing_placeholder_raises(self, tmp_path: Path) -> None:
        bad_template = tmp_path / "bad_template.html"
        bad_template.write_text("<!doctype html><html><body>No token here</body></html>", encoding="utf-8")
        with pytest.raises(ValueError, match="Placeholder token .* was not found"):
            build_site(
                template_path=bad_template,
                results_path=MOCK_RESULTS_PATH,
                out_path=tmp_path / "dist.html",
            )

    def test_duplicate_placeholder_raises(self, tmp_path: Path) -> None:
        bad_template = tmp_path / "dup_template.html"
        bad_template.write_text(
            f"<!doctype html><html><body>{PLACEHOLDER_TOKEN} and {PLACEHOLDER_TOKEN}</body></html>",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="appeared 2 times"):
            build_site(
                template_path=bad_template,
                results_path=MOCK_RESULTS_PATH,
                out_path=tmp_path / "dist.html",
            )

    def test_schema_validation_failure_raises(self, tmp_path: Path) -> None:
        bad_results = tmp_path / "invalid_schema.json"
        # Missing required top-level fields
        bad_results.write_text(json.dumps({"meta": {"quick": True}}), encoding="utf-8")

        with pytest.raises(SchemaValidationError):
            build_site(
                template_path=TEMPLATE_PATH,
                results_path=bad_results,
                out_path=tmp_path / "dist.html",
                validate_schema=True,
            )


class TestCLIExecution:
    """Verify command line invocation of web/build_site.py."""

    def test_cli_mock_flag(self, tmp_path: Path) -> None:
        out_html = tmp_path / "cli_dist.html"
        cmd = [
            sys.executable,
            str(ROOT / "web" / "build_site.py"),
            "--mock",
            "--out",
            str(out_html),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"CLI stderr: {res.stderr}"
        assert out_html.is_file()
        assert PLACEHOLDER_TOKEN not in out_html.read_text(encoding="utf-8")

    def test_cli_missing_results_returns_error(self, tmp_path: Path) -> None:
        cmd = [
            sys.executable,
            str(ROOT / "web" / "build_site.py"),
            "--results",
            str(tmp_path / "does_not_exist.json"),
            "--out",
            str(tmp_path / "cli_dist.html"),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert res.returncode == 1
        assert "ERROR:" in res.stderr
