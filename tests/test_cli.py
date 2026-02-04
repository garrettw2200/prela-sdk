"""Tests for Prela CLI."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Check if CLI dependencies are installed
try:
    import typer
    import yaml
    from rich.console import Console
    from typer.testing import CliRunner

    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False

if CLI_AVAILABLE:
    from prela.contrib.cli import (
        DEFAULT_CONFIG,
        app,
        build_span_tree,
        find_root_span,
        format_span_label,
        group_spans_by_trace,
        load_config,
        load_traces_from_file,
        parse_duration,
        save_config,
    )

pytestmark = pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI dependencies not installed")


@pytest.fixture
def cli_runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_config_file(tmp_path, monkeypatch):
    """Create temporary config file."""
    config_path = tmp_path / ".prela.yaml"
    monkeypatch.chdir(tmp_path)
    return config_path


@pytest.fixture
def sample_spans():
    """Create sample span data for testing."""
    return [
        {
            "trace_id": "trace-123",
            "span_id": "span-1",
            "parent_span_id": None,
            "name": "root_span",
            "span_type": "agent",
            "status": "success",
            "started_at": "2025-01-26T10:00:00.000000Z",
            "ended_at": "2025-01-26T10:00:02.000000Z",
            "duration_ms": 2000,
            "attributes": {"key": "value"},
            "events": [],
        },
        {
            "trace_id": "trace-123",
            "span_id": "span-2",
            "parent_span_id": "span-1",
            "name": "child_span",
            "span_type": "llm",
            "status": "success",
            "started_at": "2025-01-26T10:00:00.500000Z",
            "ended_at": "2025-01-26T10:00:01.500000Z",
            "duration_ms": 1000,
            "attributes": {"model": "gpt-4"},
            "events": [{"name": "llm.request", "timestamp": "2025-01-26T10:00:00.500000Z"}],
        },
        {
            "trace_id": "trace-456",
            "span_id": "span-3",
            "parent_span_id": None,
            "name": "another_root",
            "span_type": "tool",
            "status": "error",
            "started_at": "2025-01-26T10:05:00.000000Z",
            "ended_at": "2025-01-26T10:05:00.500000Z",
            "duration_ms": 500,
            "attributes": {},
            "events": [],
        },
    ]


@pytest.fixture
def trace_dir_with_data(tmp_path, sample_spans):
    """Create trace directory with sample data."""
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()

    # Write spans to JSONL file
    jsonl_file = trace_dir / "traces.jsonl"
    with open(jsonl_file, "w") as f:
        for span in sample_spans:
            f.write(json.dumps(span) + "\n")

    return trace_dir


class TestParseDuration:
    """Tests for parse_duration function."""

    def test_parse_seconds(self):
        """Test parsing seconds."""
        assert parse_duration("30s") == timedelta(seconds=30)

    def test_parse_minutes(self):
        """Test parsing minutes."""
        assert parse_duration("15m") == timedelta(minutes=15)

    def test_parse_hours(self):
        """Test parsing hours."""
        assert parse_duration("2h") == timedelta(hours=2)

    def test_parse_days(self):
        """Test parsing days."""
        assert parse_duration("3d") == timedelta(days=3)

    def test_invalid_unit(self):
        """Test invalid time unit."""
        with pytest.raises(ValueError, match="Unknown duration unit"):
            parse_duration("10x")

    def test_invalid_format(self):
        """Test invalid format."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("abc")

    def test_empty_duration(self):
        """Test empty duration."""
        with pytest.raises(ValueError, match="Duration cannot be empty"):
            parse_duration("")


class TestLoadConfig:
    """Tests for config loading."""

    def test_load_default_config_when_no_file(self, temp_config_file):
        """Test loading default config when file doesn't exist."""
        config = load_config()
        assert config == DEFAULT_CONFIG

    def test_load_existing_config(self, temp_config_file):
        """Test loading existing config file."""
        # Create config file
        config_data = {
            "service_name": "test-service",
            "exporter": "console",
            "sample_rate": 0.5,
        }
        with open(temp_config_file, "w") as f:
            yaml.safe_dump(config_data, f)

        config = load_config()
        assert config["service_name"] == "test-service"
        assert config["exporter"] == "console"
        assert config["sample_rate"] == 0.5
        # Should merge with defaults
        assert config["trace_dir"] == "./traces"

    def test_load_malformed_config(self, temp_config_file):
        """Test loading malformed config falls back to defaults."""
        # Create invalid YAML
        with open(temp_config_file, "w") as f:
            f.write("invalid: yaml: file: ][")

        config = load_config()
        assert config == DEFAULT_CONFIG


class TestSaveConfig:
    """Tests for config saving."""

    def test_save_config(self, temp_config_file):
        """Test saving config file."""
        config = {
            "service_name": "my-service",
            "exporter": "file",
            "trace_dir": "./traces",
            "sample_rate": 1.0,
        }

        save_config(config)

        # Verify file was created
        assert temp_config_file.exists()

        # Load and verify
        with open(temp_config_file) as f:
            loaded = yaml.safe_load(f)

        assert loaded == config


class TestLoadTracesFromFile:
    """Tests for loading traces from files."""

    def test_load_traces_from_jsonl(self, trace_dir_with_data, sample_spans):
        """Test loading traces from JSONL file."""
        traces = load_traces_from_file(trace_dir_with_data)
        assert len(traces) == len(sample_spans)
        assert traces[0]["trace_id"] == "trace-123"

    def test_load_traces_with_time_filter(self, trace_dir_with_data):
        """Test loading traces with since filter."""
        # Filter to only recent traces (within 1 minute)
        since = datetime.fromisoformat("2025-01-26T10:04:00.000000Z")
        traces = load_traces_from_file(trace_dir_with_data, since=since)

        # Should only get span-3 (started at 10:05:00)
        assert len(traces) == 1
        assert traces[0]["span_id"] == "span-3"

    def test_load_from_nonexistent_directory(self, tmp_path):
        """Test loading from directory that doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        traces = load_traces_from_file(nonexistent)
        assert traces == []

    def test_skip_invalid_json_lines(self, tmp_path):
        """Test that invalid JSON lines are skipped."""
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        jsonl_file = trace_dir / "traces.jsonl"
        with open(jsonl_file, "w") as f:
            f.write('{"valid": "json"}\n')
            f.write("invalid json line\n")
            f.write('{"another": "valid"}\n')

        traces = load_traces_from_file(trace_dir)
        assert len(traces) == 2


class TestGroupSpansByTrace:
    """Tests for grouping spans by trace ID."""

    def test_group_spans(self, sample_spans):
        """Test grouping spans by trace_id."""
        grouped = group_spans_by_trace(sample_spans)

        assert len(grouped) == 2
        assert len(grouped["trace-123"]) == 2
        assert len(grouped["trace-456"]) == 1

    def test_group_empty_spans(self):
        """Test grouping empty span list."""
        grouped = group_spans_by_trace([])
        assert grouped == {}


class TestFindRootSpan:
    """Tests for finding root span."""

    def test_find_root_span(self, sample_spans):
        """Test finding root span."""
        root = find_root_span(sample_spans)
        assert root is not None
        assert root["span_id"] == "span-1"
        assert root["parent_span_id"] is None

    def test_find_root_when_no_null_parent(self):
        """Test finding root when no span has null parent (returns first)."""
        spans = [
            {"span_id": "span-1", "parent_span_id": "parent-1"},
            {"span_id": "span-2", "parent_span_id": "parent-2"},
        ]
        root = find_root_span(spans)
        assert root["span_id"] == "span-1"

    def test_find_root_empty_list(self):
        """Test finding root in empty list."""
        root = find_root_span([])
        assert root is None


class TestBuildSpanTree:
    """Tests for building span tree."""

    def test_build_simple_tree(self, sample_spans):
        """Test building tree from flat span list."""
        tree = build_span_tree(sample_spans, parent_id=None)

        # Should have 2 root nodes
        assert len(tree) == 2

        # First root should have 1 child
        assert tree[0]["span"]["span_id"] == "span-1"
        assert len(tree[0]["children"]) == 1
        assert tree[0]["children"][0]["span"]["span_id"] == "span-2"

        # Second root has no children
        assert tree[1]["span"]["span_id"] == "span-3"
        assert len(tree[1]["children"]) == 0

    def test_build_tree_empty(self):
        """Test building tree from empty list."""
        tree = build_span_tree([])
        assert tree == []


class TestFormatSpanLabel:
    """Tests for formatting span labels."""

    def test_format_success_span(self):
        """Test formatting successful span."""
        span = {
            "name": "test_span",
            "span_type": "llm",
            "status": "success",
            "duration_ms": 1500,
        }
        label = format_span_label(span)

        assert "test_span" in label
        assert "llm" in label
        assert "success" in label
        assert "1.50s" in label  # Duration > 1000ms shows in seconds

    def test_format_error_span(self):
        """Test formatting error span."""
        span = {
            "name": "failed_span",
            "span_type": "tool",
            "status": "error",
            "duration_ms": 500,
        }
        label = format_span_label(span)

        assert "failed_span" in label
        assert "tool" in label
        assert "error" in label
        assert "500ms" in label  # Duration < 1000ms shows in milliseconds

    def test_format_span_with_missing_fields(self):
        """Test formatting span with missing fields."""
        span = {}
        label = format_span_label(span)

        # Should use defaults
        assert "unknown" in label


class TestInitCommand:
    """Tests for 'prela init' command."""

    def test_init_interactive(self, cli_runner, temp_config_file):
        """Test interactive init command."""
        result = cli_runner.invoke(
            app,
            ["init"],
            input="my-service\nfile\n./traces\n1.0\n",
        )

        assert result.exit_code == 0
        assert "Configuration saved" in result.stdout

        # Verify config file
        assert temp_config_file.exists()
        with open(temp_config_file) as f:
            config = yaml.safe_load(f)

        assert config["service_name"] == "my-service"
        assert config["exporter"] == "file"
        assert config["sample_rate"] == 1.0

    def test_init_creates_trace_directory(self, cli_runner, temp_config_file):
        """Test that init creates trace directory."""
        result = cli_runner.invoke(
            app,
            ["init"],
            input="my-service\nfile\n./my-traces\n1.0\n",
        )

        assert result.exit_code == 0
        assert Path("./my-traces").exists()


class TestListCommand:
    """Tests for 'prela trace list' command."""

    def test_list_traces(self, cli_runner, temp_config_file, trace_dir_with_data):
        """Test listing traces."""
        # Create config pointing to trace directory
        config = DEFAULT_CONFIG.copy()
        config["trace_dir"] = str(trace_dir_with_data)
        with open(temp_config_file, "w") as f:
            yaml.safe_dump(config, f)

        result = cli_runner.invoke(app, ["list"])

        assert result.exit_code == 0
        assert "trace-123" in result.stdout
        assert "trace-456" in result.stdout
        assert "root_span" in result.stdout
        assert "another_root" in result.stdout

    def test_list_with_limit(self, cli_runner, temp_config_file, trace_dir_with_data):
        """Test listing traces with limit."""
        config = DEFAULT_CONFIG.copy()
        config["trace_dir"] = str(trace_dir_with_data)
        with open(temp_config_file, "w") as f:
            yaml.safe_dump(config, f)

        result = cli_runner.invoke(app, ["list", "--limit", "1"])

        assert result.exit_code == 0
        # Should show "1 of 2"
        assert "(1 of 2)" in result.stdout or "Recent Traces" in result.stdout

    def test_list_no_traces(self, cli_runner, temp_config_file, tmp_path):
        """Test listing when no traces exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        config = DEFAULT_CONFIG.copy()
        config["trace_dir"] = str(empty_dir)
        with open(temp_config_file, "w") as f:
            yaml.safe_dump(config, f)

        result = cli_runner.invoke(app, ["list"])

        assert result.exit_code == 0
        assert "No traces found" in result.stdout


class TestShowCommand:
    """Tests for 'prela trace show' command."""

    def test_show_trace_full_id(self, cli_runner, temp_config_file, trace_dir_with_data):
        """Test showing trace with full trace ID."""
        config = DEFAULT_CONFIG.copy()
        config["trace_dir"] = str(trace_dir_with_data)
        with open(temp_config_file, "w") as f:
            yaml.safe_dump(config, f)

        result = cli_runner.invoke(app, ["show", "trace-123"])

        assert result.exit_code == 0
        assert "trace-123" in result.stdout
        assert "root_span" in result.stdout
        assert "child_span" in result.stdout
        assert "Span Details" in result.stdout

    def test_show_trace_partial_id(self, cli_runner, temp_config_file, trace_dir_with_data):
        """Test showing trace with partial trace ID."""
        config = DEFAULT_CONFIG.copy()
        config["trace_dir"] = str(trace_dir_with_data)
        with open(temp_config_file, "w") as f:
            yaml.safe_dump(config, f)

        result = cli_runner.invoke(app, ["show", "trace-456"])

        assert result.exit_code == 0
        assert "trace-456" in result.stdout
        assert "another_root" in result.stdout

    def test_show_nonexistent_trace(self, cli_runner, temp_config_file, trace_dir_with_data):
        """Test showing trace that doesn't exist."""
        config = DEFAULT_CONFIG.copy()
        config["trace_dir"] = str(trace_dir_with_data)
        with open(temp_config_file, "w") as f:
            yaml.safe_dump(config, f)

        result = cli_runner.invoke(app, ["show", "nonexistent"])

        assert result.exit_code == 1
        assert "No trace found" in result.stdout


class TestSearchCommand:
    """Tests for 'prela trace search' command."""

    def test_search_by_span_name(self, cli_runner, temp_config_file, trace_dir_with_data):
        """Test searching by span name."""
        config = DEFAULT_CONFIG.copy()
        config["trace_dir"] = str(trace_dir_with_data)
        with open(temp_config_file, "w") as f:
            yaml.safe_dump(config, f)

        result = cli_runner.invoke(app, ["search", "root_span"])

        assert result.exit_code == 0
        assert "Found 1 traces" in result.stdout or "root_span" in result.stdout

    def test_search_by_attribute(self, cli_runner, temp_config_file, trace_dir_with_data):
        """Test searching by attribute value."""
        config = DEFAULT_CONFIG.copy()
        config["trace_dir"] = str(trace_dir_with_data)
        with open(temp_config_file, "w") as f:
            yaml.safe_dump(config, f)

        result = cli_runner.invoke(app, ["search", "gpt-4"])

        assert result.exit_code == 0
        assert "Found 1 traces" in result.stdout or "trace-123" in result.stdout

    def test_search_no_results(self, cli_runner, temp_config_file, trace_dir_with_data):
        """Test search with no results."""
        config = DEFAULT_CONFIG.copy()
        config["trace_dir"] = str(trace_dir_with_data)
        with open(temp_config_file, "w") as f:
            yaml.safe_dump(config, f)

        result = cli_runner.invoke(app, ["search", "nonexistent-query"])

        assert result.exit_code == 0
        assert "No traces found matching" in result.stdout


class TestServeCommand:
    """Tests for 'prela serve' command."""

    def test_serve_placeholder(self, cli_runner):
        """Test serve command (placeholder)."""
        result = cli_runner.invoke(app, ["serve"])

        assert result.exit_code == 0
        assert "not yet implemented" in result.stdout

    def test_serve_custom_port(self, cli_runner):
        """Test serve with custom port."""
        result = cli_runner.invoke(app, ["serve", "--port", "9000"])

        assert result.exit_code == 0
        assert "9000" in result.stdout


class TestEvalCommand:
    """Tests for 'prela eval' command."""

    def test_eval_placeholder(self, cli_runner):
        """Test eval command (placeholder)."""
        result = cli_runner.invoke(app, ["eval", "suite.yaml"])

        assert result.exit_code == 0
        assert "not yet implemented" in result.stdout
        assert "suite.yaml" in result.stdout
