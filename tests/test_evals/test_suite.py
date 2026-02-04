"""Tests for prela.evals.suite module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from prela.evals.case import EvalCase, EvalExpected, EvalInput
from prela.evals.suite import EvalSuite

# Check if PyYAML is available
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class TestEvalSuite:
    """Tests for EvalSuite class."""

    def test_init_with_minimal_fields(self):
        """Test initialization with minimal fields."""
        suite = EvalSuite(name="Test Suite")
        assert suite.name == "Test Suite"
        assert suite.description == ""
        assert suite.cases == []
        assert suite.default_assertions is None
        assert suite.setup is None
        assert suite.teardown is None
        assert suite.metadata == {}

    def test_init_with_all_fields(self):
        """Test initialization with all fields."""
        case = EvalCase(
            id="test_1",
            name="Test",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
        )

        def setup_func():
            pass

        def teardown_func():
            pass

        suite = EvalSuite(
            name="Test Suite",
            description="A test suite",
            cases=[case],
            default_assertions=[{"type": "no_errors"}],
            setup=setup_func,
            teardown=teardown_func,
            metadata={"version": "1.0"},
        )
        assert suite.description == "A test suite"
        assert len(suite.cases) == 1
        assert suite.default_assertions == [{"type": "no_errors"}]
        assert suite.setup == setup_func
        assert suite.teardown == teardown_func
        assert suite.metadata == {"version": "1.0"}

    def test_init_without_name(self):
        """Test that initialization fails without name."""
        with pytest.raises(ValueError, match="must have a non-empty 'name'"):
            EvalSuite(name="")

    def test_add_case(self):
        """Test adding a case to the suite."""
        suite = EvalSuite(name="Test Suite")
        case = EvalCase(
            id="test_1",
            name="Test",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
        )
        suite.add_case(case)
        assert len(suite.cases) == 1
        assert suite.cases[0] == case

    def test_get_case_found(self):
        """Test getting a case by ID."""
        case1 = EvalCase(
            id="test_1",
            name="Test 1",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
        )
        case2 = EvalCase(
            id="test_2",
            name="Test 2",
            input=EvalInput(query="Goodbye"),
            expected=EvalExpected(contains=["Bye"]),
        )
        suite = EvalSuite(name="Test Suite", cases=[case1, case2])
        found = suite.get_case("test_2")
        assert found == case2

    def test_get_case_not_found(self):
        """Test getting a case by ID that doesn't exist."""
        case = EvalCase(
            id="test_1",
            name="Test",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
        )
        suite = EvalSuite(name="Test Suite", cases=[case])
        found = suite.get_case("test_999")
        assert found is None

    def test_filter_by_tags_single_tag(self):
        """Test filtering cases by a single tag."""
        case1 = EvalCase(
            id="test_1",
            name="Test 1",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
            tags=["qa", "greeting"],
        )
        case2 = EvalCase(
            id="test_2",
            name="Test 2",
            input=EvalInput(query="What is 2+2?"),
            expected=EvalExpected(contains=["4"]),
            tags=["qa", "math"],
        )
        case3 = EvalCase(
            id="test_3",
            name="Test 3",
            input=EvalInput(query="Goodbye"),
            expected=EvalExpected(contains=["Bye"]),
            tags=["greeting"],
        )
        suite = EvalSuite(name="Test Suite", cases=[case1, case2, case3])

        # Filter by "qa"
        qa_cases = suite.filter_by_tags(["qa"])
        assert len(qa_cases) == 2
        assert case1 in qa_cases
        assert case2 in qa_cases

    def test_filter_by_tags_multiple_tags(self):
        """Test filtering cases by multiple tags (AND logic)."""
        case1 = EvalCase(
            id="test_1",
            name="Test 1",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
            tags=["qa", "greeting"],
        )
        case2 = EvalCase(
            id="test_2",
            name="Test 2",
            input=EvalInput(query="What is 2+2?"),
            expected=EvalExpected(contains=["4"]),
            tags=["qa", "math"],
        )
        suite = EvalSuite(name="Test Suite", cases=[case1, case2])

        # Filter by both "qa" and "greeting"
        filtered = suite.filter_by_tags(["qa", "greeting"])
        assert len(filtered) == 1
        assert case1 in filtered

    def test_filter_by_tags_no_matches(self):
        """Test filtering with tags that match no cases."""
        case = EvalCase(
            id="test_1",
            name="Test",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
            tags=["qa"],
        )
        suite = EvalSuite(name="Test Suite", cases=[case])
        filtered = suite.filter_by_tags(["nonexistent"])
        assert len(filtered) == 0

    def test_len(self):
        """Test __len__ returns number of cases."""
        case1 = EvalCase(
            id="test_1",
            name="Test 1",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
        )
        case2 = EvalCase(
            id="test_2",
            name="Test 2",
            input=EvalInput(query="Goodbye"),
            expected=EvalExpected(contains=["Bye"]),
        )
        suite = EvalSuite(name="Test Suite", cases=[case1, case2])
        assert len(suite) == 2

    def test_iter(self):
        """Test __iter__ allows iteration over cases."""
        case1 = EvalCase(
            id="test_1",
            name="Test 1",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
        )
        case2 = EvalCase(
            id="test_2",
            name="Test 2",
            input=EvalInput(query="Goodbye"),
            expected=EvalExpected(contains=["Bye"]),
        )
        suite = EvalSuite(name="Test Suite", cases=[case1, case2])
        cases_list = list(suite)
        assert len(cases_list) == 2
        assert case1 in cases_list
        assert case2 in cases_list

    def test_getitem(self):
        """Test __getitem__ allows indexing."""
        case1 = EvalCase(
            id="test_1",
            name="Test 1",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
        )
        case2 = EvalCase(
            id="test_2",
            name="Test 2",
            input=EvalInput(query="Goodbye"),
            expected=EvalExpected(contains=["Bye"]),
        )
        suite = EvalSuite(name="Test Suite", cases=[case1, case2])
        assert suite[0] == case1
        assert suite[1] == case2

    def test_from_dict_with_minimal_fields(self):
        """Test from_dict with minimal fields."""
        data = {"name": "Test Suite"}
        suite = EvalSuite.from_dict(data)
        assert suite.name == "Test Suite"
        assert suite.description == ""
        assert suite.cases == []

    def test_from_dict_with_all_fields(self):
        """Test from_dict with all fields."""
        data = {
            "name": "Test Suite",
            "description": "A test suite",
            "cases": [
                {
                    "id": "test_1",
                    "name": "Test",
                    "input": {"query": "Hello"},
                    "expected": {"contains": ["Hi"]},
                }
            ],
            "default_assertions": [{"type": "no_errors"}],
            "metadata": {"version": "1.0"},
        }
        suite = EvalSuite.from_dict(data)
        assert suite.name == "Test Suite"
        assert suite.description == "A test suite"
        assert len(suite.cases) == 1
        assert suite.cases[0].id == "test_1"
        assert suite.default_assertions == [{"type": "no_errors"}]
        assert suite.metadata == {"version": "1.0"}

    def test_to_dict_with_minimal_fields(self):
        """Test to_dict with minimal fields."""
        suite = EvalSuite(name="Test Suite")
        result = suite.to_dict()
        assert result == {"name": "Test Suite"}

    def test_to_dict_with_all_fields(self):
        """Test to_dict with all fields."""
        case = EvalCase(
            id="test_1",
            name="Test",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
        )
        suite = EvalSuite(
            name="Test Suite",
            description="A test suite",
            cases=[case],
            default_assertions=[{"type": "no_errors"}],
            metadata={"version": "1.0"},
        )
        result = suite.to_dict()
        assert result["name"] == "Test Suite"
        assert result["description"] == "A test suite"
        assert len(result["cases"]) == 1
        assert result["default_assertions"] == [{"type": "no_errors"}]
        assert result["metadata"] == {"version": "1.0"}

    def test_to_dict_excludes_empty_fields(self):
        """Test that to_dict excludes empty description, cases, assertions, metadata."""
        suite = EvalSuite(name="Test Suite")
        result = suite.to_dict()
        assert "description" not in result
        assert "cases" not in result
        assert "default_assertions" not in result
        assert "metadata" not in result

    def test_to_dict_excludes_callables(self):
        """Test that to_dict excludes setup/teardown callables."""

        def setup_func():
            pass

        suite = EvalSuite(name="Test Suite", setup=setup_func)
        result = suite.to_dict()
        assert "setup" not in result
        assert "teardown" not in result

    def test_roundtrip_serialization(self):
        """Test that to_dict/from_dict roundtrip works."""
        case = EvalCase(
            id="test_1",
            name="Test",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
            tags=["qa"],
        )
        suite = EvalSuite(
            name="Test Suite",
            description="A test suite",
            cases=[case],
            default_assertions=[{"type": "no_errors"}],
            metadata={"version": "1.0"},
        )
        data = suite.to_dict()
        restored = EvalSuite.from_dict(data)
        assert restored.name == suite.name
        assert restored.description == suite.description
        assert len(restored.cases) == len(suite.cases)
        assert restored.cases[0].id == suite.cases[0].id
        assert restored.default_assertions == suite.default_assertions
        assert restored.metadata == suite.metadata


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
class TestEvalSuiteYAML:
    """Tests for YAML serialization (requires PyYAML)."""

    def test_to_yaml_creates_file(self):
        """Test that to_yaml creates a YAML file."""
        case = EvalCase(
            id="test_1",
            name="Test",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
        )
        suite = EvalSuite(name="Test Suite", cases=[case])

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "suite.yaml"
            suite.to_yaml(yaml_path)
            assert yaml_path.exists()

    def test_to_yaml_creates_nested_directories(self):
        """Test that to_yaml creates parent directories if they don't exist."""
        suite = EvalSuite(name="Test Suite")

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "nested" / "dir" / "suite.yaml"
            suite.to_yaml(yaml_path)
            assert yaml_path.exists()

    def test_to_yaml_content(self):
        """Test that to_yaml writes correct YAML content."""
        case = EvalCase(
            id="test_1",
            name="Basic QA test",
            input=EvalInput(query="What is the capital of France?"),
            expected=EvalExpected(contains=["Paris"]),
            tags=["qa", "geography"],
        )
        suite = EvalSuite(
            name="Geography QA Suite",
            description="Tests for geography knowledge",
            cases=[case],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "suite.yaml"
            suite.to_yaml(yaml_path)

            # Read and verify
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            assert data["name"] == "Geography QA Suite"
            assert data["description"] == "Tests for geography knowledge"
            assert len(data["cases"]) == 1
            assert data["cases"][0]["id"] == "test_1"

    def test_from_yaml_reads_file(self):
        """Test that from_yaml reads a YAML file."""
        case = EvalCase(
            id="test_1",
            name="Test",
            input=EvalInput(query="Hello"),
            expected=EvalExpected(contains=["Hi"]),
        )
        suite = EvalSuite(name="Test Suite", cases=[case])

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "suite.yaml"
            suite.to_yaml(yaml_path)

            # Load back
            loaded = EvalSuite.from_yaml(yaml_path)
            assert loaded.name == suite.name
            assert len(loaded.cases) == 1
            assert loaded.cases[0].id == "test_1"

    def test_from_yaml_file_not_found(self):
        """Test that from_yaml raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            EvalSuite.from_yaml("/nonexistent/path/suite.yaml")

    def test_yaml_roundtrip(self):
        """Test full YAML roundtrip preserves data."""
        case1 = EvalCase(
            id="test_1",
            name="Basic QA test",
            input=EvalInput(query="What is the capital of France?"),
            expected=EvalExpected(contains=["Paris"]),
            tags=["qa", "geography"],
        )
        case2 = EvalCase(
            id="test_2",
            name="Math test",
            input=EvalInput(query="What is 2+2?"),
            expected=EvalExpected(contains=["4"]),
            assertions=[
                {"type": "contains", "value": "4"},
                {"type": "semantic_similarity", "threshold": 0.8},
            ],
            tags=["qa", "math"],
        )
        suite = EvalSuite(
            name="RAG Quality Suite",
            description="Tests for RAG pipeline quality",
            cases=[case1, case2],
            default_assertions=[{"type": "latency", "max_ms": 5000}],
            metadata={"version": "1.0", "author": "test"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "suite.yaml"
            suite.to_yaml(yaml_path)
            loaded = EvalSuite.from_yaml(yaml_path)

            # Verify all fields
            assert loaded.name == suite.name
            assert loaded.description == suite.description
            assert len(loaded.cases) == 2
            assert loaded.cases[0].id == "test_1"
            assert loaded.cases[0].tags == ["qa", "geography"]
            assert loaded.cases[1].id == "test_2"
            assert loaded.cases[1].assertions == [
                {"type": "contains", "value": "4"},
                {"type": "semantic_similarity", "threshold": 0.8},
            ]
            assert loaded.default_assertions == [{"type": "latency", "max_ms": 5000}]
            assert loaded.metadata == {"version": "1.0", "author": "test"}

    def test_yaml_example_format(self):
        """Test YAML output matches the expected format from requirements."""
        case = EvalCase(
            id="test_basic_qa",
            name="Basic factual question",
            input=EvalInput(query="What is the capital of France?"),
            expected=EvalExpected(contains=["Paris"]),
            assertions=[
                {"type": "contains", "value": "Paris"},
                {"type": "semantic_similarity", "threshold": 0.8},
            ],
        )
        suite = EvalSuite(
            name="RAG Quality Suite",
            description="Tests for RAG pipeline quality",
            cases=[case],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "suite.yaml"
            suite.to_yaml(yaml_path)

            # Read raw YAML content
            with open(yaml_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Verify structure (basic checks)
            assert "name: RAG Quality Suite" in content
            assert "description: Tests for RAG pipeline quality" in content
            assert "id: test_basic_qa" in content
            assert "name: Basic factual question" in content
            assert "query: What is the capital of France?" in content
            assert "- Paris" in content


class TestEvalSuiteWithoutYAML:
    """Tests for YAML functionality when PyYAML is not available."""

    @pytest.mark.skipif(YAML_AVAILABLE, reason="PyYAML is installed")
    def test_from_yaml_raises_import_error(self):
        """Test that from_yaml raises ImportError when PyYAML not installed."""
        with pytest.raises(ImportError, match="PyYAML is required"):
            EvalSuite.from_yaml("suite.yaml")

    @pytest.mark.skipif(YAML_AVAILABLE, reason="PyYAML is installed")
    def test_to_yaml_raises_import_error(self):
        """Test that to_yaml raises ImportError when PyYAML not installed."""
        suite = EvalSuite(name="Test Suite")
        with pytest.raises(ImportError, match="PyYAML is required"):
            suite.to_yaml("suite.yaml")
