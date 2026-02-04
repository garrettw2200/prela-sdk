# Evaluation Assertions

This module provides a comprehensive set of assertions for testing AI agent outputs and behaviors. Assertions are the building blocks of evaluation test cases, allowing you to verify that your agent produces expected results.

## Overview

Assertions evaluate agent outputs, expected values, and execution traces to determine if they meet specified criteria. Each assertion returns an `AssertionResult` with pass/fail status, score (for partial credit), and detailed information about the evaluation.

## Assertion Types

### Structural Assertions (`structural.py`)

Text and data format validation assertions:

#### 1. `ContainsAssertion`
Check if output contains specified text.

```python
from prela.evals.assertions import ContainsAssertion

# Case-sensitive search
assertion = ContainsAssertion(text="success", case_sensitive=True)
result = assertion.evaluate(output="Operation completed successfully", expected=None, trace=None)
assert result.passed  # True

# Case-insensitive search
assertion = ContainsAssertion(text="ERROR", case_sensitive=False)
result = assertion.evaluate(output="error occurred", expected=None, trace=None)
assert result.passed  # True
```

**Config format:**
```json
{
  "text": "success",
  "case_sensitive": true
}
```

#### 2. `NotContainsAssertion`
Check if output does NOT contain specified text.

```python
from prela.evals.assertions import NotContainsAssertion

assertion = NotContainsAssertion(text="error", case_sensitive=True)
result = assertion.evaluate(output="All tests passed!", expected=None, trace=None)
assert result.passed  # True
```

**Config format:**
```json
{
  "text": "error",
  "case_sensitive": true
}
```

#### 3. `RegexAssertion`
Match output against a regular expression pattern.

```python
from prela.evals.assertions import RegexAssertion
import re

# Phone number validation
assertion = RegexAssertion(pattern=r"\d{3}-\d{3}-\d{4}")
result = assertion.evaluate(output="Call me at 555-123-4567", expected=None, trace=None)
assert result.passed  # True
assert result.details["matched_text"] == "555-123-4567"

# Case-insensitive matching
assertion = RegexAssertion(pattern=r"hello", flags=re.IGNORECASE)
result = assertion.evaluate(output="HELLO WORLD", expected=None, trace=None)
assert result.passed  # True
```

**Config format:**
```json
{
  "pattern": "\\d{3}-\\d{3}-\\d{4}",
  "flags": 0
}
```

#### 4. `LengthAssertion`
Check if output length is within specified bounds.

```python
from prela.evals.assertions import LengthAssertion

# Min and max bounds
assertion = LengthAssertion(min_length=10, max_length=100)
result = assertion.evaluate(output="This is a medium length response.", expected=None, trace=None)
assert result.passed  # True
assert result.actual == 34  # Character count

# Min only
assertion = LengthAssertion(min_length=5)
result = assertion.evaluate(output="Hi", expected=None, trace=None)
assert not result.passed  # False (too short)

# Max only
assertion = LengthAssertion(max_length=50)
result = assertion.evaluate(output="Short text", expected=None, trace=None)
assert result.passed  # True
```

**Config format:**
```json
{
  "min_length": 10,
  "max_length": 100
}
```

#### 5. `JSONValidAssertion`
Validate that output is valid JSON, optionally matching a schema.

```python
from prela.evals.assertions import JSONValidAssertion

# Basic JSON validation
assertion = JSONValidAssertion()
result = assertion.evaluate(output='{"status": "success", "count": 42}', expected=None, trace=None)
assert result.passed  # True
assert result.actual == {"status": "success", "count": 42}

# JSON schema validation (requires jsonschema library)
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"}
    },
    "required": ["name"]
}
assertion = JSONValidAssertion(schema=schema)
result = assertion.evaluate(output='{"name": "Alice", "age": 30}', expected=None, trace=None)
assert result.passed  # True

result = assertion.evaluate(output='{"age": 30}', expected=None, trace=None)
assert not result.passed  # False (missing required field "name")
```

**Config format:**
```json
{
  "schema": {
    "type": "object",
    "properties": {
      "name": {"type": "string"}
    },
    "required": ["name"]
  }
}
```

### Tool Assertions (`tool.py`)

Assertions for verifying agent tool usage based on execution traces:

#### 6. `ToolCalledAssertion`
Check if a specific tool was called during execution.

```python
from prela.evals.assertions import ToolCalledAssertion

assertion = ToolCalledAssertion(tool_name="web_search")
result = assertion.evaluate(output=None, expected=None, trace=trace_spans)
assert result.passed  # True if "web_search" tool span found in trace
assert result.details["call_count"] == 2  # Number of times called
```

**Config format:**
```json
{
  "tool_name": "web_search"
}
```

#### 7. `ToolArgsAssertion`
Check if a tool was called with expected arguments.

```python
from prela.evals.assertions import ToolArgsAssertion

# Partial match (checks that expected args are present)
assertion = ToolArgsAssertion(
    tool_name="web_search",
    expected_args={"query": "Python tutorial"},
    partial_match=True
)
result = assertion.evaluate(output=None, expected=None, trace=trace_spans)
assert result.passed  # True even if tool has additional args

# Exact match (requires exact argument match)
assertion = ToolArgsAssertion(
    tool_name="calculator",
    expected_args={"x": 5, "y": 10},
    partial_match=False
)
result = assertion.evaluate(output=None, expected=None, trace=trace_spans)
assert result.passed  # True only if args exactly match
```

**Config format:**
```json
{
  "tool_name": "web_search",
  "expected_args": {"query": "Python"},
  "partial_match": true
}
```

#### 8. `ToolSequenceAssertion`
Check if tools were called in a specific order.

```python
from prela.evals.assertions import ToolSequenceAssertion

# Non-strict mode (other tools can appear between expected sequence)
assertion = ToolSequenceAssertion(
    sequence=["search", "calculate", "summarize"],
    strict=False
)
result = assertion.evaluate(output=None, expected=None, trace=trace_spans)
assert result.passed  # True if tools appear in this order

# Strict mode (no other tools allowed between expected sequence)
assertion = ToolSequenceAssertion(
    sequence=["search", "calculate"],
    strict=True
)
result = assertion.evaluate(output=None, expected=None, trace=trace_spans)
assert result.passed  # True only if exactly these tools in order
```

**Config format:**
```json
{
  "sequence": ["search", "calculate", "summarize"],
  "strict": false
}
```

### Semantic Assertions (`semantic.py`)

Embedding-based semantic similarity comparison (requires `sentence-transformers`):

#### 9. `SemanticSimilarityAssertion`
Check if output is semantically similar to expected text using embeddings.

```python
from prela.evals.assertions import SemanticSimilarityAssertion

assertion = SemanticSimilarityAssertion(
    expected_text="The weather is nice today",
    threshold=0.8,  # Minimum similarity score (0-1)
    model_name="all-MiniLM-L6-v2"  # Sentence transformer model
)

# High similarity (different wording, same meaning)
result = assertion.evaluate(output="Today has beautiful weather", expected=None, trace=None)
assert result.passed  # True
assert result.score > 0.8  # Similarity score

# Low similarity (different meaning)
result = assertion.evaluate(output="I like pizza", expected=None, trace=None)
assert not result.passed  # False
assert result.score < 0.8
```

**Installation:**
```bash
pip install sentence-transformers
```

**Config format:**
```json
{
  "expected_text": "The weather is nice today",
  "threshold": 0.8,
  "model_name": "all-MiniLM-L6-v2"
}
```

**Performance notes:**
- First use downloads the model (~80MB for all-MiniLM-L6-v2)
- Embeddings are cached in memory for repeated evaluations
- Model is shared across all instances (class-level cache)

## AssertionResult

All assertions return an `AssertionResult` object with the following fields:

```python
@dataclass
class AssertionResult:
    passed: bool              # Whether the assertion passed
    assertion_type: str       # Type of assertion (e.g., "contains", "regex")
    message: str              # Human-readable description
    score: float | None       # Optional score 0-1 for partial credit
    expected: Any             # Expected value (if applicable)
    actual: Any               # Actual value that was evaluated
    details: dict[str, Any]   # Additional evaluation details
```

### String representation

AssertionResult has a nice string format for console output:

```python
result = assertion.evaluate(...)
print(result)
# Output: ✓ PASS [contains] Output contains 'success'
# Output: ✗ FAIL [regex] Pattern not found
# Output: ✓ PASS [semantic_similarity] Semantically similar (score: 0.87)
```

## Creating Custom Assertions

To create a custom assertion, extend `BaseAssertion`:

```python
from prela.evals.assertions.base import BaseAssertion, AssertionResult

class CustomAssertion(BaseAssertion):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def evaluate(self, output, expected, trace):
        # Your evaluation logic here
        passed = # ... check condition

        return AssertionResult(
            passed=passed,
            assertion_type="custom",
            message=f"Custom check: {passed}",
            expected=self.param1,
            actual=output,
            details={"param2": self.param2}
        )

    @classmethod
    def from_config(cls, config):
        return cls(
            param1=config["param1"],
            param2=config.get("param2", "default")
        )
```

## Config-Based Loading

All assertions support loading from configuration dictionaries:

```python
from prela.evals.assertions import ContainsAssertion

config = {
    "text": "success",
    "case_sensitive": False
}
assertion = ContainsAssertion.from_config(config)
```

This enables declarative test definitions in YAML/JSON files:

```yaml
# eval_suite.yaml
cases:
  - name: "Test successful response"
    input:
      query: "What is 2+2?"
    assertions:
      - type: contains
        config:
          text: "4"
      - type: length
        config:
          min_length: 1
          max_length: 100
      - type: tool_called
        config:
          tool_name: calculator
```

## Integration with EvalCase

Assertions are used within `EvalCase` objects:

```python
from prela.evals import EvalCase
from prela.evals.assertions import ContainsAssertion, ToolCalledAssertion

case = EvalCase(
    name="Test calculator agent",
    input={"query": "What is 15 * 23?"},
    assertions=[
        ContainsAssertion(text="345"),
        ToolCalledAssertion(tool_name="calculator")
    ]
)

# Run the case
result = case.run(agent_function=my_agent)
assert result.passed
```

## Best Practices

1. **Combine Multiple Assertions**: Use multiple assertions to verify different aspects of agent behavior

```python
assertions = [
    ContainsAssertion(text="success"),        # Check output content
    LengthAssertion(min_length=10),           # Check output length
    ToolCalledAssertion(tool_name="search"),  # Check tool usage
]
```

2. **Use Appropriate Assertion Types**:
   - Structural assertions for format validation
   - Tool assertions for agent behavior verification
   - Semantic assertions for meaning-based comparison

3. **Set Reasonable Thresholds**:
   - Semantic similarity: 0.7-0.8 for similar meaning, 0.9+ for near-identical
   - Length bounds: Consider typical output ranges

4. **Handle Optional Dependencies**:
```python
try:
    from prela.evals.assertions import SemanticSimilarityAssertion
    use_semantic = True
except ImportError:
    use_semantic = False
```

5. **Cache Semantic Embeddings**: The semantic assertion automatically caches embeddings. For long-running tests, clear cache periodically:

```python
from prela.evals.assertions import SemanticSimilarityAssertion

# After processing many cases
SemanticSimilarityAssertion.clear_cache()
```

## Performance Considerations

- **Structural assertions**: Microsecond-level performance, negligible overhead
- **Tool assertions**: Fast trace scanning, O(n) where n = number of spans
- **Semantic assertions**: First use downloads model, subsequent calls cached
  - Model loading: ~1-2 seconds
  - Embedding computation: ~10-50ms per text
  - Cached embeddings: ~1µs lookup

## Testing

Comprehensive tests are available in `tests/test_evals/test_assertions.py`:

```bash
# Run all assertion tests
pytest tests/test_evals/test_assertions.py -v

# Run specific assertion type
pytest tests/test_evals/test_assertions.py::TestContainsAssertion -v

# Skip semantic tests (if sentence-transformers not installed)
pytest tests/test_evals/test_assertions.py -v --ignore-glob="*semantic*"
```

## Examples

See `examples/assertions_demo.py` for a comprehensive demonstration of all assertion types.

## References

- Base classes: `prela.evals.assertions.base`
- Structural: `prela.evals.assertions.structural`
- Tool: `prela.evals.assertions.tool`
- Semantic: `prela.evals.assertions.semantic`
- Tests: `tests/test_evals/test_assertions.py`
