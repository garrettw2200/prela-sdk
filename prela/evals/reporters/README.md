# Evaluation Reporters

Three production-ready reporters for outputting evaluation results in different formats.

## Quick Start

```python
from prela.evals import EvalRunner
from prela.evals.reporters import ConsoleReporter, JSONReporter, JUnitReporter

# Run your evaluation
runner = EvalRunner(suite, agent)
result = runner.run()

# Report to terminal
console = ConsoleReporter(verbose=True)
console.report(result)

# Save to JSON
json_reporter = JSONReporter("results.json")
json_reporter.report(result)

# Generate JUnit XML for CI
junit = JUnitReporter("junit.xml")
junit.report(result)
```

## ConsoleReporter

Beautiful terminal output with colors and tables.

**Parameters:**
- `verbose` (bool): Show detailed failure information (default: True)
- `use_colors` (bool): Use colored output via rich library (default: True)

**Example:**
```python
reporter = ConsoleReporter(verbose=True, use_colors=True)
reporter.report(result)
```

**Output:**
```
╭──────────────────────────── ✓ Test Suite ────────────────────────────╮
│ Total: 10 | Passed: 9 (90.0%) | Failed: 1                            │
│ Duration: 2.50s                                                      │
╰──────────────────────────────────────────────────────────────────────╯
```

**Use Cases:**
- Development and debugging
- Quick visual feedback
- Local testing

## JSONReporter

Structured JSON output for programmatic access.

**Parameters:**
- `output_path` (str | Path): Path to output JSON file
- `indent` (int): JSON indentation (default: 2, use None for compact)

**Example:**
```python
reporter = JSONReporter("eval_results/run_001.json", indent=2)
reporter.report(result)
```

**Output:**
```json
{
  "suite_name": "Test Suite",
  "started_at": "2026-01-27T14:30:00+00:00",
  "summary": {
    "total_cases": 10,
    "passed_cases": 9,
    "pass_rate": 0.9
  },
  "case_results": [...]
}
```

**Use Cases:**
- Data analysis
- Historical tracking
- Programmatic processing
- Integration with analytics tools

## JUnitReporter

JUnit XML format for CI/CD integration.

**Parameters:**
- `output_path` (str | Path): Path to output XML file

**Example:**
```python
reporter = JUnitReporter("test-results/junit.xml")
reporter.report(result)
```

**Output:**
```xml
<?xml version='1.0' encoding='utf-8'?>
<testsuite name="Test Suite" tests="10" failures="1" ...>
  <testcase name="Test 1" classname="Test Suite" time="0.145">
    ...
  </testcase>
</testsuite>
```

**Use Cases:**
- CI/CD integration (Jenkins, GitHub Actions, GitLab)
- Test result visualization
- Automated failure notifications
- Test trend tracking

## Using Multiple Reporters

Report to multiple outputs simultaneously:

```python
# Run evaluation once
result = runner.run()

# Report to multiple outputs
reporters = [
    ConsoleReporter(verbose=False),           # Terminal output
    JSONReporter("results/eval.json"),         # Data export
    JUnitReporter("results/junit.xml"),        # CI integration
]

for reporter in reporters:
    reporter.report(result)
```

## CI/CD Integration Examples

### GitHub Actions

```yaml
- name: Run evaluations
  run: |
    python run_evals.py

- name: Publish test results
  uses: EnricoMi/publish-unit-test-result-action@v2
  if: always()
  with:
    files: test-results/junit.xml
```

### GitLab CI

```yaml
test:
  script:
    - python run_evals.py
  artifacts:
    when: always
    reports:
      junit: test-results/junit.xml
```

### Jenkins

```groovy
stage('Test') {
  steps {
    sh 'python run_evals.py'
  }
  post {
    always {
      junit 'test-results/junit.xml'
    }
  }
}
```

## See Also

- Full demo: `examples/reporters_demo.py`
- Tests: `tests/test_evals/test_reporters.py`
- Documentation: `/REPORTERS_IMPLEMENTATION_SUMMARY.md`
