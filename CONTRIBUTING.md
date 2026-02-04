# Contributing to Prela SDK

Thank you for your interest in contributing to Prela! We welcome contributions from the community.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- pip
- git

### Installation

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/prela-sdk.git
   cd prela-sdk
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev,all]"
   ```

   This installs:
   - Core SDK package in editable mode
   - Development tools (pytest, ruff, black, mypy)
   - All optional dependencies for integrations

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=prela --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_tracer.py
```

### Run tests matching a pattern
```bash
pytest -k "test_span"
```

## Code Style

We use several tools to maintain code quality:

### Linting
```bash
ruff check .
```

To auto-fix issues:
```bash
ruff check --fix .
```

### Formatting
```bash
black .
```

### Type Checking
```bash
mypy prela
```

### Run all checks
```bash
# Before submitting a PR, run all checks:
ruff check .
black --check .
mypy prela
pytest
```

## Making Changes

### 1. Create a branch

```bash
git checkout -b feature/my-new-feature
# or
git checkout -b fix/my-bug-fix
```

### 2. Make your changes

- Write clear, concise code
- Add tests for new functionality
- Update documentation if needed
- Follow existing code patterns

### 3. Test your changes

```bash
# Run tests
pytest

# Run linters
ruff check .
black .
mypy prela
```

### 4. Commit your changes

```bash
git add .
git commit -m "feat: add new feature"
```

**Commit message format:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

### 5. Push and create a pull request

```bash
git push origin feature/my-new-feature
```

Then go to GitHub and create a pull request.

## Pull Request Guidelines

### Before submitting

- [ ] All tests pass locally
- [ ] Code is formatted with black
- [ ] Linting passes (ruff)
- [ ] Type checking passes (mypy)
- [ ] New features have tests
- [ ] Documentation is updated

### PR Description

Include:
- Summary of changes
- Motivation/reasoning
- Related issue number (if applicable)
- Breaking changes (if any)

### Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged

## Adding New Integrations

To add support for a new AI framework:

1. **Create instrumentor file**
   ```
   prela/instrumentation/your_framework.py
   ```

2. **Implement the instrumentor class**
   ```python
   from prela.instrumentation.base import Instrumentor

   class YourFrameworkInstrumentor(Instrumentor):
       def install(self):
           # Patch/wrap framework methods
           pass

       def uninstall(self):
           # Restore original methods
           pass
   ```

3. **Register in auto.py**
   ```python
   INSTRUMENTORS = {
       ...
       "your_framework": YourFrameworkInstrumentor,
   }
   ```

4. **Add tests**
   ```
   tests/instrumentation/test_your_framework.py
   ```

5. **Update documentation**
   - Add example to `examples/`
   - Document in README

## Reporting Issues

### Bug Reports

Include:
- Prela version (`prela.__version__`)
- Python version
- Operating system
- Minimal code to reproduce
- Expected vs actual behavior
- Error messages/stack traces

### Feature Requests

Include:
- Use case/motivation
- Proposed API (if applicable)
- Examples of how it would be used

## Code of Conduct

Be respectful and inclusive. We're all here to build something great together.

## Questions?

- Open an issue for general questions
- Check existing issues/PRs first
- Join our community discussions

## License

By contributing to Prela, you agree that your contributions will be licensed under the Apache 2.0 License.

---

Thank you for contributing to Prela! 🎉
