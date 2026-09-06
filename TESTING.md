# Testing Guide

## Overview
This document provides comprehensive instructions for testing the Stable Diffusion Web UI project, including unit tests, integration tests, and performance testing.

## Testing Philosophy

### Goals
- Ensure code quality and reliability
- Maintain backward compatibility
- Improve test coverage over time
- Enable continuous integration
- Provide fast feedback on changes

### Test Types
1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test interactions between components
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test performance characteristics
5. **Load Tests**: Test system under load

## Test Structure

### Directory Structure
```
test/
├── conftest.py              # Pytest configuration and fixtures
├── test_extras.py           # Extras tab tests
├── test_face_restorers.py   # Face restoration tests
├── test_img2img.py          # img2img functionality tests
├── test_txt2img.py          # txt2img functionality tests
├── test_utils.py            # Utility function tests
├── test_files/              # Test data files
├── test_outputs/            # Test output files
└── __init__.py             # Package initialization
```

### Test Files

#### conftest.py
Pytest configuration and shared fixtures for all tests.

#### test_extras.py
Tests for the extras tab functionality (upscaling, image info, interrogation).

#### test_face_restorers.py
Tests for face restoration models (GFPGAN, CodeFormer).

#### test_img2img.py
Tests for img2img functionality including inpainting.

#### test_txt2img.py
Tests for txt2img functionality including various prompt types.

#### test_utils.py
Tests for utility functions and helper modules.

## Running Tests

### Prerequisites
- Python 3.9+
- pytest
- requests
- Pillow
- numpy

### Installation
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Or install all dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### Running Tests

#### Basic Test Execution
```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with detailed output
pytest -v --tb=long
```

#### Test Categories
```bash
# Run unit tests only
pytest test/ -k "not integration"

# Run integration tests only
pytest test/ -k "integration"

# Run specific test module
pytest test/test_txt2img.py
pytest test/test_img2img.py
pytest test/test_extras.py

# Run tests with coverage
pytest --cov=modules --cov-report=html

# Run tests with coverage and generate report
pytest --cov=modules --cov-report=term-missing
```

#### Test Fixtures
```bash
# Run tests with specific fixtures
pytest test/ -v --fixtures

# Run tests with custom markers
pytest test/ -m "slow"  # Run slow tests only
pytest test/ -m "not slow"  # Run all tests except slow ones
```

## Test Development

### Writing New Tests

#### Test Structure
```python
import pytest
from modules import shared
from modules.processing import process_images


def test_function_name():
    """Test description.
    
    Args:
        None
        
    Returns:
        None
        
    Raises:
        AssertionError: If the test fails
    """
    # Test setup
    setup_code()
    
    # Test execution
    result = function_to_test()
    
    # Assertions
    assert expected_condition
    assert result == expected_result
```

#### Test Fixtures
```python
@pytest.fixture
def sample_image():
    """Fixture that provides a sample image for testing."""
    # Create or load a sample image
    return sample_image_data


@pytest.fixture
def api_client(base_url):
    """Fixture that provides an API client for testing."""
    return APIClient(base_url)
```

### Test Best Practices

#### 1. Isolation
Each test should be independent and not rely on other tests' state.

#### 2. Determinism
Tests should produce the same results when run multiple times.

#### 3. Speed
Tests should run quickly. Use fixtures with appropriate scope.

#### 4. Coverage
Aim for high test coverage, especially for critical functions.

#### 5. Error Handling
Test error conditions and edge cases.

### Test Categories

#### Unit Tests
Test individual functions and classes in isolation.

Example:
```python
def test_option_setter():
    """Test that options can be set correctly."""
    from modules.options import Options
    
    # Create options instance
    options = Options({}, set())
    
    # Test setting a value
    result = options.set("test_option", "test_value")
    assert result is True
    assert options.test_option == "test_value"
```

#### Integration Tests
Test interactions between components.

Example:
```python
def test_api_options_endpoint(base_url):
    """Test that the API options endpoint works."""
    import requests
    
    url = f"{base_url}/sdapi/v1/options"
    response = requests.get(url)
    
    assert response.status_code == 200
    data = response.json()
    assert "send_seed" in data
```

#### Performance Tests
Test performance characteristics.

Example:
```python
import time


def test_txt2img_performance(url_txt2img, simple_txt2img_request):
    """Test that txt2img generation completes within reasonable time."""
    start_time = time.time()
    
    response = requests.post(url_txt2img, json=simple_txt2img_request)
    
    end_time = time.time()
    duration = end_time - start_time
    
    assert response.status_code == 200
    assert duration < 30.0  # Should complete within 30 seconds
```

## Test Data

### Test Files
Test files are stored in `test/test_files/`:

- `img2img_basic.png` - Basic image for img2img tests
- `mask_basic.png` - Basic mask for inpainting tests
- `two-faces.jpg` - Image with two faces for face restoration tests

### Test Outputs
Test outputs are stored in `test/test_outputs/`:

- `gfpgan.png` - Output from GFPGAN face restoration
- `codeformer.png` - Output from CodeFormer face restoration

## Test Configuration

### Pytest Configuration
```python
# test/conftest.py
import base64
import os

import pytest


test_files_path = os.path.dirname(__file__) + "/test_files"
test_outputs_path = os.path.dirname(__file__) + "/test_outputs"


def pytest_configure(config):
    # We don't want to fail on Py.test command line arguments being
    # parsed by webui:
    os.environ.setdefault("IGNORE_CMD_ARGS_ERRORS", "1")


def file_to_base64(filename):
    with open(filename, "rb") as file:
        data = file.read()

    base64_str = str(base64.b64encode(data), "utf-8")
    return "data:image/png;base64," + base64_str


@pytest.fixture(scope="session")  # session so we don't read this over and over
def img2img_basic_image_base64() -> str:
    return file_to_base64(os.path.join(test_files_path, "img2img_basic.png"))


@pytest.fixture(scope="session")  # session so we don't read this over and over
def mask_basic_image_base64() -> str:
    return file_to_base64(os.path.join(test_files_path, "mask_basic.png"))


@pytest.fixture(scope="session")
def initialize() -> None:
    import webui  # noqa: F401
```

### Test Markers
```python
# Mark slow tests
@pytest.mark.slow
def test_slow_operation():
    pass

# Mark integration tests
@pytest.mark.integration
def test_integration():
    pass

# Mark unit tests
@pytest.mark.unit
def test_unit():
    pass
```

## Continuous Integration

### GitHub Actions
The project uses GitHub Actions for continuous integration. The workflow is defined in `.github/workflows/`:

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    - name: Run tests
      run: |
        pytest
        pytest --cov=modules --cov-report=term-missing
```

### Local CI Setup
```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit checks
pre-commit run --all-files

# Or run specific checks
pre-commit run black --all-files
pre-commit run ruff --all-files
pre-commit run eslint --all-files
```

## Test Coverage

### Coverage Reports
Test coverage is generated using pytest-cov:

```bash
# Generate HTML coverage report
pytest --cov=modules --cov-report=html

# Generate terminal coverage report
pytest --cov=modules --cov-report=term-missing

# Generate JSON coverage report
pytest --cov=modules --cov-report=json
```

### Coverage Thresholds
- **Unit Tests**: 80% minimum coverage
- **Integration Tests**: 70% minimum coverage
- **Critical Functions**: 90% minimum coverage

## Test Troubleshooting

### Common Issues

#### "ModuleNotFoundError: No module named X"
```bash
# Install missing module
pip install X

# Or install all test dependencies
pip install -r requirements-test.txt
```

#### "ImportError: cannot import name 'X' from 'modules.Y'"
```bash
# Check if the module exists
ls modules/

# Check if the import is correct
grep -r "class X" modules/
```

#### "Test fails due to missing test data"
```bash
# Check if test files exist
ls test/test_files/

# If missing, create them or download from source
```

### Debugging Tests

#### Running Tests with Debug Output
```bash
# Run a single test with debug output
pytest test/test_txt2img.py::test_txt2img_simple_performed -v -s

# Run tests with pdb debugging
pytest test/ --pdb
```

#### Profiling Tests
```bash
# Profile test execution
pytest --profile --profile-sort cumulative

# Use cProfile for specific tests
python -m pytest test/test_txt2img.py -p cProfile
```

## Test Maintenance

### Updating Tests
When making changes to the codebase:

1. Update existing tests to reflect changes
2. Add new tests for new functionality
3. Remove or update tests that are no longer relevant
4. Ensure tests still pass with the updated code

### Test Removal
Remove tests that:
- Are no longer relevant
- Are too slow to run in CI
- Are redundant with other tests
- Are failing due to external dependencies

## Test Quality Metrics

### Metrics to Track
- **Test Count**: Number of test files and test functions
- **Test Coverage**: Percentage of code covered by tests
- **Test Execution Time**: Average time to run all tests
- **Test Failure Rate**: Percentage of tests that fail
- **Test Flakiness**: Percentage of tests that fail intermittently

### Monitoring
```bash
# Monitor test execution
pytest --tb=short --quiet

# Generate test report
pytest --json-report --json-report-file=test-report.json
```

## Conclusion

Testing is essential for maintaining code quality and ensuring the reliability of the Stable Diffusion Web UI. By following these guidelines, we can ensure that our tests are effective, maintainable, and provide valuable feedback on the codebase.

The testing infrastructure should be continuously improved to keep pace with the evolving codebase and to ensure that we can confidently make changes to the project.