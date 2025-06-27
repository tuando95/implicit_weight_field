# Contributing to Implicit Weight Field Compression

Thank you for your interest in contributing to the Implicit Weight Field compression project! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/implicit_weight_field.git
   cd implicit_weight_field
   ```
3. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. Set up the development environment:
   ```bash
   conda env create -f environment.yml
   conda activate implicit-weight-field
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

We follow these coding standards:

- **Python**: PEP 8 with a line length of 100 characters
- **Formatting**: Use `black` for automatic formatting
- **Imports**: Sort with `isort`
- **Type hints**: Use type annotations where appropriate
- **Docstrings**: Google-style docstrings for all public functions and classes

Run formatting and linting:
```bash
# Format code
black .
isort .

# Check linting
flake8 .
mypy .
```

## Testing

All contributions must include appropriate tests:

1. Write unit tests for new functionality
2. Ensure all tests pass:
   ```bash
   pytest tests/
   ```
3. Check test coverage:
   ```bash
   pytest tests/ --cov=core --cov=compression --cov-report=html
   ```
4. Aim for >80% coverage for new code

## Documentation

- Update docstrings for any modified functions
- Update README.md if adding new features
- Add examples in `examples/` for significant features
- Update relevant documentation in `docs/`

## Contribution Process

1. **Open an issue** first to discuss major changes
2. **Make your changes** in a feature branch
3. **Write/update tests** for your changes
4. **Run the test suite** to ensure everything passes
5. **Update documentation** as needed
6. **Submit a pull request** with a clear description

## Pull Request Guidelines

Your PR should:

- Have a clear, descriptive title
- Reference any related issues
- Include a summary of changes
- Pass all CI checks
- Have no merge conflicts

Example PR description:
```markdown
## Summary
Brief description of what this PR does.

## Changes
- Added feature X
- Fixed bug Y
- Improved performance of Z

## Testing
- Added unit tests for X
- Verified on dataset Y

Fixes #123
```

## Code Review Process

1. A maintainer will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

## Types of Contributions

### Bug Fixes
- Ensure the bug is reproducible
- Write a test that fails without your fix
- Make the minimal change to fix the issue

### New Features
- Discuss the feature in an issue first
- Keep the implementation focused
- Add comprehensive tests and documentation

### Performance Improvements
- Include benchmarks showing the improvement
- Ensure no regression in functionality
- Document any trade-offs

### Documentation
- Fix typos and clarify existing docs
- Add examples and tutorials
- Improve API documentation

## Areas Needing Contribution

- **Baseline methods**: Additional compression baselines
- **Model support**: Support for more architectures (transformers, RNNs)
- **Optimizations**: CUDA kernels for faster weight reconstruction
- **Visualizations**: Interactive compression analysis tools
- **Documentation**: Tutorials and guides

## Development Tips

### Running Experiments
```bash
# Quick test on CIFAR-10
python scripts/train_compression.py \
    --model resnet18 \
    --dataset cifar10 \
    --max-steps 100
```

### Debugging
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use breakpoints
import pdb; pdb.set_trace()
```

### Profiling
```python
# Profile compression
python -m cProfile -o profile.stats scripts/train_compression.py

# Visualize
snakeviz profile.stats
```

## Questions?

- Open an issue for bugs or feature requests
- Join discussions in existing issues
- Contact maintainers for guidance

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers this project.

Thank you for contributing to Implicit Weight Field compression!