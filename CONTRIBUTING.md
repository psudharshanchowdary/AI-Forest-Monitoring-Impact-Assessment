# Contributing to Forest Monitoring

Thank you for your interest in contributing to the Forest Monitoring project! We welcome contributions from everyone.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/forest-monitoring.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Make your changes** and commit with clear messages
5. **Push** to your fork and open a **Pull Request**

## Development Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (including dev tools)
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Format code
black src scripts deployment tests

# Run linter
flake8 src scripts deployment

# Run tests
pytest -q
```

## Code Style

- Follow **PEP 8** guidelines
- Use **type hints** for functions
- Keep lines under 100 characters
- Use descriptive variable names
- Add docstrings to functions and classes

### Example:
```python
def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Compute Normalized Difference Vegetation Index.
    
    Args:
        red: Red band values (uint8 or float)
        nir: Near-infrared band values
        
    Returns:
        NDVI values normalized to [-1, 1]
    """
    ndvi = (nir.astype(float) - red.astype(float)) / (nir + red + 1e-8)
    return ndvi
```

## Commit Messages

Write clear, concise commit messages:

```
Add: Description of what was added
Fix: Description of what was fixed
Docs: Description of documentation updates
Refactor: Code restructuring without behavior change
```

Example: `Add: NDVI caching for faster computation` or `Fix: Model weights path issue on Windows`

## Testing

- Add tests for new features
- Ensure all tests pass: `pytest -q`
- Aim for >80% code coverage

```bash
# Check coverage
pytest --cov=src tests/
```

## Areas for Contribution

### 🐛 Bug Fixes
- Report issues with clear reproduction steps
- Include Python/OS version
- Attach error logs and screenshots

### 📊 New Datasets
- Add support for Sentinel-1, Landsat, or PlanetLabs imagery
- Document new data sources in `src/forest_monitor/data/`
- Include example usage

### 🤖 Model Improvements
- Try new architectures: YOLOv9, Segment Anything, DETR
- Fine-tune on forest-specific datasets
- Share results and trained weights

### 📚 Documentation
- Improve README, docstrings, or guides
- Add tutorials or use-case examples
- Translate documentation

### ☁️ Deployment
- AWS/GCP/Azure cloud integration
- Docker containerization
- Kubernetes support

### ✅ Tests
- Increase test coverage
- Add integration tests
- Performance benchmarks

## Pull Request Process

1. **Update** documentation if needed
2. **Add tests** for new functionality
3. **Verify** all tests pass: `pytest -q`
4. **Format** code: `black src scripts`
5. **Write** a clear PR description
6. **Link** related issues: `Fixes #123`

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactoring

## Changes Made
- Change 1
- Change 2

## Testing
- [ ] Tested locally
- [ ] Added unit tests
- [ ] All tests pass

## Related Issues
Fixes #123
```

## Code Review

All submissions go through code review. Reviewers may:
- Ask for clarifications
- Request changes
- Suggest improvements

This is a normal part of the process. Be open to feedback!

## Questions?

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Start a GitHub Discussion for questions
- **Email**: Contact the maintainers

## License

By contributing, you agree your code will be licensed under the MIT License.

Thank you for making this project better! 🎉
