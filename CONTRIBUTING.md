# Contributing to Neural Operator Cryptanalysis Lab

Thank you for your interest in contributing to the Neural Operator Cryptanalysis Lab! This project aims to advance defensive security research through neural operator-based side-channel analysis.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Security Considerations](#security-considerations)

## Code of Conduct

### Our Pledge

We are committed to making participation in this project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors include:**
- Use of sexualized language or imagery
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate

### Responsible Use

All contributors must agree to use this tool for **defensive security research only**. Any malicious use or contribution that could enable attacks on unauthorized systems is strictly prohibited.

## Getting Started

### Prerequisites

- Python 3.9+ 
- Git
- Basic understanding of cryptography and side-channel analysis
- Familiarity with PyTorch and neural networks

### First-Time Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/Neural-Operator-Cryptanalysis-Lab.git
   cd Neural-Operator-Cryptanalysis-Lab
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev,research]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/ -v
   ```

## Development Environment

### Recommended IDE Setup

- **VS Code** with extensions:
  - Python
  - Pylance
  - Black Formatter
  - GitLens
  - Docker

### Environment Configuration

Create a `.env` file for local development:
```bash
NEURAL_CRYPTO_LOG_LEVEL=DEBUG
NEURAL_CRYPTO_DATA_DIR=./data
NEURAL_CRYPTO_REQUIRE_AUTHORIZATION=false  # For development only
```

### Docker Development

Use Docker for consistent development environment:
```bash
# Development environment
docker-compose --profile dev up neural-crypto-dev

# Research environment with Jupyter
docker-compose --profile research up neural-crypto-research
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

#### 🐛 Bug Reports
- Use the bug report template
- Include minimal reproduction steps
- Provide system information and logs
- Check if the issue already exists

#### ✨ Feature Requests
- Use the feature request template
- Explain the use case and motivation
- Consider security implications
- Discuss implementation approach

#### 🔧 Code Contributions
- Neural operator architectures
- New cryptographic targets
- Performance optimizations
- Testing improvements
- Documentation updates

#### 📚 Documentation
- API documentation
- Tutorials and examples
- Best practices guides
- Security guidelines

### Branch Strategy

We use a modified Git Flow:

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features and enhancements
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical fixes for production

### Commit Message Format

Follow conventional commits:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(neural-operators): add Multipole Graph Neural Operator

Implements MGNO architecture for graph-based side-channel analysis
with support for spatial EM field modeling.

Closes #123
```

## Pull Request Process

### Before Submitting

1. **Create an issue** first to discuss large changes
2. **Update documentation** for any API changes
3. **Add tests** for new functionality
4. **Ensure all tests pass** locally
5. **Run security checks** with bandit and safety
6. **Format code** with black and lint with flake8

### PR Checklist

- [ ] Branch is up-to-date with target branch
- [ ] All tests pass (`pytest tests/`)
- [ ] Code coverage is maintained or improved
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (for significant changes)
- [ ] Security implications are considered
- [ ] Responsible use guidelines are followed

### PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Security Checklist
- [ ] No hardcoded secrets or keys
- [ ] Input validation implemented
- [ ] Authorization checks in place
- [ ] Audit logging updated
- [ ] Follows responsible disclosure practices

## Additional Notes
Any additional information, concerns, or questions.
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: At least one maintainer review required
3. **Security Review**: Required for security-sensitive changes
4. **Testing**: Comprehensive testing in CI environment
5. **Documentation Review**: For changes affecting user-facing features

## Testing Requirements

### Test Structure

```
tests/
├── unit/                   # Unit tests
├── integration/           # Integration tests
├── fixtures/             # Test data and fixtures
└── conftest.py          # Pytest configuration
```

### Writing Tests

```python
import pytest
import torch
import numpy as np
from neural_cryptanalysis.neural_operators import FourierNeuralOperator

class TestFourierNeuralOperator:
    """Test Fourier Neural Operator implementation."""
    
    @pytest.fixture
    def fno_model(self):
        """Create FNO model for testing."""
        from neural_cryptanalysis.neural_operators.base import OperatorConfig
        config = OperatorConfig(input_dim=1, output_dim=256, hidden_dim=32)
        return FourierNeuralOperator(config, modes=8)
    
    def test_forward_pass(self, fno_model):
        """Test forward pass functionality."""
        x = torch.randn(4, 100, 1)
        output = fno_model(x)
        
        assert output.shape == (4, 256)
        assert not torch.isnan(output).any()
    
    def test_parameter_initialization(self, fno_model):
        """Test parameter initialization."""
        for param in fno_model.parameters():
            assert not torch.isnan(param).any()
            assert param.requires_grad
```

### Test Coverage

- Minimum 85% code coverage required
- Critical paths must have 100% coverage
- Include edge cases and error conditions
- Test both CPU and GPU execution (when available)

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/neural_cryptanalysis --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run performance benchmarks
pytest tests/benchmarks/ --benchmark-only
```

## Documentation Standards

### Code Documentation

Use Google-style docstrings:

```python
def neural_attack(traces: np.ndarray, 
                 model: torch.nn.Module,
                 strategy: str = 'direct') -> Dict[str, Any]:
    """Perform neural operator-based side-channel attack.
    
    Args:
        traces: Side-channel traces for attack
        model: Trained neural operator model
        strategy: Attack strategy ('direct', 'template', 'adaptive')
        
    Returns:
        Attack results containing success rates and recovered keys
        
    Raises:
        ValueError: If traces are invalid or model is not trained
        SecurityError: If operation is not authorized
        
    Example:
        >>> traces = load_traces('target_traces.npy')
        >>> model = load_model('trained_model.pth')
        >>> results = neural_attack(traces, model, strategy='adaptive')
        >>> print(f"Success rate: {results['success_rate']:.2%}")
    """
```

### API Documentation

- Use Sphinx with Napoleon extension
- Include type hints for all public APIs
- Provide comprehensive examples
- Document security considerations

### Tutorials and Guides

- Step-by-step tutorials for common use cases
- Best practices guides
- Security guidelines
- Performance optimization tips

## Security Considerations

### Security Review Process

All contributions undergo security review:

1. **Automated Security Scanning**: Bandit, safety, and other tools  
2. **Manual Security Review**: For security-sensitive changes
3. **Threat Modeling**: For new features
4. **Penetration Testing**: For significant changes

### Security Requirements

- **No Hardcoded Secrets**: Use environment variables or secure vaults
- **Input Validation**: Validate all user inputs
- **Authorization Checks**: Implement proper access controls
- **Audit Logging**: Log security-relevant events
- **Error Handling**: Don't leak sensitive information in errors

### Responsible Disclosure

Contributors must follow responsible disclosure:

- Report vulnerabilities privately first
- Allow 90-day embargo period
- Coordinate with maintainers on disclosure
- Provide mitigation recommendations

## Development Workflow

### Setting Up a Feature Branch

```bash
# Update your fork
git checkout develop
git pull upstream develop

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: your feature description"

# Push to your fork
git push origin feature/your-feature-name
```

### Code Quality Checks

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/neural_cryptanalysis/

# Security checks
bandit -r src/neural_cryptanalysis/
safety check
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
  
  - repo: https://github.com/pycqa/bandit
    rev: 1.7.4
    hooks:
      - id: bandit
        args: ['-r', 'src/']
```

## Getting Help

### Community Resources

- **GitHub Discussions**: For general questions and discussions
- **Issues**: For bug reports and feature requests
- **Security**: `security@terragonlabs.com` for security-related matters

### Maintainer Contact

- **Lead Maintainer**: Terragon Labs Research Team
- **Security Team**: Terragon Labs Security Team
- **Documentation**: Terragon Labs Documentation Team

### Office Hours

We hold virtual office hours every two weeks:
- **When**: Every other Friday, 2-3 PM UTC
- **Where**: Virtual meeting (link in GitHub Discussions)
- **What**: Q&A, contribution help, roadmap discussions

## Recognition

Contributors are recognized in several ways:

- **Contributors File**: Listed in CONTRIBUTORS.md
- **Release Notes**: Significant contributions mentioned in releases
- **Blog Posts**: Featured contributions highlighted in blog posts
- **Conference Talks**: Opportunity to present work at conferences

## License

By contributing to this project, you agree that your contributions will be licensed under the same GPL-3.0 license that covers the project.

---

Thank you for contributing to the Neural Operator Cryptanalysis Lab! Together, we can advance the state of defensive cryptographic security research.