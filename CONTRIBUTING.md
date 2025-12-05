# Contributing to NEMUS

Thank you for your interest in contributing to NEMUS! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/ycharankrishna/NEMUS/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Code snippet (if applicable)

### Suggesting Features

1. Open a new issue with tag `enhancement`
2. Describe the feature and its use case
3. Provide examples of how it would be used

### Code Contributions

#### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/ycharankrishna/NEMUS.git
cd nemus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev,viz,docs]"
```

#### Code Style

We use:
- **Black** for code formatting
- **flake8** for linting
- **Type hints** for function signatures

```bash
# Format code
black nemus/

# Check linting
flake8 nemus/

# Run type checking
mypy nemus/
```

#### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nemus --cov-report=html

# Run specific test
pytest tests/test_core.py::test_chronos_engine
```

#### Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes** following code style
4. **Add tests** for new functionality
5. **Run tests** to ensure nothing breaks
6. **Commit** with clear messages:
   ```bash
   git commit -m "Add amazing feature"
   ```
7. **Push** to your fork:
   ```bash
   git push origin feature/amazing-feature
   ```
8. **Open a Pull Request** on GitHub

#### Commit Message Guidelines

```
type(scope): brief description

Detailed description of changes (if needed)

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting)
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

Examples:
```
feat(plasticity): add BCM learning rule
fix(core): handle edge case in AETS prediction
docs(readme): update installation instructions
```

## Architecture Guidelines

### The Four Pillars

When adding features, respect the Four Pillars architecture:

1. **Chronos Kernel**: Event-driven, analytical precision
2. **Bio-Trace Plasticity**: Local, O(1) memory learning
3. **Morphogenesis Controller**: Dynamic topology
4. **Transmorphic Compiler**: Hardware-aware compilation

### Code Organization

```
nemus/
├── core.py          # Event engine (Pillar 1)
├── biology.py       # Neuron models
├── plasticity.py    # Learning rules (Pillar 2)
├── morphology.py    # Structural plasticity (Pillar 3)
├── silicon.py       # Hardware compilation (Pillar 4)
├── layers.py        # Network layers
├── network.py       # Network container
└── ...
```

### Adding a New Neuron Model

```python
# In nemus/biology.py

class MyNeuron:
    """
    Brief description.
    
    Math:
    -----
    Equation here
    
    Parameters:
    -----------
    param1 : type
        Description
    """
    
    def __init__(self, neurons, param1):
        self.neurons = neurons
        self.param1 = param1
        # Initialize state
        self.v = np.zeros(neurons)
    
    def predict_spike(self, idx: int, t_now: float) -> float:
        """
        Calculate exact spike time for neuron idx.
        
        Returns:
        --------
        t_spike : float
            Time of next spike, or float('inf') if no spike
        """
        # Implement analytical inversion
        # ...
        return t_spike
    
    def update(self, idx: int, t: float, I: float):
        """Update neuron state."""
        # ...
```

### Adding a Learning Rule

```python
# In nemus/plasticity.py

class MyLearningRule:
    """
    Brief description of the rule.
    
    Math:
    -----
    Weight update equation
    """
    
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def apply(self, weights, ...):
        """
        Apply learning rule.
        
        Parameters:
        -----------
        weights : ndarray
            Weight matrix to update
        """
        # Implement update
        weights += self.learning_rate * ...
```

## Documentation

### Docstring Style

Use NumPy style docstrings:

```python
def function_name(param1, param2):
    """
    Brief description.
    
    Longer description if needed.
    
    Parameters
    ----------
    param1 : type
        Description
    param2 : type
        Description
    
    Returns
    -------
    return_val : type
        Description
    
    Examples
    --------
    >>> result = function_name(1, 2)
    >>> print(result)
    3
    """
```

### Building Documentation

```bash
cd docs/
make html
# Open _build/html/index.html
```

## Questions?

- Open a [Discussion](https://github.com/ycharankrishna/NEMUS/discussions)
- Join our community chat
- Email: yellapragadacharankrishna1234@gmail.com

Thank you for contributing to NEMUS! 🚀
