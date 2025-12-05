# Installation Guide

## Quick Install (PyPI)

Once published to PyPI:

```bash
pip install nemus
```

## Development Install

### From Source

1. **Clone the repository:**
```bash
git clone https://github.com/ycharankrishna/NEMUS.git
cd nemus
```

2. **Install in development mode:**
```bash
pip install -e .
```

3. **Install with optional dependencies:**
```bash
# For visualization
pip install -e ".[viz]"

# For development (testing, linting)
pip install -e ".[dev]"

# For documentation
pip install -e ".[docs]"

# All optional dependencies
pip install -e ".[dev,viz,docs]"
```

### Using requirements.txt

```bash
pip install -r requirements.txt
```

## Verify Installation

```python
import nemus
print(nemus.__version__)

# Run a quick test
sensor = nemus.Input(shape=(2,))
net = nemus.Network(sensor)
engine = nemus.ChronosEngine(net)
print("NEMUS installed successfully!")
```

## Run Tests

```bash
# Run Phoenix Protocol
python examples/test_phoenix_protocol.py

# Expected output:
# Stage 1 (Genesis): PASS
# Stage 2 (Learning): PASS - 92%
# Stage 3 (Catastrophe): PASS
# Stage 4 (Regeneration): PASS - 87% recovery
# Stage 5 (Incarnation): PASS - 185 bytes
```

## Troubleshooting

### NumPy Installation Issues

If NumPy fails to install:

```bash
pip install --upgrade pip setuptools wheel
pip install numpy
pip install nemus
```

### scikit-learn Installation Issues

```bash
# On Windows
pip install --only-binary=:all: scikit-learn

# On Linux/Mac
pip install scikit-learn
```

### Import Errors

Make sure you're in the correct Python environment:

```bash
python -c "import sys; print(sys.executable)"
pip list | grep nemus
```

## Platform-Specific Notes

### Windows

NEMUS works on Windows 10/11 with Python 3.8+. No special configuration needed.

### Linux

Tested on Ubuntu 20.04+, CentOS 7+, Debian 10+.

```bash
# Install system dependencies (if needed)
sudo apt-get update
sudo apt-get install python3-dev build-essential
```

### macOS

Tested on macOS 11+.

```bash
# Using Homebrew
brew install python@3.9
pip3 install nemus
```

## Hardware Support

### Intel Loihi 2

To deploy to Intel Loihi 2:

1. Install Intel NxSDK (requires Intel account)
2. NEMUS will automatically detect NxSDK
3. Use `nemus.Compiler(target=nemus.Profile.LOIHI_2)`

```python
compiler = nemus.Compiler(target=nemus.Profile.LOIHI_2)
binary = compiler.compile(network)
```

## Uninstall

```bash
pip uninstall nemus
```

## Getting Help

- **Documentation**: https://nemus.readthedocs.io
- **GitHub Issues**: https://github.com/ycharankrishna/NEMUS/issues
- **Examples**: See `examples/` directory
