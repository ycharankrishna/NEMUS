"""
NEMUS - NeuroMorphic Unified System

A revolutionary event-driven neuromorphic computing library with analytical precision.

Key Features:
------------
- Analytical Event-Time Solver (AETS): Zero time-stepping with infinite temporal resolution
- Dopaminergic Eligibility Propagation (DEP): Biologically-inspired local learning
- Dynamic Sparse Rewiring (DSR): Evolving network topology during training
- Transmorphic Compiler: Compiles to neuromorphic hardware (Intel Loihi 2)

Quick Start:
-----------
>>> import nemus
>>> 
>>> # Create a simple network
>>> sensor = nemus.Input(shape=(2,), encoding="latency")
>>> hidden = nemus.Recurrent(neurons=50, model=nemus.LIF(neurons=50))
>>> output = nemus.Output(classes=1)
>>> net = nemus.Network(sensor >> hidden >> output)
>>> 
>>> # Run event-driven simulation
>>> engine = nemus.ChronosEngine(net)
>>> engine.schedule(nemus.Event(0.01, 0, 0, 0.5, hidden.name))
>>> engine.run(duration=0.1)
>>> print(f"Output spikes: {output.spikes}")

The Four Pillars:
----------------
1. Chronos Kernel (AETS): Inverts LIF equation to calculate exact spike times
2. Bio-Trace Plasticity (DEP): Eligibility traces + three-factor learning
3. Morphogenesis Controller (DSR): Pruning + synaptogenesis
4. Transmorphic Compiler: Spectral clustering + binary code generation

Learn More:
-----------
- Documentation: https://nemus.readthedocs.io
- GitHub: https://github.com/yourusername/nemus
- Examples: See examples/ directory
- Phoenix Protocol: examples/test_phoenix_protocol.py

License: MIT
"""

from .core import ChronosEngine, Event
from .network import Network
from .layers import Input, Recurrent, Output
from .biology import LIF, AdaptiveLIF, Izhikevich
from .plasticity import EligibilityTrace, ThreeFactorRule, STDP, Homeostasis
from .morphology import Pruner, Synaptogenesis
from .silicon import Compiler, Mapper, Profile
from .receptors import TimeToFirstSpike, DeltaModulation, BurstCoding
from .data import DVSGesture
from .diagnostics import van_rossum_distance, EnergyMonitor, TopologyMetrics
from .interop import NIR_Bridge

__version__ = "1.0.0"
__author__ = "Charan"
__license__ = "MIT"

__all__ = [
    # Core
    "ChronosEngine",
    "Event",
    "Network",
    
    # Layers
    "Input",
    "Recurrent",
    "Output",
    
    # Neuron Models
    "LIF",
    "AdaptiveLIF",
    "Izhikevich",
    
    # Learning
    "EligibilityTrace",
    "ThreeFactorRule",
    "STDP",
    "Homeostasis",
    
    # Morphology
    "Pruner",
    "Synaptogenesis",
    
    # Hardware
    "Compiler",
    "Mapper",
    "Profile",
    
    # Encoding
    "TimeToFirstSpike",
    "DeltaModulation",
    "BurstCoding",
    
    # Data
    "DVSGesture",
    
    # Diagnostics
    "van_rossum_distance",
    "EnergyMonitor",
    "TopologyMetrics",
    
    # Interop
    "NIR_Bridge",
]
