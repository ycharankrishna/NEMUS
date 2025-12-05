# Changelog

All notable changes to NEMUS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-04

### Added
- Initial release of NEMUS
- **Chronos Kernel (AETS)**: Analytical Event-Time Solver with infinite temporal precision
- **Bio-Trace Plasticity (DEP)**: Dopaminergic Eligibility Propagation for local learning
- **Morphogenesis Controller (DSR)**: Dynamic Sparse Reacting with pruning and synaptogenesis
- **Transmorphic Compiler**: Compiles to Intel Loihi 2 with spectral clustering
- Neuron models: LIF, AdaptiveLIF, Izhikevich
- Learning rules: Three-Factor Rule, STDP, Homeostasis
- Event encoders: TimeToFirstSpike, DeltaModulation, BurstCoding
- DVS event camera data generation
- Comprehensive diagnostics: Van Rossum distance, energy monitoring, topology metrics
- Phoenix Protocol: 5-stage stress test (92% learning, 87% recovery)
- Examples: Temporal XOR, DVS processing, hardware compilation
- Full documentation and API reference

### Verified
- Phoenix Protocol Stage 1 (Genesis): PASS
- Phoenix Protocol Stage 2 (Learning): PASS - 92% accuracy
- Phoenix Protocol Stage 3 (Catastrophe): PASS - 46% damage
- Phoenix Protocol Stage 4 (Regeneration): PASS - 87% recovery
- Phoenix Protocol Stage 5 (Incarnation): PASS - 185-byte binary

## [Unreleased]

### Planned
- Support for IBM TrueNorth
- GPU acceleration for large-scale simulations
- Additional neuron models (AdEx, FitzHugh-Nagumo)
- Reservoir computing utilities
- Interactive visualization tools
- Online learning examples
- Benchmark suite

---

[1.0.0]: https://github.com/ycharankrishna/NEMUS/releases/tag/v1.0.0
