import numpy as np
from typing import Any, Optional

class EligibilityTrace:
    """Dopaminergic Eligibility Propagation (DEP) Trace.
    
    Differential equation: de/dt = -e/tau + Pre * Post'
    where Post' is the surrogate derivative of the postsynaptic neuron.
    """
    def __init__(self, shape: tuple, tau: float = 0.05):
        self.trace = np.zeros(shape)
        self.tau = tau
        self.last_update = np.zeros(shape)

    def decay(self, t_now: float):
        """Apply exponential decay to all traces."""
        dt = t_now - self.last_update
        decay_factor = np.exp(-dt / self.tau)
        self.trace *= decay_factor
        self.last_update = np.full_like(self.last_update, t_now)

    def accumulate_pre_post(self, pre_idx: int, post_idx: int, t_now: float, surrogate_deriv: float = 1.0):
        """Accumulate trace when pre and post activity coincide.
        
        Args:
            pre_idx: Presynaptic neuron index
            post_idx: Postsynaptic neuron index  
            t_now: Current time
            surrogate_deriv: Surrogate derivative of postsynaptic activation
        """
        # Decay first
        dt = t_now - self.last_update[pre_idx, post_idx]
        self.trace[pre_idx, post_idx] *= np.exp(-dt / self.tau)
        self.last_update[pre_idx, post_idx] = t_now
        
        # Accumulate: Pre (implicit 1 for spike) * Post' (surrogate)
        self.trace[pre_idx, post_idx] += surrogate_deriv

class ThreeFactorRule:
    """Reward-Modulated Hebbian Learning (Three-Factor Rule).
    
    Weight update: ΔW = η * L * e
    where L is the global error/dopamine signal and e is the eligibility trace.
    """
    def __init__(self, learning_rate: float = 0.01, tau_plus: float = 0.02, tau_minus: float = 0.02):
        self.lr = learning_rate
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

    def apply(self, weights: np.ndarray, eligibility: np.ndarray, reward: float):
        """Apply reward-modulated weight update.
        
        Args:
            weights: Synaptic weight matrix (Input x Neurons)
            eligibility: Eligibility trace matrix (Input x Neurons)
            reward: Global reward/dopamine signal
        """
        delta_w = self.lr * reward * eligibility
        weights += delta_w

class STDP:
    """Spike-Timing-Dependent Plasticity (STDP).
    
    Classic two-factor Hebbian rule based on spike timing.
    """
    def __init__(self, learning_rate: float = 0.01, tau_plus: float = 0.02, tau_minus: float = 0.02, w_max: float = 1.0):
        self.lr = learning_rate
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_max = w_max

    def on_pre_spike(self, weights: np.ndarray, pre_idx: int, t_last_post: np.ndarray, t_now: float):
        """Long-Term Depression (LTD) applied when pre-spike arrives."""
        dt = t_now - t_last_post
        delta = -self.lr * np.exp(-np.maximum(dt, 0) / self.tau_minus)
        weights[pre_idx, :] += delta
        np.clip(weights[pre_idx, :], 0, self.w_max, out=weights[pre_idx, :])

    def on_post_spike(self, weights: np.ndarray, post_idx: int, t_last_pre: np.ndarray, t_now: float):
        """Long-Term Potentiation (LTP) applied when post-spike occurs."""
        dt = t_now - t_last_pre
        delta = self.lr * np.exp(-np.maximum(dt, 0) / self.tau_plus)
        weights[:, post_idx] += delta
        np.clip(weights[:, post_idx], 0, self.w_max, out=weights[:, post_idx])

class Homeostasis:
    """Homeostatic plasticity for activity regulation.
    
    Adjusts neuron thresholds to maintain target firing rate.
    """
    def __init__(self, target_rate: float = 10.0, adaptation_rate: float = 0.001):
        self.target = target_rate
        self.rate = adaptation_rate

    def apply(self, threshold: np.ndarray, activity_rate: np.ndarray):
        """Adjust thresholds based on deviation from target rate.
        
        Args:
            threshold: Neuron threshold array
            activity_rate: Measured firing rate array
        """
        diff = activity_rate - self.target
        threshold += self.rate * diff
        # Ensure thresholds stay positive
        np.maximum(threshold, 0.1, out=threshold)
