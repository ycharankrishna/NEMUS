import numpy as np
from typing import Optional, Tuple

class NeuronModel:
    def __init__(self, neurons: int, name: str):
        self.neurons = neurons
        self.name = name
        self.v = np.zeros(neurons)
        self.t_last = np.zeros(neurons)

    def update_state(self, idx: int, t_now: float, weight: float) -> None:
        """Updates V(t) based on input weight at t_now."""
        raise NotImplementedError

    def predict_spike(self, idx: int, t_now: float) -> float:
        """Returns absolute time of next spike, or infinity."""
        raise NotImplementedError

    def reset(self):
        """Reset neuron state."""
        self.v[:] = 0.0
        self.t_last[:] = 0.0

class LIF(NeuronModel):
    def __init__(self, neurons: int, tau_mem: float = 0.02, threshold: float = 1.0, resistance: float = 1.0):
        super().__init__(neurons, "LIF")
        self.tau_mem = tau_mem
        self.threshold = threshold
        self.resistance = resistance
        self.current = np.zeros(neurons)

    def reset(self):
        super().reset()
        self.current[:] = 0.0
        
    def update_state(self, idx: int, t_now: float, weight: float) -> None:
        # 1. Evolve V from t_last to t_now (analytical decay)
        dt = t_now - self.t_last[idx]
        decay = np.exp(-dt / self.tau_mem)
        v_inf = self.current[idx] * self.resistance
        
        self.v[idx] = self.v[idx] * decay + v_inf * (1 - decay)
        self.t_last[idx] = t_now
        
        # 2. Add Input Spike (instantaneous charge)
        self.v[idx] += weight

    def predict_spike(self, idx: int, t_now: float) -> float:
        """Analytical prediction using inverted LIF equation."""
        v_inf = self.current[idx] * self.resistance
        v_now = self.v[idx]
        
        if v_now >= self.threshold:
            return t_now
            
        if v_inf <= self.threshold:
            return float('inf')
        
        numerator = self.threshold - v_inf
        denominator = v_now - v_inf
        
        if denominator == 0:
            return float('inf')
        
        u = numerator / denominator
        
        if u <= 0:
            return float('inf')
        
        dt = -self.tau_mem * np.log(u)
        
        if dt < 0:
            return t_now
        
        return t_now + dt

class AdaptiveLIF(LIF):
    def __init__(self, neurons: int, tau_mem: float = 0.02, tau_adapt: float = 1.0, threshold: float = 1.0, beta: float = 1.0):
        super().__init__(neurons, tau_mem, threshold)
        self.tau_adapt = tau_adapt
        self.beta = beta
        self.theta_adapt = np.zeros(neurons)
        self.t_last_adapt = np.zeros(neurons)

    def reset(self):
        super().reset()
        self.theta_adapt[:] = 0.0
        self.t_last_adapt[:] = 0.0

    def update_state(self, idx: int, t_now: float, weight: float) -> None:
        # Update Adaptation (exponential decay)
        dt_adapt = t_now - self.t_last_adapt[idx]
        self.theta_adapt[idx] *= np.exp(-dt_adapt / self.tau_adapt)
        self.t_last_adapt[idx] = t_now
        
        # Update V
        super().update_state(idx, t_now, weight)

    def predict_spike(self, idx: int, t_now: float) -> float:
        """Analytical prediction with adiabatic approximation for adaptation."""
        effective_thresh = self.threshold + self.theta_adapt[idx]
        
        v_inf = self.current[idx] * self.resistance
        v_now = self.v[idx]
        
        if v_now >= effective_thresh:
            return t_now
        if v_inf <= effective_thresh:
            return float('inf')
        
        u = (effective_thresh - v_inf) / (v_now - v_inf)
        if u <= 0:
            return float('inf')
        dt = -self.tau_mem * np.log(u)
        return t_now + dt

class Izhikevich(NeuronModel):
    """Izhikevich spiking neuron model.
    
    Dynamics:
        dv/dt = 0.04v^2 + 5v + 140 - u + I
        du/dt = a(bv - u)
    
    When v >= 30mV: v <- c, u <- u + d
    """
    def __init__(self, neurons: int, a=0.02, b=0.2, c=-65, d=8):
        super().__init__(neurons, "Izhikevich")
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.u = np.zeros(neurons)
        self.v[:] = -65.0  # Resting potential
        self.current = np.zeros(neurons)

    def reset(self):
        self.v[:] = -65.0
        self.u[:] = 0.0
        self.t_last[:] = 0.0
        self.current[:] = 0.0

    def update_state(self, idx: int, t_now: float, weight: float) -> None:
        """Euler integration for Izhikevich dynamics."""
        dt = t_now - self.t_last[idx]
        
        # Numerical integration (Euler method with adaptive stepping)
        dt_step = min(dt, 0.001)  # Max 1ms step
        steps = max(1, int(dt / dt_step))
        dt_step = dt / steps
        
        v = self.v[idx]
        u = self.u[idx]
        I = self.current[idx]
        
        for _ in range(steps):
            dv = 0.04 * v**2 + 5 * v + 140 - u + I
            du = self.a * (self.b * v - u)
            
            v += dv * dt_step
            u += du * dt_step
            
            # Check for spike
            if v >= 30:
                v = self.c
                u += self.d
                break
        
        # Add input weight
        v += weight
        
        # Check again after input
        if v >= 30:
            v = self.c
            u += self.d
        
        self.v[idx] = v
        self.u[idx] = u
        self.t_last[idx] = t_now

    def predict_spike(self, idx: int, t_now: float) -> float:
        """Numerical lookahead for spike prediction.
        
        Izhikevich model doesn't have closed-form solution, so we simulate forward.
        """
        max_lookahead = 0.1  # 100ms max
        dt_step = 0.001  # 1ms steps
        
        v = self.v[idx]
        u = self.u[idx]
        I = self.current[idx]
        
        t = 0
        while t < max_lookahead:
            dv = 0.04 * v**2 + 5 * v + 140 - u + I
            du = self.a * (self.b * v - u)
            
            v += dv * dt_step
            u += du * dt_step
            t += dt_step
            
            if v >= 30:
                return t_now + t
        
        return float('inf')
