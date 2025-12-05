import numpy as np
from typing import List, Tuple

class Encoder:
    def encode(self, data: np.ndarray, dt: float) -> List[Tuple[float, int]]:
        raise NotImplementedError

class TimeToFirstSpike(Encoder):
    def __init__(self, t_max: float = 0.1):
        self.t_max = t_max

    def encode(self, data: np.ndarray, dt: float = 0.0) -> List[Tuple[float, int]]:
        # Data is array of intensities [0, 1]
        # Higher intensity -> Earlier spike
        # t = t_max * (1 - intensity)
        
        spikes = []
        for idx, intensity in enumerate(data):
            if intensity > 0:
                t = self.t_max * (1.0 - intensity)
                spikes.append((t, idx))
        
        spikes.sort(key=lambda x: x[0])
        return spikes

class DeltaModulation(Encoder):
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.last_val = None

    def encode(self, data: np.ndarray, dt: float) -> List[Tuple[float, int]]:
        # data is a frame at time t
        # Spike if abs(curr - last) > threshold
        
        if self.last_val is None:
            self.last_val = np.zeros_like(data)
            return []
            
        diff = data - self.last_val
        spikes = []
        
        # Positive spikes
        pos_idx = np.where(diff > self.threshold)[0]
        for idx in pos_idx:
            spikes.append((0.0, idx)) # Immediate spike
            self.last_val[idx] = data[idx] # Update reference
            
        # Negative spikes? (Ignored or separate channel)
        
        return spikes

class BurstCoding(Encoder):
    def __init__(self, max_spikes: int = 5, t_interval: float = 0.005):
        self.max_spikes = max_spikes
        self.t_interval = t_interval

    def encode(self, data: np.ndarray, dt: float = 0.0) -> List[Tuple[float, int]]:
        # Intensity maps to number of spikes
        spikes = []
        for idx, intensity in enumerate(data):
            num = int(intensity * self.max_spikes)
            for i in range(num):
                spikes.append((i * self.t_interval, idx))
        return spikes
