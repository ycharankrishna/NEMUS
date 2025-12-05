import heapq
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Dict, Set
import numpy as np

@dataclass(order=True)
class Event:
    timestamp: float
    priority: int 
    address: int = field(compare=False) 
    payload: Any = field(compare=False) 
    layer_name: str = field(compare=False)
    valid: bool = field(compare=False, default=True) # For lazy deletion

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._count = 0

    def push(self, event: Event):
        heapq.heappush(self._queue, event)
        self._count += 1

    def pop(self) -> Event:
        while self._queue:
            event = heapq.heappop(self._queue)
            self._count -= 1
            if event.valid:
                return event
        raise IndexError("pop from empty priority queue")

    def is_empty(self) -> bool:
        # Check if any valid events exist
        # This is O(N) in worst case if many invalid. 
        # Optimization: track valid count.
        return self._count == 0 # Approximate, might have invalid ones.
        
    def peek_time(self) -> Optional[float]:
        while self._queue and not self._queue[0].valid:
            heapq.heappop(self._queue) # Clean up lazy deleted
            self._count -= 1
            
        if self._queue:
            return self._queue[0].timestamp
        return None

class Manifold:
    def __init__(self, shape: Tuple[int, ...], metric: str = "euclidean"):
        self.shape = shape
        self.metric = metric
        self.ndim = len(shape)
        ranges = [np.linspace(0, 1, s) for s in shape]
        grids = np.meshgrid(*ranges, indexing='ij')
        self.coords = np.stack(grids, axis=-1).reshape(-1, self.ndim)

    def distance(self, idx1: int, idx2: int) -> float:
        p1 = self.coords[idx1]
        p2 = self.coords[idx2]
        return np.linalg.norm(p1 - p2)

class ChronosEngine:
    """Analytical Event-Time Solver (AETS) Kernel."""
    def __init__(self, network: Any):
        self.network = network
        self.time = 0.0
        self.queue = PriorityQueue()
        self.network.attach_engine(self)
        
        # Map (layer_name, neuron_idx) -> Event
        # To support updating predictions
        self.pending_spikes: Dict[Tuple[str, int], Event] = {}

    def schedule(self, event: Event):
        self.queue.push(event)

    def update_prediction(self, layer_name: str, neuron_idx: int, t_spike: float):
        # 1. Invalidate old prediction
        key = (layer_name, neuron_idx)
        if key in self.pending_spikes:
            self.pending_spikes[key].valid = False
            del self.pending_spikes[key]
            
        # 2. Schedule new prediction if finite
        if t_spike != float('inf'):
            # Priority 1 for spikes
            evt = Event(timestamp=t_spike, priority=1, address=neuron_idx, payload=1.0, layer_name=layer_name)
            self.queue.push(evt)
            self.pending_spikes[key] = evt

    def run(self, duration: float):
        end_time = self.time + duration
        while True:
            t_next = self.queue.peek_time()
            if t_next is None or t_next > end_time:
                break
                
            event = self.queue.pop()
            self.time = event.timestamp
            
            # If it's a spike event from a neuron, we remove it from pending
            if event.priority == 1: # Spike
                key = (event.layer_name, event.address)
                if key in self.pending_spikes and self.pending_spikes[key] == event:
                    del self.pending_spikes[key]
            
            self.network.dispatch(event)
            
    def run_until_empty(self):
        while True:
            t_next = self.queue.peek_time()
            if t_next is None: break
            
            event = self.queue.pop()
            self.time = event.timestamp
            
            if event.priority == 1:
                key = (event.layer_name, event.address)
                if key in self.pending_spikes and self.pending_spikes[key] == event:
                    del self.pending_spikes[key]
                    
            self.network.dispatch(event)
