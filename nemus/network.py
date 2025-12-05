import numpy as np
from .core import PriorityQueue, Event
from .layers import Layer, Input, Output
from typing import List, Any

class Network:
    def __init__(self, entry_layer: Layer):
        # Find true entry
        while entry_layer.prev_layer:
            entry_layer = entry_layer.prev_layer
            
        self.entry_layer = entry_layer
        self.layers = self._traverse_layers(entry_layer)
        self.queue = None
        self.stats = {'synapse_count': 0}
        
    def attach_engine(self, engine: Any):
        """Attach the Chronos engine."""
        self.queue = engine.queue
        
        # Initialize queue and engine for all layers
        for layer in self.layers:
            layer.set_queue(self.queue)
            layer.set_engine(engine)
            
            if hasattr(layer, 'weights') and layer.weights is not None:
                 self.stats['synapse_count'] += layer.weights.size

    def dispatch(self, event: Event):
        """Dispatch event to target layer."""
        target_layer = next((l for l in self.layers if l.name == event.layer_name), None)
        if target_layer:
            target_layer.receive_event(event)

    def _traverse_layers(self, start_layer: Layer) -> List[Layer]:
        layers = []
        current = start_layer
        while current:
            layers.append(current)
            current = current.next_layer
        return layers

    def reset(self):
        """Reset all layers."""
        for layer in self.layers:
            layer.reset()
