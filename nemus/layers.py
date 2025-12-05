import numpy as np
from typing import Optional, Any, Tuple
from .core import Event, PriorityQueue
from .biology import NeuronModel, LIF, AdaptiveLIF
from .plasticity import STDP, ThreeFactorRule, EligibilityTrace
from .morphology import Pruner, Synaptogenesis
from .receptors import TimeToFirstSpike, DeltaModulation

class Layer:
    def __init__(self, name: str):
        self.name = name
        self.next_layer: Optional['Layer'] = None
        self.prev_layer: Optional['Layer'] = None
        self.network_queue: Optional[PriorityQueue] = None
        self.engine: Optional[Any] = None

    def connect(self, other: 'Layer'):
        self.next_layer = other
        other.prev_layer = self
        return other 

    def __rshift__(self, other: 'Layer'):
        return self.connect(other)

    def set_queue(self, queue: PriorityQueue):
        self.network_queue = queue
        if self.next_layer:
            self.next_layer.set_queue(queue)
            
    def set_engine(self, engine: Any):
        self.engine = engine
        if self.next_layer:
            self.next_layer.set_engine(engine)

    def receive_event(self, event: Event):
        raise NotImplementedError

    def reset(self):
        """Reset layer state."""
        pass

class Input(Layer):
    def __init__(self, shape: Tuple[int, ...], encoding: str = "latency"):
        super().__init__("Input")
        self.shape = shape
        self.size = np.prod(shape)
        if encoding == "latency":
            self.encoder = TimeToFirstSpike()
        elif encoding == "delta":
            self.encoder = DeltaModulation()
        else:
            self.encoder = None

    def receive_event(self, event: Event):
        # Forward to next layer
        if self.next_layer:
            new_event = Event(
                timestamp=event.timestamp + 0.0001,
                priority=1,
                address=event.address,
                payload=event.payload,
                layer_name=self.next_layer.name
            )
            self.network_queue.push(new_event)

class Recurrent(Layer):
    """Recurrent layer with AETS support."""
    def __init__(self, neurons: int, model: NeuronModel, plasticity: Any = None, 
                 morphology: Any = None, sparsity: float = 0.1):
        super().__init__("Recurrent_" + model.name)
        self.neurons = neurons
        self.model = model
        self.plasticity = plasticity
        self.morphology = morphology
        self.sparsity = sparsity
        
        # Weights
        self.weights: Optional[np.ndarray] = None
        self.rec_weights = np.random.randn(neurons, neurons) * 0.05
        
        # Apply sparsity
        mask = np.random.rand(neurons, neurons) > sparsity
        self.rec_weights[mask] = 0
        
        # For morphology
        self.event_count = 0
        self.rewire_period = 100
        
        # For tracking last spike times (needed for STDP)
        self.t_last_spike = np.ones(neurons) * -1.0
        
        # Activity traces for morphology
        self.activity_trace = np.zeros(neurons)

    def _init_weights(self, input_size: int):
        self.weights = np.random.randn(input_size, self.neurons) * 0.1

    def receive_event(self, event: Event):
        source_idx = event.address
        t_now = event.timestamp
        
        # Priority 0 = Input spike, Priority 1 = Neuron spike
        if event.priority == 0:
            # External input
            if self.weights is None:
                self._init_weights(max(source_idx + 1, 128))
                
            if source_idx >= self.weights.shape[0]:
                old_size = self.weights.shape[0]
                new_size = max(source_idx + 1, old_size * 2)
                new_weights = np.zeros((new_size, self.neurons))
                new_weights[:old_size, :] = self.weights
                self.weights = new_weights
                
            # STDP pre-spike
            if self.plasticity and isinstance(self.plasticity, STDP):
                self.plasticity.on_pre_spike(self.weights, source_idx, self.t_last_spike, t_now)
            
            # Update all neurons that receive this input
            targets = np.where(self.weights[source_idx, :] != 0)[0]
            for idx in targets:
                weight = self.weights[source_idx, idx]
                
                # Update state
                self.model.update_state(idx, t_now, weight)
                
                # Check for immediate spike
                if self.model.v[idx] >= self.model.threshold:
                    self._fire_neuron(idx, t_now)
                else:
                    # Re-predict spike time
                    t_spike = self.model.predict_spike(idx, t_now)
                    if self.engine:
                        self.engine.update_prediction(self.name, idx, t_spike)
                        
        elif event.priority == 1:
            # Predicted neuron spike
            neuron_idx = source_idx
            self._fire_neuron(neuron_idx, t_now)
            
        # Morphology
        self.event_count += 1
        if self.morphology and self.event_count % self.rewire_period == 0:
            if isinstance(self.morphology, list):
                for rule in self.morphology:
                    if isinstance(rule, Pruner):
                        rule.apply(self.rec_weights)
                    elif isinstance(rule, Synaptogenesis):
                        rule.apply(self.rec_weights, self.activity_trace, self.activity_trace)

    def _fire_neuron(self, idx: int, t_now: float):
        """Handle neuron firing."""
        # Reset voltage
        self.model.v[idx] = 0.0
        self.t_last_spike[idx] = t_now
        
        # Update activity trace
        self.activity_trace[idx] = self.activity_trace[idx] * 0.9 + 0.1
        
        # STDP post-spike
        if self.plasticity and isinstance(self.plasticity, STDP) and self.weights is not None:
            # Get last pre-spike times (approximation: use model.t_last for inputs)
            # This is a simplification - proper STDP needs to track per-synapse timing
            t_last_pre = self.model.t_last[:self.weights.shape[0]] if hasattr(self.model, 't_last') else np.zeros(self.weights.shape[0])
            self.plasticity.on_post_spike(self.weights, idx, t_last_pre, t_now)
        
        # Propagate spike to next layer
        if self.next_layer:
            self.network_queue.push(Event(
                timestamp=t_now + 0.0001, 
                priority=0,
                address=idx,
                payload=1.0,
                layer_name=self.next_layer.name
            ))
        
        # Schedule next spike prediction
        if self.engine:
            self.engine.update_prediction(self.name, idx, t_spike)

    def reset(self):
        """Reset recurrent layer state."""
        self.model.reset()
        self.t_last_spike[:] = -1.0
        self.activity_trace[:] = 0.0
        self.event_count = 0
        # Note: We do NOT reset weights as they are learned/fixed


class Output(Layer):
    def __init__(self, classes: int):
        super().__init__("Output")
        self.classes = classes
        self.spikes = []

    def receive_event(self, event: Event):
        self.spikes.append((event.timestamp, event.address))

    def reset(self):
        self.spikes = []
