import numpy as np
from typing import Any, Dict

class LearningAlgorithm:
    def update(self, network: Any, error: float):
        raise NotImplementedError

class EligibilityProp(LearningAlgorithm):
    def __init__(self, learning_rate: float = 0.001, trace_decay: float = 0.05, feedback_alignment: str = "symmetric"):
        self.learning_rate = learning_rate
        self.trace_decay = trace_decay
        self.feedback_alignment = feedback_alignment

    def update(self, network: Any, error: float):
        # Iterate over all layers in the network
        # If layer has weights and traces, update them.
        # Delta W = eta * Trace * Error
        
        for layer in network.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'traces'):
                if layer.weights is None or layer.traces is None:
                    continue
                # Decay traces (approximated here for the batch update)
                # In a real event-driven setup, traces decay continuously.
                # Here we assume traces are accumulated during the forward pass.
                
                # Update weights
                # We assume 'traces' is a matrix of same shape as weights
                # representing the eligibility.
                
                # For this demo, we'll assume a scalar error is broadcasted
                # or a vector error if we want to be precise.
                
                # Simple Hebbian-like update with global error modulation
                delta_w = self.learning_rate * layer.traces * error
                
                layer.weights += delta_w
                
                # Decay/Reset traces for next batch?
                # In continuous learning, they decay.
                # layer.traces *= (1 - self.trace_decay) 
                pass

class Optimizer:
    def __init__(self, algorithm: LearningAlgorithm):
        self.algorithm = algorithm
        self.network = None

    def attach(self, network: Any):
        self.network = network

    def step(self, spikes: Any, target: Any) -> float:
        if not self.network:
            raise RuntimeError("Optimizer not attached to a network.")
        
        # Calculate loss/error
        # Assuming spikes is a list of (t, neuron_idx) from the output layer
        # and target is the expected class index.
        
        # Simple loss: Did the correct neuron spike most?
        # Count spikes per neuron
        spike_counts = {}
        for t, idx in spikes:
            spike_counts[idx] = spike_counts.get(idx, 0) + 1
            
        # Predicted class is max spike count
        if not spike_counts:
            predicted = -1
        else:
            predicted = max(spike_counts, key=spike_counts.get)
            
        # Error signal
        # If correct, positive reinforcement?
        # If wrong, negative?
        # Let's use a simple error scalar: 1.0 if correct, -1.0 if wrong.
        # Or CrossEntropy-like gradient.
        
        # For EligibilityProp, we often use (Target - Output)
        # Let's assume we want to increase activity of target neuron and decrease others.
        
        # We need to broadcast error back to layers.
        # But EligibilityProp with "Global Error" simplifies this to a scalar or vector broadcast.
        
        # Let's define error as 1 if correct, -1 if wrong (simple reinforcement)
        error = 1.0 if predicted == target else -1.0
        
        # Run the algorithm
        self.algorithm.update(self.network, error)
        
        return -error # Return "loss" (negative of reward)
