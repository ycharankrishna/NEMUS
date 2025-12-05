import numpy as np
from typing import Optional

class Pruner:
    """Synaptic pruning based on weight magnitude threshold."""
    def __init__(self, threshold: float = 0.001):
        self.threshold = threshold

    def apply(self, weights: np.ndarray) -> int:
        """Prune weak synapses by setting them to zero.
        
        Returns:
            Number of synapses pruned
        """
        mask = np.abs(weights) < self.threshold
        count = np.sum(mask)
        weights[mask] = 0
        return count

class Synaptogenesis:
    """Dynamic Sparse Rewiring with Hypothetical Gradient Analysis.
    
    Implements the "Hypothetical Gradient" approach: for each pair of unconnected
    neurons, estimate the potential gradient if a connection existed, and create
    synapses where this potential is high.
    """
    def __init__(self, growth_rate: float = 0.01, probability: float = 0.1, sigma: float = 10.0):
        self.growth_rate = growth_rate
        self.prob = probability
        self.sigma = sigma

    def apply(self, weights: np.ndarray, pre_activity: Optional[np.ndarray] = None, 
              post_activity: Optional[np.ndarray] = None,
              pre_coords: Optional[np.ndarray] = None, 
              post_coords: Optional[np.ndarray] = None) -> int:
        """Grow new synapses based on hypothetical gradient.
        
        The hypothetical gradient is estimated as the correlation between
        presynaptic and postsynaptic activity, weighted by spatial distance.
        
        Args:
            weights: Current weight matrix (Pre x Post)
            pre_activity: Recent activity trace of presynaptic neurons
            post_activity: Recent activity trace of postsynaptic neurons
            pre_coords: Spatial coordinates of presynaptic neurons
            post_coords: Spatial coordinates of postsynaptic neurons
            
        Returns:
            Number of synapses created
        """
        zeros = np.where(weights == 0)
        if len(zeros[0]) == 0:
            return 0
            
        num_candidates = len(zeros[0])
        num_grow = int(num_candidates * self.growth_rate)
        if num_grow == 0:
            return 0
        
        # If we have activity traces, calculate hypothetical gradients
        if pre_activity is not None and post_activity is not None:
            # Hypothetical gradient = correlation * spatial_proximity
            # For each zero connection (i,j), estimate correlation
            
            # Sample candidates (can't compute all for large networks)
            sample_size = min(num_grow * 10, num_candidates)
            sample_indices = np.random.choice(num_candidates, sample_size, replace=False)
            
            scores = np.zeros(sample_size)
            for idx, sample_idx in enumerate(sample_indices):
                pre_idx = zeros[0][sample_idx]
                post_idx = zeros[1][sample_idx]
                
                # Correlation: activity_pre * activity_post
                correlation = pre_activity[pre_idx] * post_activity[post_idx]
                
                # Spatial factor
                if pre_coords is not None and post_coords is not None:
                    dist = np.linalg.norm(pre_coords[pre_idx] - post_coords[post_idx])
                    spatial_factor = np.exp(-dist**2 / (2 * self.sigma**2))
                else:
                    spatial_factor = 1.0
                
                scores[idx] = correlation * spatial_factor
            
            # Select top scoring connections
            if np.sum(scores) > 0:
                # Normalize to probabilities
                probs = scores / np.sum(scores)
                selected = np.random.choice(sample_size, min(num_grow, sample_size), 
                                          replace=False, p=probs)
                selected_indices = sample_indices[selected]
            else:
                # Fallback to random if no activity
                selected_indices = sample_indices[:num_grow]
                
        else:
            # Fallback: spatial-only or random growth
            if pre_coords is not None and post_coords is not None:
                sample_size = min(num_grow * 3, num_candidates)
                indices = np.random.choice(num_candidates, sample_size, replace=False)
                
                accepted = []
                for idx in indices:
                    pre_idx = zeros[0][idx]
                    post_idx = zeros[1][idx]
                    dist = np.linalg.norm(pre_coords[pre_idx] - post_coords[post_idx])
                    prob = np.exp(-dist**2 / (2 * self.sigma**2))
                    
                    if np.random.rand() < prob:
                        accepted.append(idx)
                        if len(accepted) >= num_grow:
                            break
                
                selected_indices = np.array(accepted) if accepted else indices[:num_grow]
            else:
                # Pure random growth
                selected_indices = np.random.choice(num_candidates, num_grow, replace=False)
        
        # Create synapses with small initial weight
        for idx in selected_indices:
            weights[zeros[0][idx], zeros[1][idx]] = 0.01
            
        return len(selected_indices)
