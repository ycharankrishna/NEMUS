import numpy as np
from typing import List, Tuple

def van_rossum_distance(spikes_a: List[Tuple[float, int]], spikes_b: List[Tuple[float, int]], tau: float = 0.1) -> float:
    # Robust Van Rossum Distance
    # If one is empty and other is not, distance should be high.
    
    if not spikes_a and not spikes_b:
        return 0.0
    
    # If one is empty, return a penalty proportional to the number of spikes in the other
    # or a max value.
    if not spikes_a:
        return float(len(spikes_b))
    if not spikes_b:
        return float(len(spikes_a))
        
    # Extract times (ignoring neuron index for this aggregate metric, or we can match by index)
    # The user wants "Reconstruction Error".
    # In Auto-Encoder, Input Neuron I should trigger Output Neuron I.
    # So we should compare spike trains per neuron.
    
    # Group by neuron
    spikes_a_by_neuron = {}
    for t, idx in spikes_a:
        spikes_a_by_neuron.setdefault(idx, []).append(t)
        
    spikes_b_by_neuron = {}
    for t, idx in spikes_b:
        spikes_b_by_neuron.setdefault(idx, []).append(t)
        
    all_neurons = set(spikes_a_by_neuron.keys()) | set(spikes_b_by_neuron.keys())
    
    total_dist = 0.0
    
    for n in all_neurons:
        ta_list = np.array(spikes_a_by_neuron.get(n, []))
        tb_list = np.array(spikes_b_by_neuron.get(n, []))
        
        if len(ta_list) == 0:
            total_dist += len(tb_list) # Penalty for hallucination
            continue
        if len(tb_list) == 0:
            total_dist += len(ta_list) # Penalty for silence
            continue
            
        # Compute distance between two spike trains for this neuron
        # Simple implementation: sum of exponentials difference?
        # Or just match closest.
        # Let's use the "kernel" definition: integral of (f - g)^2
        # Approximated by sum of pairwise kernel evaluations.
        # D^2 = sum(sum(k(ti, tj))) + sum(sum(k(ui, uj))) - 2 sum(sum(k(ti, uj)))
        
        def kernel_sum(t1, t2):
            # Vectorized calculation
            # t1: (N,1), t2: (1,M) -> (N,M)
            diff = t1[:, None] - t2[None, :]
            return np.sum(np.exp(-np.abs(diff) / tau))
            
        term1 = kernel_sum(ta_list, ta_list)
        term2 = kernel_sum(tb_list, tb_list)
        term3 = kernel_sum(ta_list, tb_list)
        
        dist_sq = term1 + term2 - 2 * term3
        if dist_sq < 0: dist_sq = 0 # Numerical noise
        total_dist += np.sqrt(dist_sq)
        
    # Normalize by number of neurons or spikes?
    # Let's normalize by total spikes to keep it in [0, 1] range roughly?
    # No, distance is extensive.
    # Let's return average distance per active neuron.
    return total_dist / len(all_neurons) if all_neurons else 0.0
