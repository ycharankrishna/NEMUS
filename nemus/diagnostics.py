import numpy as np
from typing import List, Tuple, Any

class EnergyMonitor:
    def __init__(self):
        self.joules = 0.0
        self.spike_cost = 1e-12 # 1 pJ per spike

    def record_spike(self):
        self.joules += self.spike_cost

    def report(self) -> float:
        return self.joules

class TopologyMetrics:
    def calculate_sparsity(self, weights: np.ndarray) -> float:
        if weights is None: return 1.0
        total = weights.size
        zeros = np.count_nonzero(weights == 0)
        return zeros / total

    def small_worldness(self, weights: np.ndarray) -> float:
        # Calculate Clustering Coefficient (C) and Path Length (L)
        if weights is None: return 0.0
        
        # Binarize Adjacency Matrix
        A = (weights != 0).astype(int)
        n = A.shape[0]
        if n < 3: return 0.0
        
        # 1. Average Clustering Coefficient (C)
        # C = (1/n) * sum(local_clustering)
        # local_clustering_i = (edges between neighbors) / (k_i * (k_i - 1))
        
        # Neighbors for each node
        neighbors = [np.where(A[i, :] > 0)[0] for i in range(n)]
        degrees = np.array([len(nbrs) for nbrs in neighbors])
        
        c_i = np.zeros(n)
        for i in range(n):
            k = degrees[i]
            if k < 2:
                c_i[i] = 0.0
                continue
            
            nbrs = neighbors[i]
            # Count edges between neighbors
            # Subgraph of neighbors
            sub_A = A[np.ix_(nbrs, nbrs)]
            edges = np.sum(sub_A) # Directed edges
            
            # Max possible edges = k * (k - 1)
            c_i[i] = edges / (k * (k - 1))
            
        C = np.mean(c_i)
        
        # 2. Average Path Length (L)
        # Use BFS for unweighted shortest paths
        # This is O(N * (N+E)). For N=100-1000 it's fine.
        
        total_path_len = 0
        reachable_pairs = 0
        
        for start_node in range(n):
            # BFS
            visited = [-1] * n
            visited[start_node] = 0
            queue = [start_node]
            
            while queue:
                u = queue.pop(0)
                dist = visited[u]
                
                for v in neighbors[u]:
                    if visited[v] == -1:
                        visited[v] = dist + 1
                        queue.append(v)
                        total_path_len += visited[v]
                        reachable_pairs += 1
                        
        if reachable_pairs == 0:
            L = float('inf')
        else:
            L = total_path_len / reachable_pairs
            
        # 3. Random Graph Reference (Erdos-Renyi)
        # C_rand ~ k_avg / n
        # L_rand ~ ln(n) / ln(k_avg)
        k_avg = np.mean(degrees)
        if k_avg <= 1: return 0.0
        
        C_rand = k_avg / n
        L_rand = np.log(n) / np.log(k_avg) if k_avg > 1 else 1.0
        
        # Sigma
        if C_rand == 0 or L == 0: return 0.0
        sigma = (C / C_rand) / (L / L_rand)
        
        return sigma

def van_rossum_distance(spikes_a: List[Tuple[float, int]], spikes_b: List[Tuple[float, int]], tau: float = 0.1) -> float:
    if not spikes_a and not spikes_b:
        return 0.0
    
    if not spikes_a:
        return float(len(spikes_b))
    if not spikes_b:
        return float(len(spikes_a))
        
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
            total_dist += len(tb_list)
            continue
        if len(tb_list) == 0:
            total_dist += len(ta_list)
            continue
            
        def kernel_sum(t1, t2):
            diff = t1[:, None] - t2[None, :]
            return np.sum(np.exp(-np.abs(diff) / tau))
            
        term1 = kernel_sum(ta_list, ta_list)
        term2 = kernel_sum(tb_list, tb_list)
        term3 = kernel_sum(ta_list, tb_list)
        
        dist_sq = term1 + term2 - 2 * term3
        if dist_sq < 0: dist_sq = 0
        total_dist += np.sqrt(dist_sq)
        
    return total_dist / len(all_neurons) if all_neurons else 0.0
