from enum import Enum
from typing import Any, Dict, List
import numpy as np
import struct

class Profile(Enum):
    LOIHI_2 = "Loihi 2"
    SPINNAKER_2 = "SpiNNaker 2"
    JETSON_NANO = "Jetson Nano"
    ARM_CORTEX_M4 = "ARM Cortex-M4"

class Quantizer:
    """Weight quantization for hardware deployment."""
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.range = 2**(bits-1) - 1

    def quantize(self, weights: Any) -> tuple:
        """Quantize weights to fixed-point representation.
        
        Returns:
            (quantized_weights, scale_factor)
        """
        if weights is None:
            return None, 1.0
        
        scale = self.range / max(abs(weights.min()), abs(weights.max()) + 1e-9)
        q_weights = (weights * scale).astype(np.int8)
        return q_weights, scale

class Mapper:
    """Constraint-Aware Graph Embedding using Spectral Clustering.
    
    Uses the Fiedler vector (2nd eigenvector of graph Laplacian) to partition
    the neural network graph into hardware cores.
    """
    def __init__(self, profile: Profile):
        self.profile = profile

    def map(self, network: Any) -> Dict[str, Any]:
        """Map network to hardware cores using spectral clustering.
        
        Algorithm:
        1. Build adjacency matrix of the full network graph
        2. Compute graph Laplacian L = D - A
        3. Find eigenvectors of L
        4. Use Fiedler vector (2nd smallest eigenvalue) for bisection
        5. Recursively partition until cores are filled
        """
        neurons_per_core = 1024
        
        # Build global neuron list and adjacency
        neuron_list = []  # (layer_name, local_idx)
        adjacency = []
        
        # Map layers to global neuron indices
        global_idx = 0
        layer_to_global = {}
        
        for layer in network.layers:
            if not hasattr(layer, 'neurons'):
                continue
                
            layer_start = global_idx
            layer_to_global[layer.name] = (layer_start, global_idx + layer.neurons)
            
            for i in range(layer.neurons):
                neuron_list.append((layer.name, i))
                global_idx += 1
        
        total_neurons = len(neuron_list)
        if total_neurons == 0:
            return {}
            
        # Build adjacency matrix (sparse representation)
        # This is expensive for large networks - in production would use scipy.sparse
        adj_matrix = np.zeros((total_neurons, total_neurons))
        
        for layer in network.layers:
            if not hasattr(layer, 'weights') or layer.weights is None:
                continue
            if not hasattr(layer, 'next_layer') or layer.next_layer is None:
                continue
                
            # Forward connections
            src_start, src_end = layer_to_global.get(layer.name, (0, 0))
            dst_start, dst_end = layer_to_global.get(layer.next_layer.name, (0, 0))
            
            # Fill adjacency (binary graph)
            weights = layer.weights
            for i in range(min(weights.shape[0], src_end - src_start)):
                for j in range(min(weights.shape[1], dst_end - dst_start)):
                    if weights[i, j] != 0:
                        adj_matrix[src_start + i, dst_start + j] = 1
                        adj_matrix[dst_start + j, src_start + i] = 1  # Undirected
        
        # Spectral clustering via Laplacian eigenvectors
        mapping = self._spectral_partition(adj_matrix, neuron_list, neurons_per_core)
        
        return mapping

    def _spectral_partition(self, adj_matrix: np.ndarray, neuron_list: List, neurons_per_core: int) -> Dict:
        """Recursively partition graph using Fiedler vector.
        
        Args:
            adj_matrix: Adjacency matrix
            neuron_list: List of (layer_name, idx) tuples
            neurons_per_core: Maximum neurons per core
            
        Returns:
            Mapping dictionary {layer_name: [core_id, ...]}
        """
        n = adj_matrix.shape[0]
        
        if n <= neurons_per_core:
            # Base case: all neurons fit on one core
            mapping = {}
            for layer_name, idx in neuron_list:
                if layer_name not in mapping:
                    mapping[layer_name] = []
                mapping[layer_name].append(0)
            return mapping
        
        # Compute graph Laplacian
        degree = np.sum(adj_matrix, axis=1)
        D = np.diag(degree)
        L = D -adj_matrix
        
        # Compute eigenvectors (use 2nd smallest for Fiedler)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            # Sort by eigenvalue
            idx = eigenvalues.argsort()
            fiedler = eigenvectors[:, idx[1]]  # 2nd eigenvector
        except:
            # Fallback to sequential if eigendecomposition fails
            return self._sequential_partition(neuron_list, neurons_per_core)
        
        # Bisect based on sign of Fiedler vector
        partition_mask = fiedler >= np.median(fiedler)
        
        # Split adjacency and neuron lists
        idx_0 = np.where(~partition_mask)[0]
        idx_1 = np.where(partition_mask)[0]
        
        if len(idx_0) == 0 or len(idx_1) == 0:
            # Degenerate partition, fall back
            return self._sequential_partition(neuron_list, neurons_per_core)
        
        adj_0 = adj_matrix[np.ix_(idx_0, idx_0)]
        adj_1 = adj_matrix[np.ix_(idx_1, idx_1)]
        
        neurons_0 = [neuron_list[i] for i in idx_0]
        neurons_1 = [neuron_list[i] for i in idx_1]
        
        # Recursive partition
        map_0 = self._spectral_partition(adj_0, neurons_0, neurons_per_core)
        map_1 = self._spectral_partition(adj_1, neurons_1, neurons_per_core)
        
        # Merge mappings with offset
        max_core = max(max(cores) for cores in map_0.values() if cores) if map_0 else -1
        for layer_name, cores in map_1.items():
            map_1[layer_name] = [c + max_core + 1 for c in cores]
        
        # Combine
        result = map_0
        for layer_name, cores in map_1.items():
            if layer_name in result:
                result[layer_name].extend(cores)
            else:
                result[layer_name] = cores
                
        return result

    def _sequential_partition(self, neuron_list: List, neurons_per_core: int) -> Dict:
        """Fallback sequential partition."""
        mapping = {}
        current_core = 0
        core_load = 0
        
        for layer_name, idx in neuron_list:
            if layer_name not in mapping:
                mapping[layer_name] = []
                
            mapping[layer_name].append(current_core)
            core_load += 1
            
            if core_load >= neurons_per_core:
                current_core += 1
                core_load = 0
                
        return mapping

class Compiler:
    """Transmorphic Compiler for hardware deployment."""
    def __init__(self, target: Profile):
        self.target = target
        self.mapper = Mapper(target)
        self.quantizer = Quantizer(bits=8)

    def compile(self, network: Any) -> bytes:
        """Compile network to hardware-specific binary format.
        
        Returns:
            Binary executable with header, layer metadata, and quantized weights
        """
        magic = b'NEMU'
        target_id = list(Profile).index(self.target)
        num_layers = len(network.layers)
        
        header = struct.pack('<4sBI', magic, target_id, num_layers)
        
        layer_headers = b''
        weights_blob = b''
        current_offset = 0
        
        for i, layer in enumerate(network.layers):
            n_neurons = getattr(layer, 'neurons', 0)
            
            w_bytes = b''
            if hasattr(layer, 'weights') and layer.weights is not None:
                q_w, scale = self.quantizer.quantize(layer.weights)
                w_bytes = struct.pack('f', scale) + q_w.tobytes()
            
            w_size = len(w_bytes)
            layer_headers += struct.pack('<IIQQ', i, n_neurons, current_offset, w_size)
            weights_blob += w_bytes
            current_offset += w_size
            
        return header + layer_headers + weights_blob
