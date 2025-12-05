import numpy as np
from typing import Iterator, Tuple, List

class DVSGesture:
    """Event-based gesture dataset iterator.
    
    Generates synthetic DVS (Dynamic Vision Sensor) event streams
    with realistic temporal dynamics for neuromorphic testing.
    """
    def __init__(self, stream: bool = True, n_samples: int = 5):
        self.stream = stream
        self.n_samples = n_samples

    def __iter__(self) -> Iterator[Tuple[List[Tuple[float, int, int, int]], int]]:
        """Yields (event_batch, target_class).
        
        event_batch: list of (t, x, y, p) where:
            t: timestamp in seconds
            x, y: spatial coordinates
            p: polarity (0 or 1)
        """
        for i in range(self.n_samples):
            target_class = i % 10  # Classes 0-9
            
            # Generate temporally-structured event stream
            # Simulate a gesture: events cluster in space-time patterns
            num_events = np.random.poisson(150)  # Poisson noise
            events = []
            
            # Create spatial cluster centers (simulating moving objects)
            n_clusters = 3
            cluster_centers = [(np.random.randint(5, 23), np.random.randint(5, 23)) for _ in range(n_clusters)]
            
            for _ in range(num_events):
                # Time: exponential inter-arrival
                if not events:
                    t = np.random.exponential(0.001)
                else:
                    t = events[-1][0] + np.random.exponential(0.001)
                
                # Space: cluster around centers with Gaussian spread
                cluster_idx = np.random.choice(n_clusters)
                cx, cy = cluster_centers[cluster_idx]
                x = int(np.clip(cx + np.random.randn() * 3, 0, 27))
                y = int(np.clip(cy + np.random.randn() * 3, 0, 27))
                
                # Polarity: correlated with cluster
                p = cluster_idx % 2
                
                events.append((t, x, y, p))
                
                if t > 0.1:  # Max 100ms per sample
                    break
            
            # Sort by time (should already be sorted, but ensure)
            events.sort(key=lambda e: e[0])
            
            yield events, target_class
