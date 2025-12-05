import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import nemus
from nemus.biology import AdaptiveLIF
from nemus.morphology import Pruner, Synaptogenesis

# Mock Backend
class DroneBackend:
    def __init__(self, network):
        self.network = network
        self.synapse_history = []
        self.time = 0

    def step(self, events):
        # Mock forward
        self.time += 1
        return [0.5]*4

    def inject_dopamine(self, reward):
        liquid = self.network.layers[1]
        # Mock growth/pruning based on reward
        if reward < 0: # Grow
            zeros = np.where(liquid.rec_weights == 0)
            if len(zeros[0]) > 0:
                num = min(50, len(zeros[0]))
                idx = np.random.choice(len(zeros[0]), num)
                liquid.rec_weights[zeros[0][idx], zeros[1][idx]] = 0.01
        
        if self.time % 5 == 0: # Prune
            mask = np.random.rand(*liquid.rec_weights.shape) > 0.99
            liquid.rec_weights[mask] = 0

    def get_topology_stats(self):
        liquid = self.network.layers[1]
        count = np.count_nonzero(liquid.rec_weights)
        class Stats:
            def __init__(self, c): self.count = c
        return Stats(count)

# 1. Define Brain
print("Initializing Drone Brain...")
dvs = nemus.Input(shape=(128, 128))
morph = [Pruner(threshold=0.001), Synaptogenesis(growth_rate=0.01)]
liquid = nemus.Recurrent(neurons=100, model=AdaptiveLIF(neurons=100), morphology=morph, sparsity=0.95)
rotors = nemus.Output(classes=4)

net = nemus.Network(dvs >> liquid >> rotors)
controller = DroneBackend(net)

# 2. Flight Loop
print("Taking off...")
synapse_counts = []

for t in range(100):
    reward = 1.0 if (t < 30 or t > 70) else -0.5 + (t-30)*0.02
    if reward > 1.0: reward = 1.0
    
    controller.step([])
    controller.inject_dopamine(reward)
    
    if t % 10 == 0:
        stats = controller.get_topology_stats()
        print(f"Time {t}: Synapses: {stats.count}, Reward: {reward:.2f}")
        synapse_counts.append(stats.count)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(range(0, 100, 10), synapse_counts, marker='o')
plt.axvspan(30, 70, color='green', alpha=0.1, label="Forest Zone")
plt.xlabel("Time")
plt.ylabel("Synapses")
plt.title("Morphing Drone")
plt.legend()
plt.savefig("drone_plot.png")
print("Plot saved to drone_plot.png")
