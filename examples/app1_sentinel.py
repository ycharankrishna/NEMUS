import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import nemus

print("Initializing Sentinel Network...")

# Simplified approach: Direct weight initialization and bias currents
sensor = nemus.Input(shape=(16,), encoding="latency")

# CRITICAL: Use LIF with BIAS CURRENT to ensure continual spiking
cortex = nemus.Recurrent(
    neurons=32,
    model=nemus.LIF(
        neurons=32,
        tau_mem=0.03,
        threshold=0.2,  # Moderate threshold
        resistance=1.0
    ),
    plasticity=None  # Disable learning initially
)

# SET BIAS CURRENT (this is the key to keeping neurons alive)
cortex.model.current = np.random.uniform(0.15, 0.25, 32)  # Constant drive

recon = nemus.Output(classes=16)
net = nemus.Network(sensor >> cortex >> recon)

# Initialize with strong, clamped weights
cortex._init_weights(16)
cortex.weights = np.random.uniform(0.4, 0.9, cortex.weights.shape)

engine = nemus.ChronosEngine(net)

# Schedule INITIAL spike predictions for all neurons
print("Scheduling initial predictions...")
for idx in range(cortex.neurons):
    t_spike = cortex.model.predict_spike(idx, 0.0)
    if t_spike != float('inf'):
        engine.update_prediction(cortex.name, idx, t_spike)

# Storage
history_scores = []
history_time = []
spike_counts = []

np.random.seed(42)
normal_pattern = np.random.choice(16, 6, replace=False)
anomaly_pattern = np.random.choice(16, 12, replace=False)

print("Sentinel active. Running simulation...")

for step in range(50):
    t = step * 0.1
    
    if step % 10 == 0:
        print(f"Step {step}/50")
    
    # Pattern
    if step < 35:
        indices = normal_pattern
        pattern = "Normal"
    else:
        indices = anomaly_pattern
        pattern = "Anomaly"
    
    # Schedule input events with stronger payloads
    for idx in indices:
        evt = nemus.Event(
            timestamp=t + np.random.uniform(0, 0.005),
            priority=0,
            address=int(idx),
            payload=0.15,  # Moderate input strength
            layer_name=cortex.name
        )
        engine.schedule(evt)
    
    # Clear output
    recon.spikes = []
    
    # Run
    engine.run(duration=0.08)
    
    # Collect
    output_spikes = list(recon.spikes)
    spike_counts.append(len(output_spikes))
    
    # Every 5 steps, boost bias if network is quiet
    if step % 5 == 0 and len(output_spikes) < 5:
        cortex.model.current *= 1.05  # Gradually increase drive
        # Re-predict for quiet neurons
        for idx in range(cortex.neurons):
            if cortex.model.v[idx] < cortex.model.threshold * 0.5:
                t_spike = cortex.model.predict_spike(idx, engine.time)
                engine.update_prediction(cortex.name, idx, t_spike)
    
    # Clamp weights
    np.clip(cortex.weights, 0.3, 1.5, out=cortex.weights)
    
    # Score
    input_spikes = [(t, int(idx)) for idx in indices]
    score = nemus.van_rossum_distance(input_spikes, output_spikes)
    
    history_scores.append(score)
    history_time.append(t)
    
    if step % 10 == 0:
        avg_current = np.mean(cortex.model.current)
        avg_v = np.mean(cortex.model.v)
        print(f"  Spikes: {len(output_spikes)}, Avg I: {avg_current:.3f}, Avg V: {avg_v:.3f}")
    
    if step >= 30 and score > 4.0:
        print(f"[ALERT] Anomaly at t={t:.1f}s! Score: {score:.2f} ({pattern})")

# Visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Anomaly Score
axes[0].plot(history_time, history_scores, 'b-', linewidth=2.5, label="Reconstruction Error")
axes[0].axvline(x=3.5, color='r', linestyle='--', linewidth=2, label="Anomaly Start")
axes[0].set_xlabel("Time (s)", fontsize=12)
axes[0].set_ylabel("Van Rossum Distance", fontsize=12)
axes[0].set_title("Sentinel: Anomaly Detection", fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Network Activity
axes[1].plot(history_time, spike_counts, 'orange', linewidth=2.5, markersize=8, label="Output Spikes")
axes[1].axhline(y=10, color='green', linestyle=':', linewidth=2, alpha=0.6, label="Target (~10 spikes)")
axes[1].axhline(y=0, color='red', linestyle=':', linewidth=2, label="DEATH")
axes[1].set_xlabel("Time (s)", fontsize=12)
axes[1].set_ylabel("Spike Count per Batch", fontsize=12)
axes[1].set_title("Network Vitality - PROOF OF LIFE", fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(bottom=-5)

plt.tight_layout()
plt.savefig("sentinel_plot.png", dpi=150, bbox_inches='tight')

print("\n" + "="*70)
print("SENTINEL RESURRECTION CHECK")
print("="*70)
print(f"Average spikes per batch: {np.mean(spike_counts):.1f}")
print(f"Min spikes: {min(spike_counts)} | Max spikes: {max(spike_counts)}")
print(f"Batches with >0 spikes: {sum(1 for s in spike_counts if s > 0)}/50")
print(f"Network status: {'ALIVE!' if np.mean(spike_counts) > 2 else 'DEAD (still in coma)'}")
print(f"Score range: [{min(history_scores):.2f}, {max(history_scores):.2f}]")
print("="*70)
