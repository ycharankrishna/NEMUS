import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import nemus
from nemus.biology import LIF
from nemus.learning import EligibilityProp
from nemus.compile import SiliconBackend

# 1. Define the Bio-Physical Dynamics
input_layer = nemus.Input(shape=(28, 28), encoding="latency")

# A "Liquid" Layer: Dynamic connectivity
liquid_layer = nemus.Recurrent(
    neurons=100, # Reduced from 1000 for faster demo
    model=LIF(neurons=100, tau_mem=0.02, threshold=1.0),
    sparsity=0.1
)

readout = nemus.Output(classes=10)

# 2. Build the Event-Manifold
# Connects layers. The backend builds the analytical event solver.
net = nemus.Network(input_layer >> liquid_layer >> readout)

# 3. Configure the Learner
optimizer = nemus.Optimizer(
    algorithm=EligibilityProp(
        learning_rate=0.001,
        trace_decay=0.05,
        feedback_alignment="symmetric"
    )
)
optimizer.attach(net) # Explicitly attach network

# 4. The Data Stream
data_stream = nemus.DVSGesture(stream=True)

# 5. The Training Loop
print("Initializing Chronos Engine...")

for i, (event_batch, target) in enumerate(data_stream):
    print(f"Batch {i+1}: Processing {len(event_batch)} events...")
    
    # Forward: Solves analytical time-to-spike
    spikes = net.forward(event_batch, mode="asynchronous")
    print(f"  > Output Spikes: {len(spikes)}")
    
    # Backward: Updates weights locally using eligibility traces
    loss = optimizer.step(spikes, target)
    print(f"  > Loss: {loss}")
    
    # Structural Plasticity check
    if net.stats['synapse_count'] < 10000:
        # print("Network is growing new synapses...")
        pass

# 6. Compilation to Hardware
compiler = nemus.Compiler(target=SiliconBackend.LOIHI_2)
executable = compiler.build(net, optimization_level="power_efficient")

executable.deploy(device_id=0)
print("Deployed. Network is now learning on-chip.")
