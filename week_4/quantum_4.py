# week 4 - error correction and ibm quantum

from qiskit import QuantumCircuit, transpile
# Updated imports for noise model
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import pauli_error, depolarizing_error
import numpy as np
import matplotlib.pyplot as plt

# IBM Quantum - wrapped in try block in case package isn't installed
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    # For first time use, you need to save your token:
    QiskitRuntimeService.save_account(channel="ibm_quantum", token="cf70dacd26050c46d95405cbcc61b2ac64adc608235571dd81fa6ab1f31d8723893fb191dad37e6d528202fd949168ce5d2e0613ef0bf2f6609a62c459356ba3")
    service = QiskitRuntimeService(channel="ibm_quantum")
    print("Connected to IBM Quantum!")
    
    # list available backends
    print("Available quantum computers:")
    backends = service.backends()
    for backend in backends:
        print(f"- {backend.name}")
    
except ImportError:
    print("qiskit-ibm-runtime package not installed")
    print("To use IBM Quantum: pip install qiskit-ibm-runtime")
except Exception as e:
    print(f"Could not connect to IBM Quantum: {e}")
    print("Continuing with local simulation only...")

# noise model
noise_model = NoiseModel()

# bit flip error (X error)
bit_flip_prob = 0.05
bit_flip = pauli_error([('X', bit_flip_prob), ('I', 1 - bit_flip_prob)])
noise_model.add_all_qubit_quantum_error(bit_flip, ['x'])

# phase flip error (Z error)
phase_flip_prob = 0.05
phase_flip = pauli_error([('Z', phase_flip_prob), ('I', 1 - phase_flip_prob)])
noise_model.add_all_qubit_quantum_error(phase_flip, ['z'])

# depolarizing error (random X, Y, Z errors)
depol_prob = 0.02
depol = depolarizing_error(depol_prob, 1)
noise_model.add_all_qubit_quantum_error(depol, ['h'])

# setup local sim with noise
sim = AerSimulator(noise_model=noise_model)

# basic circuit with no error correction
basic = QuantumCircuit(1, 1)
basic.x(0)  # prepare |1⟩
basic.measure(0, 0)

# run with noise
job = transpile(basic, sim)
result = sim.run(job, shots=1000).result()
counts = result.get_counts()
print("\nNoisy circuit results:")
print(counts)
error_rate = counts.get('0', 0) / 1000
print(f"Error rate: {error_rate:.2%}")

# bit flip code
def bit_flip_code():
    qc = QuantumCircuit(3, 3)
    
    # encode logical |1⟩ state into 3 physical qubits
    qc.x(0)  # prepare |1⟩
    qc.cx(0, 1)  # copy to qubit 1
    qc.cx(0, 2)  # copy to qubit 2
    # now we have |111⟩
    
    # errors might happen here (in the noise model)
    
    # syndrome measurement
    qc.cx(0, 1)
    qc.cx(0, 2)
    
    # measure all qubits
    qc.measure([0, 1, 2], [0, 1, 2])
    
    return qc

qec = bit_flip_code()
print("\nBit-flip code circuit:")
print(qec.draw())

# run with noise
job = transpile(qec, sim)
result = sim.run(job, shots=1000).result()
counts = result.get_counts()
print("Bit-flip code results:")
print(counts)

# analyze results
total_correct = 0
for outcome, count in counts.items():
    # majority vote decoding
    ones = outcome.count('1')
    zeros = outcome.count('0')
    if ones > zeros:
        # should be |1⟩
        total_correct += count

success_rate = total_correct / 1000
print(f"Success rate with error correction: {success_rate:.2%}")

# phase flip code
def phase_flip_code():
    qc = QuantumCircuit(3, 3)
    
    # encode |+⟩ state
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    
    # Z error might happen here
    
    # syndrome measurement
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    
    # measure
    qc.measure([0, 1, 2], [0, 1, 2])
    
    return qc

phase_qec = phase_flip_code()
print("\nPhase-flip code circuit:")
print(phase_qec.draw())

# run phase flip code
job = transpile(phase_qec, sim)
result = sim.run(job, shots=1000).result()
counts = result.get_counts()
print("Phase-flip code results:")
print(counts)

# Shor code (protects against any single-qubit error)
def shor_code():
    qc = QuantumCircuit(9, 9)
    
    # encode logical |1⟩
    qc.x(0)
    
    # phase flip protection
    qc.h(0)
    qc.cx(0, 3)
    qc.cx(0, 6)
    
    # bit flip protection for each block
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(3, 4)
    qc.cx(3, 5)
    qc.cx(6, 7)
    qc.cx(6, 8)
    
    # errors might happen here
    
    # measure
    qc.measure(range(9), range(9))
    
    return qc

print("\nRunning a simple circuit locally with simulated noise...")
simple = QuantumCircuit(5, 5)
simple.h(0)
simple.cx(0, 1)
simple.cx(1, 2)
simple.cx(2, 3)
simple.cx(3, 4)
simple.measure(range(5), range(5))

# Run on local simulator with noise
job = transpile(simple, sim)
result = sim.run(job, shots=1000).result()
counts = result.get_counts()
print("Simulated noisy results:")
print(counts)

print("\nTo run on IBM Quantum:")
print("1. Install qiskit-ibm-runtime: pip install qiskit-ibm-runtime")
print("2. Get your token from IBM Quantum dashboard")
print("3. Uncomment and add your token to: QiskitRuntimeService.save_account(...)")