# week 2 - gates and algorithms

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator, Aer
import matplotlib.pyplot as plt

# setup
sim = AerSimulator()
state_sim = Aer.get_backend('statevector_simulator')
sampler = Sampler()

# testing basic gates
# X gate - quantum NOT
x_circuit = QuantumCircuit(1, 1)
x_circuit.x(0)
x_circuit.measure(0, 0)

job = sampler.run(x_circuit, shots=1000)
result = job.result()
counts = result.quasi_dists[0]
print("X gate results:", counts)

# Z gate
z_circuit = QuantumCircuit(1, 1)
z_circuit.z(0)
z_circuit.measure(0, 0)

job = sampler.run(z_circuit, shots=1000)
result = job.result()
counts = result.quasi_dists[0]
print("Z gate results:", counts)

# Z gate only visible after H gate
hz_circuit = QuantumCircuit(1, 1)
hz_circuit.h(0)
hz_circuit.z(0)
hz_circuit.h(0)
hz_circuit.measure(0, 0)

job = sampler.run(hz_circuit, shots=1000)
result = job.result()
counts = result.quasi_dists[0]
print("H->Z->H results:", counts)

# Deutsch algorithm
def deutsch(f_type):
    qc = QuantumCircuit(2, 1)
    
    # prepare |01⟩
    qc.x(1)
    
    # apply H to both qubits
    qc.h(0)
    qc.h(1)
    
    # oracle function
    if f_type == "constant_0":
        # f(x) = 0 for all x
        pass
    elif f_type == "constant_1":
        # f(x) = 1 for all x
        qc.x(1)
        qc.z(0)
        qc.x(1)
    elif f_type == "balanced_id":
        # f(x) = x
        qc.cx(0, 1)
    elif f_type == "balanced_not":
        # f(x) = NOT x
        qc.x(0)
        qc.cx(0, 1)
        qc.x(0)
    
    # apply H to control qubit
    qc.h(0)
    
    # measure control qubit
    qc.measure(0, 0)
    
    return qc

print("\nDeutsch algorithm:")
function_types = ["constant_0", "constant_1", "balanced_id", "balanced_not"]

for f_type in function_types:
    circuit = deutsch(f_type)
    job = sampler.run(circuit, shots=1000)
    result = job.result()
    counts = result.quasi_dists[0]
    print(f"{f_type}: {counts}")

# Grover's search (2-qubit example)
def grover_2qubit():
    qc = QuantumCircuit(2, 2)
    
    # initial superposition
    qc.h(0)
    qc.h(1)
    
    # oracle - marks |11⟩ state
    qc.x(0)
    qc.x(1)
    qc.h(1)
    qc.cx(0, 1)
    qc.h(1)
    qc.x(0)
    qc.x(1)
    
    # diffusion - amplifies marked state
    qc.h(0)
    qc.h(1)
    qc.x(0)
    qc.x(1)
    qc.h(1)
    qc.cx(0, 1)
    qc.h(1)
    qc.x(0)
    qc.x(1)
    qc.h(0)
    qc.h(1)
    
    # measure
    qc.measure([0, 1], [0, 1])
    
    return qc

print("\nGrover's algorithm:")
grover_circuit = grover_2qubit()
job = sampler.run(grover_circuit, shots=1000)
result = job.result()
counts = result.quasi_dists[0]
print("Results:", counts)

# combining gates in different ways
test_circuit = QuantumCircuit(1, 1)
test_circuit.x(0)
test_circuit.z(0)
test_circuit.x(0)
test_circuit.measure(0, 0)

job = sampler.run(test_circuit, shots=1000)
result = job.result()
counts = result.quasi_dists[0]
print("\nX->Z->X results:", counts)