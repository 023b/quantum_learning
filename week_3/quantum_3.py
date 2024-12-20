# week 3 - quantum fourier transform and more

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt

# setup
sim = AerSimulator()
sampler = Sampler()

# quantum fourier transform
def qft(n):
    qc = QuantumCircuit(n)
    
    for i in range(n):
        qc.h(i)
        for j in range(i+1, n):
            # controlled phase rotation
            qc.cp(np.pi/2**(j-i), j, i)
    
    # swap qubits
    for i in range(n//2):
        qc.swap(i, n-i-1)
        
    return qc

# 3-qubit QFT
n = 3
qft_circuit = qft(n)
print("QFT circuit:")
print(qft_circuit.draw())

# prepare a state and apply QFT
test_qft = QuantumCircuit(3, 3)
test_qft.x(0)  # |001⟩ state
test_qft.barrier()

# Fix: use compose() instead of +=
qft_part = qft(3)
test_qft = test_qft.compose(qft_part)

test_qft.barrier()
test_qft.measure(range(3), range(3))

job = sampler.run(test_qft, shots=1000)
result = job.result()
qft_counts = result.quasi_dists[0]
print("\nQFT on |001⟩ results:", qft_counts)

# phase estimation algorithm
def phase_estimation():
    # 3 counting qubits + 1 eigenstate qubit
    qc = QuantumCircuit(4, 3)
    
    # prepare eigenstate
    qc.x(3)
    
    # prepare counting qubits
    for i in range(3):
        qc.h(i)
    
    # controlled unitary operations
    qc.p(np.pi/4, 3)  # phase = 1/8, binary 0.001
    qc.cp(np.pi/2, 0, 3)
    qc.cp(np.pi, 1, 3)
    qc.cp(2*np.pi, 2, 3)
    
    # inverse QFT
    qc.barrier()
    
    # Fix: use compose() instead of +=
    inverse_qft = qft(3).inverse()
    qc = qc.compose(inverse_qft, qubits=[0, 1, 2])
    
    # measure counting qubits
    qc.measure(range(3), range(3))
    
    return qc

pe_circuit = phase_estimation()
print("\nPhase Estimation circuit:")
print(pe_circuit.draw())

job = sampler.run(pe_circuit, shots=1000)
result = job.result()
pe_counts = result.quasi_dists[0]
print("Phase Estimation results:", pe_counts)
print("Should be mostly |001⟩ (binary for 1/8)")

# quantum teleportation
def teleport():
    qc = QuantumCircuit(3, 2)
    
    # create Bell state between qubits 1-2
    qc.h(1)
    qc.cx(1, 2)
    
    # prepare state to teleport on qubit 0
    qc.x(0)  # teleport |1⟩
    
    # Bell measurement between qubit 0 and 1
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    
    # correction operations - using classical conditions
    # Note: These operations may not work with the Sampler primitive directly
    # but we're showing the circuit structure
    qc.x(2).c_if(0, 1)
    qc.z(2).c_if(1, 1)
    
    return qc

# can't directly run the teleport circuit with conditional ops
# in newer Qiskit with Sampler, so just show the circuit
teleport_circuit = teleport()
print("\nTeleportation circuit:")
print(teleport_circuit.draw())