from qiskit import QuantumCircuit  # Import the main class for quantum circuits
from qiskit.primitives import Sampler  # New way to execute circuits
from qiskit_aer import AerSimulator  # Simulator to mimic a real quantum computer
import matplotlib.pyplot as plt  # Might use later for visualizations

# === SIMPLE QUANTUM CIRCUIT ===
qc = QuantumCircuit(1, 1)  # One qubit, one classical bit
qc.h(0)  # Apply Hadamard gate to make superposition
qc.measure(0, 0)  # Measure the qubit

print("Quantum Circuit:")
print(qc.draw())  # Show the circuit

# === RUNNING THE CIRCUIT ===
sim = AerSimulator()  # Initialize simulator
sampler = Sampler()  # Qiskit 1.0+ method

job = sampler.run(qc, shots=1000)
result = job.result()
counts = result.quasi_dists[0]  # Get results
print("Results:", counts)

# === QUANTUM RANDOM NUMBER GENERATOR ===
def get_random(max_num):
    """
    Generates a quantum random number between 0 and max_num - 1.
    """
    qrng = QuantumCircuit(8, 8)  # 8 qubits, 8 classical bits
    
    for i in range(8):
        qrng.h(i)  # Put each qubit in superposition
    
    qrng.measure(range(8), range(8))  # Measure all qubits
    
    job = sampler.run(qrng, shots=1)
    result = job.result()
    bitstring = list(result.quasi_dists[0].keys())[0]  # Get a random bitstring
    
    return int(format(bitstring, '08b'), 2) % max_num  # Convert to integer

print("\nQuantum RNG:")
for _ in range(5):
    print(get_random(100))

# === QUANTUM DICE ===
def quantum_dice(sides=6):
    return get_random(sides) + 1  # Offset so it's 1-based

rolls = [quantum_dice() for _ in range(20)]
print("\nDice Rolls:", rolls)

# === BELL STATE (ENTANGLEMENT) ===
bell = QuantumCircuit(2, 2)  # Two qubits, two classical bits
bell.h(0)  # Hadamard gate on first qubit
bell.cx(0, 1)  # CNOT gate for entanglement
bell.measure([0, 1], [0, 1])  # Measure both qubits

print("\nBell State Circuit:")
print(bell.draw())

job = sampler.run(bell, shots=1000)
result = job.result()
bell_counts = result.quasi_dists[0]  # Get measurement results
print("\nBell State Results:", bell_counts)

# TODO: More quantum gates and algorithms
