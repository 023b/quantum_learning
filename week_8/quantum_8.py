# week 8 - simple quantum vs classical comparison

import time
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ====================== PROBLEM 1: DATABASE SEARCH ======================
# Classical: Linear search
# Quantum: Grover's algorithm concept (simplified)

def problem_1_search():
    print("\n===== PROBLEM 1: DATABASE SEARCH =====")
    print("Finding a specific item in an unsorted list")
    
    # List sizes to test
    sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    classical_times = []
    quantum_times = []
    
    for size in sizes:
        print(f"\nList size: {size} items")
        
        # Create a random list and pick a random item to find
        test_list = list(range(size))
        target = np.random.randint(0, size)
        print(f"Searching for item: {target}")
        
        # Classical linear search
        start_time = time.time()
        
        # Simple linear search
        found_index = -1
        for i in range(len(test_list)):
            if test_list[i] == target:
                found_index = i
                break
        
        classical_time = time.time() - start_time
        classical_times.append(classical_time)
        print(f"Classical search time: {classical_time:.6f} seconds")
        print(f"Classical complexity: O(N) = {size} operations")
        
        # Quantum search simulation
        # Grover's would use O(√N) operations
        # We'll simulate the time it would take
        start_time = time.time()
        
        # Instead of actually implementing Grover's,
        # we'll just take the square root of the operations
        # to simulate the quantum advantage
        grover_ops = int(np.sqrt(size))
        
        # Simulate the quantum time
        # Sleep for a time proportional to √N operations
        time.sleep(grover_ops * 0.001)  # Scale factor to make it visible
        
        quantum_time = time.time() - start_time
        quantum_times.append(quantum_time)
        print(f"Quantum search time (simulated): {quantum_time:.6f} seconds")
        print(f"Quantum complexity: O(√N) = {grover_ops} operations")
        
        # Calculate theoretical speedup
        speedup = size / grover_ops
        print(f"Theoretical speedup: {speedup:.2f}x")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, classical_times, 'ro-', label='Classical O(N)')
    plt.plot(sizes, quantum_times, 'bo-', label='Quantum O(√N)')
    plt.xlabel('List Size')
    plt.ylabel('Time (seconds)')
    plt.title('Search Time Comparison')
    plt.legend()
    plt.grid(True)
    # plt.show()
    
    return sizes, classical_times, quantum_times

# ====================== PROBLEM 2: QUANTUM SIMULATION ======================
# Classical: Exponential complexity
# Quantum: Native (linear) complexity

def problem_2_simulation():
    print("\n===== PROBLEM 2: QUANTUM STATE SIMULATION =====")
    print("Simulating a quantum system with N qubits")
    
    # Number of qubits to test
    qubit_counts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    classical_times = []
    quantum_times = []
    
    for n_qubits in qubit_counts:
        print(f"\nSystem size: {n_qubits} qubits")
        print(f"State space: 2^{n_qubits} = {2**n_qubits} dimensions")
        
        # Classical simulation time
        # For a classical computer, simulating a quantum system
        # requires exponential resources
        start_time = time.time()
        
        # Create and manipulate a state vector of size 2^n_qubits
        state_size = 2**n_qubits
        state = np.zeros(state_size, dtype=complex)
        state[0] = 1  # Initialize to |0...0⟩
        
        # Apply some random operations
        for _ in range(5):
            # Random matrix-vector multiplication
            # In practice, this would be applying gates to the state
            random_phases = np.exp(1j * np.random.random(state_size) * 2 * np.pi)
            state = state * random_phases
            state = state / np.linalg.norm(state)
        
        classical_time = time.time() - start_time
        classical_times.append(classical_time)
        print(f"Classical simulation time: {classical_time:.6f} seconds")
        print(f"Classical complexity: O(2^N) = {state_size} operations")
        
        # Quantum simulation (native)
        # On a quantum computer, simulating a quantum system
        # can be done with the same number of qubits
        start_time = time.time()
        
        # Create and run a quantum circuit
        qc = QuantumCircuit(n_qubits)
        
        # Apply some gates
        for i in range(n_qubits):
            qc.h(i)  # Hadamard on each qubit
        
        for i in range(n_qubits-1):
            qc.cx(i, i+1)  # CNOT between adjacent qubits
        
        # Instead of actually running on quantum hardware,
        # we'll just use a time proportional to N
        # to simulate the quantum advantage
        time.sleep(n_qubits * 0.01)  # Scale factor
        
        quantum_time = time.time() - start_time
        quantum_times.append(quantum_time)
        print(f"Quantum simulation time: {quantum_time:.6f} seconds")
        print(f"Quantum complexity: O(N) = {n_qubits} operations")
        
        # Calculate theoretical speedup
        speedup = state_size / n_qubits if n_qubits > 0 else float('inf')
        print(f"Theoretical speedup: {speedup:.2f}x")
    
    # Plot the results (log scale)
    plt.figure(figsize=(10, 6))
    plt.semilogy(qubit_counts, classical_times, 'ro-', label='Classical O(2^N)')
    plt.semilogy(qubit_counts, quantum_times, 'bo-', label='Quantum O(N)')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Time (seconds) - Log Scale')
    plt.title('Quantum System Simulation Comparison')
    plt.legend()
    plt.grid(True)
    # plt.show()
    
    return qubit_counts, classical_times, quantum_times

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    print("Week 8: Simple Quantum vs Classical Comparison")
    print("==============================================")
    print("This program demonstrates problems where quantum computers have a theoretical advantage")
    print("Note: We're simulating the time a quantum computer would take based on its complexity")
    print("      not actually using a quantum computer for these calculations")
    
    # Run the comparisons
    search_sizes, search_classical, search_quantum = problem_1_search()
    sim_qubits, sim_classical, sim_quantum = problem_2_simulation()
    
    # Create a summary plot
    plt.figure(figsize=(12, 10))
    
    # Search comparison - theoretical speedup
    plt.subplot(2, 1, 1)
    speedups = [size/np.sqrt(size) for size in search_sizes]
    plt.plot(search_sizes, speedups, 'go-')
    plt.xlabel('List Size')
    plt.ylabel('Speedup Factor')
    plt.title('Theoretical Quantum Speedup for Search')
    plt.grid(True)
    
    # Simulation comparison - theoretical speedup
    plt.subplot(2, 1, 2)
    speedups = [2**n/n for n in sim_qubits]
    plt.semilogy(sim_qubits, speedups, 'go-')
    plt.xlabel('Number of Qubits')
    plt.ylabel('Speedup Factor (Log Scale)')
    plt.title('Theoretical Quantum Speedup for Simulation')
    plt.grid(True)
    
    plt.tight_layout()
    # plt.show()
    
    print("\n===== SUMMARY =====")
    print("\nProblems where quantum computers are faster:")
    print("1. Searching unsorted databases (Grover's algorithm)")
    print("   - Classical: O(N) operations")
    print("   - Quantum: O(√N) operations")
    print("   - Example: Searching a database of 1 million items")
    print("     * Classical: ~1,000,000 operations")
    print("     * Quantum: ~1,000 operations (1000x faster)")
    
    print("\n2. Simulating quantum systems")
    print("   - Classical: O(2^N) operations")
    print("   - Quantum: O(N) operations")
    print("   - Example: Simulating a 50-qubit quantum system")
    print("     * Classical: ~10^15 operations (impossible!)")
    print("     * Quantum: ~50 operations (trillion+ times faster)")
    
    print("\nOther algorithms with quantum advantage (not shown):")
    print("- Shor's algorithm: Factoring large numbers")
    print("- Quantum Fourier Transform: Signal processing")
    print("- HHL algorithm: Solving linear systems")
    
    print("\nNote: This program simulates quantum advantage")
    print("Current quantum computers are still too small and noisy")
    print("to demonstrate these advantages in practice")