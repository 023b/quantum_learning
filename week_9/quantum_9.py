# week 9 - quantum computing language model (fixed)

import numpy as np
import matplotlib.pyplot as plt
import re
import random
import time
from collections import Counter

# For quantum simulation
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# ====================== QUANTUM TEXT CORPUS ======================
# A small corpus of quantum computing terms and descriptions

quantum_corpus = """
Quantum computing uses quantum bits or qubits.
Qubits can exist in superposition of states.
Quantum gates manipulate qubits in a quantum circuit.
The Hadamard gate creates superposition.
CNOT is a two-qubit gate that creates entanglement.
Quantum entanglement is a key quantum resource.
Quantum algorithms can solve certain problems faster than classical algorithms.
Grover's algorithm provides a quadratic speedup for search problems.
Shor's algorithm can factor large numbers exponentially faster.
Quantum error correction protects against decoherence.
Quantum key distribution enables secure communication.
The Bloch sphere represents a single qubit state.
Quantum supremacy is achieved when quantum computers outperform classical computers.
IBM, Google, and other companies are developing quantum computers.
Noise and decoherence are major challenges in quantum computing.
A quantum circuit consists of quantum gates applied to qubits.
Measurement collapses quantum states into classical bit values.
Quantum machine learning combines quantum computing and machine learning.
Quantum Fourier Transform is a key component in many quantum algorithms.
Quantum annealing is an approach used by D-Wave systems.
"""

# ====================== TEXT PROCESSING ======================

def preprocess_text(text):
    """Clean and tokenize text"""
    # Convert to lowercase
    text = text.lower()
    
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    
    # Remove punctuation and split into words
    words = re.findall(r'\b\w+\b', text)
    
    return words

def create_ngrams(words, n=2):
    """Create n-grams from a list of words"""
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(tuple(words[i:i+n]))
    return ngrams

def build_markov_model(ngrams):
    """Build a Markov model from n-grams"""
    model = {}
    for ngram in ngrams:
        prefix = ngram[:-1]
        suffix = ngram[-1]
        
        if prefix not in model:
            model[prefix] = []
        
        model[prefix].append(suffix)
    
    return model

def generate_text(model, start_words, length=20):
    """Generate text using a Markov model"""
    current = start_words
    result = list(current)
    
    for _ in range(length):
        if current in model:
            possible_next = model[current]
            next_word = random.choice(possible_next)
            result.append(next_word)
            # Update current tuple (remove first word and add next_word)
            current = current[1:] + (next_word,)
        else:
            # If we can't find the current n-gram, restart with a random n-gram
            current = random.choice(list(model.keys()))
            result.extend(current)
    
    return ' '.join(result)

# ====================== QUANTUM TEXT GENERATION ======================

def quantum_word_encoding(word, word_to_index, num_qubits):
    """Encode a word as a quantum state using qubit rotations"""
    # Get the index of the word in our vocabulary
    if word in word_to_index:
        index = word_to_index[word]
    else:
        index = 0  # Default for unknown words
    
    # Create a quantum circuit with measurement
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Convert index to binary representation
    binary = format(index, f'0{num_qubits}b')
    
    # Apply X gates for 1s in the binary representation
    for i, bit in enumerate(binary):
        if bit == '1':
            qc.x(i)
    
    # Add some 'quantum flavor' with superposition
    for i in range(num_qubits):
        qc.h(i)
    
    # Add some entanglement
    for i in range(num_qubits-1):
        qc.cx(i, i+1)
    
    # Add measurements to all qubits
    qc.measure(range(num_qubits), range(num_qubits))
    
    return qc

def quantum_sample_next_word(qc, model, word_to_index, index_to_word, num_qubits):
    """Use quantum measurement to sample the next word"""
    # Simulate the quantum circuit
    simulator = AerSimulator()
    job = simulator.run(qc, shots=100)
    result = job.result()
    counts = result.get_counts()
    
    # Get the most frequent bitstring
    max_bitstring = max(counts.items(), key=lambda x: x[1])[0]
    
    # Convert to an integer index (handling potential overflow)
    measured_index = int(max_bitstring, 2) % len(word_to_index)
    
    # Get the corresponding word
    current_word = index_to_word.get(measured_index, "quantum")  # Default if not found
    
    # Find possible next words from our model
    possible_next = []
    for prefix in model:
        if len(prefix) > 0 and prefix[-1] == current_word:
            possible_next.extend(model[prefix])
    
    # If no next words found, choose a random word
    if not possible_next:
        return random.choice(list(word_to_index.keys()))
    
    # Otherwise, choose based on quantum measurement probabilities
    probabilities = []
    for word in possible_next:
        # Create a simple probability based on word frequency
        prob = possible_next.count(word) / len(possible_next)
        probabilities.append(prob)
    
    # Normalize probabilities
    total = sum(probabilities)
    if total > 0:
        probabilities = [p/total for p in probabilities]
    else:
        # Equal probabilities if total is 0
        probabilities = [1.0/len(possible_next)] * len(possible_next)
    
    # Sample according to these probabilities
    return np.random.choice(possible_next, p=probabilities)

def generate_quantum_text(model, start_word, word_to_index, index_to_word, num_qubits, length=20):
    """Generate text using quantum circuits for sampling"""
    result = [start_word]
    current_word = start_word
    
    for _ in range(length):
        # Encode the current word into a quantum circuit
        qc = quantum_word_encoding(current_word, word_to_index, num_qubits)
        
        # Use quantum circuit to help sample the next word
        next_word = quantum_sample_next_word(qc, model, word_to_index, index_to_word, num_qubits)
        
        result.append(next_word)
        current_word = next_word
    
    return ' '.join(result)

# ====================== MAIN EXECUTION ======================

if __name__ == "__main__":
    print("Week 9: Quantum Computing Language Model")
    print("========================================")
    
    # Process text
    print("\nProcessing quantum computing corpus...")
    words = preprocess_text(quantum_corpus)
    print(f"Vocabulary size: {len(set(words))} unique words")
    
    # Create n-grams and Markov model
    n = 2  # Use bigrams
    ngrams = create_ngrams(words, n)
    model = build_markov_model(ngrams)
    
    # Create word-to-index mapping
    unique_words = sorted(set(words))
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    index_to_word = {i: word for i, word in enumerate(unique_words)}
    
    # Determine number of qubits needed to encode vocabulary
    num_qubits = int(np.ceil(np.log2(len(word_to_index))))
    print(f"Using {num_qubits} qubits to encode {len(word_to_index)} words")
    
    # Generate text using classical Markov model
    print("\n===== Classical Text Generation =====")
    start_tuple = random.choice(list(model.keys()))
    print(f"Starting with: {' '.join(start_tuple)}")
    
    classical_start_time = time.time()
    classical_text = generate_text(model, start_tuple, length=30)
    classical_time = time.time() - classical_start_time
    
    print("Generated text:")
    print(classical_text)
    print(f"Generation time: {classical_time:.4f} seconds")
    
    # Generate text using quantum-assisted approach
    print("\n===== Quantum-Assisted Text Generation =====")
    start_word = random.choice(unique_words)
    print(f"Starting with: {start_word}")
    
    quantum_start_time = time.time()
    quantum_text = generate_quantum_text(model, start_word, word_to_index, index_to_word, num_qubits, length=15)
    quantum_time = time.time() - quantum_start_time
    
    print("Generated text:")
    print(quantum_text)
    print(f"Generation time: {quantum_time:.4f} seconds")
    
    # Compare results
    print("\n===== Comparison =====")
    print(f"Classical generation time: {classical_time:.4f} seconds for 30 words")
    print(f"Quantum generation time: {quantum_time:.4f} seconds for 15 words")
    print(f"Quantum is {quantum_time/classical_time*2:.2f}x slower per word (due to simulation overhead)")
    
    # Analyze n-gram statistics
    print("\n===== Corpus Statistics =====")
    word_counts = Counter(words)
    top_words = word_counts.most_common(10)
    
    print("Top 10 most frequent words:")
    for word, count in top_words:
        print(f"'{word}': {count} occurrences")
    
    # Create a simple visualization of word frequencies
    plt.figure(figsize=(12, 6))
    words_for_plot, counts_for_plot = zip(*top_words)
    plt.bar(words_for_plot, counts_for_plot)
    plt.title('Most Common Words in Quantum Computing Corpus')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    # plt.show()
    
    print("\n===== Final Reflections =====")
    print("This project brought together several quantum computing concepts:")
    print("- Quantum circuits for encoding information")
    print("- Superposition and entanglement as quantum resources")
    print("- Quantum measurement for probabilistic sampling")
    print("- Simulation of quantum systems")
    print("\nA true quantum advantage in NLP would likely come from:")
    print("1. Quantum embedding of text in exponentially larger feature spaces")
    print("2. Quantum associative memory for efficient information retrieval")
    print("3. Quantum sampling for generative models")
    print("4. Quantum optimization for training language models")
    print("\nCurrent limitations:")
    print("- Small vocabulary size due to limited qubits")
    print("- Simple n-gram model rather than transformer architecture")
    print("- Simulation overhead makes classical approach faster")
    print("- Need for error correction on real quantum hardware")
    
    print("\nThank you for joining this quantum computing journey!")