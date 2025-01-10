# week 6 - quantum neural networks and AI applications

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# setup simulator
sim = AerSimulator()
sampler = Sampler()

# Quantum Neural Network (QNN) implementation
class QuantumNeuralNetwork:
    def __init__(self, n_qubits, n_layers):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = 2 * n_qubits * n_layers
        self.params = np.random.random(self.n_params) * 2 * np.pi
        
    def create_circuit(self, x, params):
        qc = QuantumCircuit(self.n_qubits, 1)
        
        # Encode input data
        for i in range(self.n_qubits):
            if i < len(x):
                qc.rx(x[i], i)
        
        # Variational layers
        param_idx = 0
        for l in range(self.n_layers):
            # Rotation layer
            for i in range(self.n_qubits):
                qc.ry(params[param_idx], i)
                param_idx += 1
                
            # Entanglement layer
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            qc.cx(self.n_qubits - 1, 0)  # close the loop
            
            # Second rotation layer
            for i in range(self.n_qubits):
                qc.ry(params[param_idx], i)
                param_idx += 1
        
        # Measurement - use first qubit for classification
        qc.measure(0, 0)
        
        return qc
    
    def predict(self, x):
        qc = self.create_circuit(x, self.params)
        job = sampler.run(qc, shots=1000)
        result = job.result()
        counts = result.quasi_dists[0]
        return counts.get(1, 0)  # Probability of class 1
    
    def loss(self, X, y):
        total_loss = 0
        for i in range(len(X)):
            pred = self.predict(X[i])
            total_loss += (pred - y[i])**2
        return total_loss / len(X)
    
    def train(self, X, y, epochs=20, learning_rate=0.1):
        history = []
        
        for epoch in range(epochs):
            # Compute numerical gradients and update parameters
            grads = np.zeros_like(self.params)
            
            # Compute loss
            current_loss = self.loss(X[:5], y[:5])  # Using only first 5 samples to speed up
            history.append(current_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}, Loss: {current_loss:.4f}")
            
            # Calculate gradients (numerical approximation)
            for i in range(len(self.params)):
                eps = 0.01
                params_plus = self.params.copy()
                params_plus[i] += eps
                
                # Save original value
                original_value = self.params[i]
                
                # Update to params_plus
                self.params[i] = params_plus[i]
                loss_plus = self.loss(X[:5], y[:5])
                
                # Restore original value
                self.params[i] = original_value
                
                # Compute gradient
                grad = (loss_plus - current_loss) / eps
                grads[i] = grad
            
            # Update parameters
            self.params -= learning_rate * grads
        
        return history

# Load and prepare data - Iris dataset for binary classification (simplified)
iris = load_iris()
X = iris.data[:100, :2]  # Take only first 2 features for simplicity, first 2 classes
y = iris.target[:100]  # Binary classes (0, 1)

# Scale features to [0, π]
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title("Iris Dataset (2 classes)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
# plt.show()

# Quantum Transfer Learning - combining classical and quantum
def create_hybrid_model(n_features):
    # Classical preprocessing
    def classical_layer(x):
        # Simplified classical layer (just a linear transformation)
        W = np.array([
            [0.5, 0.8],
            [0.2, 0.7]
        ])
        b = np.array([0.1, -0.1])
        return np.dot(x, W) + b
    
    # Quantum circuit
    def quantum_layer(x, params):
        qc = QuantumCircuit(2, 1)
        
        # Encode preprocessed features
        qc.rx(x[0], 0)
        qc.rx(x[1], 1)
        
        # Variational layer
        qc.ry(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 0)
        qc.ry(params[3], 1)
        
        # Measure
        qc.measure(0, 0)
        
        return qc
    
    # Full hybrid model
    def hybrid_predict(x, params):
        # Apply classical layer
        x_preprocessed = classical_layer(x)
        
        # Apply quantum layer
        qc = quantum_layer(x_preprocessed, params)
        job = sampler.run(qc, shots=1000)
        result = job.result()
        counts = result.quasi_dists[0]
        return counts.get(1, 0)  # Probability of class 1
    
    return hybrid_predict

# Create QNN
print("Creating a Quantum Neural Network...")
qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2)

# Display example circuit
x_sample = X_train[0]
qc_example = qnn.create_circuit(x_sample, qnn.params)
print("\nExample QNN circuit:")
print(qc_example.draw())

# Train QNN (with very few samples and epochs for demonstration)
print("\nTraining the Quantum Neural Network (simplified)...")
history = qnn.train(X_train[:5], y_train[:5], epochs=2)

# Test QNN
print("\nTesting QNN on a few samples:")
correct = 0
for i in range(5):  # Test on just 5 samples for speed
    pred_prob = qnn.predict(X_test[i])
    pred_class = 1 if pred_prob > 0.5 else 0
    print(f"Sample {i+1}: True={y_test[i]}, Predicted={pred_class}, Probability={pred_prob:.4f}")
    if pred_class == y_test[i]:
        correct += 1

print(f"\nAccuracy on test samples: {correct/5:.2%}")

# Create and demonstrate hybrid model
print("\nDemonstrating Quantum Transfer Learning:")
hybrid_model = create_hybrid_model(n_features=2)
hybrid_params = np.random.random(4) * 2 * np.pi

# Test hybrid model
print("\nHybrid model predictions:")
for i in range(3):
    hybrid_pred = hybrid_model(X_test[i], hybrid_params)
    print(f"Sample {i+1}: Predicted probability={hybrid_pred:.4f}, True class={y_test[i]}")

# Quantum Reinforcement Learning concepts
print("\nQuantum Reinforcement Learning concepts:")
print("1. State encoding: Quantum states can represent complex probability distributions")
print("2. Action selection: Quantum measurements can provide randomized policies")
print("3. Q-Learning: Quantum circuits can approximate Q-functions")
print("4. Advantage: Potential for quadratic speedup in certain RL algorithms")

print("\nNote: This implementation is simplified for learning purposes.")
print("In practice, you would use specialized quantum ML libraries.")