# week 5 - quantum machine learning

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# create a simple dataset
X, y = make_moons(n_samples=100, noise=0.3, random_state=42)

# scale data to range [0, 2π] for encoding into qubits
scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
X_scaled = scaler.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# plot the data
plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title("Dataset for Quantum Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
# plt.show()

# quantum feature map - encodes classical data into quantum state
def create_feature_map(x):
    fm = QuantumCircuit(2)
    # encode 2 features into 2 qubits
    fm.rx(x[0], 0)
    fm.rx(x[1], 1)
    # entangle features
    fm.cx(0, 1)
    fm.cx(1, 0)
    # second rotation
    fm.rx(x[0], 0)
    fm.rx(x[1], 1)
    return fm

# variational quantum classifier
def create_variational_circuit(params):
    vc = QuantumCircuit(2)
    # first rotation layer
    vc.ry(params[0], 0)
    vc.ry(params[1], 1)
    # entanglement
    vc.cx(0, 1)
    # second rotation layer
    vc.ry(params[2], 0)
    vc.ry(params[3], 1)
    return vc

# full quantum circuit for classification
def create_classifier_circuit(x, params):
    qc = QuantumCircuit(2, 1)
    
    # encode data
    fm = create_feature_map(x)
    qc.compose(fm, inplace=True)
    
    # variational part (trainable)
    vc = create_variational_circuit(params)
    qc.compose(vc, inplace=True)
    
    # measure qubit 0 for classification
    qc.measure(0, 0)
    
    return qc

# simulate the circuit
simulator = AerSimulator()

# random initial parameters
initial_params = np.random.random(4) * 2 * np.pi

# predict for a single datapoint
def predict_point(x, params):
    qc = create_classifier_circuit(x, params)
    result = simulator.run(qc, shots=1000).result()
    counts = result.get_counts()
    return counts.get('1', 0) / 1000  # probability of class 1

# predict for a few test points
print("\nPredictions with initial random parameters:")
for i in range(3):
    prob = predict_point(X_test[i], initial_params)
    print(f"Point {i}: true label={y_test[i]}, predicted prob={prob:.4f}")

# simplified gradient descent training (not using real VQE)
def train_classifier(X_train, y_train, params, epochs=20, lr=0.1):
    trajectory = [params.copy()]
    
    for epoch in range(epochs):
        gradients = np.zeros_like(params)
        total_loss = 0
        
        # compute loss and gradients for each training point
        for x, y_true in zip(X_train, y_train):
            # forward pass
            y_pred = predict_point(x, params)
            
            # simple MSE loss
            loss = (y_pred - y_true) ** 2
            total_loss += loss
            
            # compute numerical gradients
            for i in range(len(params)):
                eps = 0.01
                params_plus = params.copy()
                params_plus[i] += eps
                y_pred_plus = predict_point(x, params_plus)
                
                # finite difference
                grad = (y_pred_plus - y_pred) / eps
                gradients[i] += 2 * (y_pred - y_true) * grad
        
        # update parameters
        params = params - lr * gradients
        trajectory.append(params.copy())
        
        avg_loss = total_loss / len(X_train)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return params, trajectory

# too expensive to run in full, so just simulate 2 epochs with 5 data points
print("\nTraining quantum classifier (simplified):")
trained_params, trajectory = train_classifier(X_train[:5], y_train[:5], initial_params, epochs=2)

print("\nInitial parameters:", initial_params)
print("Trained parameters:", trained_params)

# evaluate on test set
correct = 0
for x, y_true in zip(X_test, y_test):
    y_pred = predict_point(x, trained_params)
    predicted_class = 1 if y_pred > 0.5 else 0
    if predicted_class == y_true:
        correct += 1

accuracy = correct / len(X_test)
print(f"\nTest accuracy: {accuracy:.2%}")

# quantum kernel - another approach to quantum ML
def quantum_kernel(x1, x2):
    # create circuit for x1
    qc1 = QuantumCircuit(2)
    qc1.rx(x1[0], 0)
    qc1.rx(x1[1], 1)
    qc1.cx(0, 1)
    
    # create circuit for x2
    qc2 = QuantumCircuit(2)
    qc2.rx(x2[0], 0)
    qc2.rx(x2[1], 1)
    qc2.cx(0, 1)
    
    # create hadamard test circuit
    qc = QuantumCircuit(3, 1)
    qc.h(0)
    # controlled-qc1
    qc.cx(0, 1)
    qc.rx(x1[0], 1)
    qc.rx(x1[1], 2)
    qc.cx(1, 2)
    # controlled-qc2†
    qc.cx(0, 1)
    qc.rx(-x2[0], 1)
    qc.rx(-x2[1], 2)
    qc.cx(1, 2)
    
    qc.h(0)
    qc.measure(0, 0)
    
    # simulate
    result = simulator.run(qc, shots=1000).result()
    counts = result.get_counts()
    
    # compute kernel value
    kernel_val = counts.get('0', 0) / 1000
    return kernel_val

# demonstrate quantum kernel
print("\nQuantum kernel example:")
for i in range(2):
    for j in range(2):
        k_val = quantum_kernel(X_test[i], X_test[j])
        print(f"K({i},{j}) = {k_val:.4f}")

print("\nThis is a simplified implementation for learning purposes.")
print("In practice, you would use qiskit's quantum kernel library and proper optimizers.")