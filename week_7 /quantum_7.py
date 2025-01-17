# week 7 - quantum vs classical machine learning (IBM Quantum Hardware)

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# For quantum part
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler as IBMSampler
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler as LocalSampler

# Create a dataset for binary classification
print("Generating dataset...")
X, y = make_circles(n_samples=1000, noise=0.15, factor=0.2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data for classical ML
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scale data to [0, 2π] for quantum encoding
q_scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
X_train_q = q_scaler.fit_transform(X_train)
X_test_q = q_scaler.transform(X_test)

# Plot dataset
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title("Binary Classification Dataset (Concentric Circles)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
# plt.show()

# ====================== CLASSICAL ML APPROACHES ======================

def run_classical_ml():
    print("\n===== CLASSICAL MACHINE LEARNING =====")
    results = {}
    
    # SVM Classifier
    print("Training SVM classifier...")
    start_time = time.time()
    
    # Training
    svm_train_start = time.time()
    svm = SVC(kernel='rbf', gamma='scale')
    svm.fit(X_train_scaled, y_train)
    svm_train_time = time.time() - svm_train_start
    
    # Prediction
    svm_predict_start = time.time()
    y_pred_svm = svm.predict(X_test_scaled)
    svm_predict_time = time.time() - svm_predict_start
    
    # Accuracy
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    svm_total_time = time.time() - start_time
    
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    print(f"SVM Training time: {svm_train_time:.6f} seconds")
    print(f"SVM Prediction time: {svm_predict_time:.6f} seconds")
    print(f"SVM Total time: {svm_total_time:.6f} seconds")
    
    results["svm"] = {
        "accuracy": svm_accuracy,
        "train_time": svm_train_time,
        "predict_time": svm_predict_time,
        "total_time": svm_total_time
    }
    
    # Neural Network Classifier
    print("\nTraining Neural Network classifier...")
    start_time = time.time()
    
    # Training
    nn_train_start = time.time()
    nn = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42)
    nn.fit(X_train_scaled, y_train)
    nn_train_time = time.time() - nn_train_start
    
    # Prediction
    nn_predict_start = time.time()
    y_pred_nn = nn.predict(X_test_scaled)
    nn_predict_time = time.time() - nn_predict_start
    
    # Accuracy
    nn_accuracy = accuracy_score(y_test, y_pred_nn)
    nn_total_time = time.time() - start_time
    
    print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
    print(f"Neural Network Training time: {nn_train_time:.6f} seconds")
    print(f"Neural Network Prediction time: {nn_predict_time:.6f} seconds")
    print(f"Neural Network Total time: {nn_total_time:.6f} seconds")
    
    results["nn"] = {
        "accuracy": nn_accuracy,
        "train_time": nn_train_time,
        "predict_time": nn_predict_time,
        "total_time": nn_total_time
    }
    
    return results

# ====================== QUANTUM ML APPROACH (IBM HARDWARE) ======================

def run_quantum_ml(use_real_quantum=True):
    print("\n===== QUANTUM MACHINE LEARNING =====")
    
    # Setup quantum backend
    if use_real_quantum:
        try:
            # Initialize the IBM Quantum service
            # Get your token from the IBM Quantum dashboard and save it first time:
            QiskitRuntimeService.save_account(channel="ibm_quantum", token="cf70dacd26050c46d95405cbcc61b2ac64adc608235571dd81fa6ab1f31d8723893fb191dad37e6d528202fd949168ce5d2e0613ef0bf2f6609a62c459356ba3")
            
            print("Connecting to IBM Quantum...")
            service = QiskitRuntimeService()
            
            # List available backends
            backends = service.backends()
            print("Available backends:")
            for backend in backends:
                print(f"- {backend.name}")
            
            # Select a backend - start with a simulator, then try real hardware
            backend_name = "ibmq_qasm_simulator"  # You can change this to a real quantum computer
            print(f"Using backend: {backend_name}")
            
            # Create a sampler from the IBM backend
            sampler = IBMSampler(backend=backend_name)
            
        except Exception as e:
            print(f"Error connecting to IBM Quantum: {e}")
            print("Falling back to local simulator...")
            use_real_quantum = False
    
    if not use_real_quantum:
        print("Using local quantum simulator...")
        sampler = LocalSampler()
    
    # Quantum feature map (encoding classical data into quantum state)
    def quantum_feature_map(x):
        qc = QuantumCircuit(2)
        
        # Encode 2 features
        qc.rx(x[0], 0)
        qc.rx(x[1], 1)
        
        # Entangle
        qc.cx(0, 1)
        
        # Second rotation with product encoding
        qc.rz(x[0] * x[1], 0)
        qc.rz(x[0] * x[1], 1)
        
        # More entanglement
        qc.cx(1, 0)
        
        return qc
    
    # Variational quantum classifier
    class QuantumClassifier:
        def __init__(self, sampler, n_qubits=2):
            self.n_qubits = n_qubits
            self.params = np.random.random(6) * 2 * np.pi  # 6 variational parameters
            self.feature_map = quantum_feature_map
            self.sampler = sampler
            self.is_ibm = isinstance(sampler, IBMSampler)
            
        def variational_circuit(self, params):
            qc = QuantumCircuit(self.n_qubits)
            
            # First rotation layer
            qc.ry(params[0], 0)
            qc.ry(params[1], 1)
            
            # Entanglement
            qc.cx(0, 1)
            
            # Second rotation layer
            qc.ry(params[2], 0)
            qc.ry(params[3], 1)
            
            # Entanglement
            qc.cx(0, 1)
            
            # Final rotation layer
            qc.ry(params[4], 0)
            qc.ry(params[5], 1)
            
            return qc
            
        def circuit(self, x):
            # Create quantum circuit for classification
            qc = QuantumCircuit(self.n_qubits, 1)
            
            # Add feature map
            feature_map = self.feature_map(x)
            qc = qc.compose(feature_map)
            
            # Add variational part
            var_circuit = self.variational_circuit(self.params)
            qc = qc.compose(var_circuit)
            
            # Measure first qubit for classification
            qc.measure(0, 0)
            
            return qc
            
        def predict_sample(self, x):
            qc = self.circuit(x)
            
            # IBM Quantum has a different API
            if self.is_ibm:
                job = self.sampler.run(circuits=[qc], shots=1000)
                result = job.result()
                counts = result.quasi_dists[0]
            else:
                job = self.sampler.run(qc, shots=1000)
                result = job.result()
                counts = result.quasi_dists[0]
            
            # If probability of measuring |1⟩ > 0.5, predict class 1
            prob_1 = counts.get(1, 0)
            return 1 if prob_1 > 0.5 else 0
            
        def predict(self, X):
            y_pred = []
            for x in X:
                y_pred.append(self.predict_sample(x))
            return np.array(y_pred)
            
        def train(self, X, y, epochs=10, learning_rate=0.1):
            # Use a very small subset for training if using real quantum hardware
            subset_size = 5 if self.is_ibm else 20
            print(f"Training quantum model with {subset_size} samples for {epochs} epochs...")
            
            X_subset = X[:subset_size]
            y_subset = y[:subset_size]
            
            # Simple training loop
            for epoch in range(epochs):
                correct = 0
                for i, (x, target) in enumerate(zip(X_subset, y_subset)):
                    # Make prediction
                    pred = self.predict_sample(x)
                    
                    # If incorrect, update parameters slightly
                    if pred != target:
                        # Simple parameter update - just perturb params in random direction
                        self.params += np.random.normal(0, learning_rate, size=self.params.shape)
                    else:
                        correct += 1
                
                acc = correct / len(X_subset)
                print(f"Epoch {epoch}: Accuracy = {acc:.4f}")
    
    # Run the quantum classifier
    print("Setting up quantum classifier...")
    start_time = time.time()
    
    # Create and train model
    q_train_start = time.time()
    qc = QuantumClassifier(sampler)
    
    # Shorter training for real quantum hardware
    epochs = 3 if use_real_quantum else 10
    qc.train(X_train_q, y_train, epochs=epochs, learning_rate=0.1)
    q_train_time = time.time() - q_train_start
    
    # Test on a smaller subset
    test_subset_size = 10 if use_real_quantum else 50
    X_test_subset = X_test_q[:test_subset_size]
    y_test_subset = y_test[:test_subset_size]
    
    # Make predictions
    q_predict_start = time.time()
    y_pred_q = qc.predict(X_test_subset)
    q_predict_time = time.time() - q_predict_start
    
    # Accuracy
    q_accuracy = accuracy_score(y_test_subset, y_pred_q)
    q_total_time = time.time() - start_time
    
    # Hardware type
    hw_type = "IBM Quantum Hardware" if use_real_quantum else "Local Quantum Simulator"
    
    print(f"\nQuantum Classifier ({hw_type}) Accuracy: {q_accuracy:.4f} (on {test_subset_size} test samples)")
    print(f"Quantum Training time: {q_train_time:.6f} seconds")
    print(f"Quantum Prediction time: {q_predict_time:.6f} seconds")
    print(f"Quantum Total time: {q_total_time:.6f} seconds")
    
    results = {
        "accuracy": q_accuracy,
        "train_time": q_train_time,
        "predict_time": q_predict_time,
        "total_time": q_total_time,
        "test_size": test_subset_size,
        "hardware": hw_type
    }
    
    return results

# ====================== VISUALIZATION ======================

def visualize_comparison(classical_results, quantum_results):
    # Accuracies
    methods = ['SVM', 'Neural Network', 'Quantum']
    accuracies = [
        classical_results["svm"]["accuracy"],
        classical_results["nn"]["accuracy"],
        quantum_results["accuracy"]
    ]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(methods, accuracies)
    
    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy Comparison')
    plt.grid(axis='y')
    # plt.show()
    
    # Training times
    train_times = [
        classical_results["svm"]["train_time"],
        classical_results["nn"]["train_time"],
        quantum_results["train_time"]
    ]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(methods, train_times)
    
    # Add time values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.4f}s', ha='center', va='bottom')
    
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison (log scale)')
    plt.yscale('log')
    plt.grid(axis='y')
    # plt.show()
    
    # Prediction times
    predict_times = [
        classical_results["svm"]["predict_time"],
        classical_results["nn"]["predict_time"],
        quantum_results["predict_time"]
    ]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(methods, predict_times)
    
    # Add time values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{height:.4f}s', ha='center', va='bottom')
    
    plt.ylabel('Prediction Time (seconds)')
    plt.title('Prediction Time Comparison (log scale)')
    plt.yscale('log')
    plt.grid(axis='y')
    # plt.show()
    
    # Summary
    print("\n====================== COMPARISON SUMMARY ======================")
    print("Classification accuracy:")
    print(f"SVM: {classical_results['svm']['accuracy']:.4f}")
    print(f"Neural Network: {classical_results['nn']['accuracy']:.4f}")
    print(f"Quantum Classifier ({quantum_results['hardware']}): {quantum_results['accuracy']:.4f} (on {quantum_results['test_size']} samples)")
    
    print("\nTraining time:")
    print(f"SVM: {classical_results['svm']['train_time']:.6f} seconds")
    print(f"Neural Network: {classical_results['nn']['train_time']:.6f} seconds")
    print(f"Quantum Classifier: {quantum_results['train_time']:.6f} seconds")
    
    print("\nPrediction time:")
    print(f"SVM: {classical_results['svm']['predict_time']:.6f} seconds")
    print(f"Neural Network: {classical_results['nn']['predict_time']:.6f} seconds")
    print(f"Quantum Classifier: {quantum_results['predict_time']:.6f} seconds")
    
    print("\nTotal processing time:")
    print(f"SVM: {classical_results['svm']['total_time']:.6f} seconds")
    print(f"Neural Network: {classical_results['nn']['total_time']:.6f} seconds")
    print(f"Quantum Classifier: {quantum_results['total_time']:.6f} seconds")
    
    print("\nNOTES:")
    print(f"1. The quantum classifier used {quantum_results['hardware']}.")
    print(f"2. The quantum classifier was trained on a very small subset of data")
    print(f"   while classical methods used the full dataset.")
    print("3. Queue times for IBM Quantum hardware can be substantial and are included in the timing.")
    print("4. Current quantum hardware has limitations in circuit depth and qubit count.")
    print("5. This comparison is meant to demonstrate the current state of quantum ML,")
    print("   not to claim superiority of one approach over the other.")

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    print("Week 7: Classical vs Quantum Machine Learning Comparison")
    print("======================================================")
    print("Task: Binary classification on the concentric circles dataset")
    
    # Ask if user wants to use IBM Quantum hardware
    use_real_quantum = False
    try:
        response = input("Do you want to use IBM Quantum hardware? (y/n): ")
        use_real_quantum = response.lower() == 'y'
    except:
        print("Using local simulator by default")
    
    # Run classical ML methods
    classical_results = run_classical_ml()
    
    # Run quantum ML method
    quantum_results = run_quantum_ml(use_real_quantum)
    
    # Compare and visualize results
    visualize_comparison(classical_results, quantum_results)