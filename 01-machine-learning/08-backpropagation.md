# Backpropagation

## Introduction

Backpropagation is the fundamental algorithm for training neural networks. It efficiently computes gradients of the cost function with respect to all parameters using the chain rule of calculus. This algorithm enables neural networks to learn from data by updating weights and biases in the direction that reduces the cost function.

## Mathematical Foundation

### The Chain Rule

The chain rule is the mathematical foundation of backpropagation:

$$\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$$

For composite functions:
$$\frac{d}{dx}f(g(h(x))) = f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)$$

### Forward Pass

For a neural network with $L$ layers:

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = \sigma^{(l)}(z^{(l)})$$

Where:
- $z^{(l)}$ is the weighted input to layer $l$
- $a^{(l)}$ is the activation of layer $l$
- $W^{(l)}$ is the weight matrix for layer $l$
- $b^{(l)}$ is the bias vector for layer $l$
- $\sigma^{(l)}$ is the activation function for layer $l$

### Backward Pass

The key insight is to compute gradients backward through the network:

$$\delta^{(l)} = \frac{\partial J}{\partial z^{(l)}}$$

**Output Layer:**
$$\delta^{(L)} = \frac{\partial J}{\partial a^{(L)}} \cdot \sigma'^{(L)}(z^{(L)})$$

**Hidden Layers:**
$$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \cdot \sigma'^{(l)}(z^{(l)})$$

**Parameter Gradients:**
$$\frac{\partial J}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$$
$$\frac{\partial J}{\partial b^{(l)}} = \delta^{(l)}$$

## Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU
            if activation == 'relu':
                w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i])
            else:
                w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.01
            b = np.zeros((layer_sizes[i+1], 1))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def forward(self, X):
        """Forward pass through the network"""
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], self.activations[-1]) + self.biases[i]
            self.z_values.append(z)
            
            if i == len(self.weights) - 1:
                # Output layer - use softmax for classification
                a = self.softmax(z)
            else:
                # Hidden layers
                if self.activation == 'relu':
                    a = self.relu(z)
                elif self.activation == 'sigmoid':
                    a = self.sigmoid(z)
                elif self.activation == 'tanh':
                    a = self.tanh(z)
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def compute_cost(self, y_true, y_pred):
        """Compute cross-entropy cost"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))
    
    def backward(self, X, y_true, y_pred):
        """Backward pass (backpropagation)"""
        m = X.shape[1]
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient
        # For softmax + cross-entropy, the gradient simplifies to:
        dz = y_pred - y_true
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradients for weights and biases
            dW[i] = np.dot(dz, self.activations[i].T) / m
            db[i] = np.sum(dz, axis=1, keepdims=True) / m
            
            if i > 0:
                # Gradient for previous layer
                da = np.dot(self.weights[i].T, dz)
                
                # Gradient for z (before activation)
                if self.activation == 'relu':
                    dz = da * self.relu_derivative(self.z_values[i-1])
                elif self.activation == 'sigmoid':
                    dz = da * self.sigmoid_derivative(self.z_values[i-1])
                elif self.activation == 'tanh':
                    dz = da * self.tanh_derivative(self.z_values[i-1])
        
        return dW, db
    
    def update_parameters(self, dW, db):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def train(self, X, y, epochs=1000, batch_size=32, verbose=True):
        """Train the neural network"""
        costs = []
        
        for epoch in range(epochs):
            # Mini-batch training
            for i in range(0, X.shape[1], batch_size):
                X_batch = X[:, i:i+batch_size]
                y_batch = y[:, i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Backward pass
                dW, db = self.backward(X_batch, y_batch, y_pred)
                
                # Update parameters
                self.update_parameters(dW, db)
            
            # Compute cost
            if epoch % 100 == 0:
                y_pred_full = self.forward(X)
                cost = self.compute_cost(y, y_pred_full)
                costs.append(cost)
                if verbose:
                    print(f"Epoch {epoch}, Cost: {cost:.6f}")
        
        return costs
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return np.argmax(output, axis=0)

# Generate classification data
X, y = make_classification(n_samples=1000, n_features=2, n_classes=3, 
                          n_clusters_per_class=1, n_redundant=0, 
                          random_state=42)

# Convert to one-hot encoding
y_onehot = np.zeros((y.size, 3))
y_onehot[np.arange(y.size), y] = 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).T
X_test_scaled = scaler.transform(X_test).T
y_train_scaled = y_train.T
y_test_scaled = y_test.T

# Create and train neural network
nn = NeuralNetwork([2, 10, 10, 3], activation='relu', learning_rate=0.01)
costs = nn.train(X_train_scaled, y_train_scaled, epochs=1000, batch_size=32)

# Plot training progress
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(costs)
plt.xlabel('Epoch (x100)')
plt.ylabel('Cost')
plt.title('Training Cost')
plt.grid(True)

# Make predictions
y_pred = nn.predict(X_test_scaled)
y_test_labels = np.argmax(y_test_scaled.T, axis=1)

# Plot decision boundaries
plt.subplot(1, 3, 2)
x_min, x_max = X_test_scaled[0, :].min() - 1, X_test_scaled[0, :].max() + 1
y_min, y_max = X_test_scaled[1, :].min() - 1, X_test_scaled[1, :].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Create grid for prediction
grid_points = np.c_[xx.ravel(), yy.ravel()].T
grid_predictions = nn.predict(grid_points)
grid_predictions = grid_predictions.reshape(xx.shape)

plt.contourf(xx, yy, grid_predictions, alpha=0.4)
plt.scatter(X_test_scaled[0, :], X_test_scaled[1, :], c=y_test_labels, alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundaries')
plt.colorbar()

# Confusion matrix
plt.subplot(1, 3, 3)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_labels, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1, 2])
plt.yticks([0, 1, 2])

plt.tight_layout()
plt.show()

# Evaluate performance
accuracy = accuracy_score(y_test_labels, y_pred)
print(f"Test Accuracy: {accuracy:.3f}")
```

## Gradient Computation Visualization

```python
def visualize_gradients():
    """Visualize gradient computation in backpropagation"""
    
    # Simple 2-layer network for visualization
    class SimpleNN:
        def __init__(self):
            self.W1 = np.array([[0.5, 0.3], [0.2, 0.8]])
            self.b1 = np.array([[0.1], [0.2]])
            self.W2 = np.array([[0.4, 0.6]])
            self.b2 = np.array([[0.3]])
        
        def forward(self, x):
            self.z1 = np.dot(self.W1, x) + self.b1
            self.a1 = self.sigmoid(self.z1)
            self.z2 = np.dot(self.W2, self.a1) + self.b2
            self.a2 = self.sigmoid(self.z2)
            return self.a2
        
        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_derivative(self, x):
            s = self.sigmoid(x)
            return s * (1 - s)
        
        def backward(self, x, y_true, y_pred):
            # Output layer gradient
            dz2 = (y_pred - y_true) * self.sigmoid_derivative(self.z2)
            
            # Hidden layer gradient
            da1 = np.dot(self.W2.T, dz2)
            dz1 = da1 * self.sigmoid_derivative(self.z1)
            
            # Parameter gradients
            dW2 = np.dot(dz2, self.a1.T)
            db2 = dz2
            dW1 = np.dot(dz1, x.T)
            db1 = dz1
            
            return dW1, db1, dW2, db2
    
    # Test with simple example
    nn_simple = SimpleNN()
    x = np.array([[0.5], [0.3]])
    y_true = np.array([[1.0]])
    
    # Forward pass
    y_pred = nn_simple.forward(x)
    
    # Backward pass
    dW1, db1, dW2, db2 = nn_simple.backward(x, y_true, y_pred)
    
    # Visualize
    plt.figure(figsize=(15, 10))
    
    # Network architecture
    plt.subplot(2, 3, 1)
    plt.text(0.5, 0.8, 'Input Layer', ha='center', va='center', fontsize=12)
    plt.text(0.5, 0.5, 'Hidden Layer', ha='center', va='center', fontsize=12)
    plt.text(0.5, 0.2, 'Output Layer', ha='center', va='center', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Network Architecture')
    plt.axis('off')
    
    # Forward pass values
    plt.subplot(2, 3, 2)
    plt.text(0.1, 0.8, f'x: {x.flatten()}', fontsize=10)
    plt.text(0.1, 0.6, f'z1: {nn_simple.z1.flatten()}', fontsize=10)
    plt.text(0.1, 0.4, f'a1: {nn_simple.a1.flatten()}', fontsize=10)
    plt.text(0.1, 0.2, f'z2: {nn_simple.z2.flatten()}', fontsize=10)
    plt.text(0.1, 0.0, f'y_pred: {y_pred.flatten()}', fontsize=10)
    plt.title('Forward Pass Values')
    plt.axis('off')
    
    # Backward pass gradients
    plt.subplot(2, 3, 3)
    plt.text(0.1, 0.8, f'dz2: {dW2.flatten()}', fontsize=10)
    plt.text(0.1, 0.6, f'da1: {db2.flatten()}', fontsize=10)
    plt.text(0.1, 0.4, f'dz1: {dW1.flatten()}', fontsize=10)
    plt.text(0.1, 0.2, f'dW1: {dW1.flatten()}', fontsize=10)
    plt.text(0.1, 0.0, f'dW2: {dW2.flatten()}', fontsize=10)
    plt.title('Backward Pass Gradients')
    plt.axis('off')
    
    # Weight matrices
    plt.subplot(2, 3, 4)
    plt.imshow(nn_simple.W1, cmap='viridis')
    plt.title('W1 Matrix')
    plt.colorbar()
    
    plt.subplot(2, 3, 5)
    plt.imshow(nn_simple.W2, cmap='viridis')
    plt.title('W2 Matrix')
    plt.colorbar()
    
    # Gradient matrices
    plt.subplot(2, 3, 6)
    plt.imshow(dW1, cmap='coolwarm', center=0)
    plt.title('dW1 Gradients')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    print("Gradient Computation Summary:")
    print(f"Input: {x.flatten()}")
    print(f"Target: {y_true.flatten()}")
    print(f"Prediction: {y_pred.flatten()}")
    print(f"Cost: {0.5 * (y_pred - y_true)**2}")

visualize_gradients()
```

## Numerical Gradient Checking

```python
def numerical_gradient_checking():
    """Verify backpropagation using numerical gradients"""
    
    def compute_numerical_gradient(func, x, epsilon=1e-7):
        """Compute numerical gradient using finite differences"""
        grad = np.zeros_like(x)
        
        for i in range(x.size):
            x_plus = x.copy()
            x_plus.flat[i] += epsilon
            x_minus = x.copy()
            x_minus.flat[i] -= epsilon
            
            grad.flat[i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)
        
        return grad
    
    # Simple cost function for testing
    def simple_cost_function(weights):
        """Simple cost function for gradient checking"""
        W1, W2 = weights[:4].reshape(2, 2), weights[4:6].reshape(1, 2)
        b1, b2 = weights[6:8].reshape(2, 1), weights[8:9].reshape(1, 1)
        
        # Forward pass
        z1 = np.dot(W1, x_test) + b1
        a1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(W2, a1) + b2
        a2 = 1 / (1 + np.exp(-z2))
        
        # Cost
        return 0.5 * np.mean((a2 - y_test)**2)
    
    # Test data
    x_test = np.array([[0.5, 0.3]]).T
    y_test = np.array([[1.0]])
    
    # Initial weights
    W1 = np.random.randn(2, 2) * 0.01
    W2 = np.random.randn(1, 2) * 0.01
    b1 = np.zeros((2, 1))
    b2 = np.zeros((1, 1))
    
    # Flatten weights for gradient checking
    weights = np.concatenate([W1.flatten(), W2.flatten(), b1.flatten(), b2.flatten()])
    
    # Compute numerical gradients
    numerical_grad = compute_numerical_gradient(simple_cost_function, weights)
    
    # Compute analytical gradients (simplified backpropagation)
    def compute_analytical_gradients():
        # Forward pass
        z1 = np.dot(W1, x_test) + b1
        a1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(W2, a1) + b2
        a2 = 1 / (1 + np.exp(-z2))
        
        # Backward pass
        dz2 = (a2 - y_test) * a2 * (1 - a2)
        da1 = np.dot(W2.T, dz2)
        dz1 = da1 * a1 * (1 - a1)
        
        dW2 = np.dot(dz2, a1.T)
        db2 = dz2
        dW1 = np.dot(dz1, x_test.T)
        db1 = dz1
        
        return np.concatenate([dW1.flatten(), dW2.flatten(), db1.flatten(), db2.flatten()])
    
    analytical_grad = compute_analytical_gradients()
    
    # Compare gradients
    diff = np.linalg.norm(numerical_grad - analytical_grad) / np.linalg.norm(numerical_grad + analytical_grad)
    
    plt.figure(figsize=(15, 5))
    
    # Numerical gradients
    plt.subplot(1, 3, 1)
    plt.bar(range(len(numerical_grad)), numerical_grad)
    plt.title('Numerical Gradients')
    plt.xlabel('Parameter Index')
    plt.ylabel('Gradient Value')
    plt.grid(True)
    
    # Analytical gradients
    plt.subplot(1, 3, 2)
    plt.bar(range(len(analytical_grad)), analytical_grad)
    plt.title('Analytical Gradients')
    plt.xlabel('Parameter Index')
    plt.ylabel('Gradient Value')
    plt.grid(True)
    
    # Comparison
    plt.subplot(1, 3, 3)
    plt.scatter(numerical_grad, analytical_grad, alpha=0.7)
    plt.plot([numerical_grad.min(), numerical_grad.max()], 
             [numerical_grad.min(), numerical_grad.max()], 'r--')
    plt.xlabel('Numerical Gradients')
    plt.ylabel('Analytical Gradients')
    plt.title(f'Gradient Comparison\nRelative Difference: {diff:.2e}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Gradient checking results:")
    print(f"Relative difference: {diff:.2e}")
    print(f"Gradients are {'correct' if diff < 1e-7 else 'incorrect'}")

numerical_gradient_checking()
```

## Common Issues and Solutions

### 1. Vanishing Gradients

```python
def demonstrate_vanishing_gradients():
    """Demonstrate vanishing gradient problem"""
    
    # Create a deep network
    class DeepNN:
        def __init__(self, depth=10):
            self.depth = depth
            self.weights = []
            self.biases = []
            
            # Initialize with small weights (causing vanishing gradients)
            for i in range(depth):
                w = np.random.randn(5, 5) * 0.1
                b = np.zeros((5, 1))
                self.weights.append(w)
                self.biases.append(b)
        
        def forward(self, x):
            activations = [x]
            z_values = []
            
            for i in range(self.depth):
                z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
                a = self.sigmoid(z)
                activations.append(a)
                z_values.append(z)
            
            return activations, z_values
        
        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_derivative(self, x):
            s = self.sigmoid(x)
            return s * (1 - s)
        
        def backward(self, activations, z_values, target):
            gradients = []
            
            # Output layer gradient
            dz = (activations[-1] - target) * self.sigmoid_derivative(z_values[-1])
            
            # Backpropagate through layers
            for i in range(self.depth - 1, -1, -1):
                gradients.append(dz)
                
                if i > 0:
                    da = np.dot(self.weights[i].T, dz)
                    dz = da * self.sigmoid_derivative(z_values[i-1])
            
            return list(reversed(gradients))
    
    # Test with deep network
    deep_nn = DeepNN(depth=10)
    x = np.random.randn(5, 1)
    target = np.random.randn(5, 1)
    
    # Forward pass
    activations, z_values = deep_nn.forward(x)
    
    # Backward pass
    gradients = deep_nn.backward(activations, z_values, target)
    
    # Analyze gradient magnitudes
    gradient_magnitudes = [np.linalg.norm(grad) for grad in gradients]
    
    plt.figure(figsize=(15, 5))
    
    # Gradient magnitudes
    plt.subplot(1, 3, 1)
    plt.plot(range(len(gradient_magnitudes)), gradient_magnitudes, 'o-')
    plt.xlabel('Layer (from output to input)')
    plt.ylabel('Gradient Magnitude')
    plt.title('Vanishing Gradients')
    plt.yscale('log')
    plt.grid(True)
    
    # Activation values
    plt.subplot(1, 3, 2)
    activation_means = [np.mean(np.abs(act)) for act in activations[1:]]
    plt.plot(range(len(activation_means)), activation_means, 'o-')
    plt.xlabel('Layer')
    plt.ylabel('Mean |Activation|')
    plt.title('Activation Values')
    plt.grid(True)
    
    # Weight magnitudes
    plt.subplot(1, 3, 3)
    weight_magnitudes = [np.linalg.norm(w) for w in deep_nn.weights]
    plt.plot(range(len(weight_magnitudes)), weight_magnitudes, 'o-')
    plt.xlabel('Layer')
    plt.ylabel('Weight Matrix Norm')
    plt.title('Weight Magnitudes')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("Vanishing Gradient Analysis:")
    print(f"Gradient magnitude at output layer: {gradient_magnitudes[0]:.6f}")
    print(f"Gradient magnitude at input layer: {gradient_magnitudes[-1]:.6f}")
    print(f"Gradient ratio: {gradient_magnitudes[-1] / gradient_magnitudes[0]:.2e}")

demonstrate_vanishing_gradients()
```

### 2. Exploding Gradients

```python
def demonstrate_exploding_gradients():
    """Demonstrate exploding gradient problem"""
    
    class ExplodingNN:
        def __init__(self, depth=10):
            self.depth = depth
            self.weights = []
            self.biases = []
            
            # Initialize with large weights (causing exploding gradients)
            for i in range(depth):
                w = np.random.randn(5, 5) * 2.0
                b = np.zeros((5, 1))
                self.weights.append(w)
                self.biases.append(b)
        
        def forward(self, x):
            activations = [x]
            z_values = []
            
            for i in range(self.depth):
                z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
                a = self.sigmoid(z)
                activations.append(a)
                z_values.append(z)
            
            return activations, z_values
        
        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_derivative(self, x):
            s = self.sigmoid(x)
            return s * (1 - s)
        
        def backward(self, activations, z_values, target):
            gradients = []
            
            # Output layer gradient
            dz = (activations[-1] - target) * self.sigmoid_derivative(z_values[-1])
            
            # Backpropagate through layers
            for i in range(self.depth - 1, -1, -1):
                gradients.append(dz)
                
                if i > 0:
                    da = np.dot(self.weights[i].T, dz)
                    dz = da * self.sigmoid_derivative(z_values[i-1])
            
            return list(reversed(gradients))
    
    # Test with exploding network
    exploding_nn = ExplodingNN(depth=10)
    x = np.random.randn(5, 1)
    target = np.random.randn(5, 1)
    
    # Forward pass
    activations, z_values = exploding_nn.forward(x)
    
    # Backward pass
    gradients = exploding_nn.backward(activations, z_values, target)
    
    # Analyze gradient magnitudes
    gradient_magnitudes = [np.linalg.norm(grad) for grad in gradients]
    
    plt.figure(figsize=(15, 5))
    
    # Gradient magnitudes
    plt.subplot(1, 3, 1)
    plt.plot(range(len(gradient_magnitudes)), gradient_magnitudes, 'o-')
    plt.xlabel('Layer (from output to input)')
    plt.ylabel('Gradient Magnitude')
    plt.title('Exploding Gradients')
    plt.yscale('log')
    plt.grid(True)
    
    # Activation values
    plt.subplot(1, 3, 2)
    activation_means = [np.mean(np.abs(act)) for act in activations[1:]]
    plt.plot(range(len(activation_means)), activation_means, 'o-')
    plt.xlabel('Layer')
    plt.ylabel('Mean |Activation|')
    plt.title('Activation Values')
    plt.grid(True)
    
    # Weight magnitudes
    plt.subplot(1, 3, 3)
    weight_magnitudes = [np.linalg.norm(w) for w in exploding_nn.weights]
    plt.plot(range(len(weight_magnitudes)), weight_magnitudes, 'o-')
    plt.xlabel('Layer')
    plt.ylabel('Weight Matrix Norm')
    plt.title('Weight Magnitudes')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("Exploding Gradient Analysis:")
    print(f"Gradient magnitude at output layer: {gradient_magnitudes[0]:.6f}")
    print(f"Gradient magnitude at input layer: {gradient_magnitudes[-1]:.6f}")
    print(f"Gradient ratio: {gradient_magnitudes[-1] / gradient_magnitudes[0]:.2e}")

demonstrate_exploding_gradients()
```

## Advanced Backpropagation Techniques

### 1. Batch Normalization

```python
class BatchNormalization:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.running_mean = None
        self.running_var = None
        self.gamma = None
        self.beta = None
    
    def forward(self, x, training=True):
        """Forward pass for batch normalization"""
        if self.running_mean is None:
            self.running_mean = np.zeros(x.shape[1])
            self.running_var = np.ones(x.shape[1])
            self.gamma = np.ones(x.shape[1])
            self.beta = np.zeros(x.shape[1])
        
        if training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = (self.momentum * self.running_mean + 
                               (1 - self.momentum) * batch_mean)
            self.running_var = (self.momentum * self.running_var + 
                              (1 - self.momentum) * batch_var)
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + 1e-8)
        else:
            # Use running statistics
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + 1e-8)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta

# Test batch normalization
bn = BatchNormalization()
x_bn = np.random.randn(32, 10)  # Batch of 32 samples, 10 features

# Forward pass
y_bn = bn.forward(x_bn, training=True)

plt.figure(figsize=(15, 5))

# Original data distribution
plt.subplot(1, 3, 1)
plt.hist(x_bn.flatten(), bins=30, alpha=0.7, label='Original')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Original Data Distribution')
plt.legend()
plt.grid(True)

# Normalized data distribution
plt.subplot(1, 3, 2)
plt.hist(y_bn.flatten(), bins=30, alpha=0.7, label='Normalized')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Normalized Data Distribution')
plt.legend()
plt.grid(True)

# Feature-wise statistics
plt.subplot(1, 3, 3)
original_std = np.std(x_bn, axis=0)
normalized_std = np.std(y_bn, axis=0)
plt.bar(range(len(original_std)), original_std, alpha=0.7, label='Original')
plt.bar(range(len(normalized_std)), normalized_std, alpha=0.7, label='Normalized')
plt.xlabel('Feature')
plt.ylabel('Standard Deviation')
plt.title('Feature-wise Standard Deviations')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 2. Dropout

```python
class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
    
    def forward(self, x, training=True):
        """Forward pass with dropout"""
        if training:
            # Create dropout mask
            self.mask = np.random.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p)
            return x * self.mask
        else:
            return x
    
    def backward(self, grad):
        """Backward pass with dropout"""
        return grad * self.mask

# Test dropout
dropout = Dropout(p=0.5)
x_dropout = np.random.randn(100, 10)

# Forward pass
y_dropout = dropout.forward(x_dropout, training=True)

plt.figure(figsize=(15, 5))

# Original activations
plt.subplot(1, 3, 1)
plt.imshow(x_dropout[:20, :], cmap='viridis')
plt.title('Original Activations')
plt.xlabel('Feature')
plt.ylabel('Sample')
plt.colorbar()

# Dropout mask
plt.subplot(1, 3, 2)
plt.imshow(dropout.mask[:20, :], cmap='gray')
plt.title('Dropout Mask')
plt.xlabel('Feature')
plt.ylabel('Sample')

# Dropped activations
plt.subplot(1, 3, 3)
plt.imshow(y_dropout[:20, :], cmap='viridis')
plt.title('Dropped Activations')
plt.xlabel('Feature')
plt.ylabel('Sample')
plt.colorbar()

plt.tight_layout()
plt.show()

print(f"Dropout rate: {dropout.p}")
print(f"Fraction of dropped neurons: {1 - np.mean(dropout.mask):.3f}")
```

## Real-world Example: MNIST Digit Classification

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST dataset (small subset for demonstration)
try:
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_mnist = mnist.data[:1000]  # Use first 1000 samples
    y_mnist = mnist.target[:1000].astype(int)
except:
    # Fallback to synthetic data if MNIST is not available
    print("MNIST dataset not available, using synthetic data")
    X_mnist = np.random.randn(1000, 784)
    y_mnist = np.random.randint(0, 10, 1000)

# Convert to one-hot encoding
y_onehot_mnist = np.zeros((y_mnist.size, 10))
y_onehot_mnist[np.arange(y_mnist.size), y_mnist] = 1

# Split data
X_train_mnist, X_test_mnist, y_train_mnist, y_test_mnist = train_test_split(
    X_mnist, y_onehot_mnist, test_size=0.2, random_state=42
)

# Scale features
scaler_mnist = StandardScaler()
X_train_scaled_mnist = scaler_mnist.fit_transform(X_train_mnist).T
X_test_scaled_mnist = scaler_mnist.transform(X_test_mnist).T
y_train_scaled_mnist = y_train_mnist.T
y_test_scaled_mnist = y_test_mnist.T

# Create neural network for MNIST
nn_mnist = NeuralNetwork([784, 128, 64, 10], activation='relu', learning_rate=0.01)
costs_mnist = nn_mnist.train(X_train_scaled_mnist, y_train_scaled_mnist, epochs=200, batch_size=32)

# Evaluate performance
y_pred_mnist = nn_mnist.predict(X_test_scaled_mnist)
y_test_labels_mnist = np.argmax(y_test_scaled_mnist.T, axis=1)

accuracy_mnist = accuracy_score(y_test_labels_mnist, y_pred_mnist)

# Visualize results
plt.figure(figsize=(15, 5))

# Training progress
plt.subplot(1, 3, 1)
plt.plot(costs_mnist)
plt.xlabel('Epoch (x100)')
plt.ylabel('Cost')
plt.title('MNIST Training Cost')
plt.grid(True)

# Confusion matrix
plt.subplot(1, 3, 2)
cm_mnist = confusion_matrix(y_test_labels_mnist, y_pred_mnist)
plt.imshow(cm_mnist, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('MNIST Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Sample predictions
plt.subplot(1, 3, 3)
sample_indices = np.random.choice(len(X_test_mnist), 9, replace=False)
for i, idx in enumerate(sample_indices):
    plt.subplot(3, 3, i+1)
    if X_mnist.shape[1] == 784:  # MNIST data
        plt.imshow(X_test_mnist[idx].reshape(28, 28), cmap='gray')
    else:  # Synthetic data
        plt.imshow(X_test_mnist[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Pred: {y_pred_mnist[idx]}\nTrue: {y_test_labels_mnist[idx]}')
    plt.axis('off')

plt.tight_layout()
plt.show()

print(f"MNIST Classification Accuracy: {accuracy_mnist:.3f}")

# Analyze gradient flow
def analyze_gradient_flow(nn, X, y):
    """Analyze gradient flow through the network"""
    # Forward pass
    y_pred = nn.forward(X)
    
    # Backward pass
    dW, db = nn.backward(X, y, y_pred)
    
    # Compute gradient magnitudes
    weight_grad_magnitudes = [np.linalg.norm(dw) for dw in dW]
    bias_grad_magnitudes = [np.linalg.norm(db) for db in db]
    
    return weight_grad_magnitudes, bias_grad_magnitudes

# Analyze gradients
weight_grads, bias_grads = analyze_gradient_flow(nn_mnist, X_train_scaled_mnist[:, :100], 
                                                y_train_scaled_mnist[:, :100])

plt.figure(figsize=(12, 4))

# Weight gradient magnitudes
plt.subplot(1, 2, 1)
plt.plot(weight_grads, 'o-', label='Weight Gradients')
plt.xlabel('Layer')
plt.ylabel('Gradient Magnitude')
plt.title('Weight Gradient Flow')
plt.yscale('log')
plt.legend()
plt.grid(True)

# Bias gradient magnitudes
plt.subplot(1, 2, 2)
plt.plot(bias_grads, 's-', label='Bias Gradients')
plt.xlabel('Layer')
plt.ylabel('Gradient Magnitude')
plt.title('Bias Gradient Flow')
plt.yscale('log')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Gradient Flow Analysis:")
for i, (wg, bg) in enumerate(zip(weight_grads, bias_grads)):
    print(f"Layer {i+1}: Weight grad = {wg:.6f}, Bias grad = {bg:.6f}")
```

## Conclusion

Backpropagation is the cornerstone of neural network training. Key takeaways:

### Core Concepts:
1. **Chain Rule**: Mathematical foundation for gradient computation
2. **Forward Pass**: Compute activations through the network
3. **Backward Pass**: Compute gradients using chain rule
4. **Parameter Updates**: Update weights and biases using gradients

### Common Issues:
1. **Vanishing Gradients**: Gradients become very small in deep networks
2. **Exploding Gradients**: Gradients become very large
3. **Local Minima**: Getting stuck in suboptimal solutions

### Solutions:
1. **Proper Initialization**: He/Xavier initialization
2. **Batch Normalization**: Normalize activations
3. **Dropout**: Regularization technique
4. **Learning Rate Scheduling**: Adaptive learning rates

### Best Practices:
1. **Gradient Checking**: Verify implementation correctness
2. **Monitoring**: Track gradient magnitudes and activations
3. **Regularization**: Prevent overfitting
4. **Hyperparameter Tuning**: Optimize learning rate, batch size

Backpropagation enables neural networks to learn complex patterns and has revolutionized machine learning.

## Further Reading

- **Books**: "Deep Learning" by Goodfellow et al., "Neural Networks and Deep Learning" by Michael Nielsen
- **Papers**: Original backpropagation paper by Rumelhart et al. (1986)
- **Online Resources**: CS231n course notes, Distill.pub articles
- **Practice**: Implement backpropagation from scratch, experiment with different architectures 