# Neural Networks

## Introduction

Neural networks are computational models inspired by biological neural networks, capable of learning complex patterns through multiple layers of interconnected nodes (neurons). They form the foundation of deep learning and have revolutionized machine learning across various domains.

## Mathematical Foundation

### The Artificial Neuron

A single neuron computes:

$$z = \sum_{i=1}^{n} w_i x_i + b$$

$$a = \sigma(z)$$

Where:
- $x_i$ are input features
- $w_i$ are weights
- $b$ is bias
- $\sigma$ is the activation function
- $a$ is the output

### Activation Functions

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU function"""
    return np.where(x > 0, 1, 0)

def tanh(x):
    """Hyperbolic tangent activation function"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of tanh function"""
    return 1 - np.tanh(x)**2

def softmax(x):
    """Softmax activation function"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Plot activation functions
x = np.linspace(-5, 5, 1000)

plt.figure(figsize=(15, 10))

# Activation functions
plt.subplot(2, 3, 1)
plt.plot(x, sigmoid(x), 'b-', linewidth=2, label='Sigmoid')
plt.plot(x, relu(x), 'r-', linewidth=2, label='ReLU')
plt.plot(x, tanh(x), 'g-', linewidth=2, label='Tanh')
plt.xlabel('x')
plt.ylabel('σ(x)')
plt.title('Activation Functions')
plt.legend()
plt.grid(True)

# Derivatives
plt.subplot(2, 3, 2)
plt.plot(x, sigmoid_derivative(x), 'b-', linewidth=2, label='Sigmoid\'')
plt.plot(x, relu_derivative(x), 'r-', linewidth=2, label='ReLU\'')
plt.plot(x, tanh_derivative(x), 'g-', linewidth=2, label='Tanh\'')
plt.xlabel('x')
plt.ylabel('σ\'(x)')
plt.title('Activation Function Derivatives')
plt.legend()
plt.grid(True)

# Sigmoid details
plt.subplot(2, 3, 3)
plt.plot(x, sigmoid(x), 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.title('Sigmoid Function')
plt.grid(True)

# ReLU details
plt.subplot(2, 3, 4)
plt.plot(x, relu(x), 'r-', linewidth=2)
plt.axvline(x=0, color='b', linestyle='--', alpha=0.7)
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.title('ReLU Function')
plt.grid(True)

# Tanh details
plt.subplot(2, 3, 5)
plt.plot(x, tanh(x), 'g-', linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
plt.xlabel('x')
plt.ylabel('Tanh(x)')
plt.title('Tanh Function')
plt.grid(True)

# Softmax example
plt.subplot(2, 3, 6)
x_softmax = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
softmax_output = softmax(x_softmax)
plt.bar(range(3), softmax_output[0], alpha=0.7, label='Sample 1')
plt.bar(range(3), softmax_output[1], alpha=0.7, label='Sample 2')
plt.bar(range(3), softmax_output[2], alpha=0.7, label='Sample 3')
plt.xlabel('Class')
plt.ylabel('Probability')
plt.title('Softmax Output')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Activation function properties:")
print(f"Sigmoid(0) = {sigmoid(0):.3f}")
print(f"ReLU(-1) = {relu(-1):.3f}, ReLU(1) = {relu(1):.3f}")
print(f"Tanh(0) = {tanh(0):.3f}, Tanh(∞) = {tanh(100):.3f}")
```

## Feedforward Neural Network

### Mathematical Formulation

For a network with $L$ layers:

**Forward Pass:**
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = \sigma^{(l)}(z^{(l)})$$

Where:
- $W^{(l)}$ is the weight matrix for layer $l$
- $b^{(l)}$ is the bias vector for layer $l$
- $\sigma^{(l)}$ is the activation function for layer $l$

### Implementation

```python
class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.activations = []
        self.z_values = []
        
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
    
    def forward(self, X):
        """Forward pass through the network"""
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], self.activations[-1]) + self.biases[i]
            self.z_values.append(z)
            
            if i == len(self.weights) - 1:
                # Output layer - use softmax for classification
                a = softmax(z.T).T
            else:
                # Hidden layers
                if self.activation == 'relu':
                    a = relu(z)
                elif self.activation == 'sigmoid':
                    a = sigmoid(z)
                elif self.activation == 'tanh':
                    a = tanh(z)
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y, output):
        """Backward pass (backpropagation)"""
        m = X.shape[1]
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient
        dz = output - y
        
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
                    dz = da * relu_derivative(self.z_values[i-1])
                elif self.activation == 'sigmoid':
                    dz = da * sigmoid_derivative(self.z_values[i-1])
                elif self.activation == 'tanh':
                    dz = da * tanh_derivative(self.z_values[i-1])
        
        return dW, db
    
    def update_parameters(self, dW, db):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def train(self, X, y, epochs=1000, batch_size=32):
        """Train the neural network"""
        costs = []
        
        for epoch in range(epochs):
            # Mini-batch training
            for i in range(0, X.shape[1], batch_size):
                X_batch = X[:, i:i+batch_size]
                y_batch = y[:, i:i+batch_size]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Backward pass
                dW, db = self.backward(X_batch, y_batch, output)
                
                # Update parameters
                self.update_parameters(dW, db)
            
            # Compute cost
            if epoch % 100 == 0:
                output = self.forward(X)
                cost = -np.mean(np.sum(y * np.log(output + 1e-15), axis=0))
                costs.append(cost)
                print(f"Epoch {epoch}, Cost: {cost:.6f}")
        
        return costs
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return np.argmax(output, axis=0)

# Generate classification data
X_class, y_class = make_classification(n_samples=1000, n_features=2, n_classes=3, 
                                      n_clusters_per_class=1, n_redundant=0, 
                                      random_state=42)

# Convert to one-hot encoding
y_onehot = np.zeros((y_class.size, 3))
y_onehot[np.arange(y_class.size), y_class] = 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_class, y_onehot, test_size=0.2, random_state=42)

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

## Convolutional Neural Networks (CNN)

### Mathematical Formulation

**Convolution Operation:**
$$(f * k)(p) = \sum_{s+t=p} f(s) k(t)$$

**Pooling Operation:**
$$\text{maxpool}(x) = \max_{i,j \in W} x_{i,j}$$

### Implementation

```python
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize filters
        self.filters = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.bias = np.zeros(out_channels)
    
    def forward(self, input_data):
        """Forward pass for convolution"""
        batch_size, in_channels, height, width = input_data.shape
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Add padding
        if self.padding > 0:
            padded_input = np.pad(input_data, ((0, 0), (0, 0), 
                                             (self.padding, self.padding), 
                                             (self.padding, self.padding)))
        else:
            padded_input = input_data
        
        # Initialize output
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Perform convolution
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Extract patch
                        patch = padded_input[b, :, h_start:h_end, w_start:w_end]
                        
                        # Apply filter
                        output[b, c_out, h_out, w_out] = (
                            np.sum(patch * self.filters[c_out]) + self.bias[c_out]
                        )
        
        return output

class MaxPool2D:
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
    
    def forward(self, input_data):
        """Forward pass for max pooling"""
        batch_size, channels, height, width = input_data.shape
        
        # Calculate output dimensions
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w_out * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Extract patch
                        patch = input_data[b, c, h_start:h_end, w_start:w_end]
                        
                        # Apply max pooling
                        output[b, c, h_out, w_out] = np.max(patch)
        
        return output

class Flatten:
    def forward(self, input_data):
        """Flatten the input"""
        batch_size = input_data.shape[0]
        return input_data.reshape(batch_size, -1)

# Simple CNN implementation
class SimpleCNN:
    def __init__(self):
        self.conv1 = Conv2D(1, 6, 5, padding=2)  # 1 input channel, 6 output channels, 5x5 kernel
        self.pool1 = MaxPool2D(2, 2)
        self.conv2 = Conv2D(6, 16, 5, padding=2)
        self.pool2 = MaxPool2D(2, 2)
        self.flatten = Flatten()
        self.fc1 = NeuralNetwork([16*7*7, 120, 84, 10], activation='relu')
    
    def forward(self, x):
        # Convolutional layers
        x = relu(self.conv1.forward(x))
        x = self.pool1.forward(x)
        x = relu(self.conv2.forward(x))
        x = self.pool2.forward(x)
        
        # Flatten and fully connected layers
        x = self.flatten.forward(x)
        x = self.fc1.forward(x.T).T
        
        return x

# Generate synthetic image data
def generate_image_data(n_samples=1000, image_size=28):
    """Generate synthetic image data"""
    images = np.random.rand(n_samples, 1, image_size, image_size)
    labels = np.random.randint(0, 10, n_samples)
    
    # Convert to one-hot
    one_hot = np.zeros((n_samples, 10))
    one_hot[np.arange(n_samples), labels] = 1
    
    return images, one_hot

# Test CNN
image_data, image_labels = generate_image_data(100, 28)
print(f"Image data shape: {image_data.shape}")
print(f"Labels shape: {image_labels.shape}")

# Test convolution
conv_layer = Conv2D(1, 6, 5, padding=2)
conv_output = conv_layer.forward(image_data[:5])
print(f"Convolution output shape: {conv_output.shape}")

# Test pooling
pool_layer = MaxPool2D(2, 2)
pool_output = pool_layer.forward(conv_output)
print(f"Pooling output shape: {pool_output.shape}")

# Visualize CNN operations
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(image_data[0, 0], cmap='gray')
plt.title('Original Image')
plt.axis('off')

# After convolution
plt.subplot(1, 3, 2)
plt.imshow(conv_output[0, 0], cmap='gray')
plt.title('After Convolution')
plt.axis('off')

# After pooling
plt.subplot(1, 3, 3)
plt.imshow(pool_output[0, 0], cmap='gray')
plt.title('After Pooling')
plt.axis('off')

plt.tight_layout()
plt.show()
```

## Recurrent Neural Networks (RNN)

### Mathematical Formulation

**Simple RNN:**
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

### Implementation

```python
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        
        # Initialize biases
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
    
    def forward(self, x_sequence):
        """Forward pass through RNN"""
        seq_length = x_sequence.shape[1]
        batch_size = x_sequence.shape[0]
        
        # Initialize hidden states and outputs
        h_states = np.zeros((batch_size, seq_length, self.hidden_size))
        y_outputs = np.zeros((batch_size, seq_length, self.output_size))
        
        # Initialize hidden state
        h_prev = np.zeros((batch_size, self.hidden_size))
        
        for t in range(seq_length):
            # Current input
            x_t = x_sequence[:, t, :].T  # (input_size, batch_size)
            
            # Hidden state
            h_t = np.tanh(np.dot(self.W_xh, x_t) + 
                         np.dot(self.W_hh, h_prev.T) + self.b_h)
            h_t = h_t.T  # (batch_size, hidden_size)
            
            # Output
            y_t = np.dot(self.W_hy, h_t.T) + self.b_y
            y_t = y_t.T  # (batch_size, output_size)
            
            # Store states
            h_states[:, t, :] = h_t
            y_outputs[:, t, :] = y_t
            
            # Update hidden state
            h_prev = h_t
        
        return h_states, y_outputs
    
    def backward(self, x_sequence, h_states, y_outputs, targets):
        """Backward pass through RNN"""
        seq_length = x_sequence.shape[1]
        batch_size = x_sequence.shape[0]
        
        # Initialize gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # Initialize hidden state gradient
        dh_next = np.zeros((batch_size, self.hidden_size))
        
        # Backpropagate through time
        for t in reversed(range(seq_length)):
            # Output gradient
            dy = y_outputs[:, t, :] - targets[:, t, :]
            
            # Hidden to output gradients
            dW_hy += np.dot(dy.T, h_states[:, t, :])
            db_y += np.sum(dy, axis=0, keepdims=True).T
            
            # Hidden state gradient
            dh = np.dot(dy, self.W_hy) + dh_next
            
            # Tanh gradient
            dh_raw = (1 - h_states[:, t, :]**2) * dh
            
            # Bias gradient
            db_h += np.sum(dh_raw, axis=0, keepdims=True).T
            
            # Weight gradients
            if t > 0:
                dW_hh += np.dot(dh_raw.T, h_states[:, t-1, :])
            dW_xh += np.dot(dh_raw.T, x_sequence[:, t, :])
            
            # Next hidden state gradient
            dh_next = np.dot(dh_raw, self.W_hh)
        
        return dW_xh, dW_hh, dW_hy, db_h, db_y
    
    def update_parameters(self, gradients):
        """Update parameters using gradients"""
        dW_xh, dW_hh, dW_hy, db_h, db_y = gradients
        
        self.W_xh -= self.learning_rate * dW_xh
        self.W_hh -= self.learning_rate * dW_hh
        self.W_hy -= self.learning_rate * dW_hy
        self.b_h -= self.learning_rate * db_h
        self.b_y -= self.learning_rate * db_y
    
    def train(self, x_sequences, y_sequences, epochs=100):
        """Train the RNN"""
        costs = []
        
        for epoch in range(epochs):
            total_cost = 0
            
            for i in range(len(x_sequences)):
                # Forward pass
                h_states, y_outputs = self.forward(x_sequences[i:i+1])
                
                # Compute cost
                cost = np.mean((y_outputs - y_sequences[i:i+1])**2)
                total_cost += cost
                
                # Backward pass
                gradients = self.backward(x_sequences[i:i+1], h_states, y_outputs, y_sequences[i:i+1])
                
                # Update parameters
                self.update_parameters(gradients)
            
            avg_cost = total_cost / len(x_sequences)
            costs.append(avg_cost)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Cost: {avg_cost:.6f}")
        
        return costs

# Generate sequence data
def generate_sequence_data(n_samples=100, seq_length=10, input_size=3, output_size=2):
    """Generate synthetic sequence data"""
    x_sequences = []
    y_sequences = []
    
    for _ in range(n_samples):
        # Generate random sequence
        x_seq = np.random.randn(1, seq_length, input_size)
        y_seq = np.random.randn(1, seq_length, output_size)
        
        x_sequences.append(x_seq)
        y_sequences.append(y_seq)
    
    return x_sequences, y_sequences

# Test RNN
x_seqs, y_seqs = generate_sequence_data(50, 8, 3, 2)

rnn = SimpleRNN(input_size=3, hidden_size=10, output_size=2, learning_rate=0.01)
costs = rnn.train(x_seqs, y_seqs, epochs=100)

# Plot training progress
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(costs)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('RNN Training Cost')
plt.grid(True)

# Test prediction
test_seq = x_seqs[0]
h_states, predictions = rnn.forward(test_seq)

plt.subplot(1, 2, 2)
plt.plot(predictions[0, :, 0], 'b-', label='Predicted 1')
plt.plot(predictions[0, :, 1], 'r-', label='Predicted 2')
plt.plot(y_seqs[0][0, :, 0], 'b--', label='Actual 1')
plt.plot(y_seqs[0][0, :, 1], 'r--', label='Actual 2')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('RNN Predictions vs Actual')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Long Short-Term Memory (LSTM)

### Mathematical Formulation

**LSTM Gates:**
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
$$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t * \tanh(C_t)$$

### Implementation

```python
class LSTM:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights for all gates
        self.W_f = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.W_i = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.W_C = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.W_o = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01
        
        # Initialize biases
        self.b_f = np.zeros((hidden_size, 1))
        self.b_i = np.zeros((hidden_size, 1))
        self.b_C = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
    
    def forward(self, x_sequence):
        """Forward pass through LSTM"""
        seq_length = x_sequence.shape[1]
        batch_size = x_sequence.shape[0]
        
        # Initialize states
        h_states = np.zeros((batch_size, seq_length, self.hidden_size))
        C_states = np.zeros((batch_size, seq_length, self.hidden_size))
        y_outputs = np.zeros((batch_size, seq_length, self.output_size))
        
        # Initialize hidden and cell states
        h_prev = np.zeros((batch_size, self.hidden_size))
        C_prev = np.zeros((batch_size, self.hidden_size))
        
        for t in range(seq_length):
            # Concatenate input and previous hidden state
            x_t = x_sequence[:, t, :]
            concat_input = np.concatenate([h_prev, x_t], axis=1)
            
            # Gates
            f_t = sigmoid(np.dot(concat_input, self.W_f.T) + self.b_f.T)
            i_t = sigmoid(np.dot(concat_input, self.W_i.T) + self.b_i.T)
            C_tilde = np.tanh(np.dot(concat_input, self.W_C.T) + self.b_C.T)
            o_t = sigmoid(np.dot(concat_input, self.W_o.T) + self.b_o.T)
            
            # Cell state
            C_t = f_t * C_prev + i_t * C_tilde
            
            # Hidden state
            h_t = o_t * np.tanh(C_t)
            
            # Output
            y_t = np.dot(h_t, self.W_y.T) + self.b_y.T
            
            # Store states
            h_states[:, t, :] = h_t
            C_states[:, t, :] = C_t
            y_outputs[:, t, :] = y_t
            
            # Update states
            h_prev = h_t
            C_prev = C_t
        
        return h_states, C_states, y_outputs

# Test LSTM
lstm = LSTM(input_size=3, hidden_size=10, output_size=2, learning_rate=0.01)

# Test forward pass
test_seq = x_seqs[0]
h_states, C_states, predictions = lstm.forward(test_seq)

print(f"LSTM output shape: {predictions.shape}")
print(f"Hidden states shape: {h_states.shape}")
print(f"Cell states shape: {C_states.shape}")

# Visualize LSTM states
plt.figure(figsize=(15, 5))

# Hidden states
plt.subplot(1, 3, 1)
plt.imshow(h_states[0].T, cmap='viridis', aspect='auto')
plt.title('Hidden States')
plt.xlabel('Time Step')
plt.ylabel('Hidden Unit')
plt.colorbar()

# Cell states
plt.subplot(1, 3, 2)
plt.imshow(C_states[0].T, cmap='plasma', aspect='auto')
plt.title('Cell States')
plt.xlabel('Time Step')
plt.ylabel('Cell Unit')
plt.colorbar()

# Predictions
plt.subplot(1, 3, 3)
plt.plot(predictions[0, :, 0], 'b-', label='Output 1')
plt.plot(predictions[0, :, 1], 'r-', label='Output 2')
plt.xlabel('Time Step')
plt.ylabel('Output Value')
plt.title('LSTM Predictions')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Real-world Example: Image Classification

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load digit dataset
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# Convert to one-hot encoding
y_onehot_digits = np.zeros((y_digits.size, 10))
y_onehot_digits[np.arange(y_digits.size), y_digits] = 1

# Split data
X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(
    X_digits, y_onehot_digits, test_size=0.2, random_state=42
)

# Scale features
scaler_digits = StandardScaler()
X_train_scaled_digits = scaler_digits.fit_transform(X_train_digits).T
X_test_scaled_digits = scaler_digits.transform(X_test_digits).T
y_train_scaled_digits = y_train_digits.T
y_test_scaled_digits = y_test_digits.T

# Create neural network for digit classification
nn_digits = NeuralNetwork([64, 128, 64, 10], activation='relu', learning_rate=0.01)
costs_digits = nn_digits.train(X_train_scaled_digits, y_train_scaled_digits, epochs=500, batch_size=32)

# Evaluate performance
y_pred_digits = nn_digits.predict(X_test_scaled_digits)
y_test_labels_digits = np.argmax(y_test_scaled_digits.T, axis=1)

accuracy_digits = accuracy_score(y_test_labels_digits, y_pred_digits)

# Visualize results
plt.figure(figsize=(15, 5))

# Training progress
plt.subplot(1, 3, 1)
plt.plot(costs_digits)
plt.xlabel('Epoch (x100)')
plt.ylabel('Cost')
plt.title('Training Cost')
plt.grid(True)

# Confusion matrix
plt.subplot(1, 3, 2)
cm_digits = confusion_matrix(y_test_labels_digits, y_pred_digits)
plt.imshow(cm_digits, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Sample predictions
plt.subplot(1, 3, 3)
sample_indices = np.random.choice(len(X_test_digits), 9, replace=False)
for i, idx in enumerate(sample_indices):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_test_digits[idx].reshape(8, 8), cmap='gray')
    plt.title(f'Pred: {y_pred_digits[idx]}\nTrue: {y_test_labels_digits[idx]}')
    plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Digit Classification Accuracy: {accuracy_digits:.3f}")

# Compare with different architectures
architectures = {
    'Small': [64, 32, 10],
    'Medium': [64, 128, 64, 10],
    'Large': [64, 256, 128, 64, 10]
}

results_arch = {}

for name, arch in architectures.items():
    print(f"\nTraining {name} architecture...")
    nn_temp = NeuralNetwork(arch, activation='relu', learning_rate=0.01)
    nn_temp.train(X_train_scaled_digits, y_train_scaled_digits, epochs=200, batch_size=32)
    
    y_pred_temp = nn_temp.predict(X_test_scaled_digits)
    accuracy_temp = accuracy_score(y_test_labels_digits, y_pred_temp)
    results_arch[name] = accuracy_temp

# Plot architecture comparison
plt.figure(figsize=(10, 6))
arch_names = list(results_arch.keys())
accuracies = list(results_arch.values())

plt.bar(arch_names, accuracies)
plt.ylabel('Accuracy')
plt.title('Architecture Comparison')
plt.ylim(0, 1)
plt.grid(True, axis='y')

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("Architecture Comparison:")
for name, acc in results_arch.items():
    print(f"{name}: {acc:.3f}")
```

## Conclusion

Neural networks are powerful models capable of learning complex patterns. Key takeaways:

### Benefits:
1. **Universal Approximation**: Can approximate any continuous function
2. **Feature Learning**: Automatically learn relevant features
3. **Scalability**: Can handle large datasets
4. **Flexibility**: Various architectures for different tasks

### Best Practices:
1. **Proper Initialization**: Use appropriate weight initialization
2. **Regularization**: Apply dropout, L2 regularization
3. **Batch Normalization**: Normalize activations
4. **Learning Rate Scheduling**: Adjust learning rate during training
5. **Early Stopping**: Prevent overfitting

### When to Use:
- **Complex Patterns**: When data has non-linear relationships
- **Large Datasets**: When sufficient data is available
- **Feature Learning**: When manual feature engineering is difficult
- **Multiple Tasks**: When the same architecture can handle various tasks

Neural networks have become the foundation of modern machine learning and continue to advance the field.

## Further Reading

- **Books**: "Deep Learning" by Goodfellow et al., "Neural Networks and Deep Learning" by Michael Nielsen
- **Papers**: Original backpropagation paper, CNN papers by LeCun, LSTM by Hochreiter & Schmidhuber
- **Online Resources**: CS231n course, CS224n course, PyTorch/TensorFlow tutorials
- **Practice**: ImageNet competitions, NLP tasks, reinforcement learning environments 