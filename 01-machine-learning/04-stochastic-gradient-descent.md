# Stochastic Gradient Descent

## Introduction

Stochastic Gradient Descent (SGD) is a fundamental optimization algorithm used in machine learning to minimize cost functions. Unlike batch gradient descent that uses the entire dataset to compute gradients, SGD uses random samples (or mini-batches) to estimate gradients, making it computationally efficient for large datasets.

## Mathematical Foundation

### The Optimization Problem

Given a cost function $J(\theta)$ where $\theta$ represents the model parameters, we want to find:

$$\theta^* = \arg\min_{\theta} J(\theta)$$

The cost function is typically the average loss over all training examples:

$$J(\theta) = \frac{1}{n} \sum_{i=1}^{n} L(f(x_i; \theta), y_i)$$

Where:
- $L$ is the loss function
- $f(x_i; \theta)$ is the model prediction
- $(x_i, y_i)$ are training examples

### Gradient Descent vs Stochastic Gradient Descent

**Batch Gradient Descent:**
$$\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t) = \theta_t - \alpha \frac{1}{n} \sum_{i=1}^{n} \nabla L(f(x_i; \theta_t), y_i)$$

**Stochastic Gradient Descent:**
$$\theta_{t+1} = \theta_t - \alpha \nabla L(f(x_i; \theta_t), y_i)$$

Where $i$ is randomly selected from $\{1, 2, ..., n\}$.

**Mini-batch Gradient Descent:**
$$\theta_{t+1} = \theta_t - \alpha \frac{1}{m} \sum_{i \in B_t} \nabla L(f(x_i; \theta_t), y_i)$$

Where $B_t$ is a mini-batch of size $m$.

## Algorithm Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

def generate_data(n_samples=1000, n_features=10, noise=0.1):
    """Generate synthetic data for regression"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    y = X.dot(true_weights) + noise * np.random.randn(n_samples)
    return X, y, true_weights

# Generate data
X, y, true_weights = generate_data(n_samples=1000, n_features=5)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Data shape: {X.shape}")
print(f"True weights: {true_weights}")
```

## Implementation of Different Gradient Descent Variants

### 1. Batch Gradient Descent

```python
class BatchGradientDescent:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.cost_history = []
        self.weights = None
        self.bias = None
        
    def compute_cost(self, X, y, weights, bias):
        """Compute MSE cost"""
        predictions = X.dot(weights) + bias
        return np.mean((predictions - y) ** 2)
    
    def compute_gradients(self, X, y, weights, bias):
        """Compute gradients for all samples"""
        n_samples = X.shape[0]
        predictions = X.dot(weights) + bias
        error = predictions - y
        
        # Gradients
        dw = (2/n_samples) * X.T.dot(error)
        db = (2/n_samples) * np.sum(error)
        
        return dw, db
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for iteration in range(self.max_iterations):
            # Compute gradients using all samples
            dw, db = self.compute_gradients(X, y, self.weights, self.bias)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost
            cost = self.compute_cost(X, y, self.weights, self.bias)
            self.cost_history.append(cost)
            
            # Check convergence
            if iteration > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                print(f"Converged at iteration {iteration}")
                break
                
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Cost: {cost:.6f}")
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias

# Train batch gradient descent
batch_gd = BatchGradientDescent(learning_rate=0.01, max_iterations=1000)
batch_gd.fit(X_scaled, y)

print(f"Batch GD - Final weights: {batch_gd.weights}")
print(f"Batch GD - Final bias: {batch_gd.bias:.6f}")
```

### 2. Stochastic Gradient Descent

```python
class StochasticGradientDescent:
    def __init__(self, learning_rate=0.01, max_epochs=100, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.cost_history = []
        self.weights = None
        self.bias = None
        
    def compute_cost(self, X, y, weights, bias):
        """Compute MSE cost"""
        predictions = X.dot(weights) + bias
        return np.mean((predictions - y) ** 2)
    
    def compute_gradient_single(self, x, y, weights, bias):
        """Compute gradient for a single sample"""
        prediction = x.dot(weights) + bias
        error = prediction - y
        
        # Gradients
        dw = 2 * error * x
        db = 2 * error
        
        return dw, db
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for epoch in range(self.max_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            
            for i in range(n_samples):
                # Compute gradient for single sample
                dw, db = self.compute_gradient_single(
                    X_shuffled[i], y_shuffled[i], self.weights, self.bias
                )
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Accumulate cost
                epoch_cost += (X_shuffled[i].dot(self.weights) + self.bias - y_shuffled[i]) ** 2
            
            # Average cost for the epoch
            epoch_cost /= n_samples
            self.cost_history.append(epoch_cost)
            
            # Check convergence
            if epoch > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                print(f"Converged at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Cost: {epoch_cost:.6f}")
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias

# Train stochastic gradient descent
sgd = StochasticGradientDescent(learning_rate=0.01, max_epochs=50)
sgd.fit(X_scaled, y)

print(f"SGD - Final weights: {sgd.weights}")
print(f"SGD - Final bias: {sgd.bias:.6f}")
```

### 3. Mini-batch Gradient Descent

```python
class MiniBatchGradientDescent:
    def __init__(self, learning_rate=0.01, batch_size=32, max_epochs=100, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.cost_history = []
        self.weights = None
        self.bias = None
        
    def compute_cost(self, X, y, weights, bias):
        """Compute MSE cost"""
        predictions = X.dot(weights) + bias
        return np.mean((predictions - y) ** 2)
    
    def compute_gradients_batch(self, X_batch, y_batch, weights, bias):
        """Compute gradients for a mini-batch"""
        batch_size = X_batch.shape[0]
        predictions = X_batch.dot(weights) + bias
        error = predictions - y_batch
        
        # Gradients
        dw = (2/batch_size) * X_batch.T.dot(error)
        db = (2/batch_size) * np.sum(error)
        
        return dw, db
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for epoch in range(self.max_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            n_batches = 0
            
            # Process mini-batches
            for i in range(0, n_samples, self.batch_size):
                # Get mini-batch
                end_idx = min(i + self.batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                # Compute gradients for mini-batch
                dw, db = self.compute_gradients_batch(X_batch, y_batch, self.weights, self.bias)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Accumulate cost
                batch_cost = self.compute_cost(X_batch, y_batch, self.weights, self.bias)
                epoch_cost += batch_cost
                n_batches += 1
            
            # Average cost for the epoch
            epoch_cost /= n_batches
            self.cost_history.append(epoch_cost)
            
            # Check convergence
            if epoch > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                print(f"Converged at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Cost: {epoch_cost:.6f}")
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias

# Train mini-batch gradient descent
mini_batch_gd = MiniBatchGradientDescent(learning_rate=0.01, batch_size=32, max_epochs=50)
mini_batch_gd.fit(X_scaled, y)

print(f"Mini-batch GD - Final weights: {mini_batch_gd.weights}")
print(f"Mini-batch GD - Final bias: {mini_batch_gd.bias:.6f}")
```

## Comparison of Different Variants

```python
# Compare convergence
plt.figure(figsize=(15, 5))

# Plot cost history
plt.subplot(1, 3, 1)
plt.plot(batch_gd.cost_history, label='Batch GD', linewidth=2)
plt.plot(sgd.cost_history, label='SGD', linewidth=2)
plt.plot(mini_batch_gd.cost_history, label='Mini-batch GD', linewidth=2)
plt.xlabel('Iteration/Epoch')
plt.ylabel('Cost (MSE)')
plt.title('Cost Convergence Comparison')
plt.legend()
plt.grid(True)

# Plot predictions vs actual
plt.subplot(1, 3, 2)
y_pred_batch = batch_gd.predict(X_scaled)
y_pred_sgd = sgd.predict(X_scaled)
y_pred_mini = mini_batch_gd.predict(X_scaled)

plt.scatter(y, y_pred_batch, alpha=0.6, label='Batch GD')
plt.scatter(y, y_pred_sgd, alpha=0.6, label='SGD')
plt.scatter(y, y_pred_mini, alpha=0.6, label='Mini-batch GD')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predictions vs Actual')
plt.legend()
plt.grid(True)

# Compare final weights
plt.subplot(1, 3, 3)
x_pos = np.arange(len(true_weights))
width = 0.25

plt.bar(x_pos - width, true_weights, width, label='True Weights', alpha=0.8)
plt.bar(x_pos, batch_gd.weights, width, label='Batch GD', alpha=0.8)
plt.bar(x_pos + width, sgd.weights, width, label='SGD', alpha=0.8)

plt.xlabel('Feature Index')
plt.ylabel('Weight Value')
plt.title('Weight Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare final costs
final_costs = [
    batch_gd.cost_history[-1],
    sgd.cost_history[-1],
    mini_batch_gd.cost_history[-1]
]

methods = ['Batch GD', 'SGD', 'Mini-batch GD']
plt.figure(figsize=(10, 6))
plt.bar(methods, final_costs)
plt.ylabel('Final Cost (MSE)')
plt.title('Final Cost Comparison')
plt.grid(True, axis='y')
plt.show()

print("Final Cost Comparison:")
for method, cost in zip(methods, final_costs):
    print(f"{method}: {cost:.6f}")
```

## Learning Rate Scheduling

### 1. Fixed Learning Rate

```python
def train_with_fixed_lr(learning_rate, X, y, max_epochs=50):
    """Train with fixed learning rate"""
    sgd = StochasticGradientDescent(learning_rate=learning_rate, max_epochs=max_epochs)
    sgd.fit(X, y)
    return sgd.cost_history

# Test different fixed learning rates
learning_rates = [0.001, 0.01, 0.1, 0.5]
cost_histories = {}

for lr in learning_rates:
    cost_histories[lr] = train_with_fixed_lr(lr, X_scaled, y)

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
for lr, history in cost_histories.items():
    plt.plot(history, label=f'LR = {lr}')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Fixed Learning Rate Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
final_costs = [history[-1] for history in cost_histories.values()]
plt.bar([str(lr) for lr in learning_rates], final_costs)
plt.xlabel('Learning Rate')
plt.ylabel('Final Cost')
plt.title('Final Cost vs Learning Rate')
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()
```

### 2. Learning Rate Decay

```python
class SGDWithDecay:
    def __init__(self, initial_lr=0.1, decay_rate=0.95, max_epochs=100):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.max_epochs = max_epochs
        self.cost_history = []
        self.lr_history = []
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for epoch in range(self.max_epochs):
            # Calculate current learning rate
            current_lr = self.initial_lr * (self.decay_rate ** epoch)
            self.lr_history.append(current_lr)
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            
            for i in range(n_samples):
                # Compute gradient for single sample
                prediction = X_shuffled[i].dot(self.weights) + self.bias
                error = prediction - y_shuffled[i]
                
                dw = 2 * error * X_shuffled[i]
                db = 2 * error
                
                # Update parameters with current learning rate
                self.weights -= current_lr * dw
                self.bias -= current_lr * db
                
                epoch_cost += error ** 2
            
            epoch_cost /= n_samples
            self.cost_history.append(epoch_cost)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, LR: {current_lr:.6f}, Cost: {epoch_cost:.6f}")
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias

# Train with learning rate decay
sgd_decay = SGDWithDecay(initial_lr=0.1, decay_rate=0.95, max_epochs=100)
sgd_decay.fit(X_scaled, y)

# Plot learning rate decay
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(sgd_decay.lr_history)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Decay')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(sgd_decay.cost_history)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cost with Learning Rate Decay')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 3. Adaptive Learning Rates

```python
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, max_epochs=100):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_epochs = max_epochs
        self.cost_history = []
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Initialize momentum and RMS
        m_w = np.zeros(n_features)
        v_w = np.zeros(n_features)
        m_b = 0
        v_b = 0
        
        for epoch in range(self.max_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            
            for i in range(n_samples):
                # Compute gradient
                prediction = X_shuffled[i].dot(self.weights) + self.bias
                error = prediction - y_shuffled[i]
                
                dw = 2 * error * X_shuffled[i]
                db = 2 * error
                
                # Update momentum (first moment)
                m_w = self.beta1 * m_w + (1 - self.beta1) * dw
                m_b = self.beta1 * m_b + (1 - self.beta1) * db
                
                # Update RMS (second moment)
                v_w = self.beta2 * v_w + (1 - self.beta2) * (dw ** 2)
                v_b = self.beta2 * v_b + (1 - self.beta2) * (db ** 2)
                
                # Bias correction
                m_w_corrected = m_w / (1 - self.beta1 ** (epoch * n_samples + i + 1))
                m_b_corrected = m_b / (1 - self.beta1 ** (epoch * n_samples + i + 1))
                v_w_corrected = v_w / (1 - self.beta2 ** (epoch * n_samples + i + 1))
                v_b_corrected = v_b / (1 - self.beta2 ** (epoch * n_samples + i + 1))
                
                # Update parameters
                self.weights -= self.learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)
                self.bias -= self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
                
                epoch_cost += error ** 2
            
            epoch_cost /= n_samples
            self.cost_history.append(epoch_cost)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Cost: {epoch_cost:.6f}")
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias

# Train with Adam optimizer
adam = AdamOptimizer(learning_rate=0.001, max_epochs=50)
adam.fit(X_scaled, y)

# Compare with regular SGD
sgd_regular = StochasticGradientDescent(learning_rate=0.01, max_epochs=50)
sgd_regular.fit(X_scaled, y)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(adam.cost_history, label='Adam', linewidth=2)
plt.plot(sgd_regular.cost_history, label='SGD', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Adam vs SGD Convergence')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
y_pred_adam = adam.predict(X_scaled)
y_pred_sgd = sgd_regular.predict(X_scaled)

plt.scatter(y, y_pred_adam, alpha=0.6, label='Adam')
plt.scatter(y, y_pred_sgd, alpha=0.6, label='SGD')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predictions Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Practical Considerations

### 1. Feature Scaling

```python
# Demonstrate importance of feature scaling
X_unscaled = np.random.rand(1000, 3) * 1000  # Large scale features
y_unscaled = 2 * X_unscaled[:, 0] + 3 * X_unscaled[:, 1] + 1 + np.random.normal(0, 0.1, 1000)

# Train without scaling
sgd_unscaled = StochasticGradientDescent(learning_rate=0.0001, max_epochs=50)
sgd_unscaled.fit(X_unscaled, y_unscaled)

# Train with scaling
scaler = StandardScaler()
X_scaled_unscaled = scaler.fit_transform(X_unscaled)
sgd_scaled = StochasticGradientDescent(learning_rate=0.01, max_epochs=50)
sgd_scaled.fit(X_scaled_unscaled, y_unscaled)

# Compare convergence
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(sgd_unscaled.cost_history, label='Without Scaling', linewidth=2)
plt.plot(sgd_scaled.cost_history, label='With Scaling', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Convergence with/without Scaling')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
final_costs = [sgd_unscaled.cost_history[-1], sgd_scaled.cost_history[-1]]
plt.bar(['Without Scaling', 'With Scaling'], final_costs)
plt.ylabel('Final Cost')
plt.title('Final Cost Comparison')
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()

print(f"Final cost without scaling: {sgd_unscaled.cost_history[-1]:.6f}")
print(f"Final cost with scaling: {sgd_scaled.cost_history[-1]:.6f}")
```

### 2. Early Stopping

```python
class SGDWithEarlyStopping:
    def __init__(self, learning_rate=0.01, max_epochs=100, patience=10, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.tolerance = tolerance
        self.cost_history = []
        self.weights = None
        self.bias = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        best_cost = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_cost = 0
            
            for i in range(n_samples):
                # Compute gradient for single sample
                prediction = X_shuffled[i].dot(self.weights) + self.bias
                error = prediction - y_shuffled[i]
                
                dw = 2 * error * X_shuffled[i]
                db = 2 * error
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                epoch_cost += error ** 2
            
            epoch_cost /= n_samples
            self.cost_history.append(epoch_cost)
            
            # Early stopping check
            if epoch_cost < best_cost - self.tolerance:
                best_cost = epoch_cost
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Cost: {epoch_cost:.6f}")
    
    def predict(self, X):
        return X.dot(self.weights) + self.bias

# Train with early stopping
sgd_early_stop = SGDWithEarlyStopping(learning_rate=0.01, max_epochs=200, patience=15)
sgd_early_stop.fit(X_scaled, y)

# Compare with regular SGD
sgd_regular_long = StochasticGradientDescent(learning_rate=0.01, max_epochs=200)
sgd_regular_long.fit(X_scaled, y)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(sgd_early_stop.cost_history, label='With Early Stopping', linewidth=2)
plt.plot(sgd_regular_long.cost_history, label='Without Early Stopping', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Early Stopping Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(sgd_early_stop.cost_history[-50:], label='With Early Stopping', linewidth=2)
plt.plot(sgd_regular_long.cost_history[-50:], label='Without Early Stopping', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Final 50 Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Real-world Example: Large-scale Linear Regression

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate large dataset
X_large, y_large, true_weights_large = make_regression(
    n_samples=10000, n_features=100, noise=0.1, random_state=42
)

# Split data
X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(
    X_large, y_large, test_size=0.2, random_state=42
)

# Scale features
scaler_large = StandardScaler()
X_train_scaled_large = scaler_large.fit_transform(X_train_large)
X_test_scaled_large = scaler_large.transform(X_test_large)

print(f"Training set size: {X_train_large.shape}")
print(f"Test set size: {X_test_large.shape}")

# Compare different optimizers
optimizers = {
    'Batch GD': BatchGradientDescent(learning_rate=0.01, max_iterations=100),
    'SGD': StochasticGradientDescent(learning_rate=0.01, max_epochs=50),
    'Mini-batch GD': MiniBatchGradientDescent(learning_rate=0.01, batch_size=64, max_epochs=50),
    'Adam': AdamOptimizer(learning_rate=0.001, max_epochs=50)
}

results_large = {}

for name, optimizer in optimizers.items():
    print(f"\nTraining {name}...")
    optimizer.fit(X_train_scaled_large, y_train_large)
    
    # Make predictions
    y_pred = optimizer.predict(X_test_scaled_large)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_large, y_pred)
    r2 = r2_score(y_test_large, y_pred)
    
    results_large[name] = {
        'MSE': mse,
        'R²': r2,
        'Cost History': optimizer.cost_history
    }
    
    print(f"{name} - MSE: {mse:.6f}, R²: {r2:.6f}")

# Plot results
plt.figure(figsize=(15, 5))

# Cost convergence
plt.subplot(1, 3, 1)
for name, result in results_large.items():
    plt.plot(result['Cost History'], label=name, linewidth=2)
plt.xlabel('Iteration/Epoch')
plt.ylabel('Cost')
plt.title('Cost Convergence on Large Dataset')
plt.legend()
plt.grid(True)

# MSE comparison
plt.subplot(1, 3, 2)
mse_values = [result['MSE'] for result in results_large.values()]
plt.bar(results_large.keys(), mse_values)
plt.ylabel('Mean Squared Error')
plt.title('MSE Comparison')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

# R² comparison
plt.subplot(1, 3, 3)
r2_values = [result['R²'] for result in results_large.values()]
plt.bar(results_large.keys(), r2_values)
plt.ylabel('R² Score')
plt.title('R² Comparison')
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()

# Print summary
print("\nSummary of Results:")
for name, result in results_large.items():
    print(f"{name}:")
    print(f"  MSE: {result['MSE']:.6f}")
    print(f"  R²: {result['R²']:.6f}")
    print(f"  Final Cost: {result['Cost History'][-1]:.6f}")
```

## Conclusion

Stochastic Gradient Descent is a powerful optimization algorithm that offers several advantages:

### Key Benefits:
1. **Computational Efficiency**: Scales well with large datasets
2. **Memory Efficiency**: Processes data in small batches
3. **Escape Local Minima**: Stochastic nature helps escape local optima
4. **Online Learning**: Can update model with new data

### Best Practices:
1. **Feature Scaling**: Essential for convergence
2. **Learning Rate Tuning**: Critical for performance
3. **Mini-batch Size**: Balance between efficiency and stability
4. **Early Stopping**: Prevents overfitting
5. **Adaptive Learning Rates**: Adam, RMSprop for better convergence

### When to Use:
- **Large datasets**: When batch gradient descent is too slow
- **Online learning**: When data arrives incrementally
- **Deep learning**: Foundation for training neural networks
- **Real-time systems**: When quick updates are needed

SGD remains the foundation for modern machine learning optimization, especially in deep learning where it's used in various forms (Adam, RMSprop, etc.).

## Further Reading

- **Books**: "Deep Learning" by Goodfellow et al., "Pattern Recognition and Machine Learning" by Bishop
- **Papers**: Original SGD paper by Robbins & Monro (1951), Adam paper by Kingma & Ba (2014)
- **Online Resources**: CS231n course notes, Distill.pub articles
- **Practice**: Implement SGD from scratch, experiment with different datasets 