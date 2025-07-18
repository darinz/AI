# Linear Classification

## Introduction

Linear classification extends the concept of linear regression to classification problems, where the goal is to predict discrete class labels rather than continuous values. The fundamental idea is to find a linear decision boundary that separates different classes in the feature space.

## Mathematical Foundation

### The Classification Problem

Given:
- Input features: $X \in \mathbb{R}^{n \times d}$
- Target labels: $y \in \{0, 1, ..., K-1\}$ (for K classes)
- Training data: $\{(x_i, y_i)\}_{i=1}^n$

Goal: Find a function $f: \mathbb{R}^d \rightarrow \{0, 1, ..., K-1\}$ that maps inputs to class labels.

### Linear Decision Boundary

For binary classification, the decision boundary is defined by:
$$\mathbf{w}^T \mathbf{x} + b = 0$$

Where:
- $\mathbf{w} \in \mathbb{R}^d$ is the weight vector
- $b \in \mathbb{R}$ is the bias term
- $\mathbf{x} \in \mathbb{R}^d$ is the input vector

The classification rule is:
$$f(\mathbf{x}) = \begin{cases} 
1 & \text{if } \mathbf{w}^T \mathbf{x} + b > 0 \\
0 & \text{if } \mathbf{w}^T \mathbf{x} + b \leq 0
\end{cases}$$

## Logistic Regression

### Mathematical Formulation

Logistic regression models the probability of belonging to class 1 using the sigmoid function:

$$P(y = 1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

Where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function.

### Properties of the Sigmoid Function

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Plot sigmoid function
z = np.linspace(-10, 10, 100)
sigma_z = sigmoid(z)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(z, sigma_z, 'b-', linewidth=2)
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Sigmoid Function')
plt.grid(True)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)

# Plot derivative of sigmoid
sigma_prime = sigmoid(z) * (1 - sigmoid(z))
plt.subplot(1, 2, 2)
plt.plot(z, sigma_prime, 'g-', linewidth=2)
plt.xlabel('z')
plt.ylabel("σ'(z)")
plt.title('Derivative of Sigmoid Function')
plt.grid(True)

plt.tight_layout()
plt.show()

print("Sigmoid function properties:")
print(f"σ(0) = {sigmoid(0):.3f}")
print(f"σ(∞) = {sigmoid(100):.3f}")
print(f"σ(-∞) = {sigmoid(-100):.3f}")
```

### Cost Function: Cross-Entropy Loss

For binary classification, the cross-entropy loss is:

$$J(\mathbf{w}, b) = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

Where $\hat{y}_i = \sigma(\mathbf{w}^T \mathbf{x}_i + b)$.

**Why Cross-Entropy?**

1. **Convex**: Guarantees convergence to global minimum
2. **Penalizes Wrong Predictions**: Heavily penalizes confident wrong predictions
3. **Information Theory**: Measures information loss

```python
def cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    """Compute cross-entropy loss"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Demonstrate cross-entropy behavior
y_true = np.array([1, 1, 0, 0])
y_pred_correct = np.array([0.9, 0.8, 0.1, 0.2])
y_pred_wrong = np.array([0.1, 0.2, 0.9, 0.8])

loss_correct = cross_entropy_loss(y_true, y_pred_correct)
loss_wrong = cross_entropy_loss(y_true, y_pred_wrong)

print(f"Cross-entropy loss for correct predictions: {loss_correct:.4f}")
print(f"Cross-entropy loss for wrong predictions: {loss_wrong:.4f}")
```

### Gradient of Cross-Entropy Loss

The gradient with respect to the weights is:

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) \mathbf{x}_i$$

$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$

**Python Implementation:**

```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for iteration in range(self.max_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = np.mean(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute cost
            cost = cross_entropy_loss(y, predictions)
            self.cost_history.append(cost)
            
            # Check convergence
            if iteration > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                break
                
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Cost: {cost:.6f}")
    
    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

# Generate synthetic data
np.random.seed(42)
n_samples = 200
X = np.random.randn(n_samples, 2)
# Create a linear decision boundary
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Add some noise
noise = np.random.randn(n_samples) * 0.3
y = ((X[:, 0] + X[:, 1] + noise) > 0).astype(int)

# Train logistic regression
model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
model.fit(X, y)

# Plot results
plt.figure(figsize=(15, 5))

# Plot cost history
plt.subplot(1, 3, 1)
plt.plot(model.cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost History')
plt.grid(True)

# Plot decision boundary
plt.subplot(1, 3, 2)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.colorbar()

# Plot predictions vs actual
plt.subplot(1, 3, 3)
y_pred = model.predict(X)
plt.scatter(y, model.predict_proba(X), alpha=0.6)
plt.xlabel('Actual Class')
plt.ylabel('Predicted Probability')
plt.title('Predictions vs Actual')
plt.grid(True)

plt.tight_layout()
plt.show()

# Model evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

print(f"Model Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Confusion matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1])
plt.yticks([0, 1])

# Add text annotations
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()
```

## Linear Discriminant Analysis (LDA)

### Mathematical Foundation

LDA finds a linear combination of features that maximizes the separation between classes while minimizing the variance within each class.

### Fisher's Linear Discriminant

The optimal projection direction is given by:

$$\mathbf{w} = \mathbf{S}_W^{-1}(\mathbf{\mu}_1 - \mathbf{\mu}_0)$$

Where:
- $\mathbf{S}_W$ is the within-class scatter matrix
- $\mathbf{\mu}_1, \mathbf{\mu}_0$ are the class means

**Python Implementation:**

```python
class LinearDiscriminantAnalysis:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.class_means = None
        self.prior = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        
        # Calculate class means
        self.class_means = np.zeros((n_classes, n_features))
        self.prior = np.zeros(n_classes)
        
        for i, c in enumerate(classes):
            mask = (y == c)
            self.class_means[i] = np.mean(X[mask], axis=0)
            self.prior[i] = np.sum(mask) / n_samples
        
        # Calculate within-class scatter matrix
        S_W = np.zeros((n_features, n_features))
        for i, c in enumerate(classes):
            mask = (y == c)
            X_c = X[mask] - self.class_means[i]
            S_W += X_c.T.dot(X_c)
        
        # Calculate between-class scatter matrix
        overall_mean = np.mean(X, axis=0)
        S_B = np.zeros((n_features, n_features))
        for i, c in enumerate(classes):
            diff = self.class_means[i] - overall_mean
            S_B += self.prior[i] * np.outer(diff, diff)
        
        # Solve generalized eigenvalue problem
        eigenvals, eigenvecs = np.linalg.eigh(np.linalg.inv(S_W).dot(S_B))
        
        # Sort eigenvectors by eigenvalues
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Take the first eigenvector (for binary classification)
        self.weights = eigenvecs[:, 0]
        
        # Calculate bias
        self.bias = -0.5 * (self.weights.dot(self.class_means[0]) + 
                           self.weights.dot(self.class_means[1]))
    
    def predict(self, X):
        scores = X.dot(self.weights) + self.bias
        return (scores > 0).astype(int)
    
    def predict_proba(self, X):
        scores = X.dot(self.weights) + self.bias
        return 1 / (1 + np.exp(-scores))

# Compare LDA with Logistic Regression
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X, y)

# Plot comparison
plt.figure(figsize=(15, 5))

# LDA decision boundary
plt.subplot(1, 3, 1)
Z_lda = lda_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_lda = Z_lda.reshape(xx.shape)

plt.contourf(xx, yy, Z_lda, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('LDA Decision Boundary')

# Logistic Regression decision boundary
plt.subplot(1, 3, 2)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')

# Compare accuracies
y_pred_lda = lda_model.predict(X)
accuracy_lda = accuracy_score(y, y_pred_lda)

plt.subplot(1, 3, 3)
models = ['LDA', 'Logistic Regression']
accuracies = [accuracy_lda, accuracy]
plt.bar(models, accuracies)
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

print(f"LDA Accuracy: {accuracy_lda:.3f}")
print(f"Logistic Regression Accuracy: {accuracy:.3f}")
```

## Support Vector Machines (SVM)

### Mathematical Foundation

SVM finds the optimal hyperplane that maximizes the margin between classes. The margin is the distance between the hyperplane and the nearest data points from each class.

### Hard Margin SVM

For linearly separable data, the optimization problem is:

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2$$

Subject to: $y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1$ for all $i$

### Soft Margin SVM

For non-linearly separable data, we introduce slack variables:

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$

Subject to: $y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i$ and $\xi_i \geq 0$ for all $i$

**Python Implementation:**

```python
class SupportVectorMachine:
    def __init__(self, C=1.0, learning_rate=0.01, max_iterations=1000):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        # Convert labels to {-1, 1}
        y_binary = 2 * y - 1
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for iteration in range(self.max_iterations):
            for i in range(n_samples):
                # Check if point violates margin
                condition = y_binary[i] * (np.dot(X[i], self.weights) + self.bias)
                
                if condition < 1:
                    # Update weights and bias
                    self.weights += self.learning_rate * (self.C * y_binary[i] * X[i] - self.weights)
                    self.bias += self.learning_rate * self.C * y_binary[i]
                else:
                    # Update weights only
                    self.weights += self.learning_rate * (-self.weights)
    
    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        return (scores > 0).astype(int)
    
    def decision_function(self, X):
        return np.dot(X, self.weights) + self.bias

# Train SVM
svm_model = SupportVectorMachine(C=1.0, learning_rate=0.01, max_iterations=1000)
svm_model.fit(X, y)

# Plot SVM results
plt.figure(figsize=(15, 5))

# SVM decision boundary
plt.subplot(1, 3, 1)
Z_svm = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)

plt.contourf(xx, yy, Z_svm, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary')

# Support vectors (approximate)
decision_values = svm_model.decision_function(X)
support_vector_mask = np.abs(decision_values) <= 1.1
plt.scatter(X[support_vector_mask, 0], X[support_vector_mask, 1], 
           facecolors='none', edgecolors='red', s=100, linewidth=2, label='Support Vectors')
plt.legend()

# Compare margins
plt.subplot(1, 3, 2)
margins = np.abs(decision_values)
plt.scatter(X[:, 0], X[:, 1], c=margins, cmap='viridis', alpha=0.8)
plt.colorbar(label='Distance to Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Margin Visualization')

# Model comparison
plt.subplot(1, 3, 3)
y_pred_svm = svm_model.predict(X)
accuracy_svm = accuracy_score(y, y_pred_svm)

models = ['LDA', 'Logistic Regression', 'SVM']
accuracies = [accuracy_lda, accuracy, accuracy_svm]
plt.bar(models, accuracies)
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print(f"SVM Accuracy: {accuracy_svm:.3f}")
```

## Multi-class Classification

### One-vs-Rest (OvR) Strategy

For K classes, train K binary classifiers, each distinguishing one class from all others.

### One-vs-One (OvO) Strategy

Train $\frac{K(K-1)}{2}$ binary classifiers, each distinguishing between two classes.

**Python Implementation:**

```python
class MultiClassClassifier:
    def __init__(self, base_classifier, strategy='ovr'):
        self.base_classifier = base_classifier
        self.strategy = strategy
        self.classifiers = []
        self.classes = None
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        if self.strategy == 'ovr':
            # One-vs-Rest
            for i, c in enumerate(self.classes):
                # Create binary labels
                y_binary = (y == c).astype(int)
                
                # Train classifier
                classifier = type(self.base_classifier)()
                classifier.fit(X, y_binary)
                self.classifiers.append(classifier)
                
        elif self.strategy == 'ovo':
            # One-vs-One
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    # Create binary labels for classes i and j
                    mask = (y == self.classes[i]) | (y == self.classes[j])
                    X_binary = X[mask]
                    y_binary = (y[mask] == self.classes[i]).astype(int)
                    
                    # Train classifier
                    classifier = type(self.base_classifier)()
                    classifier.fit(X_binary, y_binary)
                    self.classifiers.append((i, j, classifier))
    
    def predict(self, X):
        if self.strategy == 'ovr':
            # Get predictions from all classifiers
            predictions = np.zeros((X.shape[0], len(self.classes)))
            for i, classifier in enumerate(self.classifiers):
                predictions[:, i] = classifier.predict_proba(X)[:, 1] if hasattr(classifier, 'predict_proba') else classifier.predict(X)
            
            # Return class with highest probability
            return self.classes[np.argmax(predictions, axis=1)]
            
        elif self.strategy == 'ovo':
            # Count votes for each class
            votes = np.zeros((X.shape[0], len(self.classes)))
            
            for i, j, classifier in self.classifiers:
                pred = classifier.predict(X)
                votes[:, i] += pred
                votes[:, j] += (1 - pred)
            
            # Return class with most votes
            return self.classes[np.argmax(votes, axis=1)]

# Generate multi-class data
np.random.seed(42)
n_samples = 300
X_multi = np.random.randn(n_samples, 2)

# Create three classes
y_multi = np.zeros(n_samples)
y_multi[X_multi[:, 0] + X_multi[:, 1] > 1] = 1
y_multi[X_multi[:, 0] + X_multi[:, 1] < -1] = 2

# Train multi-class classifiers
ovr_classifier = MultiClassClassifier(LogisticRegression(), strategy='ovr')
ovo_classifier = MultiClassClassifier(LogisticRegression(), strategy='ovo')

ovr_classifier.fit(X_multi, y_multi)
ovo_classifier.fit(X_multi, y_multi)

# Plot results
plt.figure(figsize=(15, 5))

# One-vs-Rest
plt.subplot(1, 3, 1)
y_pred_ovr = ovr_classifier.predict(X_multi)
plt.scatter(X_multi[:, 0], X_multi[:, 1], c=y_pred_ovr, alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('One-vs-Rest Classification')

# One-vs-One
plt.subplot(1, 3, 2)
y_pred_ovo = ovo_classifier.predict(X_multi)
plt.scatter(X_multi[:, 0], X_multi[:, 1], c=y_pred_ovo, alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('One-vs-One Classification')

# True labels
plt.subplot(1, 3, 3)
plt.scatter(X_multi[:, 0], X_multi[:, 1], c=y_multi, alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('True Labels')

plt.tight_layout()
plt.show()

# Compare accuracies
accuracy_ovr = accuracy_score(y_multi, y_pred_ovr)
accuracy_ovo = accuracy_score(y_multi, y_pred_ovo)

print(f"One-vs-Rest Accuracy: {accuracy_ovr:.3f}")
print(f"One-vs-One Accuracy: {accuracy_ovo:.3f}")
```

## Model Evaluation for Classification

### 1. Confusion Matrix

```python
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

# Plot confusion matrices for all models
plot_confusion_matrix(y, y_pred, 'Logistic Regression')
plot_confusion_matrix(y, y_pred_lda, 'LDA')
plot_confusion_matrix(y, y_pred_svm, 'SVM')
```

### 2. ROC Curve and AUC

```python
from sklearn.metrics import roc_curve, auc

def plot_roc_curves(X, y, models, model_names):
    plt.figure(figsize=(10, 8))
    
    for model, name in zip(models, model_names):
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 1]  # Probability of positive class
        else:
            y_proba = model.decision_function(X)
        
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot ROC curves
models = [model, lda_model, svm_model]
model_names = ['Logistic Regression', 'LDA', 'SVM']
plot_roc_curves(X, y, models, model_names)
```

### 3. Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve

def plot_precision_recall_curves(X, y, models, model_names):
    plt.figure(figsize=(10, 8))
    
    for model, name in zip(models, model_names):
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 1]
        else:
            y_proba = model.decision_function(X)
        
        precision, recall, _ = precision_recall_curve(y, y_proba)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot precision-recall curves
plot_precision_recall_curves(X, y, models, model_names)
```

## Real-world Example: Breast Cancer Classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load breast cancer dataset
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models_cancer = {
    'Logistic Regression': LogisticRegression(learning_rate=0.1, max_iterations=1000),
    'LDA': LinearDiscriminantAnalysis(),
    'SVM': SupportVectorMachine(C=1.0, learning_rate=0.01, max_iterations=1000)
}

results_cancer = {}

for name, model in models_cancer.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    results_cancer[name] = {
        'Accuracy': accuracy,
        'Predictions': y_pred,
        'Model': model
    }
    
    print(f"{name}: Accuracy = {accuracy:.3f}")

# Compare models
plt.figure(figsize=(12, 4))

# Accuracy comparison
plt.subplot(1, 3, 1)
model_names = list(results_cancer.keys())
accuracies = [results_cancer[name]['Accuracy'] for name in model_names]
plt.bar(model_names, accuracies)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45)

# ROC curves
plt.subplot(1, 3, 2)
for name, result in results_cancer.items():
    model = result['Model']
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_scaled)
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
    else:
        y_proba = model.decision_function(X_test_scaled)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

# Confusion matrix for best model
best_model_name = max(results_cancer.keys(), key=lambda x: results_cancer[x]['Accuracy'])
best_predictions = results_cancer[best_model_name]['Predictions']

plt.subplot(1, 3, 3)
cm = confusion_matrix(y_test, best_predictions)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()

print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {results_cancer[best_model_name]['Accuracy']:.3f}")
```

## Conclusion

Linear classification provides a solid foundation for understanding classification problems. Key takeaways:

1. **Logistic Regression**: Probabilistic approach with cross-entropy loss
2. **LDA**: Optimal separation under Gaussian assumptions
3. **SVM**: Margin maximization with support vectors
4. **Multi-class**: OvR and OvO strategies
5. **Evaluation**: Multiple metrics provide comprehensive assessment
6. **Feature Scaling**: Important for convergence and performance

Linear classifiers are interpretable, computationally efficient, and often serve as strong baselines for classification tasks.

## Further Reading

- **Books**: "Pattern Recognition and Machine Learning" by Bishop, "The Elements of Statistical Learning" by Hastie et al.
- **Papers**: Original SVM paper by Vapnik (1995), LDA by Fisher (1936)
- **Online Resources**: StatQuest videos, scikit-learn documentation
- **Practice**: UCI Machine Learning Repository, Kaggle competitions 