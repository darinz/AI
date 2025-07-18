# 09 - Generalization in Machine Learning

## Overview

Generalization is the ability of a machine learning model to perform well on unseen data. It's the core goal of machine learning - creating models that can make accurate predictions beyond the training data.

## The Generalization Problem

### Overfitting vs Underfitting

**Overfitting**: Model learns training data too well, including noise and irrelevant patterns
**Underfitting**: Model is too simple to capture the underlying patterns in the data

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y_true = 2 * X.flatten() + 1 + 0.5 * np.sin(X.flatten())
y_noisy = y_true + np.random.normal(0, 0.5, 100)

# Create models with different complexities
models = {
    'Underfitting (Linear)': Pipeline([
        ('poly', PolynomialFeatures(degree=1)),
        ('reg', LinearRegression())
    ]),
    'Good Fit (Cubic)': Pipeline([
        ('poly', PolynomialFeatures(degree=3)),
        ('reg', LinearRegression())
    ]),
    'Overfitting (High Degree)': Pipeline([
        ('poly', PolynomialFeatures(degree=15)),
        ('reg', LinearRegression())
    ])
}

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (name, model) in enumerate(models.items()):
    model.fit(X, y_noisy)
    y_pred = model.predict(X)
    
    axes[i].scatter(X, y_noisy, alpha=0.6, label='Data')
    axes[i].plot(X, y_true, 'g-', linewidth=2, label='True Function')
    axes[i].plot(X, y_pred, 'r-', linewidth=2, label='Model Prediction')
    axes[i].set_title(f'{name}\nMSE: {mean_squared_error(y_noisy, y_pred):.3f}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Bias-Variance Tradeoff

The fundamental tradeoff in machine learning:

- **Bias**: Error due to overly simplistic assumptions
- **Variance**: Error due to model sensitivity to training data

```python
def bias_variance_decomposition(model, X_train, y_train, X_test, y_test, n_iterations=100):
    """Estimate bias and variance of a model"""
    predictions = []
    
    for _ in range(n_iterations):
        # Bootstrap sample
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot, y_boot = X_train[indices], y_train[indices]
        
        # Train model
        model.fit(X_boot, y_boot)
        pred = model.predict(X_test)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate bias and variance
    mean_pred = np.mean(predictions, axis=0)
    bias = np.mean((mean_pred - y_test) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    
    return bias, variance, bias + variance

# Example with different model complexities
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

X_train = X[:80]
y_train = y_noisy[:80]
X_test = X[80:]
y_test = y_noisy[80:]

models_complexity = {
    'Low Complexity (Ridge α=10)': Ridge(alpha=10),
    'Medium Complexity (Ridge α=1)': Ridge(alpha=1),
    'High Complexity (Random Forest)': RandomForestRegressor(n_estimators=100, max_depth=10)
}

print("Bias-Variance Decomposition:")
print("-" * 50)
for name, model in models_complexity.items():
    bias, variance, total = bias_variance_decomposition(model, X_train, y_train, X_test, y_test)
    print(f"{name}:")
    print(f"  Bias: {bias:.4f}")
    print(f"  Variance: {variance:.4f}")
    print(f"  Total Error: {total:.4f}")
    print()
```

## Cross-Validation

### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Different cross-validation strategies
cv_strategies = {
    '5-Fold CV': KFold(n_splits=5, shuffle=True, random_state=42),
    '10-Fold CV': KFold(n_splits=10, shuffle=True, random_state=42),
    'Leave-One-Out': KFold(n_splits=len(X), shuffle=True, random_state=42)
}

model = LogisticRegression(max_iter=1000)

print("Cross-Validation Results:")
print("-" * 40)
for name, cv in cv_strategies.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"{name}:")
    print(f"  Mean Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    print(f"  Individual Scores: {scores}")
    print()
```

### Stratified Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

# Stratified k-fold for classification
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=stratified_cv, scoring='accuracy')

print("Stratified 5-Fold CV:")
print(f"Mean Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

## Regularization Techniques

### L1 Regularization (Lasso)

```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Generate data with some irrelevant features
np.random.seed(42)
X = np.random.randn(100, 20)
# Only first 5 features are relevant
y = 2*X[:, 0] + 1.5*X[:, 1] - X[:, 2] + 0.5*X[:, 3] + X[:, 4] + np.random.normal(0, 0.1, 100)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lasso with different alpha values
alphas = [0.001, 0.01, 0.1, 1, 10]
lasso_models = {}

for alpha in alphas:
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X_scaled, y)
    lasso_models[alpha] = lasso

# Plot coefficient paths
plt.figure(figsize=(12, 6))
for alpha in alphas:
    plt.plot(range(20), lasso_models[alpha].coef_, 
             marker='o', label=f'α={alpha}')

plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficient Paths')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Count non-zero coefficients
print("Number of non-zero coefficients:")
for alpha in alphas:
    n_nonzero = np.sum(lasso_models[alpha].coef_ != 0)
    print(f"α={alpha}: {n_nonzero} features selected")
```

### L2 Regularization (Ridge)

```python
from sklearn.linear_model import Ridge

ridge_models = {}
for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_scaled, y)
    ridge_models[alpha] = ridge

# Compare coefficient magnitudes
plt.figure(figsize=(12, 6))
for alpha in alphas:
    plt.plot(range(20), ridge_models[alpha].coef_, 
             marker='o', label=f'α={alpha}')

plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Ridge Coefficient Paths')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Elastic Net

```python
from sklearn.linear_model import ElasticNet

# Elastic Net combines L1 and L2 regularization
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X_scaled, y)

print("Elastic Net Results:")
print(f"Non-zero coefficients: {np.sum(elastic_net.coef_ != 0)}")
print(f"L1 ratio: {elastic_net.l1_ratio}")
```

## Early Stopping

```python
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Generate data
X = np.random.randn(1000, 10)
y = np.sum(X[:, :5], axis=1) + np.random.normal(0, 0.1, 1000)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural network with early stopping
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)

mlp.fit(X_train, y_train)

print("Early Stopping Results:")
print(f"Training iterations: {mlp.n_iter_}")
print(f"Training score: {mlp.score(X_train, y_train):.3f}")
print(f"Validation score: {mlp.score(X_val, y_val):.3f}")
```

## Model Selection

### Grid Search with Cross-Validation

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Grid search for SVM parameters
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

svm = SVC(random_state=42)
grid_search = GridSearchCV(
    svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)
```

### Learning Curves

```python
from sklearn.model_selection import learning_curve

def plot_learning_curves(estimator, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title(f'Learning Curves - {title}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Compare different models
models = {
    'Linear SVM': SVC(kernel='linear', C=1),
    'RBF SVM': SVC(kernel='rbf', C=1, gamma='scale'),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    plot_learning_curves(model, X, y, name)
```

## Best Practices

### 1. Data Splitting Strategy

```python
# Proper train/validation/test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

### 2. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Always scale features for regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

### 3. Model Complexity Selection

```python
# Use validation set to select model complexity
complexities = [1, 3, 5, 10, 15, 20]
val_scores = []

for degree in complexities:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('reg', LinearRegression())
    ])
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    val_scores.append(score)

best_degree = complexities[np.argmax(val_scores)]
print(f"Best polynomial degree: {best_degree}")
```

### 4. Ensemble Methods

```python
from sklearn.ensemble import VotingClassifier, BaggingClassifier

# Ensemble of different models
estimators = [
    ('svm', SVC(probability=True, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('lr', LogisticRegression(random_state=42))
]

ensemble = VotingClassifier(estimators=estimators, voting='soft')
ensemble.fit(X_train, y_train)

print(f"Ensemble validation score: {ensemble.score(X_val, y_val):.3f}")
```

## Summary

Generalization is the cornerstone of machine learning. Key principles:

1. **Balance bias and variance** through appropriate model complexity
2. **Use cross-validation** to estimate generalization performance
3. **Apply regularization** to prevent overfitting
4. **Monitor learning curves** to detect overfitting/underfitting
5. **Use proper data splitting** (train/validation/test)
6. **Scale features** when using regularization
7. **Consider ensemble methods** for improved generalization

The goal is to create models that generalize well to unseen data while avoiding both overfitting and underfitting. 