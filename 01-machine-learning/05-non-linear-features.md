# Non-linear Features

## Introduction

Non-linear features are transformations of original input features that allow linear models to capture complex, non-linear relationships in the data. By creating these transformed features, we can maintain the computational efficiency and interpretability of linear models while extending their modeling capabilities.

## Mathematical Foundation

### The Problem with Linear Models

Linear models assume a linear relationship between features and target:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n$$

However, real-world relationships are often non-linear. Non-linear features solve this by transforming the input space:

$$y = \beta_0 + \beta_1 \phi_1(x) + \beta_2 \phi_2(x) + ... + \beta_k \phi_k(x)$$

Where $\phi_i(x)$ are non-linear transformations of the original features.

### Feature Space Transformation

The key insight is that we can transform the input space to make non-linear relationships linear in the transformed space:

$$\Phi: \mathbb{R}^d \rightarrow \mathbb{R}^k$$

Where $k$ can be larger than $d$ (dimensionality expansion).

## Common Non-linear Transformations

### 1. Polynomial Features

Polynomial features capture polynomial relationships between variables.

**Mathematical Formulation:**
$$\phi(x) = [1, x, x^2, x^3, ..., x^d]$$

**Python Implementation:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def generate_nonlinear_data(n_samples=200, noise=0.1):
    """Generate data with non-linear relationship"""
    np.random.seed(42)
    X = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
    y = 2 * X[:, 0] + 0.5 * X[:, 0]**2 + noise * np.random.randn(n_samples)
    return X, y

# Generate data
X, y = generate_nonlinear_data()

# Visualize the non-linear relationship
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X, y, alpha=0.6)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Non-linear Data')
plt.grid(True)

# Fit linear model
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

plt.subplot(1, 3, 2)
plt.scatter(X, y, alpha=0.6, label='Data')
plt.plot(X, y_pred_linear, 'r-', label='Linear Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Model Fit')
plt.legend()
plt.grid(True)

# Fit polynomial model
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)

plt.subplot(1, 3, 3)
plt.scatter(X, y, alpha=0.6, label='Data')
plt.plot(X, y_pred_poly, 'g-', label='Polynomial Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Model Fit')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare performance
mse_linear = mean_squared_error(y, y_pred_linear)
mse_poly = mean_squared_error(y, y_pred_poly)

print(f"Linear Model MSE: {mse_linear:.6f}")
print(f"Polynomial Model MSE: {mse_poly:.6f}")
print(f"Improvement: {(mse_linear - mse_poly) / mse_linear * 100:.2f}%")

# Show polynomial features
print(f"\nPolynomial features shape: {X_poly.shape}")
print(f"Feature names: {poly_features.get_feature_names_out()}")
print(f"Polynomial coefficients: {poly_model.coef_}")
```

### 2. Logarithmic Transformations

Logarithmic transformations are useful for data with exponential relationships.

**Mathematical Formulation:**
$$\phi(x) = \log(x + \epsilon)$$

Where $\epsilon$ is a small constant to handle zero values.

```python
def generate_exponential_data(n_samples=200, noise=0.1):
    """Generate data with exponential relationship"""
    np.random.seed(42)
    X = np.random.uniform(1, 10, n_samples).reshape(-1, 1)
    y = 2 * np.exp(0.5 * X[:, 0]) + noise * np.random.randn(n_samples)
    return X, y

# Generate exponential data
X_exp, y_exp = generate_exponential_data()

# Apply logarithmic transformation
X_log = np.log(X_exp)

# Compare models
linear_model_exp = LinearRegression()
linear_model_exp.fit(X_exp, y_exp)
y_pred_linear_exp = linear_model_exp.predict(X_exp)

log_model = LinearRegression()
log_model.fit(X_log, y_exp)
y_pred_log = log_model.predict(X_log)

# Visualize results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_exp, y_exp, alpha=0.6)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Exponential Data')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(X_exp, y_exp, alpha=0.6, label='Data')
plt.plot(X_exp, y_pred_linear_exp, 'r-', label='Linear Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Model on Exponential Data')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(X_exp, y_exp, alpha=0.6, label='Data')
plt.plot(X_exp, y_pred_log, 'g-', label='Log-transformed Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Log-transformed Model')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare performance
mse_linear_exp = mean_squared_error(y_exp, y_pred_linear_exp)
mse_log = mean_squared_error(y_exp, y_pred_log)

print(f"Linear Model MSE: {mse_linear_exp:.6f}")
print(f"Log-transformed Model MSE: {mse_log:.6f}")
print(f"Improvement: {(mse_linear_exp - mse_log) / mse_linear_exp * 100:.2f}%")
```

### 3. Trigonometric Features

Trigonometric features capture periodic relationships.

**Mathematical Formulation:**
$$\phi(x) = [\sin(x), \cos(x), \sin(2x), \cos(2x), ...]$$

```python
def generate_periodic_data(n_samples=200, noise=0.1):
    """Generate data with periodic relationship"""
    np.random.seed(42)
    X = np.random.uniform(0, 4*np.pi, n_samples).reshape(-1, 1)
    y = 2 * np.sin(X[:, 0]) + 0.5 * np.cos(2*X[:, 0]) + noise * np.random.randn(n_samples)
    return X, y

# Generate periodic data
X_periodic, y_periodic = generate_periodic_data()

# Create trigonometric features
X_sin = np.sin(X_periodic)
X_cos = np.cos(X_periodic)
X_sin2 = np.sin(2 * X_periodic)
X_cos2 = np.cos(2 * X_periodic)

X_trig = np.column_stack([X_periodic, X_sin, X_cos, X_sin2, X_cos2])

# Fit models
linear_model_periodic = LinearRegression()
linear_model_periodic.fit(X_periodic, y_periodic)
y_pred_linear_periodic = linear_model_periodic.predict(X_periodic)

trig_model = LinearRegression()
trig_model.fit(X_trig, y_periodic)
y_pred_trig = trig_model.predict(X_trig)

# Visualize results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_periodic, y_periodic, alpha=0.6)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Periodic Data')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(X_periodic, y_periodic, alpha=0.6, label='Data')
plt.plot(X_periodic, y_pred_linear_periodic, 'r-', label='Linear Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Model on Periodic Data')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(X_periodic, y_periodic, alpha=0.6, label='Data')
plt.plot(X_periodic, y_pred_trig, 'g-', label='Trigonometric Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Trigonometric Model')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare performance
mse_linear_periodic = mean_squared_error(y_periodic, y_pred_linear_periodic)
mse_trig = mean_squared_error(y_periodic, y_pred_trig)

print(f"Linear Model MSE: {mse_linear_periodic:.6f}")
print(f"Trigonometric Model MSE: {mse_trig:.6f}")
print(f"Improvement: {(mse_linear_periodic - mse_trig) / mse_linear_periodic * 100:.2f}%")
```

### 4. Interaction Features

Interaction features capture relationships between multiple variables.

**Mathematical Formulation:**
$$\phi(x_1, x_2) = x_1 \times x_2$$

```python
def generate_interaction_data(n_samples=200, noise=0.1):
    """Generate data with interaction effects"""
    np.random.seed(42)
    X1 = np.random.uniform(-2, 2, n_samples)
    X2 = np.random.uniform(-2, 2, n_samples)
    X = np.column_stack([X1, X2])
    y = 2 * X1 + 3 * X2 + 1.5 * X1 * X2 + noise * np.random.randn(n_samples)
    return X, y

# Generate interaction data
X_interaction, y_interaction = generate_interaction_data()

# Create interaction features
X_interaction_feat = np.column_stack([
    X_interaction,
    X_interaction[:, 0] * X_interaction[:, 1]  # Interaction term
])

# Fit models
linear_model_interaction = LinearRegression()
linear_model_interaction.fit(X_interaction, y_interaction)
y_pred_linear_interaction = linear_model_interaction.predict(X_interaction)

interaction_model = LinearRegression()
interaction_model.fit(X_interaction_feat, y_interaction)
y_pred_interaction = interaction_model.predict(X_interaction_feat)

# Visualize results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_interaction[:, 0], y_interaction, alpha=0.6, label='X1 vs y')
plt.scatter(X_interaction[:, 1], y_interaction, alpha=0.6, label='X2 vs y')
plt.xlabel('Features')
plt.ylabel('y')
plt.title('Interaction Data')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(y_interaction, y_pred_linear_interaction, alpha=0.6)
plt.plot([y_interaction.min(), y_interaction.max()], 
         [y_interaction.min(), y_interaction.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Model Predictions')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(y_interaction, y_pred_interaction, alpha=0.6)
plt.plot([y_interaction.min(), y_interaction.max()], 
         [y_interaction.min(), y_interaction.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Interaction Model Predictions')
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare performance
mse_linear_interaction = mean_squared_error(y_interaction, y_pred_linear_interaction)
mse_interaction = mean_squared_error(y_interaction, y_pred_interaction)

print(f"Linear Model MSE: {mse_linear_interaction:.6f}")
print(f"Interaction Model MSE: {mse_interaction:.6f}")
print(f"Improvement: {(mse_linear_interaction - mse_interaction) / mse_linear_interaction * 100:.2f}%")
```

## Advanced Feature Engineering Techniques

### 1. Spline Transformations

Spline transformations use piecewise polynomial functions to capture complex non-linear relationships.

```python
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import SplineTransformer

def generate_complex_data(n_samples=300, noise=0.1):
    """Generate data with complex non-linear relationship"""
    np.random.seed(42)
    X = np.random.uniform(-5, 5, n_samples).reshape(-1, 1)
    y = (X[:, 0]**3 - 2*X[:, 0]**2 + X[:, 0] + 
         0.5 * np.sin(3*X[:, 0]) + noise * np.random.randn(n_samples))
    return X, y

# Generate complex data
X_complex, y_complex = generate_complex_data()

# Apply spline transformation
spline_transformer = SplineTransformer(n_knots=5, degree=3)
X_spline = spline_transformer.fit_transform(X_complex)

# Fit models
linear_model_complex = LinearRegression()
linear_model_complex.fit(X_complex, y_complex)
y_pred_linear_complex = linear_model_complex.predict(X_complex)

spline_model = LinearRegression()
spline_model.fit(X_spline, y_complex)
y_pred_spline = spline_model.predict(X_spline)

# Visualize results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_complex, y_complex, alpha=0.6)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Complex Non-linear Data')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(X_complex, y_complex, alpha=0.6, label='Data')
plt.plot(X_complex, y_pred_linear_complex, 'r-', label='Linear Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Model')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(X_complex, y_complex, alpha=0.6, label='Data')
plt.plot(X_complex, y_pred_spline, 'g-', label='Spline Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Spline Model')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare performance
mse_linear_complex = mean_squared_error(y_complex, y_pred_linear_complex)
mse_spline = mean_squared_error(y_complex, y_pred_spline)

print(f"Linear Model MSE: {mse_linear_complex:.6f}")
print(f"Spline Model MSE: {mse_spline:.6f}")
print(f"Improvement: {(mse_linear_complex - mse_spline) / mse_linear_complex * 100:.2f}%")
```

### 2. Radial Basis Functions (RBF)

RBF features use Gaussian functions centered at specific points.

```python
def rbf_features(X, centers, gamma=1.0):
    """Create RBF features"""
    features = []
    for center in centers:
        rbf = np.exp(-gamma * np.sum((X - center)**2, axis=1))
        features.append(rbf)
    return np.column_stack(features)

# Generate data
X_rbf, y_rbf = generate_complex_data(n_samples=200)

# Create RBF features
centers = np.array([[-3, 0], [0, 0], [3, 0]])
X_rbf_features = rbf_features(X_rbf, centers, gamma=0.5)

# Fit models
linear_model_rbf = LinearRegression()
linear_model_rbf.fit(X_rbf, y_rbf)
y_pred_linear_rbf = linear_model_rbf.predict(X_rbf)

rbf_model = LinearRegression()
rbf_model.fit(X_rbf_features, y_rbf)
y_pred_rbf = rbf_model.predict(X_rbf_features)

# Visualize results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_rbf, y_rbf, alpha=0.6)
plt.xlabel('X')
plt.ylabel('y')
plt.title('RBF Data')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(X_rbf, y_rbf, alpha=0.6, label='Data')
plt.plot(X_rbf, y_pred_linear_rbf, 'r-', label='Linear Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Model')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(X_rbf, y_rbf, alpha=0.6, label='Data')
plt.plot(X_rbf, y_pred_rbf, 'g-', label='RBF Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('RBF Model')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare performance
mse_linear_rbf = mean_squared_error(y_rbf, y_pred_linear_rbf)
mse_rbf = mean_squared_error(y_rbf, y_pred_rbf)

print(f"Linear Model MSE: {mse_linear_rbf:.6f}")
print(f"RBF Model MSE: {mse_rbf:.6f}")
print(f"Improvement: {(mse_linear_rbf - mse_rbf) / mse_linear_rbf * 100:.2f}%")
```

## Feature Selection and Regularization

### 1. Ridge Regression with Non-linear Features

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Create polynomial features
poly_features_high = PolynomialFeatures(degree=5, include_bias=False)
X_poly_high = poly_features_high.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_poly_high, y, test_size=0.2, random_state=42)

# Grid search for optimal alpha
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_ridge = grid_search.best_estimator_
y_pred_ridge = best_ridge.predict(X_test)

# Compare with linear model
linear_model_simple = LinearRegression()
linear_model_simple.fit(X_train, y_train)
y_pred_linear_simple = linear_model_simple.predict(X_test)

# Visualize results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_linear_simple, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Model')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_ridge, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Ridge Regression')
plt.grid(True)

plt.subplot(1, 3, 3)
# Plot coefficient magnitudes
plt.bar(range(len(best_ridge.coef_)), np.abs(best_ridge.coef_))
plt.xlabel('Feature Index')
plt.ylabel('|Coefficient|')
plt.title('Ridge Coefficients')
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare performance
mse_linear_simple = mean_squared_error(y_test, y_pred_linear_simple)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

print(f"Linear Model MSE: {mse_linear_simple:.6f}")
print(f"Ridge Model MSE: {mse_ridge:.6f}")
print(f"Best alpha: {grid_search.best_params_['alpha']}")
```

### 2. Lasso Regression for Feature Selection

```python
from sklearn.linear_model import Lasso

# Grid search for optimal alpha
lasso = Lasso()
grid_search_lasso = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_lasso.fit(X_train, y_train)

best_lasso = grid_search_lasso.best_estimator_
y_pred_lasso = best_lasso.predict(X_test)

# Visualize feature selection
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_lasso, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Lasso Model')
plt.grid(True)

plt.subplot(1, 3, 2)
# Compare coefficient magnitudes
plt.bar(range(len(best_ridge.coef_)), np.abs(best_ridge.coef_), alpha=0.7, label='Ridge')
plt.bar(range(len(best_lasso.coef_)), np.abs(best_lasso.coef_), alpha=0.7, label='Lasso')
plt.xlabel('Feature Index')
plt.ylabel('|Coefficient|')
plt.title('Coefficient Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
# Count non-zero coefficients
ridge_nonzero = np.sum(best_ridge.coef_ != 0)
lasso_nonzero = np.sum(best_lasso.coef_ != 0)
total_features = len(best_ridge.coef_)

plt.bar(['Ridge', 'Lasso'], [ridge_nonzero, lasso_nonzero])
plt.ylabel('Non-zero Coefficients')
plt.title('Feature Selection')
plt.grid(True, axis='y')

plt.tight_layout()
plt.show()

# Compare performance
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

print(f"Ridge Model MSE: {mse_ridge:.6f}")
print(f"Lasso Model MSE: {mse_lasso:.6f}")
print(f"Ridge non-zero coefficients: {ridge_nonzero}/{total_features}")
print(f"Lasso non-zero coefficients: {lasso_nonzero}/{total_features}")
```

## Real-world Example: House Price Prediction

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load California housing dataset
housing = fetch_california_housing()
X_housing = housing.data
y_housing = housing.target

# Split data
X_train_housing, X_test_housing, y_train_housing, y_test_housing = train_test_split(
    X_housing, y_housing, test_size=0.2, random_state=42
)

# Scale features
scaler_housing = StandardScaler()
X_train_scaled_housing = scaler_housing.fit_transform(X_train_housing)
X_test_scaled_housing = scaler_housing.transform(X_test_housing)

# Create polynomial features
poly_features_housing = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly_housing = poly_features_housing.fit_transform(X_train_scaled_housing)
X_test_poly_housing = poly_features_housing.transform(X_test_scaled_housing)

print(f"Original features: {X_housing.shape[1]}")
print(f"Polynomial features: {X_train_poly_housing.shape[1]}")

# Train different models
models_housing = {
    'Linear (Original)': LinearRegression(),
    'Linear (Polynomial)': LinearRegression(),
    'Ridge (Polynomial)': Ridge(alpha=1.0),
    'Lasso (Polynomial)': Lasso(alpha=0.1)
}

results_housing = {}

# Train models
models_housing['Linear (Original)'].fit(X_train_scaled_housing, y_train_housing)
models_housing['Linear (Polynomial)'].fit(X_train_poly_housing, y_train_housing)
models_housing['Ridge (Polynomial)'].fit(X_train_poly_housing, y_train_housing)
models_housing['Lasso (Polynomial)'].fit(X_train_poly_housing, y_train_housing)

# Evaluate models
y_pred_orig = models_housing['Linear (Original)'].predict(X_test_scaled_housing)
y_pred_poly = models_housing['Linear (Polynomial)'].predict(X_test_poly_housing)
y_pred_ridge = models_housing['Ridge (Polynomial)'].predict(X_test_poly_housing)
y_pred_lasso = models_housing['Lasso (Polynomial)'].predict(X_test_poly_housing)

results_housing = {
    'Linear (Original)': {'MSE': mean_squared_error(y_test_housing, y_pred_orig),
                         'R²': r2_score(y_test_housing, y_pred_orig)},
    'Linear (Polynomial)': {'MSE': mean_squared_error(y_test_housing, y_pred_poly),
                           'R²': r2_score(y_test_housing, y_pred_poly)},
    'Ridge (Polynomial)': {'MSE': mean_squared_error(y_test_housing, y_pred_ridge),
                          'R²': r2_score(y_test_housing, y_pred_ridge)},
    'Lasso (Polynomial)': {'MSE': mean_squared_error(y_test_housing, y_pred_lasso),
                          'R²': r2_score(y_test_housing, y_pred_lasso)}
}

# Visualize results
plt.figure(figsize=(15, 5))

# MSE comparison
plt.subplot(1, 3, 1)
model_names = list(results_housing.keys())
mse_values = [results_housing[name]['MSE'] for name in model_names]
plt.bar(model_names, mse_values)
plt.ylabel('Mean Squared Error')
plt.title('MSE Comparison')
plt.xticks(rotation=45)

# R² comparison
plt.subplot(1, 3, 2)
r2_values = [results_housing[name]['R²'] for name in model_names]
plt.bar(model_names, r2_values)
plt.ylabel('R² Score')
plt.title('R² Comparison')
plt.xticks(rotation=45)

# Predictions vs actual
plt.subplot(1, 3, 3)
plt.scatter(y_test_housing, y_pred_orig, alpha=0.6, label='Linear (Original)')
plt.scatter(y_test_housing, y_pred_poly, alpha=0.6, label='Linear (Polynomial)')
plt.scatter(y_test_housing, y_pred_ridge, alpha=0.6, label='Ridge (Polynomial)')
plt.scatter(y_test_housing, y_pred_lasso, alpha=0.6, label='Lasso (Polynomial)')
plt.plot([y_test_housing.min(), y_test_housing.max()], 
         [y_test_housing.min(), y_test_housing.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predictions vs Actual')
plt.legend()

plt.tight_layout()
plt.show()

# Print results
print("Model Performance Comparison:")
for name, metrics in results_housing.items():
    print(f"{name}:")
    print(f"  MSE: {metrics['MSE']:.6f}")
    print(f"  R²: {metrics['R²']:.6f}")

# Feature importance analysis
lasso_coefs = models_housing['Lasso (Polynomial)'].coef_
non_zero_features = np.sum(lasso_coefs != 0)
print(f"\nLasso selected {non_zero_features} out of {len(lasso_coefs)} polynomial features")

# Show most important features
feature_names = poly_features_housing.get_feature_names_out()
important_features = np.argsort(np.abs(lasso_coefs))[-10:]  # Top 10 features

print("\nTop 10 most important features (Lasso):")
for idx in reversed(important_features):
    if lasso_coefs[idx] != 0:
        print(f"  {feature_names[idx]}: {lasso_coefs[idx]:.6f}")
```

## Best Practices and Guidelines

### 1. When to Use Non-linear Features

```python
def analyze_feature_importance(X, y, feature_names):
    """Analyze which features might benefit from non-linear transformations"""
    n_features = X.shape[1]
    
    plt.figure(figsize=(15, 5))
    
    for i in range(min(6, n_features)):
        plt.subplot(2, 3, i+1)
        plt.scatter(X[:, i], y, alpha=0.6)
        plt.xlabel(feature_names[i])
        plt.ylabel('Target')
        plt.title(f'{feature_names[i]} vs Target')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Analyze housing features
feature_names_housing = housing.feature_names
analyze_feature_importance(X_train_scaled_housing, y_train_housing, feature_names_housing)
```

### 2. Cross-validation for Feature Selection

```python
from sklearn.model_selection import cross_val_score

# Test different polynomial degrees
degrees = [1, 2, 3, 4]
cv_scores = []

for degree in degrees:
    poly_features_cv = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_cv = poly_features_cv.fit_transform(X_train_scaled_housing)
    
    # Use Ridge regression to avoid overfitting
    ridge_cv = Ridge(alpha=1.0)
    scores = cross_val_score(ridge_cv, X_poly_cv, y_train_housing, cv=5, scoring='r2')
    cv_scores.append(scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(degrees, cv_scores, 'o-', linewidth=2, markersize=8)
plt.xlabel('Polynomial Degree')
plt.ylabel('Cross-validation R² Score')
plt.title('Polynomial Degree Selection')
plt.grid(True)
plt.show()

print("Cross-validation results:")
for degree, score in zip(degrees, cv_scores):
    print(f"Degree {degree}: R² = {score:.4f}")

best_degree = degrees[np.argmax(cv_scores)]
print(f"\nBest polynomial degree: {best_degree}")
```

## Conclusion

Non-linear features are powerful tools for extending the capabilities of linear models. Key takeaways:

### Benefits:
1. **Maintains Interpretability**: Linear models remain interpretable
2. **Computational Efficiency**: Faster than non-linear models
3. **Flexibility**: Can capture complex relationships
4. **Regularization**: Can use L1/L2 regularization

### Best Practices:
1. **Start Simple**: Begin with polynomial features of degree 2-3
2. **Feature Scaling**: Essential for numerical stability
3. **Regularization**: Use Ridge/Lasso to prevent overfitting
4. **Cross-validation**: Validate feature transformations
5. **Domain Knowledge**: Use transformations that make sense

### When to Use:
- **Moderate Complexity**: When relationships are non-linear but not extremely complex
- **Interpretability Required**: When model interpretability is important
- **Computational Constraints**: When faster training is needed
- **Feature Engineering**: As part of a larger feature engineering pipeline

Non-linear features bridge the gap between simple linear models and complex non-linear models, providing a good balance of performance and interpretability.

## Further Reading

- **Books**: "The Elements of Statistical Learning" by Hastie et al., "Feature Engineering for Machine Learning" by Alice Zheng
- **Papers**: Original polynomial regression papers, spline transformation literature
- **Online Resources**: Scikit-learn documentation, Towards Data Science articles
- **Practice**: Kaggle competitions, real-world datasets with non-linear relationships 