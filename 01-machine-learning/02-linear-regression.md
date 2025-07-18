# Linear Regression

## Introduction

Linear regression is one of the most fundamental and widely used algorithms in machine learning. It models the relationship between a dependent variable (target) and one or more independent variables (features) using a linear function. Despite its simplicity, linear regression serves as the foundation for understanding more complex algorithms and is often the first step in any regression analysis.

## Mathematical Foundation

### The Linear Model

The linear regression model assumes that the relationship between the input features and the target variable is linear:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon$$

Where:
- $y$ is the target variable
- $x_1, x_2, ..., x_n$ are the input features
- $\beta_0$ is the intercept (bias term)
- $\beta_1, \beta_2, ..., \beta_n$ are the coefficients (weights)
- $\epsilon$ is the error term (noise)

In matrix notation:
$$y = X\beta + \epsilon$$

Where:
- $X \in \mathbb{R}^{n \times (d+1)}$ is the design matrix (with bias term)
- $\beta \in \mathbb{R}^{(d+1) \times 1}$ is the coefficient vector
- $y \in \mathbb{R}^{n \times 1}$ is the target vector
- $\epsilon \in \mathbb{R}^{n \times 1}$ is the error vector

### Assumptions

Linear regression relies on several key assumptions:

1. **Linearity**: The relationship between features and target is linear
2. **Independence**: Observations are independent of each other
3. **Homoscedasticity**: Error terms have constant variance
4. **Normality**: Error terms are normally distributed
5. **No multicollinearity**: Features are not highly correlated

## Cost Function

### Mean Squared Error (MSE)

The most common cost function for linear regression is the Mean Squared Error:

$$J(\beta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where:
- $y_i$ is the actual value
- $\hat{y}_i$ is the predicted value
- $n$ is the number of observations

In matrix notation:
$$J(\beta) = \frac{1}{n} (y - X\beta)^T(y - X\beta)$$

### Why MSE?

MSE is preferred because:
1. It penalizes large errors more heavily (quadratic penalty)
2. It's differentiable everywhere
3. It has a unique global minimum
4. It's mathematically tractable

## Optimization Methods

### 1. Normal Equation (Closed-form Solution)

The optimal coefficients can be found analytically by setting the gradient to zero:

$$\nabla J(\beta) = -\frac{2}{n} X^T(y - X\beta) = 0$$

Solving for $\beta$:
$$X^T(y - X\beta) = 0$$
$$X^Ty - X^TX\beta = 0$$
$$X^TX\beta = X^Ty$$
$$\beta = (X^TX)^{-1}X^Ty$$

**Python Implementation:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionNormalEquation:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        
    def fit(self, X, y):
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Normal equation
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]
        
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(np.r_[self.intercept, self.coefficients])

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 2) * 10
y = 3 * X[:, 0] + 2 * X[:, 1] + 1 + np.random.normal(0, 0.5, 100)

# Fit using normal equation
model_normal = LinearRegressionNormalEquation()
model_normal.fit(X, y)

# Compare with sklearn
model_sklearn = LinearRegression()
model_sklearn.fit(X, y)

print("Normal Equation Results:")
print(f"Intercept: {model_normal.intercept:.3f}")
print(f"Coefficients: {model_normal.coefficients}")
print(f"Sklearn Intercept: {model_sklearn.intercept_:.3f}")
print(f"Sklearn Coefficients: {model_sklearn.coef_}")

# Make predictions
y_pred_normal = model_normal.predict(X)
y_pred_sklearn = model_sklearn.predict(X)

print(f"MSE (Normal): {mean_squared_error(y, y_pred_normal):.6f}")
print(f"MSE (Sklearn): {mean_squared_error(y, y_pred_sklearn):.6f}")
```

### 2. Gradient Descent

For large datasets, the normal equation can be computationally expensive. Gradient descent provides an iterative solution:

**Algorithm:**
1. Initialize $\beta$ randomly
2. Repeat until convergence:
   - Compute gradient: $\nabla J(\beta) = -\frac{2}{n} X^T(y - X\beta)$
   - Update parameters: $\beta = \beta - \alpha \nabla J(\beta)$

**Python Implementation:**

```python
class LinearRegressionGradientDescent:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.coefficients = None
        self.intercept = None
        self.cost_history = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.coefficients = np.zeros(n_features)
        self.intercept = 0
        
        for iteration in range(self.max_iterations):
            # Add bias term
            X_b = np.c_[np.ones((n_samples, 1)), X]
            beta = np.r_[self.intercept, self.coefficients]
            
            # Compute predictions
            y_pred = X_b.dot(beta)
            
            # Compute gradients
            gradients = (2/n_samples) * X_b.T.dot(y_pred - y)
            
            # Update parameters
            beta_new = beta - self.learning_rate * gradients
            
            # Check convergence
            if np.linalg.norm(beta_new - beta) < self.tolerance:
                break
                
            self.intercept = beta_new[0]
            self.coefficients = beta_new[1:]
            
            # Store cost for visualization
            cost = mean_squared_error(y, y_pred)
            self.cost_history.append(cost)
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Cost: {cost:.6f}")
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(np.r_[self.intercept, self.coefficients])

# Fit using gradient descent
model_gd = LinearRegressionGradientDescent(learning_rate=0.01, max_iterations=1000)
model_gd.fit(X, y)

# Plot cost history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(model_gd.cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost (MSE)')
plt.title('Gradient Descent Convergence')
plt.grid(True)

# Compare predictions
y_pred_gd = model_gd.predict(X)
plt.subplot(1, 2, 2)
plt.scatter(y, y_pred_gd, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Prediction vs Actual')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Gradient Descent Results:")
print(f"Intercept: {model_gd.intercept:.3f}")
print(f"Coefficients: {model_gd.coefficients}")
print(f"MSE: {mean_squared_error(y, y_pred_gd):.6f}")
```

## Model Evaluation

### 1. R-squared (Coefficient of Determination)

R-squared measures the proportion of variance in the target variable that is predictable from the features:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

Where:
- $SS_{res}$ is the residual sum of squares
- $SS_{tot}$ is the total sum of squares
- $\bar{y}$ is the mean of the target variable

### 2. Adjusted R-squared

Adjusted R-squared penalizes for the number of features:

$$R^2_{adj} = 1 - (1 - R^2) \frac{n - 1}{n - p - 1}$$

Where $p$ is the number of features.

### 3. Root Mean Squared Error (RMSE)

RMSE is the square root of MSE:

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Python Implementation:**

```python
def evaluate_model(y_true, y_pred, X):
    """Comprehensive model evaluation"""
    n = len(y_true)
    p = X.shape[1]
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Adjusted R-squared
    r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    print("Model Evaluation Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R-squared: {r_squared:.6f}")
    print(f"Adjusted R-squared: {r_squared_adj:.6f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared,
        'r_squared_adj': r_squared_adj
    }

# Evaluate the model
metrics = evaluate_model(y, y_pred_gd, X)
```

## Feature Scaling

Feature scaling is crucial for gradient descent to converge efficiently:

### 1. Standardization (Z-score normalization)

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

### 2. Min-Max Scaling

$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**Python Implementation:**

```python
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self
        
    def transform(self, X):
        return (X - self.mean_) / self.std_
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# Compare with and without scaling
X_unscaled = np.random.rand(100, 2) * 1000  # Large scale features
y_unscaled = 3 * X_unscaled[:, 0] + 2 * X_unscaled[:, 1] + 1 + np.random.normal(0, 0.5, 100)

# Without scaling
model_no_scale = LinearRegressionGradientDescent(learning_rate=0.0001, max_iterations=2000)
model_no_scale.fit(X_unscaled, y_unscaled)

# With scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unscaled)

model_scaled = LinearRegressionGradientDescent(learning_rate=0.01, max_iterations=1000)
model_scaled.fit(X_scaled, y_unscaled)

# Compare convergence
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(model_no_scale.cost_history, label='No Scaling')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Convergence without Scaling')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(model_scaled.cost_history, label='With Scaling')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Convergence with Scaling')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Final cost without scaling: {model_no_scale.cost_history[-1]:.6f}")
print(f"Final cost with scaling: {model_scaled.cost_history[-1]:.6f}")
```

## Regularization

### 1. Ridge Regression (L2 Regularization)

Ridge regression adds L2 penalty to prevent overfitting:

$$J(\beta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$

**Python Implementation:**

```python
class RidgeRegression:
    def __init__(self, alpha=1.0, learning_rate=0.01, max_iterations=1000):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.coefficients = None
        self.intercept = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.coefficients = np.zeros(n_features)
        self.intercept = 0
        
        for iteration in range(self.max_iterations):
            # Add bias term
            X_b = np.c_[np.ones((n_samples, 1)), X]
            beta = np.r_[self.intercept, self.coefficients]
            
            # Compute predictions
            y_pred = X_b.dot(beta)
            
            # Compute gradients (with L2 penalty)
            gradients = (2/n_samples) * X_b.T.dot(y_pred - y)
            gradients[1:] += 2 * self.alpha * self.coefficients  # L2 penalty
            
            # Update parameters
            beta_new = beta - self.learning_rate * gradients
            self.intercept = beta_new[0]
            self.coefficients = beta_new[1:]
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(np.r_[self.intercept, self.coefficients])

# Compare regular and ridge regression
alphas = [0, 0.1, 1.0, 10.0]
models = []

for alpha in alphas:
    model = RidgeRegression(alpha=alpha)
    model.fit(X, y)
    models.append(model)

# Compare coefficients
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
coefficients = [model.coefficients for model in models]
plt.plot(alphas, [coef[0] for coef in coefficients], 'o-', label='β₁')
plt.plot(alphas, [coef[1] for coef in coefficients], 's-', label='β₂')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Shrinkage')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
mse_values = [mean_squared_error(y, model.predict(X)) for model in models]
plt.plot(alphas, mse_values, 'o-')
plt.xlabel('Alpha')
plt.ylabel('MSE')
plt.title('MSE vs Regularization')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 2. Lasso Regression (L1 Regularization)

Lasso regression adds L1 penalty, which can produce sparse solutions:

$$J(\beta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$

## Polynomial Regression

Polynomial regression extends linear regression to capture non-linear relationships:

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + ... + \beta_d x^d + \epsilon$$

**Python Implementation:**

```python
def polynomial_features(X, degree):
    """Generate polynomial features"""
    n_samples, n_features = X.shape
    features = [np.ones(n_samples)]
    
    for d in range(1, degree + 1):
        for i in range(n_features):
            features.append(X[:, i] ** d)
    
    return np.column_stack(features)

# Generate non-linear data
X_nonlinear = np.random.rand(100, 1) * 10
y_nonlinear = 2 * X_nonlinear[:, 0] + 0.5 * X_nonlinear[:, 0]**2 + np.random.normal(0, 1, 100)

# Fit polynomial models
degrees = [1, 2, 3, 5]
models_poly = []

for degree in degrees:
    X_poly = polynomial_features(X_nonlinear, degree)
    model = LinearRegression()
    model.fit(X_poly, y_nonlinear)
    models_poly.append((degree, model))

# Visualize results
plt.figure(figsize=(15, 4))
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)

for i, (degree, model) in enumerate(models_poly):
    plt.subplot(1, 4, i+1)
    plt.scatter(X_nonlinear, y_nonlinear, alpha=0.6, label='Data')
    
    X_plot_poly = polynomial_features(X_plot, degree)
    y_plot = model.predict(X_plot_poly)
    plt.plot(X_plot, y_plot, 'r-', label=f'Degree {degree}')
    
    plt.title(f'Polynomial Degree {degree}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Compare model performance
for degree, model in models_poly:
    X_poly = polynomial_features(X_nonlinear, degree)
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y_nonlinear, y_pred)
    print(f"Degree {degree}: MSE = {mse:.3f}")
```

## Real-world Example: House Price Prediction

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load California housing dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train different models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'Coefficients': model.coef_
    }
    
    print(f"\n{name}:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

# Compare coefficients
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
feature_names = housing.feature_names
for name, result in results.items():
    plt.plot(result['Coefficients'], 'o-', label=name, alpha=0.7)

plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Feature Coefficients Comparison')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
model_names = list(results.keys())
mse_values = [results[name]['MSE'] for name in model_names]
plt.bar(model_names, mse_values)
plt.ylabel('MSE')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Feature importance
linear_coefs = results['Linear Regression']['Coefficients']
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': linear_coefs,
    'Abs_Coefficient': np.abs(linear_coefs)
}).sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance:")
print(feature_importance)
```

## Assumptions and Diagnostics

### 1. Residual Analysis

```python
def residual_analysis(y_true, y_pred):
    """Perform residual analysis"""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(15, 4))
    
    # Residuals vs Predicted
    plt.subplot(1, 3, 1)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.grid(True)
    
    # Residuals histogram
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=30, alpha=0.7, density=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Residuals Distribution')
    plt.grid(True)
    
    # Q-Q plot
    plt.subplot(1, 3, 3)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    from scipy.stats import shapiro, jarque_bera
    
    # Normality test
    shapiro_stat, shapiro_p = shapiro(residuals)
    jarque_stat, jarque_p = jarque_bera(residuals)
    
    print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
    print(f"Jarque-Bera test: statistic={jarque_stat:.4f}, p-value={jarque_p:.4f}")

# Perform residual analysis
residual_analysis(y_test, results['Linear Regression']['y_pred'])
```

### 2. Multicollinearity Detection

```python
def check_multicollinearity(X, feature_names):
    """Check for multicollinearity using correlation matrix"""
    import seaborn as sns
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if abs(corr_matrix[i, j]) > 0.8:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
    
    if high_corr_pairs:
        print("Highly correlated feature pairs (|r| > 0.8):")
        for pair in high_corr_pairs:
            print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
    else:
        print("No highly correlated features detected.")

# Check multicollinearity
check_multicollinearity(X_train_scaled, feature_names)
```

## Conclusion

Linear regression is a powerful and interpretable algorithm that serves as the foundation for understanding more complex machine learning models. Key takeaways:

1. **Mathematical Foundation**: Understanding the normal equation and gradient descent
2. **Feature Scaling**: Essential for gradient descent convergence
3. **Regularization**: Ridge and Lasso help prevent overfitting
4. **Model Evaluation**: Multiple metrics provide different perspectives
5. **Assumptions**: Checking assumptions ensures model validity
6. **Polynomial Features**: Extends linear regression to capture non-linear relationships

Linear regression remains relevant in modern machine learning due to its interpretability, computational efficiency, and strong theoretical foundation.

## Further Reading

- **Books**: "Introduction to Linear Regression Analysis" by Montgomery et al.
- **Papers**: Original papers on ridge regression (Hoerl & Kennard, 1970) and lasso (Tibshirani, 1996)
- **Online Resources**: StatQuest videos, Towards Data Science articles
- **Practice**: Kaggle competitions, real-world datasets 