# 10 - Machine Learning Best Practices

## Overview

Machine learning best practices ensure robust, reliable, and maintainable models. This guide covers essential practices from data preparation to model deployment.

## Data Preprocessing Best Practices

### 1. Data Quality Assessment

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# Load data
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

def assess_data_quality(df):
    """Comprehensive data quality assessment"""
    print("=== DATA QUALITY ASSESSMENT ===\n")
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    }).sort_values('Missing Count', ascending=False)
    
    print("\nMissing Values:")
    print(missing_df[missing_df['Missing Count'] > 0])
    
    # Duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    # Data types
    print("\nData Types:")
    print(df.dtypes.value_counts())
    
    # Numerical columns statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\nNumerical columns: {len(numerical_cols)}")
    
    return missing_df

quality_report = assess_data_quality(df)
```

### 2. Outlier Detection and Treatment

```python
def detect_outliers(df, method='iqr', threshold=1.5):
    """Detect outliers using multiple methods"""
    outliers = {}
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == 'target':  # Skip target variable
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if method == 'iqr':
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers[col] = df[z_scores > threshold]
    
    return outliers

# Detect outliers
outliers = detect_outliers(df)

print("Outlier Analysis:")
for col, outlier_data in outliers.items():
    if len(outlier_data) > 0:
        print(f"{col}: {len(outlier_data)} outliers ({len(outlier_data)/len(df)*100:.2f}%)")

# Visualize outliers
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(df.select_dtypes(include=[np.number]).columns[:6]):
    if col != 'target':
        sns.boxplot(data=df, y=col, ax=axes[i])
        axes[i].set_title(f'Outliers in {col}')

plt.tight_layout()
plt.show()
```

### 3. Feature Scaling and Normalization

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

def compare_scalers(X_train, X_test):
    """Compare different scaling methods"""
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    
    results = {}
    
    for name, scaler in scalers.items():
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results[name] = {
            'train_mean': X_train_scaled.mean(),
            'train_std': X_train_scaled.std(),
            'train_min': X_train_scaled.min(),
            'train_max': X_train_scaled.max()
        }
    
    return results

# Split data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compare scalers
scaler_comparison = compare_scalers(X_train, X_test)

print("Scaler Comparison:")
for name, stats in scaler_comparison.items():
    print(f"\n{name}:")
    print(f"  Mean: {stats['train_mean']:.3f}")
    print(f"  Std: {stats['train_std']:.3f}")
    print(f"  Min: {stats['train_min']:.3f}")
    print(f"  Max: {stats['train_max']:.3f}")
```

## Model Selection Best Practices

### 1. Baseline Models

```python
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def create_baselines(X_train, X_test, y_train, y_test):
    """Create and evaluate baseline models"""
    baselines = {
        'Random': DummyClassifier(strategy='uniform', random_state=42),
        'Most Frequent': DummyClassifier(strategy='most_frequent', random_state=42),
        'Stratified': DummyClassifier(strategy='stratified', random_state=42)
    }
    
    results = {}
    
    for name, model in baselines.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
    
    return results

# Create baselines
baseline_results = create_baselines(X_train, X_test, y_train, y_test)

print("Baseline Model Performance:")
for name, accuracy in baseline_results.items():
    print(f"{name}: {accuracy:.3f}")
```

### 2. Model Comparison Framework

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

def comprehensive_model_evaluation(models, X, y, cv=5):
    """Comprehensive model evaluation with multiple metrics"""
    metrics = {
        'accuracy': 'accuracy',
        'f1': make_scorer(f1_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score)
    }
    
    results = {}
    
    for name, model in models.items():
        model_results = {}
        for metric_name, scorer in metrics.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
            model_results[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std()
            }
        results[name] = model_results
    
    return results

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42)
}

# Evaluate models
evaluation_results = comprehensive_model_evaluation(models, X, y)

print("Model Comparison Results:")
print("-" * 60)
for model_name, metrics in evaluation_results.items():
    print(f"\n{model_name}:")
    for metric_name, scores in metrics.items():
        print(f"  {metric_name}: {scores['mean']:.3f} (+/- {scores['std']*2:.3f})")
```

### 3. Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def optimize_hyperparameters(model, param_distributions, X, y, cv=5, n_iter=100):
    """Optimize hyperparameters using randomized search"""
    random_search = RandomizedSearchCV(
        model, param_distributions, n_iter=n_iter, cv=cv,
        scoring='f1', random_state=42, n_jobs=-1
    )
    
    random_search.fit(X, y)
    
    return random_search

# Example: Optimize Random Forest
rf_param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

rf_optimized = optimize_hyperparameters(
    RandomForestClassifier(random_state=42),
    rf_param_dist, X, y
)

print("Optimized Random Forest Parameters:")
print(rf_optimized.best_params_)
print(f"Best CV Score: {rf_optimized.best_score_:.3f}")
```

## Evaluation Best Practices

### 1. Multiple Evaluation Metrics

```python
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def comprehensive_evaluation(model, X_train, X_test, y_train, y_test):
    """Comprehensive model evaluation"""
    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr
    }

# Evaluate optimized model
results = comprehensive_evaluation(rf_optimized.best_estimator_, X_train, X_test, y_train, y_test)

print("Comprehensive Evaluation Results:")
print(f"Accuracy: {results['accuracy']:.3f}")
print(f"F1 Score: {results['f1']:.3f}")
print(f"Precision: {results['precision']:.3f}")
print(f"Recall: {results['recall']:.3f}")
print(f"AUC: {results['auc']:.3f}")

# Plot confusion matrix
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.subplot(1, 2, 2)
plt.plot(results['fpr'], results['tpr'])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 2. Cross-Validation Strategies

```python
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

def compare_cv_strategies(model, X, y):
    """Compare different cross-validation strategies"""
    cv_strategies = {
        'Stratified 5-Fold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        'Stratified 10-Fold': StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    }
    
    results = {}
    
    for name, cv in cv_strategies.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    return results

cv_results = compare_cv_strategies(rf_optimized.best_estimator_, X, y)

print("Cross-Validation Results:")
for name, result in cv_results.items():
    print(f"{name}: {result['mean']:.3f} (+/- {result['std']*2:.3f})")
```

## Feature Engineering Best Practices

### 1. Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression

def feature_selection_analysis(X, y):
    """Comprehensive feature selection analysis"""
    # Univariate selection
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    
    # Recursive feature elimination
    rfe = RFE(estimator=LogisticRegression(random_state=42), n_features_to_select=10)
    rfe.fit(X, y)
    rfe_features = X.columns[rfe.support_]
    
    return {
        'univariate_features': selected_features,
        'rfe_features': rfe_features,
        'feature_scores': selector.scores_
    }

feature_analysis = feature_selection_analysis(X, y)

print("Feature Selection Results:")
print(f"Univariate selected features: {len(feature_analysis['univariate_features'])}")
print(f"RFE selected features: {len(feature_analysis['rfe_features'])}")

# Plot feature importance scores
plt.figure(figsize=(12, 6))
feature_scores = pd.Series(feature_analysis['feature_scores'], index=X.columns)
feature_scores.sort_values(ascending=True).plot(kind='barh')
plt.title('Feature Importance Scores (Univariate Selection)')
plt.xlabel('F-score')
plt.tight_layout()
plt.show()
```

### 2. Feature Interaction Analysis

```python
def create_interaction_features(X):
    """Create interaction features"""
    X_interactions = X.copy()
    
    # Create polynomial features for important features
    important_features = ['mean radius', 'mean texture', 'mean perimeter']
    
    for i, feat1 in enumerate(important_features):
        for feat2 in important_features[i+1:]:
            if feat1 in X.columns and feat2 in X.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                X_interactions[interaction_name] = X[feat1] * X[feat2]
    
    return X_interactions

X_with_interactions = create_interaction_features(X)
print(f"Original features: {X.shape[1]}")
print(f"Features with interactions: {X_with_interactions.shape[1]}")
```

## Model Deployment Best Practices

### 1. Model Persistence

```python
import joblib
import pickle
from datetime import datetime

def save_model_pipeline(model, scaler, feature_names, model_name):
    """Save complete model pipeline"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = f"{model_name}_{timestamp}.joblib"
    joblib.dump(model, model_path)
    
    # Save preprocessing pipeline
    preprocessor = {
        'scaler': scaler,
        'feature_names': feature_names
    }
    preprocessor_path = f"preprocessor_{model_name}_{timestamp}.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'feature_count': len(feature_names),
        'model_type': type(model).__name__
    }
    
    with open(f"metadata_{model_name}_{timestamp}.json", 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    return model_path, preprocessor_path

# Save the optimized model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Retrain on scaled data
rf_optimized.best_estimator_.fit(X_train_scaled, y_train)

model_path, preprocessor_path = save_model_pipeline(
    rf_optimized.best_estimator_,
    scaler,
    list(X.columns),
    'breast_cancer_rf'
)

print(f"Model saved: {model_path}")
print(f"Preprocessor saved: {preprocessor_path}")
```

### 2. Model Monitoring

```python
class ModelMonitor:
    """Simple model monitoring class"""
    
    def __init__(self, model, scaler, feature_names):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.predictions_history = []
        self.confidence_history = []
    
    def predict_with_monitoring(self, X):
        """Make predictions with monitoring"""
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        confidence = np.max(probabilities, axis=1)
        
        # Store for monitoring
        self.predictions_history.extend(predictions)
        self.confidence_history.extend(confidence)
        
        return predictions, confidence
    
    def get_monitoring_stats(self):
        """Get monitoring statistics"""
        if not self.predictions_history:
            return {}
        
        return {
            'total_predictions': len(self.predictions_history),
            'avg_confidence': np.mean(self.confidence_history),
            'low_confidence_predictions': np.sum(np.array(self.confidence_history) < 0.8),
            'prediction_distribution': np.bincount(self.predictions_history)
        }

# Create monitor
monitor = ModelMonitor(rf_optimized.best_estimator_, scaler, list(X.columns))

# Simulate predictions
predictions, confidence = monitor.predict_with_monitoring(X_test[:10])

print("Monitoring Statistics:")
print(monitor.get_monitoring_stats())
```

## Ethical Considerations

### 1. Bias Detection

```python
def detect_bias_in_predictions(model, X, y, sensitive_features):
    """Detect bias in model predictions"""
    predictions = model.predict(X)
    
    bias_report = {}
    
    for feature in sensitive_features:
        if feature in X.columns:
            unique_values = X[feature].unique()
            
            for value in unique_values:
                mask = X[feature] == value
                group_predictions = predictions[mask]
                group_actual = y[mask]
                
                if len(group_predictions) > 0:
                    accuracy = accuracy_score(group_actual, group_predictions)
                    bias_report[f"{feature}_{value}"] = {
                        'count': len(group_predictions),
                        'accuracy': accuracy,
                        'positive_rate': np.mean(group_predictions)
                    }
    
    return bias_report

# Example bias detection (assuming we have demographic features)
# bias_report = detect_bias_in_predictions(model, X, y, ['age_group', 'gender'])
```

### 2. Model Interpretability

```python
from sklearn.inspection import permutation_importance

def model_interpretability_analysis(model, X, y):
    """Analyze model interpretability"""
    # Permutation importance
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    return feature_importance

# Analyze interpretability
interpretability = model_interpretability_analysis(rf_optimized.best_estimator_, X, y)

print("Top 10 Most Important Features:")
print(interpretability.head(10))

# Plot feature importance
plt.figure(figsize=(10, 8))
top_features = interpretability.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Permutation Importance')
plt.title('Feature Importance (Permutation)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

## Summary

Machine learning best practices ensure:

1. **Data Quality**: Thorough assessment and cleaning
2. **Proper Scaling**: Appropriate preprocessing for algorithms
3. **Baseline Models**: Establish performance benchmarks
4. **Comprehensive Evaluation**: Multiple metrics and validation strategies
5. **Feature Engineering**: Systematic feature selection and creation
6. **Model Persistence**: Proper saving and loading
7. **Monitoring**: Track model performance over time
8. **Ethical Considerations**: Bias detection and interpretability

Following these practices leads to robust, reliable, and maintainable machine learning systems. 