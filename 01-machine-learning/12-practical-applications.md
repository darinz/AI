# 12 - Practical Machine Learning Applications

## Overview

This guide covers practical applications of machine learning in real-world scenarios, from data preprocessing to model deployment and monitoring.

## End-to-End ML Project: Customer Churn Prediction

### 1. Problem Definition

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic customer data
np.random.seed(42)
n_customers = 10000

# Customer features
data = {
    'customer_id': range(1, n_customers + 1),
    'age': np.random.normal(45, 15, n_customers),
    'tenure': np.random.exponential(5, n_customers),
    'monthly_charges': np.random.normal(65, 20, n_customers),
    'total_charges': np.random.normal(2000, 1000, n_customers),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_customers),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers),
    'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
    'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
    'paperless_billing': np.random.choice(['Yes', 'No'], n_customers),
    'gender': np.random.choice(['Male', 'Female'], n_customers)
}

df = pd.DataFrame(data)

# Create churn target based on features
churn_prob = (
    0.1 +  # Base churn rate
    0.3 * (df['contract_type'] == 'Month-to-month') +
    0.2 * (df['payment_method'] == 'Electronic check') +
    0.15 * (df['online_security'] == 'No') +
    0.1 * (df['tech_support'] == 'No') +
    0.05 * (df['monthly_charges'] > 70) +
    0.02 * (df['tenure'] < 2)
)

df['churn'] = np.random.binomial(1, churn_prob)

print("Customer Churn Dataset Overview:")
print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['churn'].mean():.2%}")
print(f"Features: {list(df.columns[:-1])}")
```

### 2. Exploratory Data Analysis

```python
def comprehensive_eda(df):
    """Comprehensive exploratory data analysis"""
    print("=== EXPLORATORY DATA ANALYSIS ===\n")
    
    # Basic statistics
    print("1. Basic Statistics:")
    print(f"Total customers: {len(df)}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    print(f"Features: {len(df.columns) - 1}")
    
    # Missing values
    print("\n2. Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found")
    
    # Data types
    print("\n3. Data Types:")
    print(df.dtypes.value_counts())
    
    # Numerical features analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = numerical_cols[numerical_cols != 'churn']
    
    print(f"\n4. Numerical Features ({len(numerical_cols)}):")
    print(df[numerical_cols].describe())
    
    # Categorical features analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print(f"\n5. Categorical Features ({len(categorical_cols)}):")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
        print(f"Churn rate by {col}:")
        churn_by_cat = df.groupby(col)['churn'].mean().sort_values(ascending=False)
        print(churn_by_cat)

# Run EDA
comprehensive_eda(df)

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Age distribution
axes[0, 0].hist(df['age'], bins=30, alpha=0.7)
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')

# Tenure distribution
axes[0, 1].hist(df['tenure'], bins=30, alpha=0.7)
axes[0, 1].set_title('Tenure Distribution')
axes[0, 1].set_xlabel('Tenure (months)')

# Monthly charges
axes[0, 2].hist(df['monthly_charges'], bins=30, alpha=0.7)
axes[0, 2].set_title('Monthly Charges Distribution')
axes[0, 2].set_xlabel('Monthly Charges ($)')

# Contract type vs churn
contract_churn = df.groupby('contract_type')['churn'].mean()
axes[1, 0].bar(contract_churn.index, contract_churn.values)
axes[1, 0].set_title('Churn Rate by Contract Type')
axes[1, 0].tick_params(axis='x', rotation=45)

# Payment method vs churn
payment_churn = df.groupby('payment_method')['churn'].mean()
axes[1, 1].bar(payment_churn.index, payment_churn.values)
axes[1, 1].set_title('Churn Rate by Payment Method')
axes[1, 1].tick_params(axis='x', rotation=45)

# Internet service vs churn
internet_churn = df.groupby('internet_service')['churn'].mean()
axes[1, 2].bar(internet_churn.index, internet_churn.values)
axes[1, 2].set_title('Churn Rate by Internet Service')
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Correlation matrix
numerical_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(8, 6))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()
```

### 3. Feature Engineering

```python
def feature_engineering(df):
    """Comprehensive feature engineering"""
    df_engineered = df.copy()
    
    # 1. Create interaction features
    df_engineered['tenure_monthly_ratio'] = df_engineered['tenure'] / df_engineered['monthly_charges']
    df_engineered['total_monthly_ratio'] = df_engineered['total_charges'] / df_engineered['monthly_charges']
    
    # 2. Create categorical features
    df_engineered['age_group'] = pd.cut(df_engineered['age'], 
                                       bins=[0, 30, 50, 70, 100], 
                                       labels=['Young', 'Adult', 'Senior', 'Elderly'])
    
    df_engineered['tenure_group'] = pd.cut(df_engineered['tenure'], 
                                          bins=[0, 12, 24, 60, 100], 
                                          labels=['New', 'Short-term', 'Medium-term', 'Long-term'])
    
    # 3. Create binary features
    df_engineered['has_internet'] = (df_engineered['internet_service'] != 'No').astype(int)
    df_engineered['has_security'] = (df_engineered['online_security'] == 'Yes').astype(int)
    df_engineered['has_support'] = (df_engineered['tech_support'] == 'Yes').astype(int)
    df_engineered['month_to_month'] = (df_engineered['contract_type'] == 'Month-to-month').astype(int)
    df_engineered['electronic_payment'] = (df_engineered['payment_method'] == 'Electronic check').astype(int)
    
    # 4. Create aggregate features
    df_engineered['service_count'] = (df_engineered['has_internet'] + 
                                     df_engineered['has_security'] + 
                                     df_engineered['has_support'])
    
    return df_engineered

# Apply feature engineering
df_engineered = feature_engineering(df)

print("Feature Engineering Results:")
print(f"Original features: {len(df.columns)}")
print(f"Engineered features: {len(df_engineered.columns)}")
print(f"New features added: {len(df_engineered.columns) - len(df.columns)}")

# Show new features
new_features = set(df_engineered.columns) - set(df.columns)
print(f"\nNew features: {list(new_features)}")
```

### 4. Data Preprocessing

```python
def preprocess_data(df, target_col='churn'):
    """Complete data preprocessing pipeline"""
    
    # Separate features and target
    X = df.drop([target_col, 'customer_id'], axis=1, errors='ignore')
    y = df[target_col]
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    
    # Create preprocessing pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    # Numerical preprocessing
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return X, y, preprocessor

# Preprocess data
X, y, preprocessor = preprocess_data(df_engineered)

print("Data Preprocessing Results:")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Numerical features: {len(X.select_dtypes(include=[np.number]).columns)}")
print(f"Categorical features: {len(X.select_dtypes(include=['object']).columns)}")
```

### 5. Model Development

```python
def build_models(X, y, preprocessor):
    """Build and compare multiple models"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    # Create pipelines
    pipelines = {}
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipelines[name] = pipeline
    
    # Evaluate models
    results = {}
    for name, pipeline in pipelines.items():
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
        
        # Train on full dataset
        pipeline.fit(X, y)
        
        results[name] = {
            'pipeline': pipeline,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
    
    return results

# Build models
model_results = build_models(X, y, preprocessor)

print("Model Comparison Results:")
print("-" * 50)
for name, result in model_results.items():
    print(f"{name}:")
    print(f"  CV AUC: {result['cv_mean']:.3f} (+/- {result['cv_std']*2:.3f})")
    print()
```

### 6. Model Evaluation and Selection

```python
def comprehensive_evaluation(best_pipeline, X, y):
    """Comprehensive model evaluation"""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    best_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = best_pipeline.predict(X_test)
    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Print results
    print("Model Performance Metrics:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return metrics, best_pipeline

# Select best model
best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['cv_mean'])
best_pipeline = model_results[best_model_name]['pipeline']

print(f"Best model: {best_model_name}")
print(f"CV AUC: {model_results[best_model_name]['cv_mean']:.3f}")

# Comprehensive evaluation
metrics, final_model = comprehensive_evaluation(best_pipeline, X, y)
```

### 7. Feature Importance Analysis

```python
def analyze_feature_importance(model, X, preprocessor):
    """Analyze feature importance"""
    from sklearn.inspection import permutation_importance
    
    # Get feature names after preprocessing
    preprocessor.fit(X)
    feature_names = []
    
    # Numerical features
    numerical_features = X.select_dtypes(include=[np.number]).columns
    feature_names.extend(numerical_features)
    
    # Categorical features
    categorical_features = X.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        unique_values = X[feature].unique()
        for value in unique_values[1:]:  # Skip first value (dropped by OneHotEncoder)
            feature_names.append(f"{feature}_{value}")
    
    # Calculate permutation importance
    X_transformed = preprocessor.transform(X)
    perm_importance = permutation_importance(model.named_steps['classifier'], 
                                           X_transformed, y, n_repeats=10, random_state=42)
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Permutation Importance')
    plt.title('Feature Importance (Top 15)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return feature_importance

# Analyze feature importance
feature_importance = analyze_feature_importance(final_model, X, y)

print("Top 10 Most Important Features:")
print(feature_importance.head(10))
```

## Model Deployment and Monitoring

### 1. Model Serialization

```python
import joblib
import json
from datetime import datetime

def save_model_pipeline(model, preprocessor, feature_names, model_name):
    """Save complete model pipeline"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model pipeline
    model_path = f"{model_name}_{timestamp}.joblib"
    joblib.dump(model, model_path)
    
    # Save preprocessor
    preprocessor_path = f"preprocessor_{model_name}_{timestamp}.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'feature_count': len(feature_names),
        'model_type': type(model.named_steps['classifier']).__name__,
        'feature_names': list(feature_names)
    }
    
    metadata_path = f"metadata_{model_name}_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model_path, preprocessor_path, metadata_path

# Save model
model_path, preprocessor_path, metadata_path = save_model_pipeline(
    final_model, preprocessor, X.columns, 'customer_churn'
)

print(f"Model saved: {model_path}")
print(f"Preprocessor saved: {preprocessor_path}")
print(f"Metadata saved: {metadata_path}")
```

### 2. Prediction Service

```python
class ChurnPredictionService:
    """Customer churn prediction service"""
    
    def __init__(self, model_path, preprocessor_path):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.predictions_history = []
        self.confidence_history = []
    
    def predict_churn(self, customer_data):
        """Predict churn probability for a customer"""
        # Convert to DataFrame if needed
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])
        
        # Make prediction
        churn_probability = self.model.predict_proba(customer_data)[0, 1]
        churn_prediction = self.model.predict(customer_data)[0]
        
        # Store for monitoring
        self.predictions_history.append(churn_prediction)
        self.confidence_history.append(max(churn_probability, 1 - churn_probability))
        
        return {
            'churn_probability': churn_probability,
            'churn_prediction': bool(churn_prediction),
            'confidence': max(churn_probability, 1 - churn_probability)
        }
    
    def predict_batch(self, customers_data):
        """Predict churn for multiple customers"""
        results = []
        for customer in customers_data:
            result = self.predict_churn(customer)
            results.append(result)
        return results
    
    def get_monitoring_stats(self):
        """Get monitoring statistics"""
        if not self.predictions_history:
            return {}
        
        return {
            'total_predictions': len(self.predictions_history),
            'churn_rate': np.mean(self.predictions_history),
            'avg_confidence': np.mean(self.confidence_history),
            'low_confidence_predictions': np.sum(np.array(self.confidence_history) < 0.8)
        }

# Create prediction service
prediction_service = ChurnPredictionService(model_path, preprocessor_path)

# Test with sample customer
sample_customer = {
    'age': 45,
    'tenure': 24,
    'monthly_charges': 70,
    'total_charges': 2000,
    'contract_type': 'Month-to-month',
    'payment_method': 'Electronic check',
    'internet_service': 'Fiber optic',
    'online_security': 'No',
    'tech_support': 'No',
    'paperless_billing': 'Yes',
    'gender': 'Male'
}

# Add engineered features
sample_customer_engineered = feature_engineering(pd.DataFrame([sample_customer]))

prediction = prediction_service.predict_churn(sample_customer_engineered)
print("Sample Customer Prediction:")
print(f"Churn Probability: {prediction['churn_probability']:.3f}")
print(f"Churn Prediction: {prediction['churn_prediction']}")
print(f"Confidence: {prediction['confidence']:.3f}")
```

### 3. Model Monitoring Dashboard

```python
class ModelMonitoringDashboard:
    """Simple model monitoring dashboard"""
    
    def __init__(self, prediction_service):
        self.prediction_service = prediction_service
        self.performance_history = []
    
    def update_performance(self, actual_churn, predicted_churn):
        """Update performance metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        self.performance_history.append({
            'actual': actual_churn,
            'predicted': predicted_churn,
            'timestamp': datetime.now()
        })
    
    def get_performance_metrics(self, window_size=100):
        """Get recent performance metrics"""
        if len(self.performance_history) < window_size:
            recent_data = self.performance_history
        else:
            recent_data = self.performance_history[-window_size:]
        
        if not recent_data:
            return {}
        
        actual = [d['actual'] for d in recent_data]
        predicted = [d['predicted'] for d in recent_data]
        
        return {
            'accuracy': accuracy_score(actual, predicted),
            'precision': precision_score(actual, predicted),
            'recall': recall_score(actual, predicted),
            'f1': f1_score(actual, predicted),
            'sample_size': len(recent_data)
        }
    
    def plot_performance_trends(self):
        """Plot performance trends over time"""
        if len(self.performance_history) < 10:
            print("Not enough data for trend analysis")
            return
        
        # Calculate rolling metrics
        window_size = min(50, len(self.performance_history) // 2)
        rolling_accuracy = []
        timestamps = []
        
        for i in range(window_size, len(self.performance_history)):
            window_data = self.performance_history[i-window_size:i]
            actual = [d['actual'] for d in window_data]
            predicted = [d['predicted'] for d in window_data]
            accuracy = accuracy_score(actual, predicted)
            rolling_accuracy.append(accuracy)
            timestamps.append(self.performance_history[i]['timestamp'])
        
        # Plot trends
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, rolling_accuracy, 'b-', linewidth=2)
        plt.axhline(y=0.8, color='r', linestyle='--', label='Threshold (0.8)')
        plt.xlabel('Time')
        plt.ylabel('Rolling Accuracy')
        plt.title('Model Performance Trends')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Create monitoring dashboard
dashboard = ModelMonitoringDashboard(prediction_service)

# Simulate some predictions and performance updates
for _ in range(50):
    # Simulate customer data
    customer_data = df_engineered.sample(1).drop('churn', axis=1)
    prediction = prediction_service.predict_churn(customer_data)
    
    # Simulate actual outcome (in real scenario, this would come from actual data)
    actual_churn = np.random.binomial(1, prediction['churn_probability'])
    
    # Update performance
    dashboard.update_performance(actual_churn, prediction['churn_prediction'])

# Get monitoring statistics
monitoring_stats = prediction_service.get_monitoring_stats()
performance_metrics = dashboard.get_performance_metrics()

print("Model Monitoring Statistics:")
print("-" * 30)
for key, value in monitoring_stats.items():
    print(f"{key}: {value}")

print("\nPerformance Metrics (Recent):")
print("-" * 30)
for key, value in performance_metrics.items():
    print(f"{key}: {value:.3f}")

# Plot performance trends
dashboard.plot_performance_trends()
```

## Real-World Applications

### 1. Recommendation Systems

```python
def simple_recommendation_system():
    """Simple collaborative filtering recommendation system"""
    # Generate user-item interaction data
    np.random.seed(42)
    n_users = 1000
    n_items = 100
    
    # Create user-item matrix
    user_item_matrix = np.random.binomial(1, 0.1, (n_users, n_items))
    
    # Apply K-means to find user segments
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    user_segments = kmeans.fit_predict(user_item_matrix)
    
    # Calculate item popularity within segments
    recommendations = {}
    for segment in range(5):
        segment_users = user_item_matrix[user_segments == segment]
        item_popularity = segment_users.sum(axis=0)
        top_items = np.argsort(item_popularity)[-10:]  # Top 10 items
        recommendations[f'Segment {segment}'] = top_items
    
    return user_segments, recommendations

user_segments, recommendations = simple_recommendation_system()

print("Recommendation System Results:")
print(f"Number of user segments: {len(np.unique(user_segments))}")
print(f"Segment sizes: {np.bincount(user_segments)}")
print("\nTop items by segment:")
for segment, items in recommendations.items():
    print(f"{segment}: Items {items}")
```

### 2. Anomaly Detection

```python
def anomaly_detection_example():
    """Anomaly detection using isolation forest"""
    from sklearn.ensemble import IsolationForest
    
    # Generate normal and anomalous data
    np.random.seed(42)
    n_normal = 1000
    n_anomalous = 50
    
    # Normal data
    normal_data = np.random.normal(0, 1, (n_normal, 2))
    
    # Anomalous data
    anomalous_data = np.random.uniform(-5, 5, (n_anomalous, 2))
    
    # Combine data
    X_anomaly = np.vstack([normal_data, anomalous_data])
    y_anomaly = np.hstack([np.zeros(n_normal), np.ones(n_anomalous)])
    
    # Apply isolation forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    predictions = iso_forest.fit_predict(X_anomaly)
    
    # Convert predictions (-1 for anomaly, 1 for normal)
    predictions = (predictions == -1).astype(int)
    
    # Calculate performance
    from sklearn.metrics import classification_report
    print("Anomaly Detection Results:")
    print(classification_report(y_anomaly, predictions))
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_anomaly[y_anomaly == 0, 0], X_anomaly[y_anomaly == 0, 1], 
               c='blue', alpha=0.6, label='Normal')
    plt.scatter(X_anomaly[y_anomaly == 1, 0], X_anomaly[y_anomaly == 1, 1], 
               c='red', alpha=0.6, label='Anomalous')
    plt.scatter(X_anomaly[predictions == 1, 0], X_anomaly[predictions == 1, 1], 
               c='orange', marker='x', s=100, linewidths=3, label='Detected Anomalies')
    plt.title('Anomaly Detection Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

anomaly_detection_example()
```

## Best Practices Summary

### 1. Data Quality
- Always validate data quality before modeling
- Handle missing values appropriately
- Check for data leakage
- Ensure proper train/validation/test splits

### 2. Feature Engineering
- Create domain-specific features
- Handle categorical variables properly
- Scale numerical features when needed
- Remove highly correlated features

### 3. Model Selection
- Start with simple models (baselines)
- Use cross-validation for evaluation
- Compare multiple algorithms
- Consider model interpretability

### 4. Evaluation
- Use appropriate metrics for the problem
- Consider business context
- Validate on holdout set
- Monitor for data drift

### 5. Deployment
- Version control your models
- Implement proper error handling
- Monitor model performance
- Plan for model updates

### 6. Monitoring
- Track prediction accuracy
- Monitor feature distributions
- Alert on performance degradation
- Maintain audit trails

This comprehensive approach ensures robust, reliable, and maintainable machine learning systems in production environments. 