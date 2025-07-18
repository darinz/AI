# Feature Templates

## Introduction

Feature templates are systematic ways to generate features from raw data, especially useful in domains like natural language processing, computer vision, and structured data analysis. They provide a framework for creating domain-specific features that capture relevant patterns and relationships in the data.

## Mathematical Foundation

### The Feature Template Framework

A feature template is a function that maps raw data to a feature vector:

$$\phi: \mathcal{X} \rightarrow \mathbb{R}^d$$

Where:
- $\mathcal{X}$ is the input space (text, images, structured data, etc.)
- $\mathbb{R}^d$ is the feature space
- $\phi$ is the feature template function

### Template Types

1. **N-gram Templates**: For sequential data (text, time series)
2. **Convolutional Templates**: For spatial data (images, signals)
3. **Temporal Templates**: For time series data
4. **Relational Templates**: For structured/graph data

## N-gram Templates

### Mathematical Formulation

For text data, n-gram templates extract sequences of n consecutive tokens:

$$\phi_{n-gram}(x) = [f(w_1...w_n), f(w_2...w_{n+1}), ..., f(w_{N-n+1}...w_N)]$$

Where $f$ is a frequency function and $w_i$ are tokens.

### Implementation

```python
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns

class NGramTemplate:
    def __init__(self, n_range=(1, 3), max_features=1000, min_df=2):
        self.n_range = n_range
        self.max_features = max_features
        self.min_df = min_df
        self.vectorizer = None
        
    def fit_transform(self, texts):
        """Create n-gram features from text data"""
        # Configure n-gram range
        ngram_range = (self.n_range[0], self.n_range[1])
        
        # Create vectorizer
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            stop_words='english'
        )
        
        # Transform texts to features
        features = self.vectorizer.fit_transform(texts)
        return features
    
    def get_feature_names(self):
        """Get feature names (n-grams)"""
        return self.vectorizer.get_feature_names_out()
    
    def transform(self, texts):
        """Transform new texts using fitted vectorizer"""
        return self.vectorizer.transform(texts)

# Example usage
sample_texts = [
    "machine learning is amazing",
    "deep learning uses neural networks",
    "natural language processing is cool",
    "computer vision and machine learning",
    "artificial intelligence and deep learning"
]

# Create n-gram features
ngram_template = NGramTemplate(n_range=(1, 2), max_features=50)
features = ngram_template.fit_transform(sample_texts)

# Convert to dense array for visualization
features_dense = features.toarray()
feature_names = ngram_template.get_feature_names()

print(f"Feature matrix shape: {features_dense.shape}")
print(f"Number of features: {len(feature_names)}")

# Visualize feature matrix
plt.figure(figsize=(12, 8))
sns.heatmap(features_dense, 
            xticklabels=feature_names, 
            yticklabels=[f"Text {i+1}" for i in range(len(sample_texts))],
            cmap='Blues', 
            annot=True, 
            fmt='d')
plt.title('N-gram Feature Matrix')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Show most common n-grams
feature_counts = np.sum(features_dense, axis=0)
top_features_idx = np.argsort(feature_counts)[-10:]
top_features = [feature_names[i] for i in top_features_idx]
top_counts = [feature_counts[i] for i in top_features_idx]

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features)), top_counts)
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Frequency')
plt.title('Top 10 Most Common N-grams')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### TF-IDF Templates

```python
class TfidfTemplate:
    def __init__(self, n_range=(1, 2), max_features=1000, min_df=2):
        self.n_range = n_range
        self.max_features = max_features
        self.min_df = min_df
        self.vectorizer = None
        
    def fit_transform(self, texts):
        """Create TF-IDF features from text data"""
        ngram_range = (self.n_range[0], self.n_range[1])
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            stop_words='english'
        )
        
        features = self.vectorizer.fit_transform(texts)
        return features
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()
    
    def transform(self, texts):
        return self.vectorizer.transform(texts)

# Compare Count vs TF-IDF
count_template = NGramTemplate(n_range=(1, 2), max_features=50)
tfidf_template = TfidfTemplate(n_range=(1, 2), max_features=50)

count_features = count_template.fit_transform(sample_texts)
tfidf_features = tfidf_template.fit_transform(sample_texts)

# Visualize comparison
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.heatmap(count_features.toarray(), 
            xticklabels=count_template.get_feature_names(), 
            yticklabels=[f"Text {i+1}" for i in range(len(sample_texts))],
            cmap='Blues', 
            annot=True, 
            fmt='d')
plt.title('Count Features')
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 2, 2)
sns.heatmap(tfidf_features.toarray(), 
            xticklabels=tfidf_template.get_feature_names(), 
            yticklabels=[f"Text {i+1}" for i in range(len(sample_texts))],
            cmap='Blues', 
            annot=True, 
            fmt='.2f')
plt.title('TF-IDF Features')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
```

## Convolutional Templates

### Mathematical Formulation

For spatial data, convolutional templates use sliding windows to extract local patterns:

$$\phi_{conv}(x) = [f(x_{i:i+k}), f(x_{i+1:i+k+1}), ..., f(x_{N-k+1:N})]$$

Where $k$ is the window size and $f$ is a feature function.

### Implementation

```python
class ConvolutionalTemplate:
    def __init__(self, window_size=3, stride=1, feature_functions=None):
        self.window_size = window_size
        self.stride = stride
        self.feature_functions = feature_functions or ['mean', 'std', 'max', 'min']
        
    def extract_features(self, data):
        """Extract convolutional features from 1D data"""
        features = []
        
        for i in range(0, len(data) - self.window_size + 1, self.stride):
            window = data[i:i + self.window_size]
            window_features = []
            
            for func_name in self.feature_functions:
                if func_name == 'mean':
                    window_features.append(np.mean(window))
                elif func_name == 'std':
                    window_features.append(np.std(window))
                elif func_name == 'max':
                    window_features.append(np.max(window))
                elif func_name == 'min':
                    window_features.append(np.min(window))
                elif func_name == 'median':
                    window_features.append(np.median(window))
                elif func_name == 'range':
                    window_features.append(np.max(window) - np.min(window))
            
            features.append(window_features)
        
        return np.array(features)
    
    def fit_transform(self, data_list):
        """Transform multiple sequences"""
        all_features = []
        for data in data_list:
            features = self.extract_features(data)
            all_features.append(features)
        return all_features

# Generate sample time series data
np.random.seed(42)
n_samples = 5
sequence_length = 20

time_series_data = []
for i in range(n_samples):
    # Create different patterns
    if i % 2 == 0:
        # Sinusoidal pattern
        t = np.linspace(0, 4*np.pi, sequence_length)
        data = np.sin(t) + 0.1 * np.random.randn(sequence_length)
    else:
        # Linear trend with noise
        data = np.linspace(0, 2, sequence_length) + 0.1 * np.random.randn(sequence_length)
    time_series_data.append(data)

# Apply convolutional template
conv_template = ConvolutionalTemplate(window_size=5, stride=2)
conv_features = conv_template.fit_transform(time_series_data)

# Visualize results
plt.figure(figsize=(15, 10))

# Plot original time series
plt.subplot(2, 2, 1)
for i, data in enumerate(time_series_data):
    plt.plot(data, label=f'Series {i+1}', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Original Time Series')
plt.legend()
plt.grid(True)

# Plot feature extraction for one series
plt.subplot(2, 2, 2)
series_idx = 0
data = time_series_data[series_idx]
features = conv_features[series_idx]

# Show windows
for i in range(0, len(data) - 4, 2):
    window = data[i:i+5]
    plt.plot(range(i, i+5), window, 'o-', alpha=0.7, label=f'Window {i//2+1}' if i == 0 else "")

plt.plot(data, 'k-', alpha=0.3, label='Original')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Convolutional Windows')
plt.legend()
plt.grid(True)

# Plot extracted features
plt.subplot(2, 2, 3)
feature_names = conv_template.feature_functions
for i, feature_name in enumerate(feature_names):
    feature_values = [features[j][i] for j in range(len(features))]
    plt.plot(feature_values, 'o-', label=feature_name, alpha=0.7)
plt.xlabel('Window Index')
plt.ylabel('Feature Value')
plt.title('Extracted Features')
plt.legend()
plt.grid(True)

# Feature correlation matrix
plt.subplot(2, 2, 4)
feature_matrix = np.vstack([features.flatten() for features in conv_features])
correlation_matrix = np.corrcoef(feature_matrix.T)
sns.heatmap(correlation_matrix, 
            xticklabels=feature_names, 
            yticklabels=feature_names,
            cmap='coolwarm', 
            center=0,
            annot=True, 
            fmt='.2f')
plt.title('Feature Correlation Matrix')

plt.tight_layout()
plt.show()
```

## Temporal Templates

### Mathematical Formulation

Temporal templates capture time-dependent patterns:

$$\phi_{temporal}(x_t) = [f(x_{t-k:t}), f(x_{t-k+1:t+1}), ..., f(x_{t:t+k})]$$

Where $k$ is the temporal window and $f$ captures temporal statistics.

### Implementation

```python
class TemporalTemplate:
    def __init__(self, lag_features=[1, 2, 3, 5, 7], rolling_windows=[3, 5, 7]):
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows
        
    def create_lag_features(self, data):
        """Create lag features"""
        lag_data = {}
        for lag in self.lag_features:
            lag_data[f'lag_{lag}'] = np.roll(data, lag)
            lag_data[f'lag_{lag}'][:lag] = np.nan
        return lag_data
    
    def create_rolling_features(self, data):
        """Create rolling window features"""
        rolling_data = {}
        for window in self.rolling_windows:
            rolling_data[f'rolling_mean_{window}'] = pd.Series(data).rolling(window=window).mean().values
            rolling_data[f'rolling_std_{window}'] = pd.Series(data).rolling(window=window).std().values
            rolling_data[f'rolling_max_{window}'] = pd.Series(data).rolling(window=window).max().values
            rolling_data[f'rolling_min_{window}'] = pd.Series(data).rolling(window=window).min().values
        return rolling_data
    
    def create_difference_features(self, data):
        """Create difference features"""
        diff_data = {}
        diff_data['diff_1'] = np.diff(data, prepend=data[0])
        diff_data['diff_2'] = np.diff(data, n=2, prepend=[data[0], data[0]])
        return diff_data
    
    def fit_transform(self, data_list):
        """Transform multiple time series"""
        all_features = []
        
        for data in data_list:
            features = {}
            
            # Create different types of temporal features
            lag_features = self.create_lag_features(data)
            rolling_features = self.create_rolling_features(data)
            diff_features = self.create_difference_features(data)
            
            # Combine all features
            features.update(lag_features)
            features.update(rolling_features)
            features.update(diff_features)
            
            # Convert to DataFrame
            feature_df = pd.DataFrame(features)
            all_features.append(feature_df)
        
        return all_features

# Apply temporal template
temporal_template = TemporalTemplate()
temporal_features = temporal_template.fit_transform(time_series_data)

# Visualize temporal features
plt.figure(figsize=(15, 10))

# Plot lag features for one series
plt.subplot(2, 2, 1)
series_idx = 0
data = time_series_data[series_idx]
features_df = temporal_features[series_idx]

plt.plot(data, 'k-', label='Original', linewidth=2)
for lag in [1, 2, 3]:
    lag_col = f'lag_{lag}'
    if lag_col in features_df.columns:
        plt.plot(features_df[lag_col], '--', alpha=0.7, label=f'Lag {lag}')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Lag Features')
plt.legend()
plt.grid(True)

# Plot rolling features
plt.subplot(2, 2, 2)
plt.plot(data, 'k-', label='Original', linewidth=2)
for window in [3, 5, 7]:
    mean_col = f'rolling_mean_{window}'
    if mean_col in features_df.columns:
        plt.plot(features_df[mean_col], '--', alpha=0.7, label=f'Rolling Mean {window}')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Rolling Window Features')
plt.legend()
plt.grid(True)

# Plot difference features
plt.subplot(2, 2, 3)
plt.plot(data, 'k-', label='Original', linewidth=2)
for diff_type in ['diff_1', 'diff_2']:
    if diff_type in features_df.columns:
        plt.plot(features_df[diff_type], '--', alpha=0.7, label=diff_type.replace('_', ' ').title())
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Difference Features')
plt.legend()
plt.grid(True)

# Feature correlation heatmap
plt.subplot(2, 2, 4)
correlation_matrix = features_df.corr()
sns.heatmap(correlation_matrix, 
            cmap='coolwarm', 
            center=0,
            annot=False)
plt.title('Temporal Feature Correlations')

plt.tight_layout()
plt.show()

# Show feature statistics
print("Temporal Feature Statistics:")
print(features_df.describe())
```

## Relational Templates

### Mathematical Formulation

For structured data, relational templates capture relationships between entities:

$$\phi_{relational}(x) = [f(x_i, x_j) \text{ for } (i,j) \in E]$$

Where $E$ is the set of edges and $f$ is a relationship function.

### Implementation

```python
class RelationalTemplate:
    def __init__(self, relationship_types=['distance', 'similarity', 'interaction']):
        self.relationship_types = relationship_types
        
    def create_distance_features(self, data):
        """Create distance-based features"""
        n_samples = len(data)
        distance_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i, j] = np.linalg.norm(data[i] - data[j])
        
        return distance_matrix
    
    def create_similarity_features(self, data):
        """Create similarity-based features"""
        n_samples = len(data)
        similarity_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    dot_product = np.dot(data[i], data[j])
                    norm_i = np.linalg.norm(data[i])
                    norm_j = np.linalg.norm(data[j])
                    similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
                else:
                    similarity_matrix[i, j] = 1.0
        
        return similarity_matrix
    
    def create_interaction_features(self, data):
        """Create interaction features"""
        n_samples, n_features = data.shape
        interaction_features = []
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                interaction = data[:, i] * data[:, j]
                interaction_features.append(interaction)
        
        return np.column_stack(interaction_features)
    
    def fit_transform(self, data):
        """Create relational features"""
        features = {}
        
        if 'distance' in self.relationship_types:
            features['distance'] = self.create_distance_features(data)
        
        if 'similarity' in self.relationship_types:
            features['similarity'] = self.create_similarity_features(data)
        
        if 'interaction' in self.relationship_types:
            features['interaction'] = self.create_interaction_features(data)
        
        return features

# Generate sample relational data
np.random.seed(42)
n_samples = 10
n_features = 4

relational_data = np.random.randn(n_samples, n_features)

# Apply relational template
relational_template = RelationalTemplate()
relational_features = relational_template.fit_transform(relational_data)

# Visualize relational features
plt.figure(figsize=(15, 5))

# Distance matrix
plt.subplot(1, 3, 1)
sns.heatmap(relational_features['distance'], 
            cmap='viridis', 
            annot=True, 
            fmt='.2f')
plt.title('Distance Matrix')
plt.xlabel('Sample Index')
plt.ylabel('Sample Index')

# Similarity matrix
plt.subplot(1, 3, 2)
sns.heatmap(relational_features['similarity'], 
            cmap='coolwarm', 
            center=0,
            annot=True, 
            fmt='.2f')
plt.title('Similarity Matrix')
plt.xlabel('Sample Index')
plt.ylabel('Sample Index')

# Interaction features
plt.subplot(1, 3, 3)
interaction_data = relational_features['interaction']
plt.imshow(interaction_data.T, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Interaction Features')
plt.xlabel('Sample Index')
plt.ylabel('Feature Pair Index')

plt.tight_layout()
plt.show()

print(f"Distance matrix shape: {relational_features['distance'].shape}")
print(f"Similarity matrix shape: {relational_features['similarity'].shape}")
print(f"Interaction features shape: {relational_features['interaction'].shape}")
```

## Domain-Specific Templates

### Text Processing Templates

```python
class TextProcessingTemplate:
    def __init__(self):
        self.feature_functions = {
            'length': self.text_length,
            'word_count': self.word_count,
            'avg_word_length': self.avg_word_length,
            'punctuation_count': self.punctuation_count,
            'uppercase_ratio': self.uppercase_ratio,
            'digit_count': self.digit_count
        }
    
    def text_length(self, text):
        return len(text)
    
    def word_count(self, text):
        return len(text.split())
    
    def avg_word_length(self, text):
        words = text.split()
        if not words:
            return 0
        return np.mean([len(word) for word in words])
    
    def punctuation_count(self, text):
        import string
        return sum(1 for char in text if char in string.punctuation)
    
    def uppercase_ratio(self, text):
        if not text:
            return 0
        return sum(1 for char in text if char.isupper()) / len(text)
    
    def digit_count(self, text):
        return sum(1 for char in text if char.isdigit())
    
    def fit_transform(self, texts):
        """Extract text processing features"""
        features = {}
        
        for func_name, func in self.feature_functions.items():
            features[func_name] = [func(text) for text in texts]
        
        return pd.DataFrame(features)

# Apply text processing template
text_template = TextProcessingTemplate()
text_features = text_template.fit_transform(sample_texts)

# Visualize text features
plt.figure(figsize=(15, 10))

# Feature values
plt.subplot(2, 3, 1)
plt.bar(range(len(sample_texts)), text_features['length'])
plt.xlabel('Text Index')
plt.ylabel('Length')
plt.title('Text Length')
plt.xticks(range(len(sample_texts)))

plt.subplot(2, 3, 2)
plt.bar(range(len(sample_texts)), text_features['word_count'])
plt.xlabel('Text Index')
plt.ylabel('Word Count')
plt.title('Word Count')
plt.xticks(range(len(sample_texts)))

plt.subplot(2, 3, 3)
plt.bar(range(len(sample_texts)), text_features['avg_word_length'])
plt.xlabel('Text Index')
plt.ylabel('Average Word Length')
plt.title('Average Word Length')
plt.xticks(range(len(sample_texts)))

plt.subplot(2, 3, 4)
plt.bar(range(len(sample_texts)), text_features['punctuation_count'])
plt.xlabel('Text Index')
plt.ylabel('Punctuation Count')
plt.title('Punctuation Count')
plt.xticks(range(len(sample_texts)))

plt.subplot(2, 3, 5)
plt.bar(range(len(sample_texts)), text_features['uppercase_ratio'])
plt.xlabel('Text Index')
plt.ylabel('Uppercase Ratio')
plt.title('Uppercase Ratio')
plt.xticks(range(len(sample_texts)))

plt.subplot(2, 3, 6)
plt.bar(range(len(sample_texts)), text_features['digit_count'])
plt.xlabel('Text Index')
plt.ylabel('Digit Count')
plt.title('Digit Count')
plt.xticks(range(len(sample_texts)))

plt.tight_layout()
plt.show()

print("Text Processing Features:")
print(text_features)
```

### Image Processing Templates

```python
class ImageProcessingTemplate:
    def __init__(self, feature_types=['histogram', 'texture', 'edges']):
        self.feature_types = feature_types
        
    def create_histogram_features(self, image):
        """Create histogram features"""
        # Simulate image histogram (RGB channels)
        hist_features = []
        for channel in range(3):  # RGB
            # Simulate histogram bins
            hist = np.random.rand(256)  # 256 bins for each channel
            hist = hist / np.sum(hist)  # Normalize
            hist_features.extend(hist)
        return hist_features
    
    def create_texture_features(self, image):
        """Create texture features using GLCM-like approach"""
        # Simulate texture features
        texture_features = [
            np.random.rand(),  # Contrast
            np.random.rand(),  # Homogeneity
            np.random.rand(),  # Energy
            np.random.rand(),  # Correlation
        ]
        return texture_features
    
    def create_edge_features(self, image):
        """Create edge-based features"""
        # Simulate edge features
        edge_features = [
            np.random.rand(),  # Edge density
            np.random.rand(),  # Edge direction
            np.random.rand(),  # Edge strength
        ]
        return edge_features
    
    def fit_transform(self, images):
        """Extract image features"""
        all_features = []
        
        for image in images:
            features = []
            
            if 'histogram' in self.feature_types:
                features.extend(self.create_histogram_features(image))
            
            if 'texture' in self.feature_types:
                features.extend(self.create_texture_features(image))
            
            if 'edges' in self.feature_types:
                features.extend(self.create_edge_features(image))
            
            all_features.append(features)
        
        return np.array(all_features)

# Simulate image data
n_images = 5
image_template = ImageProcessingTemplate()
image_features = image_template.fit_transform(range(n_images))

# Visualize image features
plt.figure(figsize=(15, 5))

# Histogram features
plt.subplot(1, 3, 1)
hist_features = image_features[:, :768]  # First 768 features are histogram
plt.imshow(hist_features, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Histogram Features')
plt.xlabel('Histogram Bins')
plt.ylabel('Image Index')

# Texture features
plt.subplot(1, 3, 2)
texture_features = image_features[:, 768:772]  # Next 4 features are texture
plt.imshow(texture_features, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.title('Texture Features')
plt.xlabel('Texture Feature')
plt.ylabel('Image Index')

# Edge features
plt.subplot(1, 3, 3)
edge_features = image_features[:, 772:]  # Last 3 features are edge
plt.imshow(edge_features, cmap='plasma', aspect='auto')
plt.colorbar()
plt.title('Edge Features')
plt.xlabel('Edge Feature')
plt.ylabel('Image Index')

plt.tight_layout()
plt.show()

print(f"Image features shape: {image_features.shape}")
print(f"Histogram features: {hist_features.shape[1]}")
print(f"Texture features: {texture_features.shape[1]}")
print(f"Edge features: {edge_features.shape[1]}")
```

## Template Combination and Selection

### Feature Selection for Templates

```python
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

class TemplateSelector:
    def __init__(self, selection_method='mutual_info', k_best=10):
        self.selection_method = selection_method
        self.k_best = k_best
        self.selector = None
        
    def fit_transform(self, X, y):
        """Select best features from template"""
        if self.selection_method == 'mutual_info':
            self.selector = SelectKBest(score_func=mutual_info_regression, k=self.k_best)
        elif self.selection_method == 'f_regression':
            self.selector = SelectKBest(score_func=f_regression, k=self.k_best)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        
        X_selected = self.selector.fit_transform(X, y)
        return X_selected
    
    def get_support(self):
        """Get feature support mask"""
        return self.selector.get_support()
    
    def get_feature_scores(self):
        """Get feature importance scores"""
        return self.selector.scores_

# Generate sample data for feature selection
np.random.seed(42)
n_samples = 100
n_features = 50

# Create synthetic data with some informative features
X_synthetic = np.random.randn(n_samples, n_features)
y_synthetic = (2 * X_synthetic[:, 0] + 
               1.5 * X_synthetic[:, 1]**2 + 
               0.5 * X_synthetic[:, 2] * X_synthetic[:, 3] + 
               0.1 * np.random.randn(n_samples))

# Apply feature selection
selector = TemplateSelector(selection_method='mutual_info', k_best=10)
X_selected = selector.fit_transform(X_synthetic, y_synthetic)

# Visualize feature selection results
plt.figure(figsize=(15, 5))

# Feature scores
plt.subplot(1, 3, 1)
scores = selector.get_feature_scores()
top_features_idx = np.argsort(scores)[-10:]
plt.barh(range(10), scores[top_features_idx])
plt.yticks(range(10), [f'Feature {i}' for i in top_features_idx])
plt.xlabel('Mutual Information Score')
plt.title('Top 10 Feature Scores')
plt.gca().invert_yaxis()

# Selected features
plt.subplot(1, 3, 2)
support = selector.get_support()
selected_features = np.where(support)[0]
plt.bar(range(len(selected_features)), scores[selected_features])
plt.xlabel('Selected Feature Index')
plt.ylabel('Score')
plt.title('Selected Features')
plt.xticks(range(len(selected_features)), selected_features)

# Feature correlation with target
plt.subplot(1, 3, 3)
correlations = [np.corrcoef(X_synthetic[:, i], y_synthetic)[0, 1] for i in range(n_features)]
plt.bar(range(n_features), np.abs(correlations))
plt.xlabel('Feature Index')
plt.ylabel('|Correlation with Target|')
plt.title('Feature-Target Correlations')

plt.tight_layout()
plt.show()

print(f"Original features: {X_synthetic.shape[1]}")
print(f"Selected features: {X_selected.shape[1]}")
print(f"Selected feature indices: {selected_features}")
```

### Template Performance Comparison

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def evaluate_template_performance(X, y, template_name):
    """Evaluate template performance using cross-validation"""
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    return scores.mean(), scores.std()

# Compare different templates
templates = {
    'N-gram': ngram_template.fit_transform(sample_texts),
    'TF-IDF': tfidf_template.fit_transform(sample_texts),
    'Text Processing': text_features.values,
    'Temporal': temporal_features[0].values,  # Use first series
    'Relational': relational_features['distance'].flatten().reshape(-1, 1)
}

# Generate target for comparison
y_template = np.random.randn(len(sample_texts))

# Evaluate each template
results = {}
for name, features in templates.items():
    if features.shape[0] == len(y_template):
        mean_score, std_score = evaluate_template_performance(features, y_template, name)
        results[name] = {'mean': mean_score, 'std': std_score}

# Visualize results
plt.figure(figsize=(12, 5))

# Performance comparison
plt.subplot(1, 2, 1)
template_names = list(results.keys())
mean_scores = [results[name]['mean'] for name in template_names]
std_scores = [results[name]['std'] for name in template_names]

plt.bar(template_names, mean_scores, yerr=std_scores, capsize=5)
plt.ylabel('Cross-validation R² Score')
plt.title('Template Performance Comparison')
plt.xticks(rotation=45)

# Feature count comparison
plt.subplot(1, 2, 2)
feature_counts = [templates[name].shape[1] for name in template_names]
plt.bar(template_names, feature_counts)
plt.ylabel('Number of Features')
plt.title('Feature Count Comparison')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("Template Performance Results:")
for name, result in results.items():
    print(f"{name}: R² = {result['mean']:.4f} ± {result['std']:.4f}")
```

## Best Practices and Guidelines

### 1. Template Selection Strategy

```python
class TemplatePipeline:
    def __init__(self, templates=None):
        self.templates = templates or {}
        self.feature_names = []
        self.combined_features = None
        
    def add_template(self, name, template_func, **kwargs):
        """Add a template to the pipeline"""
        self.templates[name] = (template_func, kwargs)
        
    def fit_transform(self, data, **data_kwargs):
        """Apply all templates and combine features"""
        all_features = []
        
        for name, (template_func, kwargs) in self.templates.items():
            print(f"Applying {name} template...")
            features = template_func(**data_kwargs, **kwargs)
            
            if isinstance(features, np.ndarray):
                all_features.append(features)
            elif isinstance(features, pd.DataFrame):
                all_features.append(features.values)
            else:
                # Handle sparse matrices
                all_features.append(features.toarray())
            
            self.feature_names.extend([f"{name}_{i}" for i in range(features.shape[1])])
        
        self.combined_features = np.hstack(all_features)
        return self.combined_features
    
    def get_feature_names(self):
        return self.feature_names

# Example pipeline
pipeline = TemplatePipeline()

# Add different templates
pipeline.add_template('ngram', lambda texts: NGramTemplate().fit_transform(texts))
pipeline.add_template('text_proc', lambda texts: TextProcessingTemplate().fit_transform(texts))

# Apply pipeline
combined_features = pipeline.fit_transform(texts=sample_texts)

print(f"Combined features shape: {combined_features.shape}")
print(f"Feature names: {pipeline.get_feature_names()[:10]}...")  # Show first 10
```

### 2. Template Validation

```python
def validate_template(template_func, data, expected_shape=None):
    """Validate template function"""
    try:
        features = template_func(data)
        
        # Check output type
        if not isinstance(features, (np.ndarray, pd.DataFrame)):
            print("Warning: Template should return numpy array or pandas DataFrame")
        
        # Check shape
        if expected_shape and features.shape != expected_shape:
            print(f"Warning: Expected shape {expected_shape}, got {features.shape}")
        
        # Check for NaN values
        if isinstance(features, np.ndarray):
            if np.any(np.isnan(features)):
                print("Warning: Template contains NaN values")
        else:
            if features.isnull().any().any():
                print("Warning: Template contains NaN values")
        
        return True
        
    except Exception as e:
        print(f"Error in template: {e}")
        return False

# Validate templates
print("Validating templates...")
validate_template(lambda x: NGramTemplate().fit_transform(x), sample_texts)
validate_template(lambda x: TextProcessingTemplate().fit_transform(x), sample_texts)
```

## Conclusion

Feature templates provide a systematic approach to feature engineering across different domains. Key takeaways:

### Benefits:
1. **Systematic Approach**: Structured way to generate features
2. **Domain Knowledge**: Incorporates domain-specific insights
3. **Scalability**: Can be applied to large datasets
4. **Reproducibility**: Consistent feature generation

### Best Practices:
1. **Start Simple**: Begin with basic templates
2. **Validate Outputs**: Check for NaN values and expected shapes
3. **Feature Selection**: Use selection methods to reduce dimensionality
4. **Cross-validation**: Evaluate template performance
5. **Domain Expertise**: Leverage domain knowledge for template design

### When to Use:
- **Structured Data**: When data has clear patterns
- **Domain-Specific Problems**: When domain knowledge is available
- **Feature Engineering**: As part of a larger ML pipeline
- **Interpretability**: When feature interpretability is important

Feature templates bridge the gap between raw data and machine learning models, providing a systematic way to extract meaningful features.

## Further Reading

- **Books**: "Feature Engineering for Machine Learning" by Alice Zheng, "Applied Text Analysis with Python" by Benjamin Bengfort
- **Papers**: Original n-gram papers, feature engineering literature
- **Online Resources**: Scikit-learn documentation, spaCy documentation
- **Practice**: Text classification competitions, image processing projects 