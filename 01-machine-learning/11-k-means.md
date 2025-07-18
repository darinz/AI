# 11 - K-Means Clustering

## Overview

K-means is one of the most popular and widely used clustering algorithms. It partitions data into K clusters by minimizing the within-cluster sum of squares (WCSS).

## Algorithm Theory

### Mathematical Foundation

K-means minimizes the objective function:

$$J = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2$$

Where:
- $K$ is the number of clusters
- $C_i$ is the $i$-th cluster
- $\mu_i$ is the centroid of cluster $i$
- $x$ is a data point

### Algorithm Steps

1. **Initialize**: Choose K centroids randomly
2. **Assign**: Assign each point to nearest centroid
3. **Update**: Recalculate centroids as mean of assigned points
4. **Repeat**: Steps 2-3 until convergence

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import seaborn as sns

# Generate sample data
np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

def kmeans_visualization(X, kmeans, title):
    """Visualize K-means clustering results"""
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.6)
    
    # Plot centroids
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Apply K-means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Visualize results
kmeans_visualization(X, kmeans, 'K-Means Clustering (K=4)')
```

## Implementation from Scratch

```python
class KMeansScratch:
    """K-means implementation from scratch"""
    
    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        
    def fit(self, X):
        """Fit K-means to the data"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices]
        
        for iteration in range(self.max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Store old centroids
            old_centroids = self.centroids.copy()
            
            # Update centroids
            for k in range(self.n_clusters):
                if np.sum(self.labels == k) > 0:
                    self.centroids[k] = X[self.labels == k].mean(axis=0)
            
            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                break
        
        # Calculate inertia (within-cluster sum of squares)
        self.inertia_ = np.sum([np.sum((X[self.labels == k] - self.centroids[k])**2) 
                               for k in range(self.n_clusters)])
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

# Test implementation
kmeans_scratch = KMeansScratch(n_clusters=4, random_state=42)
kmeans_scratch.fit(X)

print("Scratch Implementation Results:")
print(f"Number of iterations: {kmeans_scratch.max_iters}")
print(f"Inertia: {kmeans_scratch.inertia_:.2f}")
print(f"Centroids shape: {kmeans_scratch.centroids.shape}")
```

## Initialization Methods

### 1. Random Initialization

```python
def random_initialization(X, k):
    """Random initialization of centroids"""
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, k, replace=False)
    return X[indices]

# Test random initialization
centroids_random = random_initialization(X, 4)
print("Random initialization centroids:")
print(centroids_random)
```

### 2. K-Means++ Initialization

```python
def kmeans_plus_plus(X, k):
    """K-means++ initialization"""
    n_samples, n_features = X.shape
    centroids = np.zeros((k, n_features))
    
    # Choose first centroid randomly
    first_centroid = X[np.random.randint(n_samples)]
    centroids[0] = first_centroid
    
    # Choose remaining centroids
    for i in range(1, k):
        # Calculate distances to existing centroids
        distances = np.min([np.sum((X - centroid)**2, axis=1) for centroid in centroids[:i]], axis=0)
        
        # Choose next centroid with probability proportional to distance squared
        probabilities = distances / np.sum(distances)
        cumulative_probs = np.cumsum(probabilities)
        r = np.random.random()
        
        for j, cum_prob in enumerate(cumulative_probs):
            if r <= cum_prob:
                centroids[i] = X[j]
                break
    
    return centroids

# Test K-means++ initialization
centroids_kmeans_plus = kmeans_plus_plus(X, 4)
print("K-means++ initialization centroids:")
print(centroids_kmeans_plus)
```

### 3. Comparison of Initialization Methods

```python
def compare_initializations(X, k, n_runs=10):
    """Compare different initialization methods"""
    results = {
        'Random': [],
        'K-means++': []
    }
    
    for _ in range(n_runs):
        # Random initialization
        centroids_random = random_initialization(X, k)
        kmeans_random = KMeansScratch(n_clusters=k, random_state=None)
        kmeans_random.centroids = centroids_random
        kmeans_random.fit(X)
        results['Random'].append(kmeans_random.inertia_)
        
        # K-means++ initialization
        centroids_kmeans_plus = kmeans_plus_plus(X, k)
        kmeans_plus = KMeansScratch(n_clusters=k, random_state=None)
        kmeans_plus.centroids = centroids_kmeans_plus
        kmeans_plus.fit(X)
        results['K-means++'].append(kmeans_plus.inertia_)
    
    return results

# Compare initialization methods
init_comparison = compare_initializations(X, 4)

print("Initialization Method Comparison:")
print("-" * 40)
for method, inertias in init_comparison.items():
    print(f"{method}:")
    print(f"  Mean inertia: {np.mean(inertias):.2f}")
    print(f"  Std inertia: {np.std(inertias):.2f}")
    print(f"  Min inertia: {np.min(inertias):.2f}")
    print()
```

## Determining Optimal K

### 1. Elbow Method

```python
def elbow_method(X, max_k=10):
    """Find optimal K using elbow method"""
    inertias = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    return k_values, inertias

# Apply elbow method
k_values, inertias = elbow_method(X, max_k=10)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True, alpha=0.3)
plt.show()

# Calculate elbow point
def find_elbow_point(k_values, inertias):
    """Find the elbow point using the second derivative"""
    # Calculate second derivative
    second_derivative = np.diff(np.diff(inertias))
    elbow_idx = np.argmax(second_derivative) + 2  # +2 because of double diff
    return k_values[elbow_idx]

optimal_k_elbow = find_elbow_point(k_values, inertias)
print(f"Optimal K (Elbow method): {optimal_k_elbow}")
```

### 2. Silhouette Analysis

```python
def silhouette_analysis(X, max_k=10):
    """Find optimal K using silhouette analysis"""
    silhouette_scores = []
    k_values = range(2, max_k + 1)  # Silhouette requires at least 2 clusters
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    return k_values, silhouette_scores

# Apply silhouette analysis
k_values_sil, silhouette_scores = silhouette_analysis(X, max_k=10)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_values_sil, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal K')
plt.grid(True, alpha=0.3)
plt.show()

optimal_k_silhouette = k_values_sil[np.argmax(silhouette_scores)]
print(f"Optimal K (Silhouette): {optimal_k_silhouette}")
```

### 3. Gap Statistic

```python
def gap_statistic(X, max_k=10, n_bootstrap=10):
    """Calculate gap statistic for optimal K"""
    n_samples, n_features = X.shape
    k_values = range(1, max_k + 1)
    
    # Calculate inertia for original data
    inertias_original = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias_original.append(kmeans.inertia_)
    
    # Generate reference datasets
    inertias_reference = []
    for k in k_values:
        k_inertias = []
        for _ in range(n_bootstrap):
            # Generate uniform random data in the same range
            X_min, X_max = X.min(axis=0), X.max(axis=0)
            X_uniform = np.random.uniform(X_min, X_max, (n_samples, n_features))
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_uniform)
            k_inertias.append(kmeans.inertia_)
        
        inertias_reference.append(np.mean(k_inertias))
    
    # Calculate gap statistic
    gaps = np.log(np.array(inertias_reference)) - np.log(np.array(inertias_original))
    
    return k_values, gaps

# Apply gap statistic
k_values_gap, gaps = gap_statistic(X, max_k=10)

# Plot gap statistic
plt.figure(figsize=(10, 6))
plt.plot(k_values_gap, gaps, 'go-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Gap Statistic')
plt.title('Gap Statistic for Optimal K')
plt.grid(True, alpha=0.3)
plt.show()

optimal_k_gap = k_values_gap[np.argmax(gaps)]
print(f"Optimal K (Gap Statistic): {optimal_k_gap}")
```

## Evaluation Metrics

### 1. Inertia (Within-Cluster Sum of Squares)

```python
def calculate_inertia(X, labels, centroids):
    """Calculate inertia for given clustering"""
    inertia = 0
    for k in range(len(centroids)):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            inertia += np.sum((cluster_points - centroids[k])**2)
    return inertia

# Calculate inertia for our clustering
inertia = calculate_inertia(X, kmeans.labels_, kmeans.cluster_centers_)
print(f"Inertia: {inertia:.2f}")
```

### 2. Silhouette Score

```python
def silhouette_analysis_detailed(X, labels):
    """Detailed silhouette analysis"""
    silhouette_avg = silhouette_score(X, labels)
    silhouette_vals = silhouette_samples(X, labels)
    
    print(f"Average silhouette score: {silhouette_avg:.3f}")
    
    # Plot silhouette for each cluster
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_lower = 10
    for i in range(len(np.unique(labels))):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = len(cluster_silhouette_vals)
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.viridis(i / len(np.unique(labels)))
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, cluster_silhouette_vals,
                        facecolor=color, edgecolor=color, alpha=0.7)
        
        y_lower = y_upper + 10
    
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title("Silhouette Plot")
    plt.show()

# Apply detailed silhouette analysis
silhouette_analysis_detailed(X, kmeans.labels_)
```

### 3. Calinski-Harabasz Index

```python
def calinski_harabasz_analysis(X, max_k=10):
    """Analyze Calinski-Harabasz index for different K values"""
    ch_scores = []
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = calinski_harabasz_score(X, labels)
        ch_scores.append(score)
    
    return k_values, ch_scores

# Apply Calinski-Harabasz analysis
k_values_ch, ch_scores = calinski_harabasz_analysis(X, max_k=10)

# Plot Calinski-Harabasz scores
plt.figure(figsize=(10, 6))
plt.plot(k_values_ch, ch_scores, 'mo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Calinski-Harabasz Score')
plt.title('Calinski-Harabasz Index for Optimal K')
plt.grid(True, alpha=0.3)
plt.show()

optimal_k_ch = k_values_ch[np.argmax(ch_scores)]
print(f"Optimal K (Calinski-Harabasz): {optimal_k_ch}")
```

## Advanced K-Means Variants

### 1. Mini-Batch K-Means

```python
from sklearn.cluster import MiniBatchKMeans

def compare_kmeans_variants(X, k=4):
    """Compare standard K-means vs Mini-batch K-means"""
    # Standard K-means
    kmeans_std = KMeans(n_clusters=k, random_state=42, n_init=10)
    start_time = time.time()
    kmeans_std.fit(X)
    std_time = time.time() - start_time
    
    # Mini-batch K-means
    kmeans_mb = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=100)
    start_time = time.time()
    kmeans_mb.fit(X)
    mb_time = time.time() - start_time
    
    return {
        'Standard': {'inertia': kmeans_std.inertia_, 'time': std_time},
        'Mini-batch': {'inertia': kmeans_mb.inertia_, 'time': mb_time}
    }

import time
comparison = compare_kmeans_variants(X, k=4)

print("K-means Variants Comparison:")
for variant, results in comparison.items():
    print(f"{variant}:")
    print(f"  Inertia: {results['inertia']:.2f}")
    print(f"  Time: {results['time']:.4f} seconds")
    print()
```

### 2. K-Means with Different Distance Metrics

```python
def kmeans_custom_distance(X, k, distance_metric='euclidean'):
    """K-means with custom distance metrics"""
    if distance_metric == 'manhattan':
        def distance_func(X, centroids):
            return np.sum(np.abs(X[:, np.newaxis] - centroids), axis=2)
    elif distance_metric == 'euclidean':
        def distance_func(X, centroids):
            return np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    else:
        raise ValueError("Unsupported distance metric")
    
    # Custom K-means implementation
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(100):
        distances = distance_func(X, centroids)
        labels = np.argmin(distances, axis=0)
        
        old_centroids = centroids.copy()
        for i in range(k):
            if np.sum(labels == i) > 0:
                centroids[i] = X[labels == i].mean(axis=0)
        
        if np.allclose(old_centroids, centroids):
            break
    
    return labels, centroids

# Test different distance metrics
labels_euclidean, centroids_euclidean = kmeans_custom_distance(X, 4, 'euclidean')
labels_manhattan, centroids_manhattan = kmeans_custom_distance(X, 4, 'manhattan')

print("Distance Metrics Comparison:")
print(f"Euclidean distance - Inertia: {calculate_inertia(X, labels_euclidean, centroids_euclidean):.2f}")
print(f"Manhattan distance - Inertia: {calculate_inertia(X, labels_manhattan, centroids_manhattan):.2f}")
```

## Practical Applications

### 1. Image Segmentation

```python
from sklearn.datasets import load_sample_image

def image_segmentation_example():
    """K-means for image segmentation"""
    # Load sample image
    china = load_sample_image("china.jpg")
    
    # Reshape image to 2D array
    china_2d = china.reshape(china.shape[0] * china.shape[1], china.shape[2])
    
    # Apply K-means
    kmeans_img = KMeans(n_clusters=8, random_state=42)
    labels_img = kmeans_img.fit_predict(china_2d)
    
    # Reconstruct image
    segmented_img = kmeans_img.cluster_centers_[labels_img]
    segmented_img = segmented_img.reshape(china.shape)
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(china)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(segmented_img.astype(np.uint8))
    axes[1].set_title('Segmented Image (8 clusters)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Uncomment to run image segmentation example
# image_segmentation_example()
```

### 2. Customer Segmentation

```python
def customer_segmentation_example():
    """K-means for customer segmentation"""
    # Generate customer data
    np.random.seed(42)
    n_customers = 1000
    
    # Customer features: age, income, spending_score
    age = np.random.normal(35, 10, n_customers)
    income = np.random.normal(50000, 20000, n_customers)
    spending_score = np.random.normal(50, 15, n_customers)
    
    # Create customer data
    customer_data = np.column_stack([age, income, spending_score])
    
    # Apply K-means
    kmeans_customers = KMeans(n_clusters=4, random_state=42)
    customer_labels = kmeans_customers.fit_predict(customer_data)
    
    # Analyze segments
    segments = {}
    for i in range(4):
        segment_data = customer_data[customer_labels == i]
        segments[f'Segment {i+1}'] = {
            'size': len(segment_data),
            'avg_age': segment_data[:, 0].mean(),
            'avg_income': segment_data[:, 1].mean(),
            'avg_spending': segment_data[:, 2].mean()
        }
    
    # Print segment analysis
    print("Customer Segmentation Results:")
    print("-" * 50)
    for segment, stats in segments.items():
        print(f"\n{segment}:")
        print(f"  Size: {stats['size']} customers")
        print(f"  Average Age: {stats['avg_age']:.1f}")
        print(f"  Average Income: ${stats['avg_income']:,.0f}")
        print(f"  Average Spending Score: {stats['avg_spending']:.1f}")
    
    # Visualize segments
    fig = plt.figure(figsize=(15, 5))
    
    # Age vs Income
    ax1 = fig.add_subplot(131)
    scatter = ax1.scatter(customer_data[:, 0], customer_data[:, 1], 
                         c=customer_labels, cmap='viridis', alpha=0.6)
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Income')
    ax1.set_title('Age vs Income')
    
    # Age vs Spending
    ax2 = fig.add_subplot(132)
    ax2.scatter(customer_data[:, 0], customer_data[:, 2], 
               c=customer_labels, cmap='viridis', alpha=0.6)
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Spending Score')
    ax2.set_title('Age vs Spending Score')
    
    # Income vs Spending
    ax3 = fig.add_subplot(133)
    ax3.scatter(customer_data[:, 1], customer_data[:, 2], 
               c=customer_labels, cmap='viridis', alpha=0.6)
    ax3.set_xlabel('Income')
    ax3.set_ylabel('Spending Score')
    ax3.set_title('Income vs Spending Score')
    
    plt.tight_layout()
    plt.show()

# Run customer segmentation example
customer_segmentation_example()
```

## Summary

K-means clustering is a powerful and widely-used algorithm for unsupervised learning:

1. **Algorithm**: Iteratively assigns points to nearest centroids and updates centroids
2. **Initialization**: K-means++ provides better initialization than random
3. **Optimal K**: Use elbow method, silhouette analysis, or gap statistic
4. **Evaluation**: Inertia, silhouette score, and Calinski-Harabasz index
5. **Variants**: Mini-batch K-means for large datasets
6. **Applications**: Image segmentation, customer segmentation, data preprocessing

Key advantages:
- Simple and fast
- Scales well to large datasets
- Guaranteed convergence

Key limitations:
- Requires specifying number of clusters
- Sensitive to initialization
- Assumes spherical clusters
- May converge to local optima 