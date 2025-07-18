# Particle Filtering

## Overview

Particle filtering, also known as Sequential Monte Carlo (SMC), is a powerful approximate inference method for dynamic Bayesian networks. It represents the posterior distribution using a set of weighted particles and is particularly effective for non-linear, non-Gaussian systems.

## What is Particle Filtering?

Particle filtering is a Monte Carlo method that:
- Represents the posterior distribution as a set of weighted particles
- Updates the particle set sequentially as new observations arrive
- Handles non-linear dynamics and non-Gaussian noise
- Provides approximate but often very accurate results

## Mathematical Foundation

### State Space Model

A dynamic Bayesian network can be represented as:
- **State transition**: x_t ~ p(x_t | x_{t-1})
- **Observation model**: y_t ~ p(y_t | x_t)
- **Initial state**: x_0 ~ p(x_0)

### Particle Filter Algorithm

The particle filter maintains a set of N particles {x_t^(i), w_t^(i)} where:
- x_t^(i) is the i-th particle (state sample)
- w_t^(i) is the i-th particle weight

## Basic Particle Filter Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from collections import defaultdict

class ParticleFilter:
    def __init__(self, n_particles=1000):
        self.n_particles = n_particles
        self.particles = None
        self.weights = None
        self.history = []
    
    def initialize_particles(self, initial_distribution):
        """Initialize particles from initial distribution"""
        self.particles = initial_distribution.rvs(size=self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.history = [(self.particles.copy(), self.weights.copy())]
    
    def predict_step(self, transition_model):
        """Predict step: propagate particles through state transition"""
        # Sample new particles from transition model
        new_particles = np.zeros_like(self.particles)
        
        for i in range(self.n_particles):
            # Sample from p(x_t | x_{t-1})
            new_particles[i] = transition_model.sample(self.particles[i])
        
        self.particles = new_particles
    
    def update_step(self, observation, observation_model):
        """Update step: weight particles based on observation"""
        # Compute likelihood weights
        likelihoods = np.zeros(self.n_particles)
        
        for i in range(self.n_particles):
            likelihoods[i] = observation_model.likelihood(observation, self.particles[i])
        
        # Update weights
        self.weights *= likelihoods
        
        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            # If all weights are zero, reset to uniform
            self.weights = np.ones(self.n_particles) / self.n_particles
    
    def resample(self, method='systematic'):
        """Resample particles to prevent degeneracy"""
        if method == 'systematic':
            self._systematic_resample()
        elif method == 'multinomial':
            self._multinomial_resample()
        elif method == 'stratified':
            self._stratified_resample()
    
    def _systematic_resample(self):
        """Systematic resampling"""
        # Compute cumulative weights
        cumsum_weights = np.cumsum(self.weights)
        
        # Generate systematic samples
        u = np.random.uniform(0, 1/self.n_particles)
        u += np.arange(self.n_particles) / self.n_particles
        
        # Resample particles
        new_particles = np.zeros_like(self.particles)
        new_weights = np.ones(self.n_particles) / self.n_particles
        
        j = 0
        for i in range(self.n_particles):
            while u[i] > cumsum_weights[j]:
                j += 1
            new_particles[i] = self.particles[j]
        
        self.particles = new_particles
        self.weights = new_weights
    
    def _multinomial_resample(self):
        """Multinomial resampling"""
        # Sample indices based on weights
        indices = np.random.choice(self.n_particles, size=self.n_particles, p=self.weights)
        
        # Resample particles
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def _stratified_resample(self):
        """Stratified resampling"""
        # Divide [0,1] into n_particles strata
        u = np.random.uniform(0, 1/self.n_particles, self.n_particles)
        u += np.arange(self.n_particles) / self.n_particles
        
        # Compute cumulative weights
        cumsum_weights = np.cumsum(self.weights)
        
        # Resample particles
        new_particles = np.zeros_like(self.particles)
        new_weights = np.ones(self.n_particles) / self.n_particles
        
        j = 0
        for i in range(self.n_particles):
            while u[i] > cumsum_weights[j]:
                j += 1
            new_particles[i] = self.particles[j]
        
        self.particles = new_particles
        self.weights = new_weights
    
    def step(self, observation, transition_model, observation_model, resample_threshold=0.5):
        """Perform one step of particle filtering"""
        # Predict
        self.predict_step(transition_model)
        
        # Update
        self.update_step(observation, observation_model)
        
        # Check effective sample size
        ess = self.effective_sample_size()
        
        # Resample if necessary
        if ess < resample_threshold * self.n_particles:
            self.resample()
        
        # Store history
        self.history.append((self.particles.copy(), self.weights.copy()))
    
    def effective_sample_size(self):
        """Compute effective sample size"""
        return 1.0 / np.sum(self.weights ** 2)
    
    def estimate_state(self):
        """Estimate current state using weighted average"""
        return np.average(self.particles, weights=self.weights, axis=0)
    
    def estimate_variance(self):
        """Estimate variance of current state"""
        mean = self.estimate_state()
        variance = np.average((self.particles - mean) ** 2, weights=self.weights, axis=0)
        return variance
    
    def get_posterior_samples(self):
        """Get samples from current posterior"""
        return self.particles, self.weights

# Example: Linear Gaussian System
class LinearGaussianTransition:
    def __init__(self, A, Q):
        self.A = A  # State transition matrix
        self.Q = Q  # Process noise covariance
    
    def sample(self, x):
        """Sample from p(x_t | x_{t-1})"""
        mean = self.A @ x
        return multivariate_normal.rvs(mean=mean, cov=self.Q)

class LinearGaussianObservation:
    def __init__(self, H, R):
        self.H = H  # Observation matrix
        self.R = R  # Observation noise covariance
    
    def likelihood(self, y, x):
        """Compute p(y_t | x_t)"""
        mean = self.H @ x
        return multivariate_normal.pdf(y, mean=mean, cov=self.R)

def linear_gaussian_example():
    """Example of particle filtering for linear Gaussian system"""
    
    # System parameters
    dt = 0.1
    A = np.array([[1, dt], [0, 1]])  # Constant velocity model
    Q = np.array([[0.1, 0], [0, 0.1]])  # Process noise
    H = np.array([[1, 0]])  # Observe position only
    R = np.array([[1.0]])  # Observation noise
    
    # Create models
    transition_model = LinearGaussianTransition(A, Q)
    observation_model = LinearGaussianObservation(H, R)
    
    # Create particle filter
    pf = ParticleFilter(n_particles=1000)
    
    # Initialize particles
    initial_distribution = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    pf.initialize_particles(initial_distribution)
    
    # Generate true trajectory and observations
    np.random.seed(42)
    T = 100
    true_states = np.zeros((T, 2))
    observations = np.zeros(T)
    
    # Generate true trajectory
    true_states[0] = [0, 1]  # Initial state: position=0, velocity=1
    for t in range(1, T):
        true_states[t] = transition_model.sample(true_states[t-1])
        observations[t] = observation_model.H @ true_states[t] + np.random.normal(0, np.sqrt(observation_model.R[0, 0]))
    
    # Run particle filter
    estimated_states = np.zeros((T, 2))
    ess_history = []
    
    for t in range(T):
        if t > 0:
            pf.step(observations[t], transition_model, observation_model)
        
        estimated_states[t] = pf.estimate_state()
        ess_history.append(pf.effective_sample_size())
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot true vs estimated trajectory
    plt.subplot(2, 2, 1)
    plt.plot(true_states[:, 0], label='True Position')
    plt.plot(estimated_states[:, 0], label='Estimated Position')
    plt.plot(observations, 'r.', alpha=0.5, label='Observations')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Position Tracking')
    plt.legend()
    
    # Plot velocity
    plt.subplot(2, 2, 2)
    plt.plot(true_states[:, 1], label='True Velocity')
    plt.plot(estimated_states[:, 1], label='Estimated Velocity')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Velocity Tracking')
    plt.legend()
    
    # Plot effective sample size
    plt.subplot(2, 2, 3)
    plt.plot(ess_history)
    plt.xlabel('Time')
    plt.ylabel('Effective Sample Size')
    plt.title('Effective Sample Size')
    
    # Plot particle distribution at final time
    plt.subplot(2, 2, 4)
    final_particles, final_weights = pf.get_posterior_samples()
    plt.scatter(final_particles[:, 0], final_particles[:, 1], 
                c=final_weights, alpha=0.6, s=1)
    plt.plot(true_states[-1, 0], true_states[-1, 1], 'r*', markersize=10, label='True State')
    plt.plot(estimated_states[-1, 0], estimated_states[-1, 1], 'g*', markersize=10, label='Estimated State')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Final Particle Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Compute tracking error
    position_error = np.sqrt(np.mean((true_states[:, 0] - estimated_states[:, 0]) ** 2))
    velocity_error = np.sqrt(np.mean((true_states[:, 1] - estimated_states[:, 1]) ** 2))
    
    print(f"Position RMSE: {position_error:.3f}")
    print(f"Velocity RMSE: {velocity_error:.3f}")
    
    return pf, true_states, observations, estimated_states

# Run the example
if __name__ == "__main__":
    pf, true_states, observations, estimated_states = linear_gaussian_example()
```

## Advanced Particle Filtering

### Rao-Blackwellized Particle Filter

Rao-Blackwellized particle filtering combines particle filtering with analytical integration for parts of the state that can be marginalized analytically.

```python
class RaoBlackwellizedParticleFilter:
    def __init__(self, n_particles=1000):
        self.n_particles = n_particles
        self.particles = None  # Discrete states
        self.continuous_states = None  # Continuous states (analytically marginalized)
        self.weights = None
        self.history = []
    
    def initialize_particles(self, discrete_states, continuous_distribution):
        """Initialize particles with discrete and continuous components"""
        self.particles = discrete_states
        self.continuous_states = continuous_distribution.rvs(size=self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.history = [(self.particles.copy(), self.continuous_states.copy(), self.weights.copy())]
    
    def predict_step(self, discrete_transition, continuous_transition):
        """Predict step with Rao-Blackwellization"""
        # Predict discrete states
        new_discrete_particles = np.zeros_like(self.particles)
        for i in range(self.n_particles):
            new_discrete_particles[i] = discrete_transition.sample(self.particles[i])
        
        # Update continuous states analytically
        new_continuous_states = np.zeros_like(self.continuous_states)
        for i in range(self.n_particles):
            new_continuous_states[i] = continuous_transition.update(self.continuous_states[i], 
                                                                  self.particles[i], 
                                                                  new_discrete_particles[i])
        
        self.particles = new_discrete_particles
        self.continuous_states = new_continuous_states
    
    def update_step(self, observation, observation_model):
        """Update step with Rao-Blackwellization"""
        # Compute likelihood weights
        likelihoods = np.zeros(self.n_particles)
        
        for i in range(self.n_particles):
            # Marginalize over continuous state analytically
            likelihoods[i] = observation_model.marginal_likelihood(observation, self.particles[i])
        
        # Update weights
        self.weights *= likelihoods
        
        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles
    
    def resample(self):
        """Resample particles"""
        indices = np.random.choice(self.n_particles, size=self.n_particles, p=self.weights)
        
        self.particles = self.particles[indices]
        self.continuous_states = self.continuous_states[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles
    
    def step(self, observation, discrete_transition, continuous_transition, observation_model, resample_threshold=0.5):
        """Perform one step of Rao-Blackwellized particle filtering"""
        # Predict
        self.predict_step(discrete_transition, continuous_transition)
        
        # Update
        self.update_step(observation, observation_model)
        
        # Check effective sample size
        ess = self.effective_sample_size()
        
        # Resample if necessary
        if ess < resample_threshold * self.n_particles:
            self.resample()
        
        # Store history
        self.history.append((self.particles.copy(), self.continuous_states.copy(), self.weights.copy()))
    
    def effective_sample_size(self):
        """Compute effective sample size"""
        return 1.0 / np.sum(self.weights ** 2)
    
    def estimate_discrete_state(self):
        """Estimate discrete state using weighted average"""
        return np.average(self.particles, weights=self.weights)
    
    def estimate_continuous_state(self):
        """Estimate continuous state using weighted average"""
        return np.average(self.continuous_states, weights=self.weights, axis=0)

# Example: Jump Markov Linear System
class JumpMarkovTransition:
    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix
    
    def sample(self, current_state):
        """Sample next discrete state"""
        return np.random.choice(len(self.transition_matrix), p=self.transition_matrix[current_state])

class JumpMarkovContinuousTransition:
    def __init__(self, A_matrices, Q_matrices):
        self.A_matrices = A_matrices  # State transition matrices for each discrete state
        self.Q_matrices = Q_matrices  # Process noise covariances for each discrete state
    
    def update(self, continuous_state, old_discrete_state, new_discrete_state):
        """Update continuous state analytically"""
        A = self.A_matrices[new_discrete_state]
        Q = self.Q_matrices[new_discrete_state]
        
        mean = A @ continuous_state
        return multivariate_normal.rvs(mean=mean, cov=Q)

class JumpMarkovObservation:
    def __init__(self, H_matrices, R_matrices):
        self.H_matrices = H_matrices  # Observation matrices for each discrete state
        self.R_matrices = R_matrices  # Observation noise covariances for each discrete state
    
    def marginal_likelihood(self, observation, discrete_state):
        """Compute marginal likelihood p(y_t | discrete_state_t)"""
        H = self.H_matrices[discrete_state]
        R = self.R_matrices[discrete_state]
        
        # For simplicity, assume uniform prior on continuous state
        # In practice, this would be computed analytically
        return multivariate_normal.pdf(observation, mean=np.zeros_like(observation), cov=H @ H.T + R)

def jump_markov_example():
    """Example of Rao-Blackwellized particle filtering for jump Markov system"""
    
    # System parameters
    n_discrete_states = 2
    n_continuous_states = 2
    
    # Discrete state transition matrix
    transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
    
    # Continuous state dynamics for each discrete state
    A_matrices = [
        np.array([[1, 0.1], [0, 0.9]]),  # Slow dynamics
        np.array([[1, 0.1], [0, 1.1]])   # Fast dynamics
    ]
    
    Q_matrices = [
        np.array([[0.1, 0], [0, 0.1]]),  # Low noise
        np.array([[0.5, 0], [0, 0.5]])   # High noise
    ]
    
    # Observation models for each discrete state
    H_matrices = [
        np.array([[1, 0]]),  # Observe position only
        np.array([[1, 0]])
    ]
    
    R_matrices = [
        np.array([[1.0]]),  # Low observation noise
        np.array([[2.0]])   # High observation noise
    ]
    
    # Create models
    discrete_transition = JumpMarkovTransition(transition_matrix)
    continuous_transition = JumpMarkovContinuousTransition(A_matrices, Q_matrices)
    observation_model = JumpMarkovObservation(H_matrices, R_matrices)
    
    # Create Rao-Blackwellized particle filter
    rbpf = RaoBlackwellizedParticleFilter(n_particles=500)
    
    # Initialize particles
    discrete_states = np.random.choice(n_discrete_states, size=rbpf.n_particles)
    continuous_distribution = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    rbpf.initialize_particles(discrete_states, continuous_distribution)
    
    # Generate true trajectory and observations
    np.random.seed(42)
    T = 100
    true_discrete_states = np.zeros(T, dtype=int)
    true_continuous_states = np.zeros((T, n_continuous_states))
    observations = np.zeros(T)
    
    # Generate true trajectory
    true_discrete_states[0] = 0
    true_continuous_states[0] = [0, 1]
    
    for t in range(1, T):
        # Update discrete state
        true_discrete_states[t] = discrete_transition.sample(true_discrete_states[t-1])
        
        # Update continuous state
        A = A_matrices[true_discrete_states[t]]
        Q = Q_matrices[true_discrete_states[t]]
        true_continuous_states[t] = multivariate_normal.rvs(
            mean=A @ true_continuous_states[t-1], cov=Q
        )
        
        # Generate observation
        H = H_matrices[true_discrete_states[t]]
        R = R_matrices[true_discrete_states[t]]
        observations[t] = H @ true_continuous_states[t] + np.random.normal(0, np.sqrt(R[0, 0]))
    
    # Run Rao-Blackwellized particle filter
    estimated_discrete_states = np.zeros(T, dtype=int)
    estimated_continuous_states = np.zeros((T, n_continuous_states))
    
    for t in range(T):
        if t > 0:
            rbpf.step(observations[t], discrete_transition, continuous_transition, observation_model)
        
        estimated_discrete_states[t] = int(rbpf.estimate_discrete_state())
        estimated_continuous_states[t] = rbpf.estimate_continuous_state()
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot discrete states
    plt.subplot(2, 2, 1)
    plt.plot(true_discrete_states, label='True Discrete State')
    plt.plot(estimated_discrete_states, label='Estimated Discrete State')
    plt.xlabel('Time')
    plt.ylabel('Discrete State')
    plt.title('Discrete State Tracking')
    plt.legend()
    
    # Plot continuous states
    plt.subplot(2, 2, 2)
    plt.plot(true_continuous_states[:, 0], label='True Position')
    plt.plot(estimated_continuous_states[:, 0], label='Estimated Position')
    plt.plot(observations, 'r.', alpha=0.5, label='Observations')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Position Tracking')
    plt.legend()
    
    # Plot velocity
    plt.subplot(2, 2, 3)
    plt.plot(true_continuous_states[:, 1], label='True Velocity')
    plt.plot(estimated_continuous_states[:, 1], label='Estimated Velocity')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Velocity Tracking')
    plt.legend()
    
    # Plot discrete state accuracy
    plt.subplot(2, 2, 4)
    accuracy = np.cumsum(true_discrete_states == estimated_discrete_states) / np.arange(1, T + 1)
    plt.plot(accuracy)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Accuracy')
    plt.title('Discrete State Estimation Accuracy')
    
    plt.tight_layout()
    plt.show()
    
    # Compute performance metrics
    discrete_accuracy = np.mean(true_discrete_states == estimated_discrete_states)
    position_rmse = np.sqrt(np.mean((true_continuous_states[:, 0] - estimated_continuous_states[:, 0]) ** 2))
    velocity_rmse = np.sqrt(np.mean((true_continuous_states[:, 1] - estimated_continuous_states[:, 1]) ** 2))
    
    print(f"Discrete state accuracy: {discrete_accuracy:.3f}")
    print(f"Position RMSE: {position_rmse:.3f}")
    print(f"Velocity RMSE: {velocity_rmse:.3f}")
    
    return rbpf, true_discrete_states, true_continuous_states, observations, estimated_discrete_states, estimated_continuous_states

# Run the example
if __name__ == "__main__":
    rbpf, true_discrete, true_continuous, observations, estimated_discrete, estimated_continuous = jump_markov_example()
```

## Resampling Strategies

### Comparison of Resampling Methods

```python
class ResamplingComparison:
    def __init__(self, n_particles=1000):
        self.n_particles = n_particles
    
    def compare_resampling_methods(self, weights, n_trials=1000):
        """Compare different resampling methods"""
        methods = {
            'Systematic': self._systematic_resample,
            'Multinomial': self._multinomial_resample,
            'Stratified': self._stratified_resample,
            'Residual': self._residual_resample
        }
        
        results = {}
        
        for method_name, method_func in methods.items():
            # Run multiple trials
            ess_values = []
            variance_values = []
            
            for _ in range(n_trials):
                # Resample
                new_weights = method_func(weights.copy())
                
                # Compute effective sample size
                ess = 1.0 / np.sum(new_weights ** 2)
                ess_values.append(ess)
                
                # Compute variance of weights
                variance = np.var(new_weights)
                variance_values.append(variance)
            
            results[method_name] = {
                'mean_ess': np.mean(ess_values),
                'std_ess': np.std(ess_values),
                'mean_variance': np.mean(variance_values),
                'std_variance': np.std(variance_values)
            }
        
        return results
    
    def _systematic_resample(self, weights):
        """Systematic resampling"""
        cumsum_weights = np.cumsum(weights)
        u = np.random.uniform(0, 1/self.n_particles)
        u += np.arange(self.n_particles) / self.n_particles
        
        new_weights = np.zeros(self.n_particles)
        j = 0
        for i in range(self.n_particles):
            while u[i] > cumsum_weights[j]:
                j += 1
            new_weights[i] += 1
        
        return new_weights / self.n_particles
    
    def _multinomial_resample(self, weights):
        """Multinomial resampling"""
        indices = np.random.choice(self.n_particles, size=self.n_particles, p=weights)
        new_weights = np.bincount(indices, minlength=self.n_particles)
        return new_weights / self.n_particles
    
    def _stratified_resample(self, weights):
        """Stratified resampling"""
        u = np.random.uniform(0, 1/self.n_particles, self.n_particles)
        u += np.arange(self.n_particles) / self.n_particles
        
        cumsum_weights = np.cumsum(weights)
        new_weights = np.zeros(self.n_particles)
        
        j = 0
        for i in range(self.n_particles):
            while u[i] > cumsum_weights[j]:
                j += 1
            new_weights[i] += 1
        
        return new_weights / self.n_particles
    
    def _residual_resample(self, weights):
        """Residual resampling"""
        # Deterministic part
        deterministic_weights = np.floor(self.n_particles * weights)
        n_deterministic = int(np.sum(deterministic_weights))
        
        # Stochastic part
        residual_weights = self.n_particles * weights - deterministic_weights
        residual_weights /= np.sum(residual_weights)
        
        n_stochastic = self.n_particles - n_deterministic
        stochastic_indices = np.random.choice(self.n_particles, size=n_stochastic, p=residual_weights)
        
        # Combine
        new_weights = deterministic_weights.copy()
        for idx in stochastic_indices:
            new_weights[idx] += 1
        
        return new_weights / self.n_particles

def resampling_comparison_example():
    """Compare different resampling methods"""
    
    # Create comparison object
    comparison = ResamplingComparison(n_particles=1000)
    
    # Create test weights (some particles have high weights, others low)
    np.random.seed(42)
    weights = np.random.exponential(1, 1000)
    weights /= np.sum(weights)
    
    # Compare methods
    results = comparison.compare_resampling_methods(weights, n_trials=100)
    
    # Print results
    print("Resampling Method Comparison:")
    print("=" * 60)
    print(f"{'Method':<12} {'Mean ESS':<12} {'Std ESS':<12} {'Mean Var':<12} {'Std Var':<12}")
    print("-" * 60)
    
    for method, metrics in results.items():
        print(f"{method:<12} {metrics['mean_ess']:<12.2f} {metrics['std_ess']:<12.2f} "
              f"{metrics['mean_variance']:<12.6f} {metrics['std_variance']:<12.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot effective sample sizes
    plt.subplot(1, 2, 1)
    methods = list(results.keys())
    mean_ess = [results[method]['mean_ess'] for method in methods]
    std_ess = [results[method]['std_ess'] for method in methods]
    
    plt.bar(methods, mean_ess, yerr=std_ess, capsize=5)
    plt.ylabel('Effective Sample Size')
    plt.title('Effective Sample Size Comparison')
    plt.xticks(rotation=45)
    
    # Plot weight variances
    plt.subplot(1, 2, 2)
    mean_var = [results[method]['mean_variance'] for method in methods]
    std_var = [results[method]['std_variance'] for method in methods]
    
    plt.bar(methods, mean_var, yerr=std_var, capsize=5)
    plt.ylabel('Weight Variance')
    plt.title('Weight Variance Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return comparison, results

# Run the example
if __name__ == "__main__":
    comparison, results = resampling_comparison_example()
```

## Applications

### 1. Robot Localization

```python
class RobotLocalizationPF:
    def __init__(self, n_particles=1000, map_size=(100, 100)):
        self.n_particles = n_particles
        self.map_size = map_size
        self.pf = ParticleFilter(n_particles)
    
    def initialize_particles(self):
        """Initialize particles uniformly over the map"""
        particles = np.random.uniform(0, self.map_size, size=(self.n_particles, 2))
        weights = np.ones(self.n_particles) / self.n_particles
        self.pf.particles = particles
        self.pf.weights = weights
    
    def motion_model(self, particles, control):
        """Robot motion model"""
        # Simple constant velocity model with noise
        dt = 0.1
        velocity, angular_velocity = control
        
        new_particles = particles.copy()
        
        for i in range(self.n_particles):
            x, y = particles[i]
            theta = np.random.uniform(0, 2*np.pi)  # Random orientation
            
            # Update position
            new_x = x + velocity * dt * np.cos(theta) + np.random.normal(0, 0.1)
            new_y = y + velocity * dt * np.sin(theta) + np.random.normal(0, 0.1)
            
            # Keep particles within map bounds
            new_x = np.clip(new_x, 0, self.map_size[0])
            new_y = np.clip(new_y, 0, self.map_size[1])
            
            new_particles[i] = [new_x, new_y]
        
        return new_particles
    
    def observation_model(self, observation, particles):
        """Robot observation model"""
        # Simple landmark-based observation model
        landmark_positions = np.array([[20, 20], [80, 20], [20, 80], [80, 80]])
        
        likelihoods = np.zeros(self.n_particles)
        
        for i in range(self.n_particles):
            particle_pos = particles[i]
            
            # Compute expected distances to landmarks
            expected_distances = np.linalg.norm(landmark_positions - particle_pos, axis=1)
            
            # Compute likelihood based on observed vs expected distances
            if len(observation) == len(expected_distances):
                likelihood = 1.0
                for obs_dist, exp_dist in zip(observation, expected_distances):
                    likelihood *= norm.pdf(obs_dist, exp_dist, 1.0)
                likelihoods[i] = likelihood
        
        return likelihoods
    
    def localize(self, controls, observations):
        """Perform robot localization"""
        self.initialize_particles()
        
        estimated_positions = []
        
        for control, observation in zip(controls, observations):
            # Predict
            self.pf.particles = self.motion_model(self.pf.particles, control)
            
            # Update
            likelihoods = self.observation_model(observation, self.pf.particles)
            self.pf.weights *= likelihoods
            
            if np.sum(self.pf.weights) > 0:
                self.pf.weights /= np.sum(self.pf.weights)
            
            # Resample if necessary
            ess = self.pf.effective_sample_size()
            if ess < 0.5 * self.n_particles:
                self.pf.resample()
            
            # Estimate position
            estimated_position = np.average(self.pf.particles, weights=self.pf.weights, axis=0)
            estimated_positions.append(estimated_position)
        
        return np.array(estimated_positions)

def robot_localization_example():
    """Example of robot localization using particle filtering"""
    
    # Create robot localization system
    robot_pf = RobotLocalizationPF(n_particles=1000)
    
    # Generate synthetic trajectory and observations
    np.random.seed(42)
    T = 50
    
    # True robot trajectory (circular motion)
    true_positions = []
    t = np.linspace(0, 4*np.pi, T)
    for i in range(T):
        x = 50 + 20 * np.cos(t[i])
        y = 50 + 20 * np.sin(t[i])
        true_positions.append([x, y])
    
    true_positions = np.array(true_positions)
    
    # Generate controls and observations
    controls = []
    observations = []
    
    for i in range(T-1):
        # Control: constant velocity, varying angular velocity
        velocity = 2.0
        angular_velocity = 0.5
        controls.append([velocity, angular_velocity])
        
        # Observation: distances to landmarks
        landmark_positions = np.array([[20, 20], [80, 20], [20, 80], [80, 80]])
        true_pos = true_positions[i+1]
        
        distances = np.linalg.norm(landmark_positions - true_pos, axis=1)
        noisy_distances = distances + np.random.normal(0, 1.0, size=len(distances))
        observations.append(noisy_distances)
    
    # Perform localization
    estimated_positions = robot_pf.localize(controls, observations)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot trajectory
    plt.subplot(2, 2, 1)
    plt.plot(true_positions[:, 0], true_positions[:, 1], 'b-', label='True Trajectory')
    plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'r--', label='Estimated Trajectory')
    
    # Plot landmarks
    landmark_positions = np.array([[20, 20], [80, 20], [20, 80], [80, 80]])
    plt.scatter(landmark_positions[:, 0], landmark_positions[:, 1], c='g', s=100, marker='^', label='Landmarks')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Robot Localization')
    plt.legend()
    plt.grid(True)
    
    # Plot position errors
    plt.subplot(2, 2, 2)
    position_errors = np.linalg.norm(true_positions[1:] - estimated_positions, axis=1)
    plt.plot(position_errors)
    plt.xlabel('Time Step')
    plt.ylabel('Position Error')
    plt.title('Localization Error')
    plt.grid(True)
    
    # Plot particle distribution at final time
    plt.subplot(2, 2, 3)
    plt.scatter(robot_pf.pf.particles[:, 0], robot_pf.pf.particles[:, 1], 
                c=robot_pf.pf.weights, alpha=0.6, s=1)
    plt.plot(true_positions[-1, 0], true_positions[-1, 1], 'r*', markersize=10, label='True Position')
    plt.plot(estimated_positions[-1, 0], estimated_positions[-1, 1], 'g*', markersize=10, label='Estimated Position')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Final Particle Distribution')
    plt.legend()
    plt.grid(True)
    
    # Plot effective sample size
    plt.subplot(2, 2, 4)
    ess_history = []
    for i in range(len(controls)):
        ess = robot_pf.pf.effective_sample_size()
        ess_history.append(ess)
    
    plt.plot(ess_history)
    plt.xlabel('Time Step')
    plt.ylabel('Effective Sample Size')
    plt.title('Effective Sample Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Compute performance metrics
    mean_error = np.mean(position_errors)
    final_error = position_errors[-1]
    
    print(f"Mean localization error: {mean_error:.3f}")
    print(f"Final localization error: {final_error:.3f}")
    
    return robot_pf, true_positions, estimated_positions, position_errors

# Run the example
if __name__ == "__main__":
    robot_pf, true_positions, estimated_positions, position_errors = robot_localization_example()
```

## Key Takeaways

1. **Particle filtering** provides approximate inference for complex dynamic systems
2. **Resampling** prevents particle degeneracy but introduces variance
3. **Rao-Blackwellization** can improve efficiency by analytical marginalization
4. **Effective sample size** is a key diagnostic for particle filter performance
5. **Applications** include robot localization, target tracking, and financial modeling
6. **Trade-offs** exist between accuracy, computational cost, and particle count

## Exercises

1. Implement particle filtering for a non-linear system
2. Compare different resampling strategies on a benchmark problem
3. Apply particle filtering to a real-world tracking problem
4. Implement Rao-Blackwellized particle filtering for a hybrid system
5. Analyze the effect of particle count on filter performance 