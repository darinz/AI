# Smoothing in Bayesian Networks

## Overview

Smoothing is a technique for estimating past states using both past and future observations. It provides more accurate estimates than filtering (which only uses past observations) by incorporating all available information.

## What is Smoothing?

Smoothing computes the posterior probability of a state at time t given all observations up to time T:
$$P(x_t | y_{1:T})$$

This is more accurate than filtering because it uses future observations to improve estimates of past states.

## Types of Smoothing

### 1. Fixed-Interval Smoothing

Fixed-interval smoothing estimates all states given all observations in a fixed time window.

### 2. Fixed-Lag Smoothing

Fixed-lag smoothing estimates states with a fixed delay, using observations up to time t+L.

### 3. Fixed-Point Smoothing

Fixed-point smoothing estimates a specific state at time t given all observations.

## Mathematical Foundation

### Forward-Backward Algorithm for Smoothing

The smoothing probability can be computed using the forward-backward algorithm:

$$P(x_t | y_{1:T}) = \frac{P(x_t, y_{1:T})}{P(y_{1:T})} = \frac{\alpha_t(x_t) \beta_t(x_t)}{\sum_{x_t} \alpha_t(x_t) \beta_t(x_t)}$$

Where:
- $\alpha_t(x_t) = P(x_t, y_{1:t})$ (forward variable)
- $\beta_t(x_t) = P(y_{t+1:T} | x_t)$ (backward variable)

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from collections import defaultdict

class SmoothingAlgorithm:
    def __init__(self, transition_model, observation_model):
        self.transition_model = transition_model
        self.observation_model = observation_model
    
    def fixed_interval_smoothing(self, observations):
        """Perform fixed-interval smoothing using forward-backward algorithm"""
        T = len(observations)
        n_states = self.transition_model.n_states
        
        # Forward pass
        alpha = self._forward_pass(observations)
        
        # Backward pass
        beta = self._backward_pass(observations)
        
        # Compute smoothing probabilities
        gamma = np.zeros((T, n_states))
        
        for t in range(T):
            # Normalization constant
            normalization = np.sum(alpha[t, :] * beta[t, :])
            
            if normalization > 0:
                gamma[t, :] = (alpha[t, :] * beta[t, :]) / normalization
            else:
                gamma[t, :] = alpha[t, :] / np.sum(alpha[t, :])
        
        return gamma, alpha, beta
    
    def _forward_pass(self, observations):
        """Compute forward variables"""
        T = len(observations)
        n_states = self.transition_model.n_states
        
        alpha = np.zeros((T, n_states))
        
        # Initialization
        for i in range(n_states):
            alpha[0, i] = (self.transition_model.initial_prob[i] * 
                          self.observation_model.likelihood(observations[0], i))
        
        # Recursion
        for t in range(1, T):
            for i in range(n_states):
                alpha[t, i] = 0
                for j in range(n_states):
                    alpha[t, i] += (alpha[t-1, j] * 
                                  self.transition_model.transition_prob[j, i])
                alpha[t, i] *= self.observation_model.likelihood(observations[t], i)
        
        return alpha
    
    def _backward_pass(self, observations):
        """Compute backward variables"""
        T = len(observations)
        n_states = self.transition_model.n_states
        
        beta = np.zeros((T, n_states))
        
        # Initialization
        beta[T-1, :] = 1.0
        
        # Recursion
        for t in range(T-2, -1, -1):
            for i in range(n_states):
                beta[t, i] = 0
                for j in range(n_states):
                    beta[t, i] += (self.transition_model.transition_prob[i, j] * 
                                 self.observation_model.likelihood(observations[t+1], j) * 
                                 beta[t+1, j])
        
        return beta
    
    def fixed_lag_smoothing(self, observations, lag=5):
        """Perform fixed-lag smoothing"""
        T = len(observations)
        n_states = self.transition_model.n_states
        
        smoothed_states = np.zeros((T, n_states))
        
        for t in range(T):
            # Use observations up to min(t + lag, T)
            end_time = min(t + lag, T)
            window_observations = observations[:end_time]
            
            # Perform smoothing on this window
            gamma, _, _ = self.fixed_interval_smoothing(window_observations)
            
            # Take the estimate for time t
            if t < len(gamma):
                smoothed_states[t, :] = gamma[t, :]
            else:
                # Use filtering if t is beyond the window
                alpha = self._forward_pass(window_observations)
                smoothed_states[t, :] = alpha[-1, :] / np.sum(alpha[-1, :])
        
        return smoothed_states
    
    def rauch_tung_striebel_smoothing(self, observations):
        """Rauch-Tung-Striebel smoothing for linear Gaussian systems"""
        T = len(observations)
        n_states = self.transition_model.n_states
        
        # Forward pass (Kalman filter)
        filtered_means = np.zeros((T, n_states))
        filtered_covariances = np.zeros((T, n_states, n_states))
        
        # Initialize
        filtered_means[0] = self.transition_model.initial_mean
        filtered_covariances[0] = self.transition_model.initial_covariance
        
        # Forward pass
        for t in range(1, T):
            # Predict
            predicted_mean = (self.transition_model.transition_matrix @ 
                            filtered_means[t-1])
            predicted_covariance = (self.transition_model.transition_matrix @ 
                                  filtered_covariances[t-1] @ 
                                  self.transition_model.transition_matrix.T + 
                                  self.transition_model.process_noise)
            
            # Update
            kalman_gain = (predicted_covariance @ 
                          self.observation_model.observation_matrix.T @ 
                          np.linalg.inv(self.observation_model.observation_matrix @ 
                                       predicted_covariance @ 
                                       self.observation_model.observation_matrix.T + 
                                       self.observation_model.observation_noise))
            
            filtered_means[t] = (predicted_mean + 
                               kalman_gain @ (observations[t] - 
                                            self.observation_model.observation_matrix @ 
                                            predicted_mean))
            filtered_covariances[t] = ((np.eye(n_states) - 
                                      kalman_gain @ self.observation_model.observation_matrix) @ 
                                     predicted_covariance)
        
        # Backward pass (smoothing)
        smoothed_means = np.zeros((T, n_states))
        smoothed_covariances = np.zeros((T, n_states, n_states))
        
        # Initialize
        smoothed_means[T-1] = filtered_means[T-1]
        smoothed_covariances[T-1] = filtered_covariances[T-1]
        
        # Backward pass
        for t in range(T-2, -1, -1):
            # Predict
            predicted_mean = (self.transition_model.transition_matrix @ 
                            filtered_means[t])
            predicted_covariance = (self.transition_model.transition_matrix @ 
                                  filtered_covariances[t] @ 
                                  self.transition_model.transition_matrix.T + 
                                  self.transition_model.process_noise)
            
            # Smoothing gain
            smoothing_gain = (filtered_covariances[t] @ 
                            self.transition_model.transition_matrix.T @ 
                            np.linalg.inv(predicted_covariance))
            
            # Update
            smoothed_means[t] = (filtered_means[t] + 
                               smoothing_gain @ (smoothed_means[t+1] - predicted_mean))
            smoothed_covariances[t] = (filtered_covariances[t] + 
                                     smoothing_gain @ (smoothed_covariances[t+1] - 
                                                     predicted_covariance) @ 
                                     smoothing_gain.T)
        
        return smoothed_means, smoothed_covariances

# Example: Linear Gaussian System
class LinearGaussianTransition:
    def __init__(self, n_states=2):
        self.n_states = n_states
        self.transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        self.process_noise = np.array([[0.1, 0], [0, 0.1]])
        self.initial_mean = np.array([0, 0])
        self.initial_covariance = np.array([[1, 0], [0, 1]])
        self.initial_prob = np.array([0.5, 0.5])
    
    @property
    def transition_prob(self):
        return self.transition_matrix

class LinearGaussianObservation:
    def __init__(self, n_states=2):
        self.n_states = n_states
        self.observation_matrix = np.array([[1, 0]])
        self.observation_noise = np.array([[1.0]])
    
    def likelihood(self, observation, state):
        """Compute observation likelihood"""
        mean = self.observation_matrix @ np.array([state, 0])
        return norm.pdf(observation, mean[0], np.sqrt(self.observation_noise[0, 0]))

def linear_gaussian_smoothing_example():
    """Example of smoothing for linear Gaussian system"""
    
    # Create models
    transition_model = LinearGaussianTransition()
    observation_model = LinearGaussianObservation()
    
    # Create smoother
    smoother = SmoothingAlgorithm(transition_model, observation_model)
    
    # Generate synthetic data
    np.random.seed(42)
    T = 100
    
    # Generate true states
    true_states = np.zeros(T, dtype=int)
    true_states[0] = np.random.choice([0, 1], p=transition_model.initial_prob)
    
    for t in range(1, T):
        true_states[t] = np.random.choice([0, 1], p=transition_model.transition_prob[true_states[t-1]])
    
    # Generate observations
    observations = np.zeros(T)
    for t in range(T):
        mean = observation_model.observation_matrix @ np.array([true_states[t], 0])
        observations[t] = np.random.normal(mean[0], np.sqrt(observation_model.observation_noise[0, 0]))
    
    # Perform smoothing
    gamma, alpha, beta = smoother.fixed_interval_smoothing(observations)
    
    # Get most likely states
    smoothed_states = np.argmax(gamma, axis=1)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot observations
    plt.subplot(3, 2, 1)
    plt.plot(observations, 'b.', alpha=0.5, label='Observations')
    plt.xlabel('Time')
    plt.ylabel('Observation')
    plt.title('Observations')
    plt.legend()
    
    # Plot true vs smoothed states
    plt.subplot(3, 2, 2)
    plt.plot(true_states, 'b-', label='True States', linewidth=2)
    plt.plot(smoothed_states, 'r--', label='Smoothed States', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('State Estimation')
    plt.legend()
    
    # Plot smoothing probabilities
    plt.subplot(3, 2, 3)
    plt.plot(gamma[:, 0], 'b-', label='P(State=0)')
    plt.plot(gamma[:, 1], 'r-', label='P(State=1)')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.title('Smoothing Probabilities')
    plt.legend()
    
    # Plot forward variables
    plt.subplot(3, 2, 4)
    plt.plot(alpha[:, 0], 'b-', label='α(State=0)')
    plt.plot(alpha[:, 1], 'r-', label='α(State=1)')
    plt.xlabel('Time')
    plt.ylabel('Forward Variable')
    plt.title('Forward Variables')
    plt.legend()
    
    # Plot backward variables
    plt.subplot(3, 2, 5)
    plt.plot(beta[:, 0], 'b-', label='β(State=0)')
    plt.plot(beta[:, 1], 'r-', label='β(State=1)')
    plt.xlabel('Time')
    plt.ylabel('Backward Variable')
    plt.title('Backward Variables')
    plt.legend()
    
    # Plot estimation accuracy
    plt.subplot(3, 2, 6)
    accuracy = np.cumsum(true_states == smoothed_states) / np.arange(1, T + 1)
    plt.plot(accuracy)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Accuracy')
    plt.title('Estimation Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Compute performance metrics
    accuracy = np.mean(true_states == smoothed_states)
    print(f"Smoothing accuracy: {accuracy:.3f}")
    
    return smoother, true_states, observations, gamma, smoothed_states

# Run the example
if __name__ == "__main__":
    smoother, true_states, observations, gamma, smoothed_states = linear_gaussian_smoothing_example()
```

## Fixed-Lag Smoothing

Fixed-lag smoothing provides estimates with a fixed delay, useful for real-time applications.

```python
class FixedLagSmoother:
    def __init__(self, transition_model, observation_model, lag=5):
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.lag = lag
        self.buffer = []
        self.estimates = []
    
    def update(self, observation):
        """Update with new observation"""
        self.buffer.append(observation)
        
        if len(self.buffer) > self.lag:
            # Remove oldest observation
            self.buffer.pop(0)
        
        if len(self.buffer) >= 2:
            # Perform smoothing on the buffer
            smoother = SmoothingAlgorithm(self.transition_model, self.observation_model)
            gamma, _, _ = smoother.fixed_interval_smoothing(self.buffer)
            
            # Return estimate for the oldest time in buffer
            estimate = gamma[0, :]
            self.estimates.append(estimate)
            
            return estimate
        else:
            # Not enough observations yet
            return None
    
    def get_all_estimates(self):
        """Get all smoothed estimates"""
        return np.array(self.estimates)

def fixed_lag_smoothing_example():
    """Example of fixed-lag smoothing"""
    
    # Create models
    transition_model = LinearGaussianTransition()
    observation_model = LinearGaussianObservation()
    
    # Create fixed-lag smoother
    lag = 5
    fixed_lag_smoother = FixedLagSmoother(transition_model, observation_model, lag)
    
    # Generate synthetic data
    np.random.seed(42)
    T = 100
    
    # Generate true states
    true_states = np.zeros(T, dtype=int)
    true_states[0] = np.random.choice([0, 1], p=transition_model.initial_prob)
    
    for t in range(1, T):
        true_states[t] = np.random.choice([0, 1], p=transition_model.transition_prob[true_states[t-1]])
    
    # Generate observations
    observations = np.zeros(T)
    for t in range(T):
        mean = observation_model.observation_matrix @ np.array([true_states[t], 0])
        observations[t] = np.random.normal(mean[0], np.sqrt(observation_model.observation_noise[0, 0]))
    
    # Perform fixed-lag smoothing
    fixed_lag_estimates = []
    
    for t in range(T):
        estimate = fixed_lag_smoother.update(observations[t])
        if estimate is not None:
            fixed_lag_estimates.append(estimate)
    
    fixed_lag_estimates = np.array(fixed_lag_estimates)
    fixed_lag_states = np.argmax(fixed_lag_estimates, axis=1)
    
    # Compare with batch smoothing
    smoother = SmoothingAlgorithm(transition_model, observation_model)
    batch_gamma, _, _ = smoother.fixed_interval_smoothing(observations)
    batch_states = np.argmax(batch_gamma, axis=1)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot observations
    plt.subplot(3, 2, 1)
    plt.plot(observations, 'b.', alpha=0.5, label='Observations')
    plt.xlabel('Time')
    plt.ylabel('Observation')
    plt.title('Observations')
    plt.legend()
    
    # Plot true states
    plt.subplot(3, 2, 2)
    plt.plot(true_states, 'b-', label='True States', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('True States')
    plt.legend()
    
    # Plot batch smoothing
    plt.subplot(3, 2, 3)
    plt.plot(true_states, 'b-', label='True States', linewidth=2)
    plt.plot(batch_states, 'r--', label='Batch Smoothed', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('Batch Smoothing')
    plt.legend()
    
    # Plot fixed-lag smoothing
    plt.subplot(3, 2, 4)
    plt.plot(true_states[lag:], 'b-', label='True States', linewidth=2)
    plt.plot(fixed_lag_states, 'g--', label='Fixed-Lag Smoothed', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title(f'Fixed-Lag Smoothing (lag={lag})')
    plt.legend()
    
    # Plot accuracy comparison
    plt.subplot(3, 2, 5)
    batch_accuracy = np.cumsum(true_states == batch_states) / np.arange(1, T + 1)
    fixed_lag_accuracy = np.cumsum(true_states[lag:] == fixed_lag_states) / np.arange(1, T - lag + 1)
    
    plt.plot(batch_accuracy, 'r-', label='Batch Smoothing')
    plt.plot(range(lag, T), fixed_lag_accuracy, 'g-', label='Fixed-Lag Smoothing')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Accuracy')
    plt.title('Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot delay comparison
    plt.subplot(3, 2, 6)
    plt.plot(range(T), [0] * T, 'b-', label='True States')
    plt.plot(range(T), [0] * T, 'r--', label='Batch Smoothing')
    plt.plot(range(lag, T), [0] * (T - lag), 'g--', label='Fixed-Lag Smoothing')
    plt.xlabel('Time')
    plt.ylabel('Delay')
    plt.title('Processing Delay')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Compute performance metrics
    batch_accuracy = np.mean(true_states == batch_states)
    fixed_lag_accuracy = np.mean(true_states[lag:] == fixed_lag_states)
    
    print(f"Batch smoothing accuracy: {batch_accuracy:.3f}")
    print(f"Fixed-lag smoothing accuracy: {fixed_lag_accuracy:.3f}")
    print(f"Fixed-lag delay: {lag} time steps")
    
    return fixed_lag_smoother, true_states, observations, batch_states, fixed_lag_states

# Run the example
if __name__ == "__main__":
    fixed_lag_smoother, true_states, observations, batch_states, fixed_lag_states = fixed_lag_smoothing_example()
```

## Rauch-Tung-Striebel Algorithm

The Rauch-Tung-Striebel algorithm is an efficient smoothing algorithm for linear Gaussian systems.

```python
class RauchTungStriebelSmoother:
    def __init__(self, transition_matrix, process_noise, observation_matrix, observation_noise):
        self.transition_matrix = transition_matrix
        self.process_noise = process_noise
        self.observation_matrix = observation_matrix
        self.observation_noise = observation_noise
    
    def smooth(self, observations, initial_mean, initial_covariance):
        """Perform Rauch-Tung-Striebel smoothing"""
        T = len(observations)
        n_states = self.transition_matrix.shape[0]
        
        # Forward pass (Kalman filter)
        filtered_means = np.zeros((T, n_states))
        filtered_covariances = np.zeros((T, n_states, n_states))
        
        # Initialize
        filtered_means[0] = initial_mean
        filtered_covariances[0] = initial_covariance
        
        # Forward pass
        for t in range(1, T):
            # Predict
            predicted_mean = self.transition_matrix @ filtered_means[t-1]
            predicted_covariance = (self.transition_matrix @ 
                                  filtered_covariances[t-1] @ 
                                  self.transition_matrix.T + 
                                  self.process_noise)
            
            # Update
            kalman_gain = (predicted_covariance @ 
                          self.observation_matrix.T @ 
                          np.linalg.inv(self.observation_matrix @ 
                                       predicted_covariance @ 
                                       self.observation_matrix.T + 
                                       self.observation_noise))
            
            filtered_means[t] = (predicted_mean + 
                               kalman_gain @ (observations[t] - 
                                            self.observation_matrix @ 
                                            predicted_mean))
            filtered_covariances[t] = ((np.eye(n_states) - 
                                      kalman_gain @ self.observation_matrix) @ 
                                     predicted_covariance)
        
        # Backward pass (smoothing)
        smoothed_means = np.zeros((T, n_states))
        smoothed_covariances = np.zeros((T, n_states, n_states))
        
        # Initialize
        smoothed_means[T-1] = filtered_means[T-1]
        smoothed_covariances[T-1] = filtered_covariances[T-1]
        
        # Backward pass
        for t in range(T-2, -1, -1):
            # Predict
            predicted_mean = self.transition_matrix @ filtered_means[t]
            predicted_covariance = (self.transition_matrix @ 
                                  filtered_covariances[t] @ 
                                  self.transition_matrix.T + 
                                  self.process_noise)
            
            # Smoothing gain
            smoothing_gain = (filtered_covariances[t] @ 
                            self.transition_matrix.T @ 
                            np.linalg.inv(predicted_covariance))
            
            # Update
            smoothed_means[t] = (filtered_means[t] + 
                               smoothing_gain @ (smoothed_means[t+1] - predicted_mean))
            smoothed_covariances[t] = (filtered_covariances[t] + 
                                     smoothing_gain @ (smoothed_covariances[t+1] - 
                                                     predicted_covariance) @ 
                                     smoothing_gain.T)
        
        return smoothed_means, smoothed_covariances, filtered_means, filtered_covariances

def rts_smoothing_example():
    """Example of Rauch-Tung-Striebel smoothing"""
    
    # System parameters
    dt = 0.1
    transition_matrix = np.array([[1, dt], [0, 1]])  # Constant velocity model
    process_noise = np.array([[0.1, 0], [0, 0.1]])
    observation_matrix = np.array([[1, 0]])  # Observe position only
    observation_noise = np.array([[1.0]])
    
    # Create RTS smoother
    rts_smoother = RauchTungStriebelSmoother(
        transition_matrix, process_noise, observation_matrix, observation_noise
    )
    
    # Generate synthetic data
    np.random.seed(42)
    T = 100
    
    # Generate true trajectory
    true_positions = np.zeros(T)
    true_velocities = np.zeros(T)
    
    # Initial conditions
    true_positions[0] = 0
    true_velocities[0] = 1
    
    # Generate trajectory
    for t in range(1, T):
        true_positions[t] = true_positions[t-1] + dt * true_velocities[t-1]
        true_velocities[t] = true_velocities[t-1] + np.random.normal(0, 0.1)
    
    # Generate observations
    observations = true_positions + np.random.normal(0, 1, T)
    
    # Perform RTS smoothing
    initial_mean = np.array([0, 1])
    initial_covariance = np.array([[1, 0], [0, 1]])
    
    smoothed_means, smoothed_covariances, filtered_means, filtered_covariances = rts_smoother.smooth(
        observations, initial_mean, initial_covariance
    )
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot observations and true trajectory
    plt.subplot(2, 3, 1)
    plt.plot(observations, 'b.', alpha=0.5, label='Observations')
    plt.plot(true_positions, 'g-', label='True Position', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Observations and True Trajectory')
    plt.legend()
    
    # Plot position estimates
    plt.subplot(2, 3, 2)
    plt.plot(true_positions, 'g-', label='True Position', linewidth=2)
    plt.plot(filtered_means[:, 0], 'r--', label='Filtered Position', linewidth=2)
    plt.plot(smoothed_means[:, 0], 'b--', label='Smoothed Position', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Position Estimation')
    plt.legend()
    
    # Plot velocity estimates
    plt.subplot(2, 3, 3)
    plt.plot(true_velocities, 'g-', label='True Velocity', linewidth=2)
    plt.plot(filtered_means[:, 1], 'r--', label='Filtered Velocity', linewidth=2)
    plt.plot(smoothed_means[:, 1], 'b--', label='Smoothed Velocity', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Velocity Estimation')
    plt.legend()
    
    # Plot position errors
    plt.subplot(2, 3, 4)
    filtered_position_errors = np.abs(true_positions - filtered_means[:, 0])
    smoothed_position_errors = np.abs(true_positions - smoothed_means[:, 0])
    
    plt.plot(filtered_position_errors, 'r-', label='Filtered Error')
    plt.plot(smoothed_position_errors, 'b-', label='Smoothed Error')
    plt.xlabel('Time')
    plt.ylabel('Position Error')
    plt.title('Position Estimation Error')
    plt.legend()
    plt.grid(True)
    
    # Plot velocity errors
    plt.subplot(2, 3, 5)
    filtered_velocity_errors = np.abs(true_velocities - filtered_means[:, 1])
    smoothed_velocity_errors = np.abs(true_velocities - smoothed_means[:, 1])
    
    plt.plot(filtered_velocity_errors, 'r-', label='Filtered Error')
    plt.plot(smoothed_velocity_errors, 'b-', label='Smoothed Error')
    plt.xlabel('Time')
    plt.ylabel('Velocity Error')
    plt.title('Velocity Estimation Error')
    plt.legend()
    plt.grid(True)
    
    # Plot uncertainty
    plt.subplot(2, 3, 6)
    filtered_position_std = np.sqrt(filtered_covariances[:, 0, 0])
    smoothed_position_std = np.sqrt(smoothed_covariances[:, 0, 0])
    
    plt.plot(filtered_position_std, 'r-', label='Filtered Uncertainty')
    plt.plot(smoothed_position_std, 'b-', label='Smoothed Uncertainty')
    plt.xlabel('Time')
    plt.ylabel('Position Standard Deviation')
    plt.title('Position Estimation Uncertainty')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Compute performance metrics
    filtered_position_rmse = np.sqrt(np.mean((true_positions - filtered_means[:, 0]) ** 2))
    smoothed_position_rmse = np.sqrt(np.mean((true_positions - smoothed_means[:, 0]) ** 2))
    filtered_velocity_rmse = np.sqrt(np.mean((true_velocities - filtered_means[:, 1]) ** 2))
    smoothed_velocity_rmse = np.sqrt(np.mean((true_velocities - smoothed_means[:, 1]) ** 2))
    
    print("RTS Smoothing Performance:")
    print(f"Position RMSE - Filtered: {filtered_position_rmse:.3f}, Smoothed: {smoothed_position_rmse:.3f}")
    print(f"Velocity RMSE - Filtered: {filtered_velocity_rmse:.3f}, Smoothed: {smoothed_velocity_rmse:.3f}")
    print(f"Improvement - Position: {((filtered_position_rmse - smoothed_position_rmse) / filtered_position_rmse * 100):.1f}%")
    print(f"Improvement - Velocity: {((filtered_velocity_rmse - smoothed_velocity_rmse) / filtered_velocity_rmse * 100):.1f}%")
    
    return rts_smoother, true_positions, true_velocities, observations, smoothed_means, filtered_means

# Run the example
if __name__ == "__main__":
    rts_smoother, true_positions, true_velocities, observations, smoothed_means, filtered_means = rts_smoothing_example()
```

## Applications

### 1. Signal Processing

```python
class SignalSmoothing:
    def __init__(self, signal_type='gaussian'):
        self.signal_type = signal_type
    
    def smooth_signal(self, signal, noise_level=0.1, smoothing_window=5):
        """Smooth a noisy signal"""
        if self.signal_type == 'gaussian':
            return self._gaussian_smoothing(signal, noise_level, smoothing_window)
        elif self.signal_type == 'kalman':
            return self._kalman_smoothing(signal, noise_level)
        else:
            raise ValueError(f"Unknown signal type: {self.signal_type}")
    
    def _gaussian_smoothing(self, signal, noise_level, window_size):
        """Gaussian smoothing"""
        # Add noise to signal
        noisy_signal = signal + np.random.normal(0, noise_level, len(signal))
        
        # Apply Gaussian smoothing
        smoothed_signal = np.zeros_like(noisy_signal)
        half_window = window_size // 2
        
        for i in range(len(signal)):
            start = max(0, i - half_window)
            end = min(len(signal), i + half_window + 1)
            
            weights = np.exp(-0.5 * ((np.arange(start, end) - i) / (window_size / 4)) ** 2)
            weights /= np.sum(weights)
            
            smoothed_signal[i] = np.sum(noisy_signal[start:end] * weights)
        
        return noisy_signal, smoothed_signal
    
    def _kalman_smoothing(self, signal, noise_level):
        """Kalman smoothing"""
        # Add noise to signal
        noisy_signal = signal + np.random.normal(0, noise_level, len(signal))
        
        # Simple Kalman filter parameters
        transition_matrix = np.array([[1]])
        process_noise = np.array([[0.01]])
        observation_matrix = np.array([[1]])
        observation_noise = np.array([[noise_level ** 2]])
        
        # Create RTS smoother
        rts_smoother = RauchTungStriebelSmoother(
            transition_matrix, process_noise, observation_matrix, observation_noise
        )
        
        # Perform smoothing
        initial_mean = np.array([signal[0]])
        initial_covariance = np.array([[1]])
        
        smoothed_means, _, _, _ = rts_smoother.smooth(
            noisy_signal.reshape(-1, 1), initial_mean, initial_covariance
        )
        
        return noisy_signal, smoothed_means.flatten()

def signal_smoothing_example():
    """Example of signal smoothing"""
    
    # Generate synthetic signal
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    
    # Create a complex signal
    signal = (np.sin(2 * np.pi * t) + 
              0.5 * np.sin(4 * np.pi * t) + 
              0.3 * np.sin(6 * np.pi * t))
    
    # Create signal smoother
    signal_smoother = SignalSmoothing(signal_type='gaussian')
    
    # Smooth signal
    noisy_signal, smoothed_signal = signal_smoother.smooth_signal(signal, noise_level=0.3, smoothing_window=11)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot original signal
    plt.subplot(2, 2, 1)
    plt.plot(t, signal, 'b-', label='Original Signal', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Original Signal')
    plt.legend()
    plt.grid(True)
    
    # Plot noisy signal
    plt.subplot(2, 2, 2)
    plt.plot(t, noisy_signal, 'r.', alpha=0.5, label='Noisy Signal')
    plt.plot(t, signal, 'b-', label='Original Signal', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Noisy Signal')
    plt.legend()
    plt.grid(True)
    
    # Plot smoothed signal
    plt.subplot(2, 2, 3)
    plt.plot(t, noisy_signal, 'r.', alpha=0.3, label='Noisy Signal')
    plt.plot(t, smoothed_signal, 'g-', label='Smoothed Signal', linewidth=2)
    plt.plot(t, signal, 'b-', label='Original Signal', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Smoothed Signal')
    plt.legend()
    plt.grid(True)
    
    # Plot error comparison
    plt.subplot(2, 2, 4)
    noisy_error = np.abs(signal - noisy_signal)
    smoothed_error = np.abs(signal - smoothed_signal)
    
    plt.plot(t, noisy_error, 'r-', label='Noisy Error', alpha=0.7)
    plt.plot(t, smoothed_error, 'g-', label='Smoothed Error', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Absolute Error')
    plt.title('Error Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Compute performance metrics
    noisy_rmse = np.sqrt(np.mean((signal - noisy_signal) ** 2))
    smoothed_rmse = np.sqrt(np.mean((signal - smoothed_signal) ** 2))
    
    print("Signal Smoothing Performance:")
    print(f"Noisy signal RMSE: {noisy_rmse:.3f}")
    print(f"Smoothed signal RMSE: {smoothed_rmse:.3f}")
    print(f"Improvement: {((noisy_rmse - smoothed_rmse) / noisy_rmse * 100):.1f}%")
    
    return signal_smoother, signal, noisy_signal, smoothed_signal

# Run the example
if __name__ == "__main__":
    signal_smoother, signal, noisy_signal, smoothed_signal = signal_smoothing_example()
```

## Key Takeaways

1. **Smoothing** provides more accurate estimates than filtering by using future observations
2. **Fixed-interval smoothing** estimates all states given all observations
3. **Fixed-lag smoothing** provides estimates with a fixed delay for real-time applications
4. **Rauch-Tung-Striebel algorithm** is efficient for linear Gaussian systems
5. **Applications** include signal processing, target tracking, and time series analysis
6. **Trade-offs** exist between accuracy, computational cost, and delay

## Exercises

1. Implement smoothing for a custom Hidden Markov Model
2. Compare different smoothing algorithms on the same dataset
3. Apply smoothing to a real-world time series dataset
4. Implement fixed-lag smoothing with variable lag
5. Analyze the effect of noise level on smoothing performance 