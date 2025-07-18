# Forward-Backward Algorithm

## Overview

The Forward-Backward algorithm is a dynamic programming algorithm for computing posterior probabilities in Hidden Markov Models (HMMs), which are a special case of Bayesian networks for sequential data. It efficiently computes the probability of being in each state at each time step given the entire observation sequence.

## Hidden Markov Models

### Definition

A Hidden Markov Model consists of:
- **Hidden states**: Unobservable states that follow a Markov process
- **Observations**: Observable outputs that depend on the hidden states
- **Transition probabilities**: P(state_t | state_{t-1})
- **Emission probabilities**: P(observation_t | state_t)
- **Initial state probabilities**: P(state_1)

### Mathematical Formulation

For an HMM with:
- N hidden states: S = {s₁, s₂, ..., s_N}
- M observations: O = {o₁, o₂, ..., o_T}
- Transition matrix A: A[i,j] = P(s_j | s_i)
- Emission matrix B: B[i,k] = P(o_k | s_i)
- Initial probabilities π: π[i] = P(s₁ = s_i)

## Forward Algorithm

The forward algorithm computes the probability of observations up to time t and being in state i at time t.

### Mathematical Definition

Forward variable: α_t(i) = P(o₁, o₂, ..., o_t, q_t = s_i | λ)

Recursion:
1. **Initialization**: α₁(i) = π[i] · B[i, o₁]
2. **Recursion**: α_t(i) = [∑ⱼ α_{t-1}(j) · A[j,i]] · B[i, o_t]
3. **Termination**: P(O | λ) = ∑ᵢ α_T(i)

### Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class HiddenMarkovModel:
    def __init__(self, n_states, n_observations):
        self.n_states = n_states
        self.n_observations = n_observations
        
        # Initialize parameters
        self.pi = np.random.dirichlet(np.ones(n_states))  # Initial probabilities
        self.A = np.random.dirichlet(np.ones(n_states), size=n_states)  # Transition matrix
        self.B = np.random.dirichlet(np.ones(n_observations), size=n_states)  # Emission matrix
    
    def set_parameters(self, pi, A, B):
        """Set HMM parameters"""
        self.pi = np.array(pi)
        self.A = np.array(A)
        self.B = np.array(B)
    
    def forward_algorithm(self, observations):
        """Compute forward probabilities"""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # Initialization
        for i in range(self.n_states):
            alpha[0, i] = self.pi[i] * self.B[i, observations[0]]
        
        # Recursion
        for t in range(1, T):
            for i in range(self.n_states):
                alpha[t, i] = 0
                for j in range(self.n_states):
                    alpha[t, i] += alpha[t-1, j] * self.A[j, i]
                alpha[t, i] *= self.B[i, observations[t]]
        
        # Termination
        likelihood = np.sum(alpha[T-1, :])
        
        return alpha, likelihood
    
    def backward_algorithm(self, observations):
        """Compute backward probabilities"""
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # Initialization
        beta[T-1, :] = 1.0
        
        # Recursion
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = 0
                for j in range(self.n_states):
                    beta[t, i] += self.A[i, j] * self.B[j, observations[t+1]] * beta[t+1, j]
        
        return beta

# Example: Weather HMM
def create_weather_hmm():
    """Create a weather HMM example"""
    hmm = HiddenMarkovModel(n_states=2, n_observations=3)
    
    # States: 0 = Sunny, 1 = Rainy
    # Observations: 0 = Walk, 1 = Shop, 2 = Clean
    
    # Initial probabilities: 60% sunny, 40% rainy
    pi = [0.6, 0.4]
    
    # Transition matrix
    # P(Sunny|Sunny) = 0.8, P(Rainy|Sunny) = 0.2
    # P(Sunny|Rainy) = 0.4, P(Rainy|Rainy) = 0.6
    A = [[0.8, 0.2],
         [0.4, 0.6]]
    
    # Emission matrix
    # P(Walk|Sunny) = 0.6, P(Shop|Sunny) = 0.3, P(Clean|Sunny) = 0.1
    # P(Walk|Rainy) = 0.1, P(Shop|Rainy) = 0.4, P(Clean|Rainy) = 0.5
    B = [[0.6, 0.3, 0.1],
         [0.1, 0.4, 0.5]]
    
    hmm.set_parameters(pi, A, B)
    return hmm

def forward_algorithm_example():
    """Example of using the forward algorithm"""
    
    # Create weather HMM
    hmm = create_weather_hmm()
    
    # Observation sequence: Walk, Shop, Clean
    observations = [0, 1, 2]
    
    # Run forward algorithm
    alpha, likelihood = hmm.forward_algorithm(observations)
    
    print("Forward Algorithm Results:")
    print(f"Observation sequence: {observations}")
    print(f"Likelihood P(O|λ) = {likelihood:.6f}")
    print("\nForward probabilities α_t(i):")
    print("Time\tState\tα_t(i)")
    print("-" * 25)
    
    for t in range(len(observations)):
        for i in range(hmm.n_states):
            state_name = "Sunny" if i == 0 else "Rainy"
            print(f"{t+1}\t{state_name}\t{alpha[t, i]:.6f}")
    
    return hmm, alpha, likelihood

# Run the example
if __name__ == "__main__":
    hmm, alpha, likelihood = forward_algorithm_example()
```

## Backward Algorithm

The backward algorithm computes the probability of observations from time t+1 to T given being in state i at time t.

### Mathematical Definition

Backward variable: β_t(i) = P(o_{t+1}, o_{t+2}, ..., o_T | q_t = s_i, λ)

Recursion:
1. **Initialization**: β_T(i) = 1
2. **Recursion**: β_t(i) = ∑ⱼ A[i,j] · B[j, o_{t+1}] · β_{t+1}(j)

### Python Implementation

```python
def backward_algorithm_example():
    """Example of using the backward algorithm"""
    
    # Create weather HMM
    hmm = create_weather_hmm()
    
    # Observation sequence: Walk, Shop, Clean
    observations = [0, 1, 2]
    
    # Run backward algorithm
    beta = hmm.backward_algorithm(observations)
    
    print("Backward Algorithm Results:")
    print(f"Observation sequence: {observations}")
    print("\nBackward probabilities β_t(i):")
    print("Time\tState\tβ_t(i)")
    print("-" * 25)
    
    for t in range(len(observations)):
        for i in range(hmm.n_states):
            state_name = "Sunny" if i == 0 else "Rainy"
            print(f"{t+1}\t{state_name}\t{beta[t, i]:.6f}")
    
    return hmm, beta

# Run the example
if __name__ == "__main__":
    hmm, beta = backward_algorithm_example()
```

## Forward-Backward Algorithm

The forward-backward algorithm combines both algorithms to compute posterior probabilities.

### Mathematical Definition

Posterior probability: γ_t(i) = P(q_t = s_i | O, λ)

Using forward and backward variables:
γ_t(i) = (α_t(i) · β_t(i)) / P(O | λ)

### Python Implementation

```python
class ForwardBackwardAlgorithm:
    def __init__(self, hmm):
        self.hmm = hmm
    
    def compute_posterior_probabilities(self, observations):
        """Compute posterior probabilities using forward-backward algorithm"""
        T = len(observations)
        
        # Run forward algorithm
        alpha, likelihood = self.hmm.forward_algorithm(observations)
        
        # Run backward algorithm
        beta = self.hmm.backward_algorithm(observations)
        
        # Compute posterior probabilities
        gamma = np.zeros((T, self.hmm.n_states))
        
        for t in range(T):
            for i in range(self.hmm.n_states):
                gamma[t, i] = (alpha[t, i] * beta[t, i]) / likelihood
        
        return gamma, alpha, beta, likelihood
    
    def compute_state_sequence_probability(self, observations, state_sequence):
        """Compute probability of a specific state sequence"""
        T = len(observations)
        
        # Compute joint probability P(O, Q | λ)
        prob = self.hmm.pi[state_sequence[0]] * self.hmm.B[state_sequence[0], observations[0]]
        
        for t in range(1, T):
            prob *= (self.hmm.A[state_sequence[t-1], state_sequence[t]] * 
                    self.hmm.B[state_sequence[t], observations[t]])
        
        return prob
    
    def find_most_likely_states(self, observations):
        """Find the most likely state at each time step"""
        gamma, _, _, _ = self.compute_posterior_probabilities(observations)
        
        most_likely_states = np.argmax(gamma, axis=1)
        return most_likely_states, gamma

def forward_backward_example():
    """Example of using the forward-backward algorithm"""
    
    # Create weather HMM
    hmm = create_weather_hmm()
    
    # Create forward-backward algorithm
    fb = ForwardBackwardAlgorithm(hmm)
    
    # Observation sequence: Walk, Shop, Clean
    observations = [0, 1, 2]
    
    # Compute posterior probabilities
    gamma, alpha, beta, likelihood = fb.compute_posterior_probabilities(observations)
    
    print("Forward-Backward Algorithm Results:")
    print(f"Observation sequence: {observations}")
    print(f"Likelihood P(O|λ) = {likelihood:.6f}")
    print("\nPosterior probabilities γ_t(i):")
    print("Time\tState\tγ_t(i)")
    print("-" * 25)
    
    for t in range(len(observations)):
        for i in range(hmm.n_states):
            state_name = "Sunny" if i == 0 else "Rainy"
            print(f"{t+1}\t{state_name}\t{gamma[t, i]:.6f}")
    
    # Find most likely states
    most_likely_states, _ = fb.find_most_likely_states(observations)
    
    print(f"\nMost likely state sequence: {most_likely_states}")
    state_names = ["Sunny", "Rainy"]
    state_sequence = [state_names[s] for s in most_likely_states]
    print(f"State sequence: {state_sequence}")
    
    return fb, gamma, most_likely_states

# Run the example
if __name__ == "__main__":
    fb, gamma, states = forward_backward_example()
```

## Smoothing

Smoothing uses the forward-backward algorithm to estimate past states using future observations.

### Mathematical Definition

Smoothing probability: P(q_t = s_i | o₁, o₂, ..., o_T)

This is exactly what the forward-backward algorithm computes: γ_t(i)

### Python Implementation

```python
class SmoothingAlgorithm:
    def __init__(self, hmm):
        self.hmm = hmm
        self.fb = ForwardBackwardAlgorithm(hmm)
    
    def smooth_states(self, observations):
        """Perform smoothing to estimate past states"""
        gamma, _, _, _ = self.fb.compute_posterior_probabilities(observations)
        
        # Find most likely state at each time step
        smoothed_states = np.argmax(gamma, axis=1)
        
        return smoothed_states, gamma
    
    def compute_smoothing_accuracy(self, observations, true_states):
        """Compute accuracy of smoothing compared to true states"""
        smoothed_states, _ = self.smooth_states(observations)
        
        accuracy = np.mean(smoothed_states == true_states)
        return accuracy, smoothed_states
    
    def plot_smoothing_results(self, observations, true_states=None):
        """Plot smoothing results"""
        smoothed_states, gamma = self.smooth_states(observations)
        
        T = len(observations)
        time_steps = range(1, T + 1)
        
        plt.figure(figsize=(12, 8))
        
        # Plot observations
        plt.subplot(3, 1, 1)
        observation_names = ["Walk", "Shop", "Clean"]
        obs_names = [observation_names[obs] for obs in observations]
        plt.plot(time_steps, observations, 'bo-', label='Observations')
        plt.yticks(range(3), observation_names)
        plt.title('Observations')
        plt.xlabel('Time')
        plt.ylabel('Observation')
        plt.legend()
        
        # Plot posterior probabilities
        plt.subplot(3, 1, 2)
        plt.plot(time_steps, gamma[:, 0], 'r-', label='P(Sunny)')
        plt.plot(time_steps, gamma[:, 1], 'b-', label='P(Rainy)')
        plt.title('Posterior Probabilities')
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.legend()
        
        # Plot smoothed states
        plt.subplot(3, 1, 3)
        state_names = ["Sunny", "Rainy"]
        smoothed_names = [state_names[s] for s in smoothed_states]
        plt.plot(time_steps, smoothed_states, 'go-', label='Smoothed States')
        plt.yticks(range(2), state_names)
        plt.title('Smoothed States')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.legend()
        
        if true_states is not None:
            plt.plot(time_steps, true_states, 'ro-', label='True States', alpha=0.7)
            plt.legend()
        
        plt.tight_layout()
        plt.show()

def smoothing_example():
    """Example of smoothing with HMM"""
    
    # Create weather HMM
    hmm = create_weather_hmm()
    
    # Create smoothing algorithm
    smoother = SmoothingAlgorithm(hmm)
    
    # Generate synthetic data
    np.random.seed(42)
    T = 10
    
    # Generate true state sequence
    true_states = []
    current_state = np.random.choice([0, 1], p=hmm.pi)
    true_states.append(current_state)
    
    for t in range(1, T):
        current_state = np.random.choice([0, 1], p=hmm.A[current_state])
        true_states.append(current_state)
    
    # Generate observations
    observations = []
    for state in true_states:
        obs = np.random.choice([0, 1, 2], p=hmm.B[state])
        observations.append(obs)
    
    # Perform smoothing
    smoothed_states, gamma = smoother.smooth_states(observations)
    
    # Compute accuracy
    accuracy, _ = smoother.compute_smoothing_accuracy(observations, true_states)
    
    print("Smoothing Results:")
    print(f"True states: {true_states}")
    print(f"Observations: {observations}")
    print(f"Smoothed states: {smoothed_states}")
    print(f"Smoothing accuracy: {accuracy:.3f}")
    
    # Plot results
    smoother.plot_smoothing_results(observations, true_states)
    
    return smoother, true_states, observations, smoothed_states

# Run the example
if __name__ == "__main__":
    smoother, true_states, observations, smoothed_states = smoothing_example()
```

## Applications

### 1. Speech Recognition

```python
class SpeechRecognitionHMM:
    def __init__(self, n_phonemes, n_features):
        self.n_phonemes = n_phonemes
        self.n_features = n_features
        self.hmm = HiddenMarkovModel(n_phonemes, n_features)
    
    def recognize_speech(self, feature_sequence):
        """Recognize speech using forward-backward algorithm"""
        fb = ForwardBackwardAlgorithm(self.hmm)
        
        # Find most likely phoneme sequence
        most_likely_states, gamma = fb.find_most_likely_states(feature_sequence)
        
        return most_likely_states, gamma
    
    def compute_phoneme_probabilities(self, feature_sequence):
        """Compute probability of each phoneme at each time step"""
        fb = ForwardBackwardAlgorithm(self.hmm)
        gamma, _, _, _ = fb.compute_posterior_probabilities(feature_sequence)
        
        return gamma

# Example: Speech Recognition
def speech_recognition_example():
    """Example of speech recognition using HMM"""
    
    # Create speech recognition HMM
    n_phonemes = 5  # Example: a, e, i, o, u
    n_features = 10  # MFCC features
    
    speech_hmm = SpeechRecognitionHMM(n_phonemes, n_features)
    
    # Generate synthetic feature sequence
    np.random.seed(42)
    feature_sequence = np.random.randint(0, n_features, size=20)
    
    # Recognize speech
    phoneme_sequence, gamma = speech_hmm.recognize_speech(feature_sequence)
    
    print("Speech Recognition Results:")
    print(f"Feature sequence: {feature_sequence}")
    print(f"Recognized phonemes: {phoneme_sequence}")
    
    return speech_hmm, feature_sequence, phoneme_sequence

# Run the example
if __name__ == "__main__":
    speech_hmm, features, phonemes = speech_recognition_example()
```

### 2. Bioinformatics

```python
class BioinformaticsHMM:
    def __init__(self, n_states, n_symbols):
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.hmm = HiddenMarkovModel(n_states, n_symbols)
    
    def analyze_sequence(self, sequence):
        """Analyze biological sequence using HMM"""
        fb = ForwardBackwardAlgorithm(self.hmm)
        
        # Compute posterior probabilities
        gamma, _, _, likelihood = fb.compute_posterior_probabilities(sequence)
        
        # Find most likely state sequence
        most_likely_states, _ = fb.find_most_likely_states(sequence)
        
        return most_likely_states, gamma, likelihood
    
    def identify_motifs(self, sequence):
        """Identify motifs in biological sequence"""
        most_likely_states, gamma, _ = self.analyze_sequence(sequence)
        
        # Find regions with high probability of being in motif state
        motif_threshold = 0.7
        motif_regions = []
        
        for t in range(len(sequence)):
            if gamma[t, 1] > motif_threshold:  # Assuming state 1 is motif state
                motif_regions.append(t)
        
        return motif_regions

# Example: Bioinformatics
def bioinformatics_example():
    """Example of bioinformatics analysis using HMM"""
    
    # Create bioinformatics HMM
    n_states = 2  # Background, Motif
    n_symbols = 4  # A, C, G, T
    
    bio_hmm = BioinformaticsHMM(n_states, n_symbols)
    
    # Generate synthetic DNA sequence
    np.random.seed(42)
    sequence = np.random.randint(0, n_symbols, size=50)
    
    # Analyze sequence
    states, gamma, likelihood = bio_hmm.analyze_sequence(sequence)
    
    # Identify motifs
    motif_regions = bio_hmm.identify_motifs(sequence)
    
    print("Bioinformatics Analysis Results:")
    print(f"Sequence: {sequence}")
    print(f"States: {states}")
    print(f"Motif regions: {motif_regions}")
    print(f"Likelihood: {likelihood:.6f}")
    
    return bio_hmm, sequence, states, motif_regions

# Run the example
if __name__ == "__main__":
    bio_hmm, sequence, states, motifs = bioinformatics_example()
```

## Key Takeaways

1. **Forward Algorithm**: Computes P(observations up to t, state at t)
2. **Backward Algorithm**: Computes P(observations from t+1 to T | state at t)
3. **Forward-Backward Algorithm**: Combines both to compute posterior probabilities
4. **Smoothing**: Uses future observations to improve estimates of past states
5. **Applications**: Speech recognition, bioinformatics, time series analysis
6. **Efficiency**: O(TN²) time complexity for T observations and N states

## Exercises

1. Implement the forward-backward algorithm for a custom HMM
2. Compare smoothing with filtering (using only past observations)
3. Apply the algorithm to a real-world time series dataset
4. Implement the Viterbi algorithm for finding the most likely state sequence
5. Build an HMM for a specific domain (e.g., weather prediction, stock prices) 