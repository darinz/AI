# Gibbs Sampling in Markov Networks

## Introduction

Gibbs sampling is a Markov Chain Monte Carlo (MCMC) method for sampling from complex probability distributions, particularly those defined by Markov networks. It's a powerful technique for approximate inference when exact computation is intractable.

## What is Gibbs Sampling?

### Definition

Gibbs sampling is an MCMC algorithm that generates samples from a multivariate probability distribution by sampling each variable in turn, conditional on the current values of all other variables.

### Key Idea

Instead of sampling from the joint distribution directly (which is often intractable), Gibbs sampling:
1. Starts with an initial assignment to all variables
2. Iteratively samples each variable from its conditional distribution
3. Uses the Markov property to compute conditional distributions efficiently
4. After sufficient iterations, samples approximate the target distribution

## Mathematical Foundation

### Conditional Distributions

In a Markov network, the conditional distribution of a variable Xᵢ given all other variables is:

P(Xᵢ | X_{-i}) = P(Xᵢ | N(Xᵢ))

Where N(Xᵢ) are the neighbors of Xᵢ in the graph.

### Detailed Balance

Gibbs sampling satisfies the detailed balance condition:

P(x) P(x' | x) = P(x') P(x | x')

This ensures that the Markov chain converges to the target distribution P(x).

### Convergence

Under certain conditions (ergodicity), the chain converges to the target distribution regardless of the initial state. The rate of convergence depends on the structure of the graph and the strength of dependencies.

## Python Implementation

### Basic Gibbs Sampler

```python
import numpy as np
from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass
import random
from collections import defaultdict

@dataclass
class GibbsSampler:
    """Gibbs sampler for Markov networks."""
    
    def __init__(self, markov_network, burn_in: int = 1000, thinning: int = 10):
        self.mn = markov_network
        self.burn_in = burn_in
        self.thinning = thinning
    
    def sample(self, num_samples: int, initial_state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate samples using Gibbs sampling."""
        # Initialize state
        if initial_state is None:
            current_state = self._generate_random_initial_state()
        else:
            current_state = initial_state.copy()
        
        samples = []
        iteration = 0
        
        while len(samples) < num_samples:
            # Update each variable in turn
            for variable in self.mn.variables:
                # Compute conditional distribution
                conditional_dist = self._compute_conditional_distribution(variable, current_state)
                
                # Sample new value
                new_value = self._sample_from_distribution(conditional_dist)
                current_state[variable] = new_value
            
            iteration += 1
            
            # Skip burn-in period
            if iteration <= self.burn_in:
                continue
            
            # Apply thinning
            if (iteration - self.burn_in) % self.thinning == 0:
                samples.append(current_state.copy())
        
        return samples
    
    def _generate_random_initial_state(self) -> Dict[str, Any]:
        """Generate a random initial state."""
        state = {}
        for variable in self.mn.variables:
            state[variable] = random.choice(self.mn.domains[variable])
        return state
    
    def _compute_conditional_distribution(self, variable: str, current_state: Dict[str, Any]) -> Dict[Any, float]:
        """Compute the conditional distribution of a variable given current state."""
        # Get neighbors of the variable
        neighbors = set(self.mn.graph.neighbors(variable))
        
        # Compute unnormalized probabilities for each possible value
        unnormalized_probs = {}
        
        for value in self.mn.domains[variable]:
            # Create test state with this value
            test_state = current_state.copy()
            test_state[variable] = value
            
            # Compute probability (only need to consider potentials involving this variable)
            prob = 1.0
            for potential in self.mn.potentials:
                if variable in potential.variables:
                    prob *= potential.get_value(test_state)
            
            unnormalized_probs[value] = prob
        
        # Normalize
        total = sum(unnormalized_probs.values())
        if total == 0:
            # If all probabilities are zero, use uniform distribution
            num_values = len(self.mn.domains[variable])
            return {value: 1.0 / num_values for value in self.mn.domains[variable]}
        
        return {value: prob / total for value, prob in unnormalized_probs.items()}
    
    def _sample_from_distribution(self, distribution: Dict[Any, float]) -> Any:
        """Sample from a discrete distribution."""
        values = list(distribution.keys())
        probabilities = list(distribution.values())
        
        # Use numpy's multinomial sampling
        return np.random.choice(values, p=probabilities)

# Example: Gibbs sampling for a simple Markov network
def create_ising_model():
    """Create a simple Ising model (2D grid Markov network)."""
    from typing import List, Dict, Set, Tuple, Any
    
    class IsingModel:
        """Simple Ising model implementation."""
        
        def __init__(self, size: int, beta: float = 1.0):
            self.size = size
            self.beta = beta  # Inverse temperature
            self.variables = [f"X_{i}_{j}" for i in range(size) for j in range(size)]
            self.domains = {var: [-1, 1] for var in self.variables}
            self.graph = nx.Grid2dGraph(size, size)
            
            # Rename nodes to match our variable names
            mapping = {(i, j): f"X_{i}_{j}" for i in range(size) for j in range(size)}
            self.graph = nx.relabel_nodes(self.graph, mapping)
        
        def get_neighbors(self, variable: str) -> List[str]:
            """Get neighbors of a variable in the grid."""
            return list(self.graph.neighbors(variable))
        
        def compute_energy(self, state: Dict[str, int]) -> float:
            """Compute the energy of a state."""
            energy = 0.0
            
            # Sum over all edges
            for edge in self.graph.edges():
                var1, var2 = edge
                energy -= self.beta * state[var1] * state[var2]
            
            return energy
        
        def compute_conditional_probability(self, variable: str, state: Dict[str, int]) -> Dict[int, float]:
            """Compute conditional probability for a variable."""
            neighbors = self.get_neighbors(variable)
            
            # Compute local field
            local_field = sum(state[neighbor] for neighbor in neighbors)
            
            # Compute probabilities
            prob_plus = np.exp(self.beta * local_field)
            prob_minus = np.exp(-self.beta * local_field)
            
            total = prob_plus + prob_minus
            
            return {
                1: prob_plus / total,
                -1: prob_minus / total
            }

def test_gibbs_sampling():
    """Test Gibbs sampling on a simple Markov network."""
    # Create a simple Markov network
    variables = ['A', 'B', 'C']
    domains = {
        'A': [0, 1],
        'B': [0, 1],
        'C': [0, 1]
    }
    
    # Create Markov network (same as in overview)
    mn = MarkovNetwork(variables, domains)
    
    # Add potential functions
    ab_potential = PotentialFunction(
        variables=['A', 'B'],
        values={(0, 0): 2.0, (0, 1): 0.5, (1, 0): 0.5, (1, 1): 2.0}
    )
    
    bc_potential = PotentialFunction(
        variables=['B', 'C'],
        values={(0, 0): 2.0, (0, 1): 0.5, (1, 0): 0.5, (1, 1): 2.0}
    )
    
    ac_potential = PotentialFunction(
        variables=['A', 'C'],
        values={(0, 0): 0.5, (0, 1): 2.0, (1, 0): 2.0, (1, 1): 0.5}
    )
    
    mn.add_potential(ab_potential)
    mn.add_potential(bc_potential)
    mn.add_potential(ac_potential)
    
    # Create Gibbs sampler
    sampler = GibbsSampler(mn, burn_in=100, thinning=5)
    
    # Generate samples
    samples = sampler.sample(num_samples=100)
    
    # Analyze results
    print(f"Generated {len(samples)} samples")
    
    # Count frequency of each assignment
    assignment_counts = defaultdict(int)
    for sample in samples:
        key = tuple(sorted(sample.items()))
        assignment_counts[key] += 1
    
    print("\nSample frequencies:")
    for assignment, count in sorted(assignment_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{dict(assignment)}: {count}/{len(samples)} = {count/len(samples):.3f}")
    
    return mn, sampler, samples

if __name__ == "__main__":
    test_gibbs_sampling()
```

## Advanced Gibbs Sampling Techniques

### Block Gibbs Sampling

Instead of sampling one variable at a time, block Gibbs sampling updates groups of variables simultaneously.

```python
class BlockGibbsSampler(GibbsSampler):
    """Block Gibbs sampler for Markov networks."""
    
    def __init__(self, markov_network, blocks: List[List[str]], burn_in: int = 1000, thinning: int = 10):
        super().__init__(markov_network, burn_in, thinning)
        self.blocks = blocks
    
    def sample(self, num_samples: int, initial_state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate samples using block Gibbs sampling."""
        if initial_state is None:
            current_state = self._generate_random_initial_state()
        else:
            current_state = initial_state.copy()
        
        samples = []
        iteration = 0
        
        while len(samples) < num_samples:
            # Update each block in turn
            for block in self.blocks:
                # Compute joint conditional distribution for the block
                conditional_dist = self._compute_block_conditional_distribution(block, current_state)
                
                # Sample new values for the entire block
                new_values = self._sample_from_block_distribution(conditional_dist)
                for var, value in zip(block, new_values):
                    current_state[var] = value
            
            iteration += 1
            
            if iteration <= self.burn_in:
                continue
            
            if (iteration - self.burn_in) % self.thinning == 0:
                samples.append(current_state.copy())
        
        return samples
    
    def _compute_block_conditional_distribution(self, block: List[str], current_state: Dict[str, Any]) -> Dict[Tuple, float]:
        """Compute conditional distribution for a block of variables."""
        # This is a simplified implementation
        # In practice, this would be more complex for large blocks
        unnormalized_probs = {}
        
        # Generate all possible assignments for the block
        block_assignments = self._generate_block_assignments(block)
        
        for assignment in block_assignments:
            # Create test state
            test_state = current_state.copy()
            for var, value in zip(block, assignment):
                test_state[var] = value
            
            # Compute probability
            prob = 1.0
            for potential in self.mn.potentials:
                if any(var in potential.variables for var in block):
                    prob *= potential.get_value(test_state)
            
            unnormalized_probs[assignment] = prob
        
        # Normalize
        total = sum(unnormalized_probs.values())
        if total == 0:
            # Uniform distribution
            num_assignments = len(block_assignments)
            return {assignment: 1.0 / num_assignments for assignment in block_assignments}
        
        return {assignment: prob / total for assignment, prob in unnormalized_probs.items()}
    
    def _generate_block_assignments(self, block: List[str]) -> List[Tuple]:
        """Generate all possible assignments for a block."""
        if not block:
            return [()]
        
        var = block[0]
        remaining = block[1:]
        
        assignments = []
        for value in self.mn.domains[var]:
            sub_assignments = self._generate_block_assignments(remaining)
            for sub_assignment in sub_assignments:
                assignments.append((value,) + sub_assignment)
        
        return assignments
    
    def _sample_from_block_distribution(self, distribution: Dict[Tuple, float]) -> Tuple:
        """Sample from a block distribution."""
        assignments = list(distribution.keys())
        probabilities = list(distribution.values())
        
        return np.random.choice(assignments, p=probabilities)
```

### Adaptive Gibbs Sampling

Adaptive Gibbs sampling adjusts the sampling strategy based on the current state of the chain.

```python
class AdaptiveGibbsSampler(GibbsSampler):
    """Adaptive Gibbs sampler that adjusts sampling strategy."""
    
    def __init__(self, markov_network, burn_in: int = 1000, thinning: int = 10):
        super().__init__(markov_network, burn_in, thinning)
        self.acceptance_rates = defaultdict(list)
    
    def sample(self, num_samples: int, initial_state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate samples using adaptive Gibbs sampling."""
        if initial_state is None:
            current_state = self._generate_random_initial_state()
        else:
            current_state = initial_state.copy()
        
        samples = []
        iteration = 0
        
        while len(samples) < num_samples:
            # Determine variable ordering based on acceptance rates
            variable_order = self._get_adaptive_variable_order()
            
            for variable in variable_order:
                # Compute conditional distribution
                conditional_dist = self._compute_conditional_distribution(variable, current_state)
                
                # Sample new value
                new_value = self._sample_from_distribution(conditional_dist)
                
                # Track acceptance rate
                old_value = current_state[variable]
                if new_value != old_value:
                    self.acceptance_rates[variable].append(1)
                else:
                    self.acceptance_rates[variable].append(0)
                
                current_state[variable] = new_value
            
            iteration += 1
            
            if iteration <= self.burn_in:
                continue
            
            if (iteration - self.burn_in) % self.thinning == 0:
                samples.append(current_state.copy())
        
        return samples
    
    def _get_adaptive_variable_order(self) -> List[str]:
        """Get variable ordering based on acceptance rates."""
        if not any(self.acceptance_rates.values()):
            return self.mn.variables
        
        # Order variables by recent acceptance rate (most active first)
        recent_rates = {}
        window_size = 100
        
        for variable in self.mn.variables:
            recent = self.acceptance_rates[variable][-window_size:]
            if recent:
                recent_rates[variable] = np.mean(recent)
            else:
                recent_rates[variable] = 0.0
        
        # Sort by acceptance rate (descending)
        return sorted(self.mn.variables, key=lambda v: recent_rates[v], reverse=True)
```

## Convergence Diagnostics

### Monitoring Convergence

```python
class ConvergenceDiagnostics:
    """Tools for monitoring Gibbs sampling convergence."""
    
    def __init__(self):
        self.trace_plots = defaultdict(list)
        self.autocorrelations = defaultdict(list)
    
    def add_sample(self, sample: Dict[str, Any], iteration: int):
        """Add a sample for monitoring."""
        for variable, value in sample.items():
            self.trace_plots[variable].append((iteration, value))
    
    def compute_autocorrelation(self, variable: str, max_lag: int = 50) -> List[float]:
        """Compute autocorrelation for a variable."""
        values = [v for _, v in self.trace_plots[variable]]
        
        if len(values) < max_lag:
            return []
        
        autocorr = []
        for lag in range(1, max_lag + 1):
            if lag >= len(values):
                break
            
            # Compute autocorrelation at this lag
            numerator = 0.0
            denominator = 0.0
            mean_val = np.mean(values)
            
            for i in range(len(values) - lag):
                numerator += (values[i] - mean_val) * (values[i + lag] - mean_val)
                denominator += (values[i] - mean_val) ** 2
            
            if denominator > 0:
                autocorr.append(numerator / denominator)
            else:
                autocorr.append(0.0)
        
        return autocorr
    
    def plot_trace(self, variable: str):
        """Plot trace plot for a variable."""
        import matplotlib.pyplot as plt
        
        iterations, values = zip(*self.trace_plots[variable])
        
        plt.figure(figsize=(12, 4))
        plt.plot(iterations, values, alpha=0.7)
        plt.title(f"Trace Plot for {variable}")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_autocorrelation(self, variable: str):
        """Plot autocorrelation function for a variable."""
        import matplotlib.pyplot as plt
        
        autocorr = self.compute_autocorrelation(variable)
        lags = list(range(1, len(autocorr) + 1))
        
        plt.figure(figsize=(8, 4))
        plt.bar(lags, autocorr, alpha=0.7)
        plt.title(f"Autocorrelation Function for {variable}")
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.grid(True, alpha=0.3)
        plt.show()
```

## Applications

### Posterior Inference

Gibbs sampling is commonly used for computing posterior distributions in Bayesian models.

### Expectation Maximization

In the E-step of EM algorithms, Gibbs sampling can be used to compute expected sufficient statistics.

### Model Learning

Gibbs sampling can be used for parameter estimation in graphical models.

### Decision Making

Gibbs sampling enables computation of expected utilities under uncertainty.

## Summary

Gibbs sampling is a powerful technique for approximate inference in Markov networks:

1. **Simplicity**: Easy to implement and understand
2. **Generality**: Applicable to a wide range of models
3. **Efficiency**: Uses local structure for computational efficiency
4. **Convergence**: Guaranteed to converge under mild conditions

Key considerations:
- **Burn-in Period**: Initial samples should be discarded
- **Thinning**: Sample every k-th iteration to reduce autocorrelation
- **Convergence Monitoring**: Use diagnostics to assess convergence
- **Computational Cost**: Can be expensive for large models

Understanding Gibbs sampling is essential for:
- Performing approximate inference in complex models
- Learning parameters in graphical models
- Making decisions under uncertainty
- Building probabilistic AI systems 