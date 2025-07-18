# Supervised Learning in Bayesian Networks

## Overview

Supervised learning in Bayesian networks involves learning the parameters and structure of the network from labeled training data. This includes parameter learning (estimating CPTs) and structure learning (discovering the network topology).

## Parameter Learning

### Maximum Likelihood Estimation (MLE)

MLE finds the parameters that maximize the likelihood of the observed data.

#### Mathematical Foundation

For a Bayesian network with parameters θ, the likelihood is:
$$L(θ) = P(D | θ) = \prod_{i=1}^{n} P(x_i | θ)$$

The MLE estimate is:
$$\hat{θ} = \arg\max_θ L(θ)$$

#### Python Implementation

```python
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

class BayesianNetworkLearner:
    def __init__(self, structure=None):
        self.structure = structure
        self.cpts = {}
        self.learned_parameters = {}
    
    def learn_parameters_mle(self, data):
        """Learn parameters using Maximum Likelihood Estimation"""
        if self.structure is None:
            raise ValueError("Network structure must be provided for parameter learning")
        
        for node in self.structure.nodes:
            parents = self.structure.get_parents(node)
            self._learn_node_parameters_mle(node, parents, data)
    
    def _learn_node_parameters_mle(self, node, parents, data):
        """Learn parameters for a single node using MLE"""
        if not parents:
            # Root node - learn marginal distribution
            value_counts = data[node].value_counts()
            total = len(data)
            
            cpt = {}
            for value, count in value_counts.items():
                cpt[value] = count / total
            
            self.cpts[node] = cpt
        else:
            # Child node - learn conditional distribution
            cpt = {}
            
            # Get all unique parent configurations
            parent_configs = data[parents].drop_duplicates().values
            
            for parent_config in parent_configs:
                # Filter data for this parent configuration
                mask = True
                for parent, value in zip(parents, parent_config):
                    mask &= (data[parent] == value)
                
                filtered_data = data[mask]
                
                if len(filtered_data) > 0:
                    # Count values of the target variable
                    value_counts = filtered_data[node].value_counts()
                    total = len(filtered_data)
                    
                    cpt[parent_config] = {}
                    for value, count in value_counts.items():
                        cpt[parent_config][value] = count / total
                else:
                    # No data for this configuration - use uniform distribution
                    unique_values = data[node].unique()
                    cpt[parent_config] = {value: 1.0/len(unique_values) for value in unique_values}
            
            self.cpts[node] = cpt
    
    def compute_log_likelihood(self, data):
        """Compute log-likelihood of data given learned parameters"""
        log_likelihood = 0.0
        
        for _, row in data.iterrows():
            for node in self.structure.nodes:
                parents = self.structure.get_parents(node)
                
                if not parents:
                    # Root node
                    prob = self.cpts[node].get(row[node], 0.0)
                else:
                    # Child node
                    parent_values = tuple(row[parent] for parent in parents)
                    node_value = row[node]
                    
                    if parent_values in self.cpts[node]:
                        prob = self.cpts[node][parent_values].get(node_value, 0.0)
                    else:
                        prob = 0.0
                
                if prob > 0:
                    log_likelihood += np.log(prob)
                else:
                    log_likelihood = -np.inf
                    break
        
        return log_likelihood
    
    def predict(self, evidence):
        """Predict probability distribution for unobserved variables"""
        # Simple prediction using learned parameters
        predictions = {}
        
        for node in self.structure.nodes:
            if node not in evidence:
                parents = self.structure.get_parents(node)
                
                if not parents:
                    # Root node
                    predictions[node] = self.cpts[node]
                else:
                    # Child node
                    parent_values = tuple(evidence[parent] for parent in parents)
                    if parent_values in self.cpts[node]:
                        predictions[node] = self.cpts[node][parent_values]
                    else:
                        # Use uniform distribution if parent configuration not seen
                        unique_values = list(self.cpts[node].values())[0].keys()
                        predictions[node] = {value: 1.0/len(unique_values) for value in unique_values}
        
        return predictions

# Example: Weather Network Parameter Learning
def weather_parameter_learning_example():
    """Example of learning parameters for the weather Bayesian network"""
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # True parameters
    rain_prob = 0.2
    sprinkler_prob_no_rain = 0.4
    sprinkler_prob_rain = 0.01
    wetgrass_prob_rain_sprinkler = 0.99
    wetgrass_prob_rain_no_sprinkler = 0.9
    wetgrass_prob_no_rain_sprinkler = 0.9
    wetgrass_prob_no_rain_no_sprinkler = 0.0
    
    # Generate data
    data = []
    for _ in range(n_samples):
        # Generate Rain
        rain = np.random.choice([True, False], p=[rain_prob, 1-rain_prob])
        
        # Generate Sprinkler
        if rain:
            sprinkler = np.random.choice([True, False], p=[sprinkler_prob_rain, 1-sprinkler_prob_rain])
        else:
            sprinkler = np.random.choice([True, False], p=[sprinkler_prob_no_rain, 1-sprinkler_prob_no_rain])
        
        # Generate WetGrass
        if rain and sprinkler:
            wetgrass = np.random.choice([True, False], p=[wetgrass_prob_rain_sprinkler, 1-wetgrass_prob_rain_sprinkler])
        elif rain and not sprinkler:
            wetgrass = np.random.choice([True, False], p=[wetgrass_prob_rain_no_sprinkler, 1-wetgrass_prob_rain_no_sprinkler])
        elif not rain and sprinkler:
            wetgrass = np.random.choice([True, False], p=[wetgrass_prob_no_rain_sprinkler, 1-wetgrass_prob_no_rain_sprinkler])
        else:
            wetgrass = np.random.choice([True, False], p=[wetgrass_prob_no_rain_no_sprinkler, 1-wetgrass_prob_no_rain_no_sprinkler])
        
        data.append({'Rain': rain, 'Sprinkler': sprinkler, 'WetGrass': wetgrass})
    
    data_df = pd.DataFrame(data)
    
    # Create network structure
    from bayesian_network_definitions import BayesianNetworkStructure
    structure = BayesianNetworkStructure()
    structure.add_node("Rain")
    structure.add_node("Sprinkler")
    structure.add_node("WetGrass")
    structure.add_edge("Rain", "Sprinkler")
    structure.add_edge("Rain", "WetGrass")
    structure.add_edge("Sprinkler", "WetGrass")
    
    # Learn parameters
    learner = BayesianNetworkLearner(structure)
    learner.learn_parameters_mle(data_df)
    
    # Print learned parameters
    print("Learned Parameters (MLE):")
    print("=" * 40)
    
    for node, cpt in learner.cpts.items():
        print(f"\n{node}:")
        for parent_config, probs in cpt.items():
            if parent_config == ():
                print(f"  P({node}) = {probs}")
            else:
                print(f"  P({node} | {list(structure.get_parents(node))}={parent_config}) = {probs}")
    
    # Compare with true parameters
    print("\nTrue vs Learned Parameters:")
    print("=" * 40)
    
    print(f"P(Rain=True): True={rain_prob:.3f}, Learned={learner.cpts['Rain'][True]:.3f}")
    print(f"P(Sprinkler=True|Rain=False): True={sprinkler_prob_no_rain:.3f}, Learned={learner.cpts['Sprinkler'][(False,)][True]:.3f}")
    print(f"P(Sprinkler=True|Rain=True): True={sprinkler_prob_rain:.3f}, Learned={learner.cpts['Sprinkler'][(True,)][True]:.3f}")
    
    # Compute log-likelihood
    log_likelihood = learner.compute_log_likelihood(data_df)
    print(f"\nLog-likelihood: {log_likelihood:.2f}")
    
    return learner, data_df

# Run the example
if __name__ == "__main__":
    learner, data = weather_parameter_learning_example()
```

### Bayesian Learning

Bayesian learning incorporates prior knowledge and provides uncertainty estimates for parameters.

```python
class BayesianParameterLearner:
    def __init__(self, structure, priors=None):
        self.structure = structure
        self.priors = priors or {}
        self.posterior_parameters = {}
    
    def set_prior(self, node, prior_type='dirichlet', **kwargs):
        """Set prior for a node"""
        if prior_type == 'dirichlet':
            # Dirichlet prior for categorical distributions
            alpha = kwargs.get('alpha', 1.0)
            n_values = kwargs.get('n_values', 2)
            self.priors[node] = {'type': 'dirichlet', 'alpha': alpha, 'n_values': n_values}
        elif prior_type == 'beta':
            # Beta prior for binary variables
            alpha = kwargs.get('alpha', 1.0)
            beta = kwargs.get('beta', 1.0)
            self.priors[node] = {'type': 'beta', 'alpha': alpha, 'beta': beta}
    
    def learn_parameters_bayesian(self, data):
        """Learn parameters using Bayesian approach"""
        for node in self.structure.nodes:
            parents = self.structure.get_parents(node)
            self._learn_node_parameters_bayesian(node, parents, data)
    
    def _learn_node_parameters_bayesian(self, node, parents, data):
        """Learn parameters for a single node using Bayesian approach"""
        if not parents:
            # Root node
            if node in self.priors and self.priors[node]['type'] == 'beta':
                # Beta prior for binary variable
                alpha_prior = self.priors[node]['alpha']
                beta_prior = self.priors[node]['beta']
                
                # Count observations
                n_true = (data[node] == True).sum()
                n_false = (data[node] == False).sum()
                
                # Posterior parameters
                alpha_post = alpha_prior + n_true
                beta_post = beta_prior + n_false
                
                # Posterior mean
                prob_true = alpha_post / (alpha_post + beta_post)
                prob_false = beta_post / (alpha_post + beta_post)
                
                self.posterior_parameters[node] = {
                    'type': 'beta',
                    'alpha': alpha_post,
                    'beta': beta_post,
                    'mean': {True: prob_true, False: prob_false}
                }
            else:
                # Dirichlet prior
                alpha = self.priors.get(node, {}).get('alpha', 1.0)
                value_counts = data[node].value_counts()
                
                posterior_params = {}
                total_count = len(data)
                
                for value, count in value_counts.items():
                    posterior_params[value] = alpha + count
                
                # Normalize to get posterior means
                total = sum(posterior_params.values())
                posterior_means = {value: count/total for value, count in posterior_params.items()}
                
                self.posterior_parameters[node] = {
                    'type': 'dirichlet',
                    'params': posterior_params,
                    'mean': posterior_means
                }
        else:
            # Child node
            cpt = {}
            
            # Get all unique parent configurations
            parent_configs = data[parents].drop_duplicates().values
            
            for parent_config in parent_configs:
                # Filter data for this parent configuration
                mask = True
                for parent, value in zip(parents, parent_config):
                    mask &= (data[parent] == value)
                
                filtered_data = data[mask]
                
                if len(filtered_data) > 0:
                    # Use Dirichlet prior for conditional distribution
                    alpha = self.priors.get(node, {}).get('alpha', 1.0)
                    value_counts = filtered_data[node].value_counts()
                    
                    posterior_params = {}
                    for value, count in value_counts.items():
                        posterior_params[value] = alpha + count
                    
                    # Normalize to get posterior means
                    total = sum(posterior_params.values())
                    posterior_means = {value: count/total for value, count in posterior_params.items()}
                    
                    cpt[parent_config] = {
                        'params': posterior_params,
                        'mean': posterior_means
                    }
                else:
                    # No data - use prior
                    alpha = self.priors.get(node, {}).get('alpha', 1.0)
                    unique_values = data[node].unique()
                    uniform_prob = 1.0 / len(unique_values)
                    
                    cpt[parent_config] = {
                        'params': {value: alpha for value in unique_values},
                        'mean': {value: uniform_prob for value in unique_values}
                    }
            
            self.posterior_parameters[node] = cpt
    
    def sample_parameters(self, n_samples=1000):
        """Sample parameters from posterior distributions"""
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            
            for node in self.structure.nodes:
                parents = self.structure.get_parents(node)
                
                if not parents:
                    # Root node
                    if node in self.posterior_parameters:
                        params = self.posterior_parameters[node]
                        if params['type'] == 'beta':
                            # Sample from Beta distribution
                            alpha = params['alpha']
                            beta = params['beta']
                            prob_true = np.random.beta(alpha, beta)
                            sample[node] = {True: prob_true, False: 1-prob_true}
                        else:
                            # Sample from Dirichlet distribution
                            dirichlet_params = list(params['params'].values())
                            probs = np.random.dirichlet(dirichlet_params)
                            values = list(params['params'].keys())
                            sample[node] = dict(zip(values, probs))
                else:
                    # Child node
                    sample[node] = {}
                    for parent_config in self.posterior_parameters[node]:
                        params = self.posterior_parameters[node][parent_config]
                        dirichlet_params = list(params['params'].values())
                        probs = np.random.dirichlet(dirichlet_params)
                        values = list(params['params'].keys())
                        sample[node][parent_config] = dict(zip(values, probs))
            
            samples.append(sample)
        
        return samples
    
    def predict_with_uncertainty(self, evidence, n_samples=1000):
        """Predict with uncertainty using posterior samples"""
        samples = self.sample_parameters(n_samples)
        
        predictions = defaultdict(list)
        
        for sample_params in samples:
            # Use the sampled parameters to make a prediction
            # This is a simplified version - in practice, you'd need to implement
            # proper inference with the sampled parameters
            pass
        
        return predictions

def bayesian_learning_example():
    """Example of Bayesian parameter learning"""
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    
    # True parameters
    rain_prob = 0.3
    
    # Generate data
    data = []
    for _ in range(n_samples):
        rain = np.random.choice([True, False], p=[rain_prob, 1-rain_prob])
        data.append({'Rain': rain})
    
    data_df = pd.DataFrame(data)
    
    # Create network structure
    from bayesian_network_definitions import BayesianNetworkStructure
    structure = BayesianNetworkStructure()
    structure.add_node("Rain")
    
    # Create Bayesian learner
    bayesian_learner = BayesianParameterLearner(structure)
    
    # Set priors
    bayesian_learner.set_prior("Rain", prior_type='beta', alpha=2, beta=3)  # Prior belief: 40% chance of rain
    
    # Learn parameters
    bayesian_learner.learn_parameters_bayesian(data_df)
    
    # Print results
    print("Bayesian Learning Results:")
    print("=" * 40)
    
    rain_params = bayesian_learner.posterior_parameters["Rain"]
    print(f"Prior: Beta(α=2, β=3)")
    print(f"Data: {n_samples} observations, {(data_df['Rain'] == True).sum()} rainy days")
    print(f"Posterior: Beta(α={rain_params['alpha']:.1f}, β={rain_params['beta']:.1f})")
    print(f"Posterior mean: P(Rain=True) = {rain_params['mean'][True]:.3f}")
    
    # Compare with MLE
    mle_prob = (data_df['Rain'] == True).sum() / len(data_df)
    print(f"MLE estimate: P(Rain=True) = {mle_prob:.3f}")
    
    # Sample from posterior
    samples = bayesian_learner.sample_parameters(n_samples=1000)
    rain_samples = [sample["Rain"][True] for sample in samples]
    
    print(f"Posterior samples - Mean: {np.mean(rain_samples):.3f}, Std: {np.std(rain_samples):.3f}")
    
    # Plot posterior distribution
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(rain_samples, bins=30, alpha=0.7, density=True)
    plt.axvline(rain_params['mean'][True], color='red', linestyle='--', label='Posterior Mean')
    plt.axvline(mle_prob, color='green', linestyle='--', label='MLE')
    plt.axvline(rain_prob, color='black', linestyle='--', label='True Value')
    plt.xlabel('P(Rain=True)')
    plt.ylabel('Density')
    plt.title('Posterior Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    from scipy.stats import beta
    x = np.linspace(0, 1, 100)
    prior_pdf = beta.pdf(x, 2, 3)
    posterior_pdf = beta.pdf(x, rain_params['alpha'], rain_params['beta'])
    
    plt.plot(x, prior_pdf, label='Prior')
    plt.plot(x, posterior_pdf, label='Posterior')
    plt.xlabel('P(Rain=True)')
    plt.ylabel('Density')
    plt.title('Prior vs Posterior')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return bayesian_learner, data_df

# Run the example
if __name__ == "__main__":
    bayesian_learner, data = bayesian_learning_example()
```

## Structure Learning

### Score-Based Structure Learning

Score-based methods evaluate different network structures using a scoring function.

```python
class StructureLearner:
    def __init__(self, variables, max_parents=3):
        self.variables = variables
        self.max_parents = max_parents
        self.best_structure = None
        self.best_score = -np.inf
    
    def learn_structure_greedy(self, data, score_function='bic'):
        """Learn structure using greedy search"""
        from itertools import combinations
        
        # Start with empty graph
        current_structure = {var: [] for var in self.variables}
        current_score = self._compute_score(current_structure, data, score_function)
        
        improved = True
        iteration = 0
        
        while improved and iteration < 100:
            improved = False
            best_move = None
            best_move_score = current_score
            
            # Try adding edges
            for var in self.variables:
                potential_parents = [v for v in self.variables if v != var and v not in current_structure[var]]
                
                for parent in potential_parents:
                    # Check if adding this parent would exceed max_parents
                    if len(current_structure[var]) >= self.max_parents:
                        continue
                    
                    # Check if adding this edge would create a cycle
                    if self._would_create_cycle(current_structure, var, parent):
                        continue
                    
                    # Try adding the edge
                    new_structure = self._copy_structure(current_structure)
                    new_structure[var].append(parent)
                    
                    new_score = self._compute_score(new_structure, data, score_function)
                    
                    if new_score > best_move_score:
                        best_move_score = new_score
                        best_move = (var, parent, 'add')
            
            # Try removing edges
            for var in self.variables:
                for parent in current_structure[var]:
                    new_structure = self._copy_structure(current_structure)
                    new_structure[var].remove(parent)
                    
                    new_score = self._compute_score(new_structure, data, score_function)
                    
                    if new_score > best_move_score:
                        best_move_score = new_score
                        best_move = (var, parent, 'remove')
            
            # Apply best move
            if best_move is not None:
                var, parent, action = best_move
                if action == 'add':
                    current_structure[var].append(parent)
                else:
                    current_structure[var].remove(parent)
                
                current_score = best_move_score
                improved = True
            
            iteration += 1
        
        self.best_structure = current_structure
        self.best_score = current_score
        
        return current_structure, current_score
    
    def _compute_score(self, structure, data, score_function):
        """Compute score for a network structure"""
        if score_function == 'bic':
            return self._compute_bic_score(structure, data)
        elif score_function == 'aic':
            return self._compute_aic_score(structure, data)
        elif score_function == 'log_likelihood':
            return self._compute_log_likelihood_score(structure, data)
        else:
            raise ValueError(f"Unknown score function: {score_function}")
    
    def _compute_bic_score(self, structure, data):
        """Compute BIC score"""
        log_likelihood = self._compute_log_likelihood_score(structure, data)
        n_params = self._count_parameters(structure, data)
        n_samples = len(data)
        
        bic = log_likelihood - 0.5 * n_params * np.log(n_samples)
        return bic
    
    def _compute_aic_score(self, structure, data):
        """Compute AIC score"""
        log_likelihood = self._compute_log_likelihood_score(structure, data)
        n_params = self._count_parameters(structure, data)
        
        aic = log_likelihood - n_params
        return aic
    
    def _compute_log_likelihood_score(self, structure, data):
        """Compute log-likelihood score"""
        # Learn parameters for this structure
        learner = BayesianNetworkLearner(structure)
        learner.learn_parameters_mle(data)
        
        # Compute log-likelihood
        return learner.compute_log_likelihood(data)
    
    def _count_parameters(self, structure, data):
        """Count number of parameters in the network"""
        n_params = 0
        
        for var in self.variables:
            parents = structure[var]
            
            if not parents:
                # Root node
                n_values = len(data[var].unique())
                n_params += n_values - 1
            else:
                # Child node
                n_values = len(data[var].unique())
                n_parent_configs = 1
                for parent in parents:
                    n_parent_values = len(data[parent].unique())
                    n_parent_configs *= n_parent_values
                
                n_params += (n_values - 1) * n_parent_configs
        
        return n_params
    
    def _would_create_cycle(self, structure, var, parent):
        """Check if adding an edge would create a cycle"""
        # Simple cycle detection - in practice, use more sophisticated algorithms
        visited = set()
        
        def dfs(node, target):
            if node == target:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            for child in structure[node]:
                if dfs(child, target):
                    return True
            
            return False
        
        return dfs(parent, var)
    
    def _copy_structure(self, structure):
        """Create a copy of the structure"""
        return {var: parents.copy() for var, parents in structure.items()}

def structure_learning_example():
    """Example of structure learning"""
    
    # Generate synthetic data from a known structure
    np.random.seed(42)
    n_samples = 1000
    
    # True structure: A -> B, A -> C, B -> D
    data = []
    for _ in range(n_samples):
        # Generate A (root)
        a = np.random.choice([0, 1], p=[0.6, 0.4])
        
        # Generate B (depends on A)
        if a == 0:
            b = np.random.choice([0, 1], p=[0.7, 0.3])
        else:
            b = np.random.choice([0, 1], p=[0.3, 0.7])
        
        # Generate C (depends on A)
        if a == 0:
            c = np.random.choice([0, 1], p=[0.8, 0.2])
        else:
            c = np.random.choice([0, 1], p=[0.2, 0.8])
        
        # Generate D (depends on B)
        if b == 0:
            d = np.random.choice([0, 1], p=[0.9, 0.1])
        else:
            d = np.random.choice([0, 1], p=[0.1, 0.9])
        
        data.append({'A': a, 'B': b, 'C': c, 'D': d})
    
    data_df = pd.DataFrame(data)
    
    # Learn structure
    variables = ['A', 'B', 'C', 'D']
    structure_learner = StructureLearner(variables, max_parents=2)
    
    learned_structure, score = structure_learner.learn_structure_greedy(data_df, score_function='bic')
    
    # Print results
    print("Structure Learning Results:")
    print("=" * 40)
    print(f"Learned structure: {learned_structure}")
    print(f"BIC score: {score:.2f}")
    
    # Compare with true structure
    true_structure = {
        'A': [],
        'B': ['A'],
        'C': ['A'],
        'D': ['B']
    }
    
    print(f"\nTrue structure: {true_structure}")
    
    # Compute accuracy
    correct_edges = 0
    total_edges = 0
    
    for var in variables:
        true_parents = set(true_structure[var])
        learned_parents = set(learned_structure[var])
        
        correct_edges += len(true_parents & learned_parents)
        total_edges += len(true_parents | learned_parents)
    
    accuracy = correct_edges / total_edges if total_edges > 0 else 1.0
    print(f"Structure accuracy: {accuracy:.3f}")
    
    # Visualize structures
    import networkx as nx
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # True structure
    G_true = nx.DiGraph()
    for var, parents in true_structure.items():
        for parent in parents:
            G_true.add_edge(parent, var)
    
    pos = nx.spring_layout(G_true)
    nx.draw(G_true, pos, with_labels=True, node_color='lightblue', 
            node_size=1000, font_size=12, font_weight='bold', ax=ax1)
    ax1.set_title('True Structure')
    
    # Learned structure
    G_learned = nx.DiGraph()
    for var, parents in learned_structure.items():
        for parent in parents:
            G_learned.add_edge(parent, var)
    
    pos = nx.spring_layout(G_learned)
    nx.draw(G_learned, pos, with_labels=True, node_color='lightgreen', 
            node_size=1000, font_size=12, font_weight='bold', ax=ax2)
    ax2.set_title('Learned Structure')
    
    plt.tight_layout()
    plt.show()
    
    return structure_learner, data_df, learned_structure

# Run the example
if __name__ == "__main__":
    structure_learner, data, learned_structure = structure_learning_example()
```

## Constraint-Based Structure Learning

Constraint-based methods use conditional independence tests to learn the network structure.

```python
class ConstraintBasedLearner:
    def __init__(self, variables, alpha=0.05):
        self.variables = variables
        self.alpha = alpha
        self.independence_tests = {}
    
    def learn_structure_pc(self, data):
        """Learn structure using PC algorithm"""
        # Initialize complete graph
        graph = {var: set(self.variables) - {var} for var in self.variables}
        
        # Phase 1: Remove edges based on unconditional independence
        for var1 in self.variables:
            for var2 in self.variables:
                if var1 != var2:
                    if self._test_independence(var1, var2, set(), data):
                        graph[var1].discard(var2)
                        graph[var2].discard(var1)
        
        # Phase 2: Remove edges based on conditional independence
        n = 1
        while True:
            edges_removed = 0
            
            for var1 in self.variables:
                for var2 in list(graph[var1]):
                    # Find subsets of neighbors of size n
                    neighbors = graph[var1] - {var2}
                    if len(neighbors) >= n:
                        for subset in self._get_subsets(neighbors, n):
                            if self._test_independence(var1, var2, subset, data):
                                graph[var1].discard(var2)
                                graph[var2].discard(var1)
                                edges_removed += 1
                                break
            
            if edges_removed == 0:
                break
            
            n += 1
        
        # Phase 3: Orient edges (simplified)
        # In practice, this would use more sophisticated orientation rules
        
        return graph
    
    def _test_independence(self, var1, var2, conditioning_set, data):
        """Test conditional independence using chi-square test"""
        if not conditioning_set:
            # Unconditional independence
            contingency_table = pd.crosstab(data[var1], data[var2])
            chi2, p_value, _, _ = scipy.stats.chi2_contingency(contingency_table)
        else:
            # Conditional independence
            # This is a simplified version - in practice, use more sophisticated tests
            p_value = 0.1  # Placeholder
        
        return p_value > self.alpha
    
    def _get_subsets(self, elements, size):
        """Get all subsets of given size"""
        from itertools import combinations
        return list(combinations(elements, size))

def constraint_based_example():
    """Example of constraint-based structure learning"""
    
    # Use the same synthetic data as before
    np.random.seed(42)
    n_samples = 1000
    
    data = []
    for _ in range(n_samples):
        a = np.random.choice([0, 1], p=[0.6, 0.4])
        
        if a == 0:
            b = np.random.choice([0, 1], p=[0.7, 0.3])
        else:
            b = np.random.choice([0, 1], p=[0.3, 0.7])
        
        if a == 0:
            c = np.random.choice([0, 1], p=[0.8, 0.2])
        else:
            c = np.random.choice([0, 1], p=[0.2, 0.8])
        
        if b == 0:
            d = np.random.choice([0, 1], p=[0.9, 0.1])
        else:
            d = np.random.choice([0, 1], p=[0.1, 0.9])
        
        data.append({'A': a, 'B': b, 'C': c, 'D': d})
    
    data_df = pd.DataFrame(data)
    
    # Learn structure using constraint-based method
    variables = ['A', 'B', 'C', 'D']
    constraint_learner = ConstraintBasedLearner(variables, alpha=0.05)
    
    learned_graph = constraint_learner.learn_structure_pc(data_df)
    
    print("Constraint-Based Structure Learning Results:")
    print("=" * 50)
    print(f"Learned graph: {learned_graph}")
    
    return constraint_learner, data_df, learned_graph

# Run the example
if __name__ == "__main__":
    constraint_learner, data, learned_graph = constraint_based_example()
```

## Model Selection and Validation

```python
class ModelSelection:
    def __init__(self, data, variables):
        self.data = data
        self.variables = variables
    
    def cross_validate_parameters(self, structure, n_folds=5):
        """Cross-validate parameter learning"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, test_idx in kf.split(self.data):
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]
            
            # Learn parameters on training data
            learner = BayesianNetworkLearner(structure)
            learner.learn_parameters_mle(train_data)
            
            # Evaluate on test data
            score = learner.compute_log_likelihood(test_data)
            scores.append(score)
        
        return np.mean(scores), np.std(scores)
    
    def compare_structures(self, structures, n_folds=5):
        """Compare different network structures"""
        results = {}
        
        for name, structure in structures.items():
            mean_score, std_score = self.cross_validate_parameters(structure, n_folds)
            results[name] = {
                'mean_score': mean_score,
                'std_score': std_score,
                'n_params': self._count_parameters(structure)
            }
        
        return results
    
    def _count_parameters(self, structure):
        """Count parameters in a structure"""
        n_params = 0
        
        for var in self.variables:
            parents = structure.get(var, [])
            
            if not parents:
                n_values = len(self.data[var].unique())
                n_params += n_values - 1
            else:
                n_values = len(self.data[var].unique())
                n_parent_configs = 1
                for parent in parents:
                    n_parent_values = len(self.data[parent].unique())
                    n_parent_configs *= n_parent_values
                
                n_params += (n_values - 1) * n_parent_configs
        
        return n_params

def model_selection_example():
    """Example of model selection"""
    
    # Generate data
    np.random.seed(42)
    n_samples = 1000
    
    data = []
    for _ in range(n_samples):
        a = np.random.choice([0, 1], p=[0.6, 0.4])
        b = np.random.choice([0, 1], p=[0.7, 0.3]) if a == 0 else np.random.choice([0, 1], p=[0.3, 0.7])
        c = np.random.choice([0, 1], p=[0.8, 0.2]) if a == 0 else np.random.choice([0, 1], p=[0.2, 0.8])
        data.append({'A': a, 'B': b, 'C': c})
    
    data_df = pd.DataFrame(data)
    
    # Define different structures to compare
    structures = {
        'Independent': {'A': [], 'B': [], 'C': []},
        'A->B': {'A': [], 'B': ['A'], 'C': []},
        'A->C': {'A': [], 'B': [], 'C': ['A']},
        'A->B, A->C': {'A': [], 'B': ['A'], 'C': ['A']},
        'A->B->C': {'A': [], 'B': ['A'], 'C': ['B']},
        'Complete': {'A': [], 'B': ['A'], 'C': ['A', 'B']}
    }
    
    # Compare structures
    model_selector = ModelSelection(data_df, ['A', 'B', 'C'])
    results = model_selector.compare_structures(structures)
    
    # Print results
    print("Model Selection Results:")
    print("=" * 50)
    print(f"{'Structure':<20} {'Mean Score':<12} {'Std Score':<12} {'Params':<8}")
    print("-" * 50)
    
    for name, result in results.items():
        print(f"{name:<20} {result['mean_score']:<12.2f} {result['std_score']:<12.2f} {result['n_params']:<8}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['mean_score'])
    print(f"\nBest model: {best_model[0]} (score: {best_model[1]['mean_score']:.2f})")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot scores
    plt.subplot(1, 2, 1)
    names = list(results.keys())
    scores = [results[name]['mean_score'] for name in names]
    errors = [results[name]['std_score'] for name in names]
    
    plt.bar(names, scores, yerr=errors, capsize=5)
    plt.ylabel('Cross-validation Score')
    plt.title('Model Comparison')
    plt.xticks(rotation=45)
    
    # Plot parameters vs score
    plt.subplot(1, 2, 2)
    params = [results[name]['n_params'] for name in names]
    plt.scatter(params, scores)
    
    for i, name in enumerate(names):
        plt.annotate(name, (params[i], scores[i]), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Number of Parameters')
    plt.ylabel('Cross-validation Score')
    plt.title('Complexity vs Performance')
    
    plt.tight_layout()
    plt.show()
    
    return model_selector, results

# Run the example
if __name__ == "__main__":
    model_selector, results = model_selection_example()
```

## Key Takeaways

1. **Parameter Learning**: MLE provides point estimates, Bayesian learning provides uncertainty
2. **Structure Learning**: Score-based methods optimize a scoring function, constraint-based methods use independence tests
3. **Model Selection**: Cross-validation helps choose between different structures
4. **Trade-offs**: More complex models may fit better but risk overfitting
5. **Applications**: Supervised learning enables building Bayesian networks from data
6. **Validation**: Always validate learned models on held-out data

## Exercises

1. Implement MLE parameter learning for a custom Bayesian network
2. Compare MLE and Bayesian learning on a synthetic dataset
3. Implement greedy structure learning with different scoring functions
4. Apply structure learning to a real-world dataset
5. Perform model selection to find the best network structure 