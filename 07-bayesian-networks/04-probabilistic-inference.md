# Probabilistic Inference in Bayesian Networks

## Overview

Probabilistic inference is the process of computing posterior probabilities of variables given evidence in a Bayesian network. This is a fundamental operation that enables reasoning under uncertainty.

## What is Probabilistic Inference?

Given a Bayesian network and some evidence (observed variables), probabilistic inference computes:
- **Marginal probabilities**: P(X | evidence)
- **Joint probabilities**: P(X, Y | evidence)  
- **Conditional probabilities**: P(X | Y, evidence)

## Exact Inference Methods

### Variable Elimination

Variable elimination is an exact inference algorithm that systematically eliminates variables from the joint distribution.

#### Mathematical Foundation

The algorithm works by:
1. **Factor multiplication**: Combine factors (CPTs) into larger factors
2. **Summing out**: Eliminate variables by summing over their values
3. **Normalization**: Ensure probabilities sum to 1

For a query P(Q | E), we compute:
$$P(Q | E) = \frac{P(Q, E)}{P(E)} = \frac{\sum_{H} P(Q, E, H)}{\sum_{Q, H} P(Q, E, H)}$$

Where H are the hidden variables.

#### Python Implementation

```python
import numpy as np
from collections import defaultdict
import itertools

class VariableElimination:
    def __init__(self, bayesian_network):
        self.bn = bayesian_network
        self.factors = {}
        self.initialize_factors()
    
    def initialize_factors(self):
        """Initialize factors from CPTs"""
        for node in self.bn.structure.nodes:
            cpt = self.bn.cpts[node]
            parents = self.bn.structure.get_parents(node)
            
            # Create factor for this node
            factor = Factor(node, parents, cpt)
            self.factors[node] = factor
    
    def eliminate_variable(self, var, factors):
        """Eliminate a variable from a set of factors"""
        # Find factors that contain the variable
        relevant_factors = [f for f in factors if var in f.variables]
        
        if not relevant_factors:
            return factors
        
        # Multiply relevant factors
        product_factor = relevant_factors[0]
        for factor in relevant_factors[1:]:
            product_factor = self.multiply_factors(product_factor, factor)
        
        # Sum out the variable
        result_factor = self.sum_out_variable(product_factor, var)
        
        # Remove original factors and add result
        new_factors = [f for f in factors if f not in relevant_factors]
        new_factors.append(result_factor)
        
        return new_factors
    
    def multiply_factors(self, factor1, factor2):
        """Multiply two factors"""
        # Get all variables
        all_vars = list(set(factor1.variables + factor2.variables))
        
        # Create new factor
        new_factor = Factor("product", all_vars, {})
        
        # Generate all possible assignments
        var_values = {var: [True, False] for var in all_vars}  # Assume binary
        
        for assignment in self.generate_assignments(all_vars, var_values):
            # Get values for each factor
            factor1_vals = {var: assignment[var] for var in factor1.variables}
            factor2_vals = {var: assignment[var] for var in factor2.variables}
            
            # Get probabilities
            prob1 = factor1.get_probability(factor1_vals)
            prob2 = factor2.get_probability(factor2_vals)
            
            # Set probability in new factor
            new_factor.set_probability(assignment, prob1 * prob2)
        
        return new_factor
    
    def sum_out_variable(self, factor, var):
        """Sum out a variable from a factor"""
        remaining_vars = [v for v in factor.variables if v != var]
        
        new_factor = Factor("sum", remaining_vars, {})
        
        # Generate assignments for remaining variables
        var_values = {var: [True, False] for var in remaining_vars}
        
        for assignment in self.generate_assignments(remaining_vars, var_values):
            total_prob = 0.0
            
            # Sum over the variable to be eliminated
            for var_val in [True, False]:
                full_assignment = assignment.copy()
                full_assignment[var] = var_val
                total_prob += factor.get_probability(full_assignment)
            
            new_factor.set_probability(assignment, total_prob)
        
        return new_factor
    
    def generate_assignments(self, variables, var_values):
        """Generate all possible assignments for variables"""
        assignments = []
        for values in itertools.product(*[var_values[var] for var in variables]):
            assignment = dict(zip(variables, values))
            assignments.append(assignment)
        return assignments
    
    def query(self, query_vars, evidence=None):
        """Compute P(query_vars | evidence) using variable elimination"""
        if evidence is None:
            evidence = {}
        
        # Start with all factors
        factors = list(self.factors.values())
        
        # Set evidence in factors
        for factor in factors:
            factor.set_evidence(evidence)
        
        # Get variables to eliminate (hidden variables)
        all_vars = set()
        for factor in factors:
            all_vars.update(factor.variables)
        
        hidden_vars = [var for var in all_vars 
                      if var not in query_vars and var not in evidence]
        
        # Eliminate hidden variables
        for var in hidden_vars:
            factors = self.eliminate_variable(var, factors)
        
        # Multiply remaining factors
        if len(factors) > 1:
            result_factor = factors[0]
            for factor in factors[1:]:
                result_factor = self.multiply_factors(result_factor, factor)
        else:
            result_factor = factors[0]
        
        # Normalize
        result_factor.normalize()
        
        return result_factor

class Factor:
    def __init__(self, name, variables, cpt):
        self.name = name
        self.variables = variables
        self.probabilities = {}
        self.evidence = {}
        
        # Initialize from CPT if provided
        if cpt:
            self.probabilities = cpt.probabilities.copy()
    
    def set_evidence(self, evidence):
        """Set evidence for this factor"""
        self.evidence = evidence.copy()
    
    def get_probability(self, assignment):
        """Get probability for a variable assignment"""
        # Check if assignment is consistent with evidence
        for var, val in self.evidence.items():
            if var in assignment and assignment[var] != val:
                return 0.0
        
        # Get parent values for CPT lookup
        parent_values = tuple(assignment[var] for var in self.variables[:-1])
        var_value = assignment[self.variables[-1]]
        
        return self.probabilities.get(parent_values, {}).get(var_value, 0.0)
    
    def set_probability(self, assignment, probability):
        """Set probability for a variable assignment"""
        parent_values = tuple(assignment[var] for var in self.variables[:-1])
        var_value = assignment[self.variables[-1]]
        
        if parent_values not in self.probabilities:
            self.probabilities[parent_values] = {}
        
        self.probabilities[parent_values][var_value] = probability
    
    def normalize(self):
        """Normalize the factor"""
        total = 0.0
        for parent_config in self.probabilities:
            total += sum(self.probabilities[parent_config].values())
        
        if total > 0:
            for parent_config in self.probabilities:
                for var_value in self.probabilities[parent_config]:
                    self.probabilities[parent_config][var_value] /= total

# Example: Variable Elimination
def variable_elimination_example():
    """Example of using variable elimination for inference"""
    
    # Create a simple Bayesian network
    from bayesian_network_definitions import create_complete_weather_network
    weather_bn = create_complete_weather_network()
    
    # Create variable elimination engine
    ve = VariableElimination(weather_bn)
    
    # Query: P(WetGrass | Rain=True)
    evidence = {"Rain": True}
    query_vars = ["WetGrass"]
    
    result_factor = ve.query(query_vars, evidence)
    
    print("Variable Elimination Results:")
    print(f"P(WetGrass | Rain=True):")
    for assignment in result_factor.generate_assignments(query_vars, {var: [True, False] for var in query_vars}):
        prob = result_factor.get_probability(assignment)
        print(f"  P(WetGrass={assignment['WetGrass']} | Rain=True) = {prob:.4f}")
    
    return ve

# Run the example
if __name__ == "__main__":
    ve_engine = variable_elimination_example()
```

### Junction Tree Algorithm

The junction tree algorithm is another exact inference method that works by:
1. **Moralization**: Convert DAG to undirected graph
2. **Triangulation**: Add edges to make the graph chordal
3. **Clique tree construction**: Build a tree of cliques
4. **Message passing**: Propagate probabilities through the tree

#### Python Implementation

```python
import networkx as nx
from collections import defaultdict

class JunctionTree:
    def __init__(self, bayesian_network):
        self.bn = bayesian_network
        self.moral_graph = None
        self.triangulated_graph = None
        self.clique_tree = None
        self.clique_potentials = {}
    
    def moralize(self):
        """Convert DAG to moral graph (undirected)"""
        self.moral_graph = self.bn.structure.graph.to_undirected()
        
        # Add edges between parents of each node
        for node in self.bn.structure.nodes:
            parents = self.bn.structure.get_parents(node)
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    self.moral_graph.add_edge(parents[i], parents[j])
        
        return self.moral_graph
    
    def triangulate(self):
        """Triangulate the moral graph"""
        self.triangulated_graph = self.moral_graph.copy()
        
        # Simple triangulation: add edges to make graph chordal
        # This is a simplified version - real implementations use more sophisticated algorithms
        
        # Find cycles of length > 3 and add chords
        cycles = list(nx.simple_cycles(self.triangulated_graph))
        for cycle in cycles:
            if len(cycle) > 3:
                # Add chord between non-adjacent vertices
                for i in range(len(cycle)):
                    for j in range(i + 2, len(cycle)):
                        if not self.triangulated_graph.has_edge(cycle[i], cycle[j]):
                            self.triangulated_graph.add_edge(cycle[i], cycle[j])
        
        return self.triangulated_graph
    
    def find_cliques(self):
        """Find maximal cliques in the triangulated graph"""
        cliques = list(nx.find_cliques(self.triangulated_graph))
        return cliques
    
    def build_clique_tree(self, cliques):
        """Build a tree from cliques"""
        # Create a graph where nodes are cliques
        clique_graph = nx.Graph()
        
        for i, clique1 in enumerate(cliques):
            for j, clique2 in enumerate(cliques[i+1:], i+1):
                # Add edge if cliques share variables
                intersection = set(clique1) & set(clique2)
                if intersection:
                    clique_graph.add_edge(i, j, weight=len(intersection))
        
        # Find maximum spanning tree
        self.clique_tree = nx.maximum_spanning_tree(clique_graph)
        
        return self.clique_tree
    
    def initialize_potentials(self, cliques):
        """Initialize clique potentials from CPTs"""
        for i, clique in enumerate(cliques):
            self.clique_potentials[i] = defaultdict(float)
            
            # Generate all assignments for this clique
            var_values = {var: [True, False] for var in clique}
            
            for assignment in self.generate_assignments(clique, var_values):
                # Compute potential as product of relevant CPTs
                potential = 1.0
                
                for node in clique:
                    if node in self.bn.cpts:
                        cpt = self.bn.cpts[node]
                        parents = self.bn.structure.get_parents(node)
                        
                        # Get parent values
                        parent_values = tuple(assignment[parent] for parent in parents)
                        node_value = assignment[node]
                        
                        # Get probability from CPT
                        prob = cpt.get_probability(parent_values, node_value)
                        potential *= prob
                
                self.clique_potentials[i][tuple(assignment.items())] = potential
    
    def message_passing(self, evidence=None):
        """Perform message passing to compute marginals"""
        if evidence is None:
            evidence = {}
        
        # Set evidence in potentials
        for clique_id, potential in self.clique_potentials.items():
            for assignment, value in list(potential.items()):
                assignment_dict = dict(assignment)
                
                # Check if assignment is consistent with evidence
                for var, val in evidence.items():
                    if var in assignment_dict and assignment_dict[var] != val:
                        potential[assignment] = 0.0
        
        # Perform message passing (simplified version)
        # In practice, this involves more sophisticated algorithms
        
        return self.clique_potentials
    
    def query(self, query_vars, evidence=None):
        """Query the junction tree"""
        # Build junction tree
        self.moralize()
        self.triangulate()
        cliques = self.find_cliques()
        self.build_clique_tree(cliques)
        self.initialize_potentials(cliques)
        
        # Perform inference
        potentials = self.message_passing(evidence)
        
        # Find clique containing query variables
        query_clique = None
        for i, clique in enumerate(cliques):
            if all(var in clique for var in query_vars):
                query_clique = i
                break
        
        if query_clique is None:
            raise ValueError("Query variables not found in any clique")
        
        # Compute marginal
        marginal = defaultdict(float)
        for assignment, potential in potentials[query_clique].items():
            assignment_dict = dict(assignment)
            query_assignment = tuple(assignment_dict[var] for var in query_vars)
            marginal[query_assignment] += potential
        
        # Normalize
        total = sum(marginal.values())
        if total > 0:
            for assignment in marginal:
                marginal[assignment] /= total
        
        return marginal
    
    def generate_assignments(self, variables, var_values):
        """Generate all possible assignments for variables"""
        assignments = []
        for values in itertools.product(*[var_values[var] for var in variables]):
            assignment = dict(zip(variables, values))
            assignments.append(assignment)
        return assignments

# Example: Junction Tree
def junction_tree_example():
    """Example of using junction tree algorithm"""
    
    # Create Bayesian network
    from bayesian_network_definitions import create_complete_weather_network
    weather_bn = create_complete_weather_network()
    
    # Create junction tree
    jt = JunctionTree(weather_bn)
    
    # Query: P(WetGrass | Rain=True)
    evidence = {"Rain": True}
    query_vars = ["WetGrass"]
    
    marginal = jt.query(query_vars, evidence)
    
    print("Junction Tree Results:")
    print(f"P(WetGrass | Rain=True):")
    for assignment, prob in marginal.items():
        print(f"  P(WetGrass={assignment[0]} | Rain=True) = {prob:.4f}")
    
    return jt

# Run the example
if __name__ == "__main__":
    jt_engine = junction_tree_example()
```

## Approximate Inference Methods

### Sampling-Based Methods

#### Gibbs Sampling

Gibbs sampling is a Markov Chain Monte Carlo (MCMC) method that samples from the posterior distribution.

```python
import numpy as np
from collections import defaultdict

class GibbsSampler:
    def __init__(self, bayesian_network):
        self.bn = bayesian_network
        self.samples = []
    
    def sample(self, evidence=None, n_samples=1000, burn_in=100):
        """Generate samples using Gibbs sampling"""
        if evidence is None:
            evidence = {}
        
        # Initialize sample
        sample = self.initialize_sample(evidence)
        
        # Burn-in phase
        for _ in range(burn_in):
            sample = self.gibbs_step(sample, evidence)
        
        # Sampling phase
        for _ in range(n_samples):
            sample = self.gibbs_step(sample, evidence)
            self.samples.append(sample.copy())
        
        return self.samples
    
    def initialize_sample(self, evidence):
        """Initialize a sample randomly"""
        sample = {}
        
        # Set evidence
        sample.update(evidence)
        
        # Initialize other variables randomly
        for node in self.bn.structure.nodes:
            if node not in evidence:
                sample[node] = np.random.choice([True, False])
        
        return sample
    
    def gibbs_step(self, sample, evidence):
        """Perform one Gibbs sampling step"""
        new_sample = sample.copy()
        
        # Update each variable in turn
        for node in self.bn.structure.nodes:
            if node in evidence:
                continue  # Skip evidence variables
            
            # Compute conditional probability P(node | all other variables)
            prob = self.compute_conditional_probability(node, new_sample)
            
            # Sample new value
            new_sample[node] = np.random.choice([True, False], p=[prob, 1-prob])
        
        return new_sample
    
    def compute_conditional_probability(self, node, sample):
        """Compute P(node=True | all other variables)"""
        # Get parents and children
        parents = self.bn.structure.get_parents(node)
        children = self.bn.structure.get_children(node)
        
        # Compute prior probability P(node | parents)
        if node in self.bn.cpts:
            cpt = self.bn.cpts[node]
            parent_values = tuple(sample[parent] for parent in parents)
            prior_prob = cpt.get_probability(parent_values, True)
        else:
            prior_prob = 0.5  # Default uniform prior
        
        # Compute likelihood from children
        likelihood = 1.0
        for child in children:
            if child in self.bn.cpts:
                child_cpt = self.bn.cpts[child]
                child_parents = self.bn.structure.get_parents(child)
                child_parent_values = tuple(sample[parent] for parent in child_parents)
                
                # Compute P(child | parents) for both values of node
                child_value = sample[child]
                
                # Create parent assignment with node=True
                parent_assignment_true = list(child_parent_values)
                for i, parent in enumerate(child_parents):
                    if parent == node:
                        parent_assignment_true[i] = True
                
                # Create parent assignment with node=False
                parent_assignment_false = list(child_parent_values)
                for i, parent in enumerate(child_parents):
                    if parent == node:
                        parent_assignment_false[i] = False
                
                prob_true = child_cpt.get_probability(tuple(parent_assignment_true), child_value)
                prob_false = child_cpt.get_probability(tuple(parent_assignment_false), child_value)
                
                likelihood *= prob_true / prob_false
        
        # Compute posterior probability
        posterior_prob = prior_prob * likelihood
        posterior_prob_false = (1 - prior_prob) * 1.0  # likelihood for False is 1.0
        
        # Normalize
        total = posterior_prob + posterior_prob_false
        if total > 0:
            return posterior_prob / total
        else:
            return 0.5
    
    def estimate_probability(self, query_vars, evidence=None):
        """Estimate probability from samples"""
        if not self.samples:
            raise ValueError("No samples available. Run sample() first.")
        
        # Filter samples consistent with evidence
        if evidence:
            filtered_samples = []
            for sample in self.samples:
                if all(sample[var] == val for var, val in evidence.items()):
                    filtered_samples.append(sample)
        else:
            filtered_samples = self.samples
        
        if not filtered_samples:
            return 0.0
        
        # Count samples where query variables are True
        count = 0
        for sample in filtered_samples:
            if all(sample[var] for var in query_vars):
                count += 1
        
        return count / len(filtered_samples)

# Example: Gibbs Sampling
def gibbs_sampling_example():
    """Example of using Gibbs sampling for inference"""
    
    # Create Bayesian network
    from bayesian_network_definitions import create_complete_weather_network
    weather_bn = create_complete_weather_network()
    
    # Create Gibbs sampler
    gibbs = GibbsSampler(weather_bn)
    
    # Generate samples
    evidence = {"Rain": True}
    samples = gibbs.sample(evidence=evidence, n_samples=1000, burn_in=100)
    
    # Estimate probability
    prob = gibbs.estimate_probability(["WetGrass"], evidence)
    
    print("Gibbs Sampling Results:")
    print(f"P(WetGrass=True | Rain=True) ≈ {prob:.4f}")
    print(f"Generated {len(samples)} samples")
    
    return gibbs

# Run the example
if __name__ == "__main__":
    gibbs_sampler = gibbs_sampling_example()
```

### Variational Inference

Variational inference approximates the posterior by finding the closest distribution from a family of tractable distributions.

```python
class VariationalInference:
    def __init__(self, bayesian_network):
        self.bn = bayesian_network
        self.variational_params = {}
    
    def initialize_variational_params(self):
        """Initialize variational parameters"""
        for node in self.bn.structure.nodes:
            # Use uniform initialization
            self.variational_params[node] = 0.5
    
    def compute_elbo(self, evidence=None):
        """Compute Evidence Lower BOund (ELBO)"""
        if evidence is None:
            evidence = {}
        
        elbo = 0.0
        
        # Expected log likelihood
        for node in self.bn.structure.nodes:
            if node in evidence:
                continue
            
            parents = self.bn.structure.get_parents(node)
            
            if not parents:
                # Root node
                if node in self.bn.cpts:
                    cpt = self.bn.cpts[node]
                    prob_true = cpt.get_probability((), True)
                    prob_false = cpt.get_probability((), False)
                    
                    q_prob = self.variational_params[node]
                    elbo += q_prob * np.log(prob_true) + (1 - q_prob) * np.log(prob_false)
            else:
                # Child node
                if node in self.bn.cpts:
                    cpt = self.bn.cpts[node]
                    
                    # Compute expected log probability
                    for parent_assignment in self.generate_parent_assignments(parents):
                        prob_true = cpt.get_probability(parent_assignment, True)
                        prob_false = cpt.get_probability(parent_assignment, False)
                        
                        q_prob = self.variational_params[node]
                        elbo += q_prob * np.log(prob_true) + (1 - q_prob) * np.log(prob_false)
        
        # Entropy term
        for node in self.bn.structure.nodes:
            if node in evidence:
                continue
            
            q_prob = self.variational_params[node]
            if q_prob > 0 and q_prob < 1:
                elbo += q_prob * np.log(q_prob) + (1 - q_prob) * np.log(1 - q_prob)
        
        return elbo
    
    def update_variational_params(self, evidence=None, learning_rate=0.01, n_iterations=100):
        """Update variational parameters using gradient ascent"""
        if evidence is None:
            evidence = {}
        
        self.initialize_variational_params()
        
        for iteration in range(n_iterations):
            # Compute gradients and update parameters
            for node in self.bn.structure.nodes:
                if node in evidence:
                    continue
                
                # Compute gradient (simplified)
                gradient = self.compute_gradient(node, evidence)
                
                # Update parameter
                self.variational_params[node] += learning_rate * gradient
                
                # Ensure parameter is in [0, 1]
                self.variational_params[node] = np.clip(self.variational_params[node], 0, 1)
            
            # Compute ELBO
            elbo = self.compute_elbo(evidence)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, ELBO: {elbo:.4f}")
    
    def compute_gradient(self, node, evidence):
        """Compute gradient for a variational parameter (simplified)"""
        # This is a simplified gradient computation
        # In practice, automatic differentiation would be used
        
        parents = self.bn.structure.get_parents(node)
        
        if not parents:
            # Root node
            if node in self.bn.cpts:
                cpt = self.bn.cpts[node]
                prob_true = cpt.get_probability((), True)
                prob_false = cpt.get_probability((), False)
                return np.log(prob_true) - np.log(prob_false)
        
        return 0.0
    
    def generate_parent_assignments(self, parents):
        """Generate all possible parent assignments"""
        assignments = []
        for values in itertools.product([True, False], repeat=len(parents)):
            assignments.append(values)
        return assignments
    
    def query(self, query_vars, evidence=None):
        """Query using variational approximation"""
        if evidence is None:
            evidence = {}
        
        # Update variational parameters
        self.update_variational_params(evidence)
        
        # Return variational probabilities
        result = {}
        for var in query_vars:
            if var in evidence:
                result[var] = evidence[var]
            else:
                result[var] = self.variational_params[var]
        
        return result

# Example: Variational Inference
def variational_inference_example():
    """Example of using variational inference"""
    
    # Create Bayesian network
    from bayesian_network_definitions import create_complete_weather_network
    weather_bn = create_complete_weather_network()
    
    # Create variational inference engine
    vi = VariationalInference(weather_bn)
    
    # Query: P(WetGrass | Rain=True)
    evidence = {"Rain": True}
    query_vars = ["WetGrass"]
    
    result = vi.query(query_vars, evidence)
    
    print("Variational Inference Results:")
    print(f"P(WetGrass=True | Rain=True) ≈ {result['WetGrass']:.4f}")
    
    return vi

# Run the example
if __name__ == "__main__":
    vi_engine = variational_inference_example()
```

## Complexity Analysis

### Computational Complexity

| Method | Time Complexity | Space Complexity | Accuracy |
|--------|----------------|------------------|----------|
| Variable Elimination | O(n · 2^w) | O(n · 2^w) | Exact |
| Junction Tree | O(n · 2^w) | O(n · 2^w) | Exact |
| Gibbs Sampling | O(n · m) | O(n) | Approximate |
| Variational Inference | O(n · k) | O(n) | Approximate |

Where:
- n = number of variables
- w = treewidth of the graph
- m = number of samples
- k = number of iterations

### Trade-offs in Practice

```python
class InferenceComparison:
    def __init__(self, bayesian_network):
        self.bn = bayesian_network
    
    def compare_methods(self, query_vars, evidence, n_samples=1000):
        """Compare different inference methods"""
        results = {}
        times = {}
        
        # Variable Elimination
        import time
        start_time = time.time()
        try:
            ve = VariableElimination(self.bn)
            ve_result = ve.query(query_vars, evidence)
            results['Variable Elimination'] = ve_result
            times['Variable Elimination'] = time.time() - start_time
        except Exception as e:
            print(f"Variable Elimination failed: {e}")
        
        # Gibbs Sampling
        start_time = time.time()
        try:
            gibbs = GibbsSampler(self.bn)
            gibbs.sample(evidence=evidence, n_samples=n_samples)
            gibbs_result = gibbs.estimate_probability(query_vars, evidence)
            results['Gibbs Sampling'] = gibbs_result
            times['Gibbs Sampling'] = time.time() - start_time
        except Exception as e:
            print(f"Gibbs Sampling failed: {e}")
        
        # Variational Inference
        start_time = time.time()
        try:
            vi = VariationalInference(self.bn)
            vi_result = vi.query(query_vars, evidence)
            results['Variational Inference'] = vi_result
            times['Variational Inference'] = time.time() - start_time
        except Exception as e:
            print(f"Variational Inference failed: {e}")
        
        # Print results
        print("Inference Method Comparison:")
        print("=" * 50)
        for method, result in results.items():
            print(f"{method}:")
            print(f"  Result: {result}")
            print(f"  Time: {times[method]:.4f} seconds")
            print()
        
        return results, times

# Example: Method Comparison
def inference_comparison_example():
    """Compare different inference methods"""
    
    # Create Bayesian network
    from bayesian_network_definitions import create_complete_weather_network
    weather_bn = create_complete_weather_network()
    
    # Create comparison engine
    comparison = InferenceComparison(weather_bn)
    
    # Compare methods
    query_vars = ["WetGrass"]
    evidence = {"Rain": True}
    
    results, times = comparison.compare_methods(query_vars, evidence)
    
    return comparison

# Run the example
if __name__ == "__main__":
    comparison_engine = inference_comparison_example()
```

## Key Takeaways

1. **Exact inference** methods (Variable Elimination, Junction Tree) provide precise results but may be computationally expensive
2. **Approximate inference** methods (Gibbs Sampling, Variational Inference) trade accuracy for efficiency
3. **Variable Elimination** is efficient for small networks but scales poorly with treewidth
4. **Gibbs Sampling** is simple to implement and works well for many problems
5. **Variational Inference** provides fast approximate results but may miss complex posterior structure
6. **Method choice** depends on the specific problem requirements (accuracy vs. speed)

## Exercises

1. Implement variable elimination for a custom Bayesian network
2. Compare exact and approximate inference methods on the same network
3. Analyze the computational complexity of different inference algorithms
4. Implement Gibbs sampling with custom proposal distributions
5. Build a variational inference engine with automatic differentiation 