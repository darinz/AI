# Bayesian Network Definitions

## Overview

This guide covers the fundamental definitions and mathematical foundations of Bayesian networks. Understanding these core concepts is essential for building and working with Bayesian network models.

## Bayesian Network Structure

A Bayesian network is a directed acyclic graph (DAG) where:
- **Nodes** represent random variables
- **Edges** represent direct probabilistic dependencies
- **Acyclicity** ensures no directed cycles exist

### Mathematical Definition

A Bayesian network is a pair $(G, \Theta)$ where:
- $G = (V, E)$ is a directed acyclic graph with vertices $V$ and edges $E$
- $\Theta$ is a set of parameters that specify the conditional probability distributions

### Python Implementation of Network Structure

```python
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class BayesianNetworkStructure:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = set()
        self.edges = set()
    
    def add_node(self, node):
        """Add a node to the Bayesian network"""
        self.graph.add_node(node)
        self.nodes.add(node)
    
    def add_edge(self, from_node, to_node):
        """Add a directed edge from from_node to to_node"""
        if self._would_create_cycle(from_node, to_node):
            raise ValueError(f"Adding edge {from_node} -> {to_node} would create a cycle")
        
        self.graph.add_edge(from_node, to_node)
        self.edges.add((from_node, to_node))
    
    def _would_create_cycle(self, from_node, to_node):
        """Check if adding an edge would create a cycle"""
        # Temporarily add the edge
        self.graph.add_edge(from_node, to_node)
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(self.graph))
            has_cycle = len(cycles) > 0
        except nx.NetworkXNoCycle:
            has_cycle = False
        
        # Remove the temporary edge
        self.graph.remove_edge(from_node, to_node)
        
        return has_cycle
    
    def get_parents(self, node):
        """Get all parents of a node"""
        return list(self.graph.predecessors(node))
    
    def get_children(self, node):
        """Get all children of a node"""
        return list(self.graph.successors(node))
    
    def get_ancestors(self, node):
        """Get all ancestors of a node"""
        return list(nx.ancestors(self.graph, node))
    
    def get_descendants(self, node):
        """Get all descendants of a node"""
        return list(nx.descendants(self.graph, node))
    
    def is_root(self, node):
        """Check if a node is a root (has no parents)"""
        return len(self.get_parents(node)) == 0
    
    def is_leaf(self, node):
        """Check if a node is a leaf (has no children)"""
        return len(self.get_children(node)) == 0
    
    def get_topological_order(self):
        """Get nodes in topological order"""
        return list(nx.topological_sort(self.graph))
    
    def visualize(self, title="Bayesian Network Structure"):
        """Visualize the Bayesian network"""
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightblue', 
                              node_size=2000)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_weight='bold')
        
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    def print_structure_info(self):
        """Print information about the network structure"""
        print("Bayesian Network Structure Information:")
        print(f"Number of nodes: {len(self.nodes)}")
        print(f"Number of edges: {len(self.edges)}")
        print(f"Root nodes: {[n for n in self.nodes if self.is_root(n)]}")
        print(f"Leaf nodes: {[n for n in self.nodes if self.is_leaf(n)]}")
        print("\nNode relationships:")
        for node in sorted(self.nodes):
            parents = self.get_parents(node)
            children = self.get_children(node)
            print(f"  {node}: parents={parents}, children={children}")

# Example: Creating a Bayesian Network Structure
def create_medical_network():
    """Create a medical diagnosis Bayesian network structure"""
    bn = BayesianNetworkStructure()
    
    # Add nodes
    nodes = ["Age", "Gender", "Disease", "Symptom1", "Symptom2", "Test1", "Test2"]
    for node in nodes:
        bn.add_node(node)
    
    # Add edges
    edges = [
        ("Age", "Disease"),
        ("Gender", "Disease"),
        ("Disease", "Symptom1"),
        ("Disease", "Symptom2"),
        ("Symptom1", "Test1"),
        ("Symptom2", "Test2")
    ]
    
    for from_node, to_node in edges:
        bn.add_edge(from_node, to_node)
    
    return bn

# Run the example
if __name__ == "__main__":
    medical_bn = create_medical_network()
    medical_bn.visualize("Medical Diagnosis Network")
    medical_bn.print_structure_info()
```

## Conditional Probability Tables (CPTs)

Conditional Probability Tables specify the probability distribution of each variable given its parents.

### Mathematical Definition

For a variable $X_i$ with parents $\text{Pa}(X_i)$, the CPT contains:
$$P(X_i | \text{Pa}(X_i))$$

### Python Implementation of CPTs

```python
class ConditionalProbabilityTable:
    def __init__(self, variable, parents=None):
        self.variable = variable
        self.parents = parents or []
        self.probabilities = {}
    
    def set_probability(self, parent_values, variable_value, probability):
        """
        Set probability P(variable=variable_value | parents=parent_values)
        parent_values: tuple of parent values in the same order as self.parents
        """
        if parent_values not in self.probabilities:
            self.probabilities[parent_values] = {}
        
        self.probabilities[parent_values][variable_value] = probability
    
    def get_probability(self, parent_values, variable_value):
        """Get probability P(variable=variable_value | parents=parent_values)"""
        return self.probabilities.get(parent_values, {}).get(variable_value, 0.0)
    
    def get_distribution(self, parent_values):
        """Get the full conditional distribution for given parent values"""
        return self.probabilities.get(parent_values, {})
    
    def normalize(self):
        """Ensure probabilities sum to 1 for each parent configuration"""
        for parent_values in self.probabilities:
            total = sum(self.probabilities[parent_values].values())
            if total > 0:
                for var_value in self.probabilities[parent_values]:
                    self.probabilities[parent_values][var_value] /= total
    
    def sample(self, parent_values):
        """Sample a value from the conditional distribution"""
        import random
        
        distribution = self.get_distribution(parent_values)
        if not distribution:
            raise ValueError(f"No distribution found for parent values {parent_values}")
        
        values = list(distribution.keys())
        probabilities = list(distribution.values())
        
        return random.choices(values, weights=probabilities)[0]

# Example: Creating CPTs for a Simple Network
def create_weather_cpts():
    """Create CPTs for a simple weather Bayesian network"""
    cpts = {}
    
    # P(Rain)
    rain_cpt = ConditionalProbabilityTable("Rain")
    rain_cpt.set_probability((), True, 0.2)
    rain_cpt.set_probability((), False, 0.8)
    cpts["Rain"] = rain_cpt
    
    # P(Sprinkler | Rain)
    sprinkler_cpt = ConditionalProbabilityTable("Sprinkler", ["Rain"])
    sprinkler_cpt.set_probability((True,), True, 0.01)
    sprinkler_cpt.set_probability((True,), False, 0.99)
    sprinkler_cpt.set_probability((False,), True, 0.4)
    sprinkler_cpt.set_probability((False,), False, 0.6)
    cpts["Sprinkler"] = sprinkler_cpt
    
    # P(WetGrass | Rain, Sprinkler)
    wetgrass_cpt = ConditionalProbabilityTable("WetGrass", ["Rain", "Sprinkler"])
    wetgrass_cpt.set_probability((True, True), True, 0.99)
    wetgrass_cpt.set_probability((True, True), False, 0.01)
    wetgrass_cpt.set_probability((True, False), True, 0.9)
    wetgrass_cpt.set_probability((True, False), False, 0.1)
    wetgrass_cpt.set_probability((False, True), True, 0.9)
    wetgrass_cpt.set_probability((False, True), False, 0.1)
    wetgrass_cpt.set_probability((False, False), True, 0.0)
    wetgrass_cpt.set_probability((False, False), False, 1.0)
    cpts["WetGrass"] = wetgrass_cpt
    
    return cpts

# Test the CPTs
if __name__ == "__main__":
    weather_cpts = create_weather_cpts()
    
    print("Weather Network CPTs:")
    for var, cpt in weather_cpts.items():
        print(f"\n{var}:")
        for parent_config, probs in cpt.probabilities.items():
            print(f"  P({var} | {cpt.parents}={parent_config}) = {probs}")
```

## Joint Probability Distribution

The joint probability distribution is factorized using the chain rule and conditional independence:

$$P(X_1, X_2, ..., X_n) = \prod_{i=1}^{n} P(X_i | \text{Pa}(X_i))$$

### Python Implementation of Joint Distribution

```python
class BayesianNetwork:
    def __init__(self, structure, cpts):
        self.structure = structure
        self.cpts = cpts
    
    def get_joint_probability(self, assignment):
        """
        Calculate joint probability P(X1=x1, X2=x2, ..., Xn=xn)
        assignment: dict mapping variable names to their values
        """
        joint_prob = 1.0
        
        # Get nodes in topological order
        for node in self.structure.get_topological_order():
            if node in self.cpts:
                cpt = self.cpts[node]
                
                # Get parent values
                parent_values = tuple(assignment[parent] for parent in cpt.parents)
                
                # Get variable value
                var_value = assignment[node]
                
                # Get conditional probability
                prob = cpt.get_probability(parent_values, var_value)
                joint_prob *= prob
        
        return joint_prob
    
    def get_marginal_probability(self, query_vars, evidence=None):
        """
        Calculate marginal probability P(query_vars | evidence)
        Using enumeration (exact inference)
        """
        if evidence is None:
            evidence = {}
        
        # Get all variables
        all_vars = list(self.structure.nodes)
        
        # Variables to sum over (hidden variables)
        hidden_vars = [var for var in all_vars 
                      if var not in query_vars and var not in evidence]
        
        # Generate all possible assignments for hidden variables
        from itertools import product
        
        # Get possible values for each variable (simplified - assume binary)
        possible_values = {var: [True, False] for var in hidden_vars}
        
        total_prob = 0.0
        
        # Enumerate all possible assignments
        for hidden_assignment in product(*[possible_values[var] for var in hidden_vars]):
            # Create full assignment
            full_assignment = evidence.copy()
            full_assignment.update(dict(zip(hidden_vars, hidden_assignment)))
            
            # Add query variables (we'll sum over all their values)
            for query_assignment in product(*[possible_values[var] for var in query_vars]):
                full_assignment.update(dict(zip(query_vars, query_assignment)))
                
                # Calculate joint probability
                joint_prob = self.get_joint_probability(full_assignment)
                total_prob += joint_prob
        
        return total_prob
    
    def sample(self, n_samples=1000):
        """Generate samples from the Bayesian network"""
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            
            # Sample in topological order
            for node in self.structure.get_topological_order():
                if node in self.cpts:
                    cpt = self.cpts[node]
                    
                    # Get parent values
                    parent_values = tuple(sample[parent] for parent in cpt.parents)
                    
                    # Sample from conditional distribution
                    sample[node] = cpt.sample(parent_values)
            
            samples.append(sample)
        
        return samples

# Example: Complete Bayesian Network
def create_complete_weather_network():
    """Create a complete weather Bayesian network with structure and CPTs"""
    # Create structure
    structure = BayesianNetworkStructure()
    structure.add_node("Rain")
    structure.add_node("Sprinkler")
    structure.add_node("WetGrass")
    structure.add_edge("Rain", "Sprinkler")
    structure.add_edge("Rain", "WetGrass")
    structure.add_edge("Sprinkler", "WetGrass")
    
    # Create CPTs
    cpts = create_weather_cpts()
    
    # Create Bayesian network
    bn = BayesianNetwork(structure, cpts)
    
    return bn

# Test the complete network
if __name__ == "__main__":
    weather_bn = create_complete_weather_network()
    
    # Calculate joint probability
    assignment = {"Rain": True, "Sprinkler": False, "WetGrass": True}
    joint_prob = weather_bn.get_joint_probability(assignment)
    print(f"P(Rain=True, Sprinkler=False, WetGrass=True) = {joint_prob:.6f}")
    
    # Calculate marginal probability
    marginal_prob = weather_bn.get_marginal_probability(["WetGrass"], {"Rain": True})
    print(f"P(WetGrass | Rain=True) = {marginal_prob:.6f}")
    
    # Generate samples
    samples = weather_bn.sample(n_samples=1000)
    
    # Estimate probabilities from samples
    wet_grass_count = sum(1 for s in samples if s["WetGrass"])
    print(f"Estimated P(WetGrass=True) = {wet_grass_count/1000:.3f}")
```

## Parameters and Structure

### Structure Learning vs Parameter Learning

1. **Structure Learning**: Discovering the DAG structure from data
2. **Parameter Learning**: Estimating CPTs given a known structure

### Python Implementation of Parameter Learning

```python
import pandas as pd
from collections import Counter

class ParameterLearner:
    def __init__(self, structure):
        self.structure = structure
    
    def learn_parameters(self, data):
        """
        Learn CPTs from data
        data: pandas DataFrame with columns for each variable
        """
        cpts = {}
        
        for node in self.structure.get_topological_order():
            parents = self.structure.get_parents(node)
            cpt = ConditionalProbabilityTable(node, parents)
            
            if not parents:
                # Root node - learn marginal distribution
                value_counts = data[node].value_counts()
                total = len(data)
                
                for value, count in value_counts.items():
                    cpt.set_probability((), value, count / total)
            else:
                # Learn conditional distribution
                for parent_config in self._get_parent_configurations(data, parents):
                    # Filter data for this parent configuration
                    mask = True
                    for parent, value in zip(parents, parent_config):
                        mask &= (data[parent] == value)
                    
                    filtered_data = data[mask]
                    
                    if len(filtered_data) > 0:
                        # Count values of the target variable
                        value_counts = filtered_data[node].value_counts()
                        total = len(filtered_data)
                        
                        for value, count in value_counts.items():
                            cpt.set_probability(parent_config, value, count / total)
            
            cpts[node] = cpt
        
        return cpts
    
    def _get_parent_configurations(self, data, parents):
        """Get all unique parent value configurations in the data"""
        return data[parents].drop_duplicates().values

# Example: Learning Parameters from Data
def generate_synthetic_data(n_samples=1000):
    """Generate synthetic data from the weather network"""
    np.random.seed(42)
    
    # Generate Rain
    rain = np.random.choice([True, False], size=n_samples, p=[0.2, 0.8])
    
    # Generate Sprinkler based on Rain
    sprinkler = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        if rain[i]:
            sprinkler[i] = np.random.choice([True, False], p=[0.01, 0.99])
        else:
            sprinkler[i] = np.random.choice([True, False], p=[0.4, 0.6])
    
    # Generate WetGrass based on Rain and Sprinkler
    wetgrass = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        if rain[i] and sprinkler[i]:
            wetgrass[i] = np.random.choice([True, False], p=[0.99, 0.01])
        elif rain[i] and not sprinkler[i]:
            wetgrass[i] = np.random.choice([True, False], p=[0.9, 0.1])
        elif not rain[i] and sprinkler[i]:
            wetgrass[i] = np.random.choice([True, False], p=[0.9, 0.1])
        else:
            wetgrass[i] = np.random.choice([True, False], p=[0.0, 1.0])
    
    return pd.DataFrame({
        'Rain': rain,
        'Sprinkler': sprinkler,
        'WetGrass': wetgrass
    })

# Test parameter learning
if __name__ == "__main__":
    # Generate synthetic data
    data = generate_synthetic_data(1000)
    print("Generated data sample:")
    print(data.head())
    
    # Create structure
    structure = BayesianNetworkStructure()
    structure.add_node("Rain")
    structure.add_node("Sprinkler")
    structure.add_node("WetGrass")
    structure.add_edge("Rain", "Sprinkler")
    structure.add_edge("Rain", "WetGrass")
    structure.add_edge("Sprinkler", "WetGrass")
    
    # Learn parameters
    learner = ParameterLearner(structure)
    learned_cpts = learner.learn_parameters(data)
    
    print("\nLearned CPTs:")
    for var, cpt in learned_cpts.items():
        print(f"\n{var}:")
        for parent_config, probs in cpt.probabilities.items():
            print(f"  P({var} | {cpt.parents}={parent_config}) = {probs}")
```

## Key Takeaways

1. **Bayesian Network Structure**: A DAG representing variable dependencies
2. **CPTs**: Quantify the strength of dependencies between variables
3. **Joint Distribution**: Factorized using the chain rule and conditional independence
4. **Parameter Learning**: Estimating CPTs from data given a known structure
5. **Structure Learning**: Discovering the DAG structure from data (advanced topic)

## Exercises

1. Create a Bayesian network structure for a student performance model
2. Implement CPTs for a simple diagnostic system
3. Calculate joint probabilities for various assignments
4. Learn parameters from synthetic data and compare with true parameters
5. Implement marginal probability calculation using enumeration 