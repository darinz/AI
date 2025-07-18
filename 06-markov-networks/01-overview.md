# Overview of Markov Networks

## Introduction

Markov networks, also known as Markov random fields (MRFs), are undirected graphical models that represent complex probability distributions over sets of random variables. They are fundamental tools in probabilistic modeling, machine learning, and artificial intelligence for representing uncertainty and dependencies between variables.

## What are Markov Networks?

### Definition

A Markov network is an undirected graph G = (V, E) where:
- **V** is a set of nodes (vertices), each representing a random variable
- **E** is a set of edges representing dependencies between variables
- The joint probability distribution is defined in terms of potential functions over cliques

### Key Characteristics

- **Undirected Structure**: Unlike Bayesian networks, Markov networks use undirected edges
- **Symmetric Dependencies**: Relationships between variables are symmetric
- **Clique-Based**: Probability is defined in terms of functions over complete subgraphs (cliques)
- **Markov Property**: Variables are conditionally independent of non-neighbors given their neighbors

## Mathematical Foundation

### Joint Probability Distribution

The joint probability distribution in a Markov network is defined as:

P(X) = (1/Z) ∏ᵢ φᵢ(Cᵢ)

Where:
- **X** = (X₁, X₂, ..., Xₙ) is the set of random variables
- **φᵢ** are potential functions (non-negative functions)
- **Cᵢ** are cliques in the graph
- **Z** is the partition function (normalization constant)

### Partition Function

The partition function Z ensures that the distribution sums to 1:

Z = ∑ₓ ∏ᵢ φᵢ(Cᵢ)

This is often the most computationally expensive part of working with Markov networks.

### Potential Functions

Potential functions φᵢ assign non-negative values to configurations of variables in cliques. They encode the "compatibility" or "energy" of different variable assignments.

## Python Implementation

### Basic Markov Network Structure

```python
import numpy as np
from typing import List, Dict, Set, Tuple, Any
from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt

@dataclass
class PotentialFunction:
    """Represents a potential function over a set of variables."""
    variables: List[str]
    values: Dict[Tuple, float]  # Maps variable assignments to potential values
    
    def __post_init__(self):
        if not self.values:
            raise ValueError("Potential function must have at least one value")
    
    def get_value(self, assignment: Dict[str, Any]) -> float:
        """Get the potential value for a given assignment."""
        # Extract values for variables in this potential function
        key = tuple(assignment[var] for var in self.variables)
        return self.values.get(key, 0.0)

class MarkovNetwork:
    """Basic implementation of a Markov network."""
    
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]]):
        self.variables = variables
        self.domains = domains
        self.potentials: List[PotentialFunction] = []
        self.graph = nx.Graph()
        
        # Initialize graph with nodes
        for var in variables:
            self.graph.add_node(var)
    
    def add_potential(self, potential: PotentialFunction):
        """Add a potential function to the network."""
        self.potentials.append(potential)
        
        # Add edges to graph based on potential function
        for i, var1 in enumerate(potential.variables):
            for var2 in potential.variables[i+1:]:
                self.graph.add_edge(var1, var2)
    
    def get_cliques(self) -> List[List[str]]:
        """Get all cliques in the graph."""
        return list(nx.find_cliques(self.graph))
    
    def compute_joint_probability(self, assignment: Dict[str, Any]) -> float:
        """Compute the joint probability of an assignment."""
        # Compute product of potential functions
        product = 1.0
        for potential in self.potentials:
            product *= potential.get_value(assignment)
        
        # Note: This is unnormalized probability
        # Computing the partition function Z is often intractable
        return product
    
    def compute_partition_function(self) -> float:
        """Compute the partition function Z (exponential in number of variables)."""
        Z = 0.0
        
        # Generate all possible assignments
        assignments = self._generate_all_assignments()
        
        for assignment in assignments:
            Z += self.compute_joint_probability(assignment)
        
        return Z
    
    def _generate_all_assignments(self) -> List[Dict[str, Any]]:
        """Generate all possible assignments of variables."""
        if not self.variables:
            return [{}]
        
        var = self.variables[0]
        remaining_vars = self.variables[1:]
        
        assignments = []
        for value in self.domains[var]:
            sub_assignments = self._generate_all_assignments_helper(remaining_vars)
            for sub_assignment in sub_assignments:
                assignment = {var: value}
                assignment.update(sub_assignment)
                assignments.append(assignment)
        
        return assignments
    
    def _generate_all_assignments_helper(self, variables: List[str]) -> List[Dict[str, Any]]:
        """Helper function to generate assignments for remaining variables."""
        if not variables:
            return [{}]
        
        var = variables[0]
        remaining_vars = variables[1:]
        
        assignments = []
        for value in self.domains[var]:
            sub_assignments = self._generate_all_assignments_helper(remaining_vars)
            for sub_assignment in sub_assignments:
                assignment = {var: value}
                assignment.update(sub_assignment)
                assignments.append(assignment)
        
        return assignments
    
    def visualize(self, assignment: Dict[str, Any] = None):
        """Visualize the Markov network."""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=2000, node_color='lightblue')
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos)
        
        # Draw labels
        if assignment:
            labels = {var: f"{var}={assignment[var]}" for var in self.variables}
        else:
            labels = {var: var for var in self.variables}
        
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=12, font_weight='bold')
        
        plt.title("Markov Network")
        plt.axis('off')
        plt.show()

# Example: Simple Markov Network
def create_simple_markov_network():
    """Create a simple Markov network example."""
    # Define variables and their domains
    variables = ['A', 'B', 'C']
    domains = {
        'A': [0, 1],
        'B': [0, 1],
        'C': [0, 1]
    }
    
    # Create Markov network
    mn = MarkovNetwork(variables, domains)
    
    # Add potential functions
    # Potential for A and B (favoring same values)
    ab_potential = PotentialFunction(
        variables=['A', 'B'],
        values={
            (0, 0): 2.0,  # High potential when both are 0
            (0, 1): 0.5,  # Low potential when different
            (1, 0): 0.5,  # Low potential when different
            (1, 1): 2.0   # High potential when both are 1
        }
    )
    
    # Potential for B and C (favoring same values)
    bc_potential = PotentialFunction(
        variables=['B', 'C'],
        values={
            (0, 0): 2.0,
            (0, 1): 0.5,
            (1, 0): 0.5,
            (1, 1): 2.0
        }
    )
    
    # Potential for A and C (favoring different values)
    ac_potential = PotentialFunction(
        variables=['A', 'C'],
        values={
            (0, 0): 0.5,  # Low potential when same
            (0, 1): 2.0,  # High potential when different
            (1, 0): 2.0,  # High potential when different
            (1, 1): 0.5   # Low potential when same
        }
    )
    
    mn.add_potential(ab_potential)
    mn.add_potential(bc_potential)
    mn.add_potential(ac_potential)
    
    return mn

def test_markov_network():
    """Test the Markov network implementation."""
    mn = create_simple_markov_network()
    
    print("Markov Network Structure:")
    print(f"Variables: {mn.variables}")
    print(f"Cliques: {mn.get_cliques()}")
    print(f"Number of potentials: {len(mn.potentials)}")
    
    # Test joint probability computation
    test_assignment = {'A': 0, 'B': 0, 'C': 1}
    unnormalized_prob = mn.compute_joint_probability(test_assignment)
    print(f"\nUnnormalized probability of {test_assignment}: {unnormalized_prob}")
    
    # Compute partition function (for small networks)
    Z = mn.compute_partition_function()
    print(f"Partition function Z: {Z}")
    
    # Compute normalized probability
    normalized_prob = unnormalized_prob / Z
    print(f"Normalized probability: {normalized_prob}")
    
    # Visualize the network
    mn.visualize(test_assignment)
    
    return mn

if __name__ == "__main__":
    test_markov_network()
```

## Markov Properties

### Local Markov Property

A variable is conditionally independent of all other variables given its neighbors in the graph.

**Mathematical Definition**: For any variable Xᵢ and its neighbors N(Xᵢ):
Xᵢ ⊥ {Xⱼ : Xⱼ ∉ N(Xᵢ)} | N(Xᵢ)

### Pairwise Markov Property

Any two non-adjacent variables are conditionally independent given all other variables.

**Mathematical Definition**: For any non-adjacent variables Xᵢ and Xⱼ:
Xᵢ ⊥ Xⱼ | {Xₖ : k ≠ i, j}

### Global Markov Property

Variables in separated sets are conditionally independent given the separating set.

**Mathematical Definition**: If sets A, B, and S are such that S separates A from B in the graph, then:
A ⊥ B | S

```python
class MarkovProperties:
    """Implementation of Markov properties for testing conditional independence."""
    
    def __init__(self, markov_network: MarkovNetwork):
        self.mn = markov_network
    
    def get_neighbors(self, variable: str) -> Set[str]:
        """Get neighbors of a variable in the graph."""
        return set(self.mn.graph.neighbors(variable))
    
    def test_local_markov_property(self, variable: str, assignment: Dict[str, Any]) -> bool:
        """Test if local Markov property holds for a variable."""
        neighbors = self.get_neighbors(variable)
        non_neighbors = set(self.mn.variables) - {variable} - neighbors
        
        # For simplicity, we'll just check if the property structure is correct
        # In practice, this would require computing conditional probabilities
        return len(non_neighbors) >= 0  # Placeholder
    
    def test_pairwise_markov_property(self, var1: str, var2: str) -> bool:
        """Test if pairwise Markov property holds for two variables."""
        # Check if variables are non-adjacent
        return not self.mn.graph.has_edge(var1, var2)
    
    def test_global_markov_property(self, set_a: Set[str], set_b: Set[str], 
                                   separating_set: Set[str]) -> bool:
        """Test if global Markov property holds for given sets."""
        # Check if separating_set separates set_a from set_b
        # This is a simplified check - in practice would require path analysis
        return True  # Placeholder
```

## Comparison with Other Models

### Markov Networks vs Bayesian Networks

| Aspect | Markov Networks | Bayesian Networks |
|--------|----------------|-------------------|
| **Structure** | Undirected | Directed |
| **Dependencies** | Symmetric | Asymmetric |
| **Conditional Independence** | Based on graph separation | Based on d-separation |
| **Parameterization** | Potential functions | Conditional probability tables |
| **Inference** | Often more complex | Often simpler |

### When to Use Markov Networks

- **Symmetric Dependencies**: When relationships between variables are naturally symmetric
- **Undirected Structure**: When causal direction is unclear or irrelevant
- **Complex Interactions**: When variables interact in complex, non-hierarchical ways
- **Spatial/Temporal Data**: For modeling spatial or temporal dependencies

## Applications

### Computer Vision

- **Image Segmentation**: Modeling pixel dependencies
- **Object Recognition**: Modeling object-part relationships
- **Stereo Vision**: Modeling depth consistency

### Natural Language Processing

- **Part-of-Speech Tagging**: Modeling word-tag dependencies
- **Named Entity Recognition**: Modeling entity boundaries
- **Dependency Parsing**: Modeling syntactic relationships

### Bioinformatics

- **Protein Structure Prediction**: Modeling amino acid interactions
- **Gene Regulation**: Modeling regulatory networks
- **Sequence Alignment**: Modeling evolutionary relationships

### Social Network Analysis

- **Community Detection**: Modeling group affiliations
- **Influence Modeling**: Modeling information spread
- **Link Prediction**: Modeling friendship formation

## Summary

Markov networks provide a powerful framework for modeling complex probability distributions:

1. **Flexible Structure**: Can represent complex, non-linear dependencies
2. **Symmetric Relationships**: Natural for modeling mutual influences
3. **Graphical Interpretation**: Intuitive representation of dependencies
4. **Mathematical Rigor**: Well-founded probabilistic framework

Key concepts:
- **Undirected Graph Structure**: Represents symmetric dependencies
- **Potential Functions**: Encode compatibility between variable assignments
- **Markov Properties**: Enable efficient inference and learning
- **Partition Function**: Normalizes the distribution (often computationally expensive)

Understanding Markov networks is essential for:
- Modeling complex systems with symmetric dependencies
- Performing inference in undirected graphical models
- Learning structure and parameters from data
- Building probabilistic AI systems 