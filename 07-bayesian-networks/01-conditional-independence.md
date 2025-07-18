# Conditional Independence in Bayesian Networks

## Overview

Conditional independence is a fundamental concept in Bayesian networks that allows us to represent complex joint probability distributions efficiently. Understanding when variables are conditionally independent given other variables is crucial for both understanding and constructing Bayesian networks.

## What is Conditional Independence?

Two random variables X and Y are **conditionally independent** given a third variable Z if knowing the value of Z makes X and Y independent. Mathematically, this is expressed as:

$$P(X, Y | Z) = P(X | Z) \cdot P(Y | Z)$$

Or equivalently:

$$P(X | Y, Z) = P(X | Z)$$

This means that once we know Z, learning about Y provides no additional information about X.

## D-Separation: Graphical Criteria for Conditional Independence

D-separation is a graphical criterion that allows us to determine conditional independence directly from the structure of a Bayesian network.

### D-Separation Rules

1. **Chain Structure** (X → Z → Y): X and Y are d-separated given Z
2. **Fork Structure** (X ← Z → Y): X and Y are d-separated given Z  
3. **Collider Structure** (X → Z ← Y): X and Y are d-separated given Z, but NOT d-separated given Z and any descendant of Z

### Python Implementation of D-Separation

```python
import networkx as nx
import matplotlib.pyplot as plt

class BayesianNetwork:
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def add_edge(self, from_node, to_node):
        """Add a directed edge from from_node to to_node"""
        self.graph.add_edge(from_node, to_node)
    
    def is_d_separated(self, x, y, z):
        """
        Check if nodes x and y are d-separated given evidence set z
        """
        # Get all paths between x and y
        paths = list(nx.all_simple_paths(self.graph, x, y))
        
        for path in paths:
            if not self._is_path_blocked(path, z):
                return False
        return True
    
    def _is_path_blocked(self, path, evidence):
        """
        Check if a path is blocked given evidence
        """
        for i in range(1, len(path) - 1):
            prev_node = path[i-1]
            curr_node = path[i]
            next_node = path[i+1]
            
            # Chain or Fork structure
            if (prev_node, curr_node) in self.graph.edges() and \
               (curr_node, next_node) in self.graph.edges():
                # Chain: prev -> curr -> next
                if curr_node in evidence:
                    return True  # Path is blocked
            elif (curr_node, prev_node) in self.graph.edges() and \
                 (curr_node, next_node) in self.graph.edges():
                # Fork: prev <- curr -> next
                if curr_node in evidence:
                    return True  # Path is blocked
            elif (prev_node, curr_node) in self.graph.edges() and \
                 (next_node, curr_node) in self.graph.edges():
                # Collider: prev -> curr <- next
                if curr_node not in evidence and \
                   not self._has_evidence_descendant(curr_node, evidence):
                    return True  # Path is blocked
        
        return False
    
    def _has_evidence_descendant(self, node, evidence):
        """Check if node has any descendant in evidence"""
        descendants = nx.descendants(self.graph, node)
        return bool(descendants.intersection(set(evidence)))
    
    def visualize(self):
        """Visualize the Bayesian network"""
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', 
                node_size=2000, font_size=10, font_weight='bold',
                arrows=True, arrowsize=20)
        plt.title("Bayesian Network Structure")
        plt.show()

# Example: Medical Diagnosis Network
def medical_diagnosis_example():
    """Example demonstrating conditional independence in medical diagnosis"""
    bn = BayesianNetwork()
    
    # Add nodes and edges
    bn.add_edge("Disease", "Symptom1")
    bn.add_edge("Disease", "Symptom2")
    bn.add_edge("Age", "Disease")
    bn.add_edge("Symptom1", "Test1")
    bn.add_edge("Symptom2", "Test2")
    
    print("Medical Diagnosis Bayesian Network:")
    bn.visualize()
    
    # Check conditional independence
    print(f"Symptom1 and Symptom2 d-separated given Disease: {bn.is_d_separated('Symptom1', 'Symptom2', ['Disease'])}")
    print(f"Symptom1 and Symptom2 d-separated given empty set: {bn.is_d_separated('Symptom1', 'Symptom2', [])}")
    print(f"Test1 and Test2 d-separated given Disease: {bn.is_d_separated('Test1', 'Test2', ['Disease'])}")
    
    return bn

# Run the example
if __name__ == "__main__":
    medical_network = medical_diagnosis_example()
```

## Markov Properties

Bayesian networks satisfy three important Markov properties:

### 1. Local Markov Property
Each node is conditionally independent of its non-descendants given its parents:

$$X_i \perp \text{NonDescendants}(X_i) | \text{Parents}(X_i)$$

### 2. Global Markov Property
If two sets of nodes X and Y are d-separated by a third set Z, then X and Y are conditionally independent given Z.

### 3. Pairwise Markov Property
Any two non-adjacent nodes are conditionally independent given all other nodes.

## Factorization of Joint Probability

The conditional independence structure of a Bayesian network allows us to factorize the joint probability distribution efficiently:

$$P(X_1, X_2, ..., X_n) = \prod_{i=1}^{n} P(X_i | \text{Parents}(X_i))$$

### Python Implementation of Factorization

```python
import numpy as np
from collections import defaultdict

class BayesianNetworkFactorized:
    def __init__(self):
        self.nodes = []
        self.parents = defaultdict(list)
        self.cpts = {}  # Conditional Probability Tables
    
    def add_node(self, node, parents=None, cpt=None):
        """Add a node with its parents and conditional probability table"""
        self.nodes.append(node)
        if parents:
            self.parents[node] = parents
        if cpt is not None:
            self.cpts[node] = cpt
    
    def get_joint_probability(self, assignment):
        """
        Calculate joint probability using factorization
        assignment: dict mapping node names to their values
        """
        joint_prob = 1.0
        
        for node in self.nodes:
            if node in self.parents:
                # Get parent values
                parent_values = tuple(assignment[parent] for parent in self.parents[node])
                # Get conditional probability
                node_value = assignment[node]
                prob = self.cpts[node][parent_values][node_value]
            else:
                # Root node - marginal probability
                prob = self.cpts[node][node_value]
            
            joint_prob *= prob
        
        return joint_prob
    
    def sample(self, n_samples=1000):
        """Generate samples from the Bayesian network"""
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            # Sample in topological order (parents before children)
            for node in self.nodes:
                if node in self.parents:
                    # Get parent values
                    parent_values = tuple(sample[parent] for parent in self.parents[node])
                    # Sample from conditional distribution
                    probs = self.cpts[node][parent_values]
                    sample[node] = np.random.choice(list(probs.keys()), p=list(probs.values()))
                else:
                    # Root node - sample from marginal distribution
                    probs = self.cpts[node]
                    sample[node] = np.random.choice(list(probs.keys()), p=list(probs.values()))
            
            samples.append(sample)
        
        return samples

# Example: Simple Bayesian Network
def simple_bayesian_network_example():
    """Example of a simple Bayesian network with factorization"""
    bn = BayesianNetworkFactorized()
    
    # Add nodes with their CPTs
    # P(Rain) = 0.2
    bn.add_node("Rain", cpt={True: 0.2, False: 0.8})
    
    # P(Sprinkler | Rain)
    bn.add_node("Sprinkler", parents=["Rain"], 
                cpt={(True,): {True: 0.01, False: 0.99},
                     (False,): {True: 0.4, False: 0.6}})
    
    # P(WetGrass | Rain, Sprinkler)
    bn.add_node("WetGrass", parents=["Rain", "Sprinkler"],
                cpt={(True, True): {True: 0.99, False: 0.01},
                     (True, False): {True: 0.9, False: 0.1},
                     (False, True): {True: 0.9, False: 0.1},
                     (False, False): {True: 0.0, False: 1.0}})
    
    # Calculate joint probability
    assignment = {"Rain": True, "Sprinkler": False, "WetGrass": True}
    joint_prob = bn.get_joint_probability(assignment)
    print(f"P(Rain=True, Sprinkler=False, WetGrass=True) = {joint_prob:.6f}")
    
    # Generate samples
    samples = bn.sample(n_samples=1000)
    
    # Estimate probabilities from samples
    rain_count = sum(1 for s in samples if s["Rain"])
    print(f"Estimated P(Rain=True) = {rain_count/1000:.3f}")
    
    return bn

# Run the example
if __name__ == "__main__":
    simple_network = simple_bayesian_network_example()
```

## Practical Applications

### 1. Medical Diagnosis
In medical diagnosis, symptoms are often conditionally independent given the disease:
- P(Symptom1, Symptom2 | Disease) = P(Symptom1 | Disease) × P(Symptom2 | Disease)

### 2. Sensor Networks
In sensor networks, sensor readings may be conditionally independent given the true state:
- P(Sensor1, Sensor2 | TrueState) = P(Sensor1 | TrueState) × P(Sensor2 | TrueState)

### 3. Natural Language Processing
In language models, words may be conditionally independent given their context:
- P(Word1, Word2 | Context) = P(Word1 | Context) × P(Word2 | Context)

## Key Takeaways

1. **Conditional independence** is the foundation of Bayesian networks
2. **D-separation** provides graphical criteria for determining conditional independence
3. **Markov properties** ensure the network structure reflects independence relationships
4. **Factorization** enables efficient representation and computation of joint probabilities
5. **Understanding conditional independence** is crucial for building accurate Bayesian network models

## Exercises

1. Draw a Bayesian network for a simple weather model and identify all conditional independence relationships
2. Implement the d-separation algorithm for a given network structure
3. Calculate joint probabilities using factorization for a small Bayesian network
4. Generate samples from a Bayesian network and verify the conditional independence properties 