# Conditional Independence in Markov Networks

## Introduction

Conditional independence is a fundamental concept in Markov networks that enables efficient inference and learning. It captures the intuitive idea that certain variables are independent of each other once we know the values of other variables.

## What is Conditional Independence?

### Definition

Two random variables X and Y are conditionally independent given a third variable Z (denoted X ⊥ Y | Z) if:

P(X, Y | Z) = P(X | Z) P(Y | Z)

This means that once we know the value of Z, knowing the value of X provides no additional information about Y, and vice versa.

### Intuition

Conditional independence captures the idea that:
- X and Y may be dependent in general
- But this dependence is "explained away" by Z
- Once we know Z, X and Y become independent

## Mathematical Foundation

### Markov Properties

Markov networks satisfy three equivalent Markov properties:

#### 1. Pairwise Markov Property

Any two non-adjacent variables are conditionally independent given all other variables.

**Mathematical Definition**: For any non-adjacent variables Xᵢ and Xⱼ:
Xᵢ ⊥ Xⱼ | {Xₖ : k ≠ i, j}

#### 2. Local Markov Property

A variable is conditionally independent of all other variables given its neighbors.

**Mathematical Definition**: For any variable Xᵢ and its neighbors N(Xᵢ):
Xᵢ ⊥ {Xⱼ : Xⱼ ∉ N(Xᵢ)} | N(Xᵢ)

#### 3. Global Markov Property

Variables in separated sets are conditionally independent given the separating set.

**Mathematical Definition**: If sets A, B, and S are such that S separates A from B in the graph, then:
A ⊥ B | S

### Separation in Undirected Graphs

In an undirected graph, a set S separates sets A and B if every path from a node in A to a node in B passes through at least one node in S.

## Python Implementation

### Basic Conditional Independence Testing

```python
import numpy as np
from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict

class ConditionalIndependence:
    """Tools for testing conditional independence in Markov networks."""
    
    def __init__(self, markov_network):
        self.mn = markov_network
        self.graph = markov_network.graph
    
    def test_pairwise_independence(self, var1: str, var2: str) -> bool:
        """Test if two variables are non-adjacent (pairwise Markov property)."""
        return not self.graph.has_edge(var1, var2)
    
    def test_local_independence(self, variable: str) -> Set[str]:
        """Find variables that are locally independent of the given variable."""
        neighbors = set(self.graph.neighbors(variable))
        all_variables = set(self.mn.variables)
        
        # Variables that are not neighbors are locally independent
        return all_variables - {variable} - neighbors
    
    def test_global_independence(self, set_a: Set[str], set_b: Set[str], 
                                separating_set: Set[str]) -> bool:
        """Test if separating_set separates set_a from set_b."""
        # Check if all paths from A to B go through the separating set
        for var_a in set_a:
            for var_b in set_b:
                if not self._is_separated(var_a, var_b, separating_set):
                    return False
        return True
    
    def _is_separated(self, var_a: str, var_b: str, separating_set: Set[str]) -> bool:
        """Check if var_a and var_b are separated by separating_set."""
        # Use networkx to find all simple paths
        try:
            paths = list(nx.all_simple_paths(self.graph, var_a, var_b))
        except nx.NetworkXNoPath:
            return True  # No path exists, so they are separated
        
        # Check if all paths go through the separating set
        for path in paths:
            # Path goes from var_a to var_b, check intermediate nodes
            intermediate_nodes = set(path[1:-1])  # Exclude start and end
            if not intermediate_nodes.intersection(separating_set):
                return False  # Found a path that doesn't go through separating set
        
        return True
    
    def find_separating_sets(self, var_a: str, var_b: str) -> List[Set[str]]:
        """Find all minimal separating sets between two variables."""
        # This is a simplified implementation
        # In practice, more sophisticated algorithms would be used
        
        all_variables = set(self.mn.variables) - {var_a, var_b}
        separating_sets = []
        
        # Try all possible subsets (for small networks)
        for size in range(len(all_variables) + 1):
            for subset in self._get_subsets(all_variables, size):
                if self._is_separated(var_a, var_b, subset):
                    separating_sets.append(subset)
        
        # Return minimal separating sets
        minimal_sets = []
        for s in separating_sets:
            is_minimal = True
            for other_s in separating_sets:
                if other_s != s and other_s.issubset(s):
                    is_minimal = False
                    break
            if is_minimal:
                minimal_sets.append(s)
        
        return minimal_sets
    
    def _get_subsets(self, elements: Set[str], size: int) -> List[Set[str]]:
        """Get all subsets of given size."""
        elements_list = list(elements)
        if size > len(elements_list):
            return []
        
        from itertools import combinations
        return [set(combo) for combo in combinations(elements_list, size)]

# Example: Testing conditional independence
def create_test_network():
    """Create a test Markov network for conditional independence analysis."""
    variables = ['A', 'B', 'C', 'D', 'E']
    domains = {
        'A': [0, 1],
        'B': [0, 1],
        'C': [0, 1],
        'D': [0, 1],
        'E': [0, 1]
    }
    
    # Create network with structure: A -- B -- C -- D -- E
    mn = MarkovNetwork(variables, domains)
    
    # Add potentials to create the chain structure
    potentials = [
        PotentialFunction(['A', 'B'], {(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): 1.0}),
        PotentialFunction(['B', 'C'], {(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): 1.0}),
        PotentialFunction(['C', 'D'], {(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): 1.0}),
        PotentialFunction(['D', 'E'], {(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): 1.0})
    ]
    
    for potential in potentials:
        mn.add_potential(potential)
    
    return mn

def test_conditional_independence():
    """Test conditional independence properties."""
    mn = create_test_network()
    ci = ConditionalIndependence(mn)
    
    print("Conditional Independence Analysis:")
    print("=" * 40)
    
    # Test pairwise Markov property
    print("\nPairwise Markov Property:")
    for i, var1 in enumerate(mn.variables):
        for var2 in mn.variables[i+1:]:
            is_independent = ci.test_pairwise_independence(var1, var2)
            print(f"  {var1} ⊥ {var2} | others: {is_independent}")
    
    # Test local Markov property
    print("\nLocal Markov Property:")
    for variable in mn.variables:
        independent_vars = ci.test_local_independence(variable)
        print(f"  {variable} ⊥ {independent_vars} | neighbors")
    
    # Test global Markov property
    print("\nGlobal Markov Property:")
    # A and E should be independent given {B, C, D}
    is_separated = ci.test_global_independence({'A'}, {'E'}, {'B', 'C', 'D'})
    print(f"  A ⊥ E | {{B, C, D}}: {is_separated}")
    
    # A and C should be independent given B
    is_separated = ci.test_global_independence({'A'}, {'C'}, {'B'})
    print(f"  A ⊥ C | {{B}}: {is_separated}")
    
    # Find separating sets
    print("\nSeparating Sets:")
    separating_sets = ci.find_separating_sets('A', 'E')
    print(f"  Minimal separating sets between A and E: {separating_sets}")
    
    return mn, ci

if __name__ == "__main__":
    test_conditional_independence()
```

## Statistical Testing of Conditional Independence

### Chi-Square Test

```python
class StatisticalIndependenceTest:
    """Statistical tests for conditional independence."""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        self.variables = list(data[0].keys()) if data else []
    
    def chi_square_test(self, var1: str, var2: str, conditioning_vars: List[str] = None) -> Tuple[float, float]:
        """Perform chi-square test for conditional independence."""
        if conditioning_vars is None:
            conditioning_vars = []
        
        # Group data by conditioning variables
        grouped_data = self._group_by_conditioning(var1, var2, conditioning_vars)
        
        total_chi_square = 0.0
        total_df = 0
        
        for condition, group_data in grouped_data.items():
            if len(group_data) < 10:  # Need sufficient data
                continue
            
            # Create contingency table
            contingency_table = self._create_contingency_table(group_data, var1, var2)
            
            # Perform chi-square test
            chi_square, p_value, df = self._chi_square_contingency(contingency_table)
            
            total_chi_square += chi_square
            total_df += df
        
        # Combine p-values (simplified approach)
        combined_p_value = self._combine_p_values([p for _, p, _ in grouped_data.values()])
        
        return total_chi_square, combined_p_value
    
    def _group_by_conditioning(self, var1: str, var2: str, conditioning_vars: List[str]) -> Dict[Tuple, List[Dict[str, Any]]]:
        """Group data by conditioning variable values."""
        grouped = defaultdict(list)
        
        for sample in self.data:
            condition = tuple(sample[var] for var in conditioning_vars)
            grouped[condition].append(sample)
        
        return dict(grouped)
    
    def _create_contingency_table(self, data: List[Dict[str, Any]], var1: str, var2: str) -> Dict[Tuple, int]:
        """Create contingency table for two variables."""
        table = defaultdict(int)
        
        for sample in data:
            key = (sample[var1], sample[var2])
            table[key] += 1
        
        return dict(table)
    
    def _chi_square_contingency(self, table: Dict[Tuple, int]) -> Tuple[float, float, int]:
        """Compute chi-square statistic for contingency table."""
        # Get unique values
        values1 = set(key[0] for key in table.keys())
        values2 = set(key[1] for key in table.keys())
        
        # Compute row and column totals
        row_totals = defaultdict(int)
        col_totals = defaultdict(int)
        total = 0
        
        for (val1, val2), count in table.items():
            row_totals[val1] += count
            col_totals[val2] += count
            total += count
        
        # Compute expected frequencies
        chi_square = 0.0
        for (val1, val2), observed in table.items():
            expected = (row_totals[val1] * col_totals[val2]) / total
            if expected > 0:
                chi_square += (observed - expected) ** 2 / expected
        
        # Degrees of freedom
        df = (len(values1) - 1) * (len(values2) - 1)
        
        # P-value (simplified - in practice, use scipy.stats)
        p_value = 0.05  # Placeholder
        
        return chi_square, p_value, df
    
    def _combine_p_values(self, p_values: List[float]) -> float:
        """Combine p-values from multiple tests."""
        # Fisher's method (simplified)
        if not p_values:
            return 1.0
        
        # Log-sum method
        log_sum = sum(-2 * np.log(p) for p in p_values if p > 0)
        combined_p = 1.0  # Placeholder - would use chi-square distribution
        
        return combined_p

# Example: Statistical testing
def create_test_data():
    """Create test data for statistical independence testing."""
    # Simulate data where A and C are independent given B
    data = []
    
    for _ in range(1000):
        # Generate B first
        b = np.random.choice([0, 1])
        
        # Generate A and C independently given B
        if b == 0:
            a = np.random.choice([0, 1], p=[0.7, 0.3])
            c = np.random.choice([0, 1], p=[0.6, 0.4])
        else:
            a = np.random.choice([0, 1], p=[0.3, 0.7])
            c = np.random.choice([0, 1], p=[0.4, 0.6])
        
        data.append({'A': a, 'B': b, 'C': c})
    
    return data

def test_statistical_independence():
    """Test statistical independence."""
    data = create_test_data()
    test = StatisticalIndependenceTest(data)
    
    print("Statistical Independence Testing:")
    print("=" * 35)
    
    # Test A and C without conditioning
    chi_square, p_value = test.chi_square_test('A', 'C')
    print(f"A ⊥ C: chi-square={chi_square:.3f}, p-value={p_value:.3f}")
    
    # Test A and C given B
    chi_square, p_value = test.chi_square_test('A', 'C', ['B'])
    print(f"A ⊥ C | B: chi-square={chi_square:.3f}, p-value={p_value:.3f}")
    
    return test
```

## Inference Using Conditional Independence

### Variable Elimination

```python
class VariableElimination:
    """Variable elimination using conditional independence."""
    
    def __init__(self, markov_network):
        self.mn = markov_network
    
    def eliminate_variable(self, variable: str, evidence: Dict[str, Any] = None) -> Dict[Any, float]:
        """Eliminate a variable by summing over its values."""
        if evidence is None:
            evidence = {}
        
        # Get all potentials involving this variable
        relevant_potentials = []
        for potential in self.mn.potentials:
            if variable in potential.variables:
                relevant_potentials.append(potential)
        
        # Create combined potential
        combined_potential = self._combine_potentials(relevant_potentials, evidence)
        
        # Sum out the variable
        result = {}
        for value in self.mn.domains[variable]:
            # Create assignment with this value
            assignment = evidence.copy()
            assignment[variable] = value
            
            # Compute potential value
            potential_value = combined_potential.get_value(assignment)
            
            # Add to result
            for other_assignment, other_value in combined_potential.values.items():
                if self._assignments_compatible(other_assignment, assignment, [variable]):
                    key = self._remove_variable_from_assignment(other_assignment, variable)
                    result[key] = result.get(key, 0.0) + other_value
        
        return result
    
    def _combine_potentials(self, potentials: List[PotentialFunction], 
                           evidence: Dict[str, Any]) -> PotentialFunction:
        """Combine multiple potentials into one."""
        if not potentials:
            return PotentialFunction([], {(): 1.0})
        
        # Get all variables involved
        all_variables = set()
        for potential in potentials:
            all_variables.update(potential.variables)
        
        # Remove evidence variables
        query_variables = list(all_variables - set(evidence.keys()))
        
        # Create combined potential
        combined_values = {}
        
        # Generate all assignments for query variables
        for assignment in self._generate_assignments(query_variables):
            # Add evidence
            full_assignment = assignment.copy()
            full_assignment.update(evidence)
            
            # Compute product of potentials
            value = 1.0
            for potential in potentials:
                value *= potential.get_value(full_assignment)
            
            combined_values[tuple(sorted(assignment.items()))] = value
        
        return PotentialFunction(query_variables, combined_values)
    
    def _assignments_compatible(self, assignment1: Tuple, assignment2: Dict[str, Any], 
                               exclude_vars: List[str]) -> bool:
        """Check if two assignments are compatible (excluding specified variables)."""
        # Convert assignment1 to dict
        assignment1_dict = dict(assignment1)
        
        for var, value in assignment2.items():
            if var not in exclude_vars and var in assignment1_dict:
                if assignment1_dict[var] != value:
                    return False
        
        return True
    
    def _remove_variable_from_assignment(self, assignment: Tuple, variable: str) -> Tuple:
        """Remove a variable from an assignment tuple."""
        assignment_dict = dict(assignment)
        if variable in assignment_dict:
            del assignment_dict[variable]
        return tuple(sorted(assignment_dict.items()))
    
    def _generate_assignments(self, variables: List[str]) -> List[Dict[str, Any]]:
        """Generate all possible assignments for given variables."""
        if not variables:
            return [{}]
        
        var = variables[0]
        remaining = variables[1:]
        
        assignments = []
        for value in self.mn.domains[var]:
            sub_assignments = self._generate_assignments(remaining)
            for sub_assignment in sub_assignments:
                assignment = {var: value}
                assignment.update(sub_assignment)
                assignments.append(assignment)
        
        return assignments

# Example: Variable elimination
def test_variable_elimination():
    """Test variable elimination."""
    mn = create_test_network()
    eliminator = VariableElimination(mn)
    
    print("Variable Elimination:")
    print("=" * 20)
    
    # Eliminate variable B
    result = eliminator.eliminate_variable('B')
    print(f"Eliminated B, result has {len(result)} entries")
    
    return eliminator
```

## Learning Structure Using Conditional Independence

### Constraint-Based Learning

```python
class ConstraintBasedLearning:
    """Learn Markov network structure using conditional independence tests."""
    
    def __init__(self, data: List[Dict[str, Any]], alpha: float = 0.05):
        self.data = data
        self.alpha = alpha
        self.variables = list(data[0].keys()) if data else []
        self.independence_tests = StatisticalIndependenceTest(data)
    
    def learn_structure(self) -> nx.Graph:
        """Learn the structure of a Markov network."""
        # Initialize complete graph
        graph = nx.complete_graph(self.variables)
        
        # Test all pairs of variables
        for i, var1 in enumerate(self.variables):
            for var2 in self.variables[i+1:]:
                # Test independence with different conditioning sets
                if self._test_independence_with_conditioning(var1, var2, graph):
                    # Remove edge if variables are independent
                    graph.remove_edge(var1, var2)
        
        return graph
    
    def _test_independence_with_conditioning(self, var1: str, var2: str, graph: nx.Graph) -> bool:
        """Test independence with different conditioning sets."""
        # Get neighbors of both variables
        neighbors1 = set(graph.neighbors(var1)) - {var2}
        neighbors2 = set(graph.neighbors(var2)) - {var1}
        
        # Test with different conditioning sets
        conditioning_sets = [
            [],  # No conditioning
            list(neighbors1),  # Condition on neighbors of var1
            list(neighbors2),  # Condition on neighbors of var2
            list(neighbors1 | neighbors2)  # Condition on all neighbors
        ]
        
        for conditioning_set in conditioning_sets:
            chi_square, p_value = self.independence_tests.chi_square_test(var1, var2, conditioning_set)
            
            if p_value > self.alpha:
                # Variables are independent given this conditioning set
                return True
        
        return False

# Example: Structure learning
def test_structure_learning():
    """Test structure learning from data."""
    # Create data from a known structure
    data = create_test_data()
    learner = ConstraintBasedLearning(data)
    
    print("Structure Learning:")
    print("=" * 18)
    
    # Learn structure
    learned_graph = learner.learn_structure()
    
    print(f"Learned graph has {learned_graph.number_of_edges()} edges")
    print(f"Edges: {list(learned_graph.edges())}")
    
    return learner, learned_graph
```

## Summary

Conditional independence is a fundamental concept in Markov networks:

1. **Mathematical Foundation**: Three equivalent Markov properties
2. **Graphical Interpretation**: Separation in undirected graphs
3. **Statistical Testing**: Chi-square tests and other methods
4. **Computational Applications**: Variable elimination and structure learning

Key insights:
- **Local Structure**: Conditional independence enables efficient local computations
- **Global Structure**: Separation properties guide global inference
- **Learning**: Independence tests can be used to learn network structure
- **Inference**: Conditional independence enables tractable inference algorithms

Understanding conditional independence is essential for:
- Performing efficient inference in Markov networks
- Learning network structure from data
- Understanding the relationship between graph structure and probability distributions
- Building scalable probabilistic models 