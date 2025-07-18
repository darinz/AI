# Overview of Constraint Satisfaction Problems

## Introduction

Constraint Satisfaction Problems (CSPs) are a fundamental class of problems in artificial intelligence that involve finding values for a set of variables that satisfy a given set of constraints. CSPs provide a powerful framework for modeling and solving a wide variety of real-world problems, from scheduling and resource allocation to configuration and planning.

## What is a Constraint Satisfaction Problem?

A Constraint Satisfaction Problem consists of three components:

1. **Variables**: A set of variables that need to be assigned values
2. **Domains**: The set of possible values for each variable
3. **Constraints**: Rules that specify which combinations of values are allowed

### Mathematical Definition

Formally, a CSP is defined as a triple (X, D, C):

- **X** = {X₁, X₂, ..., Xₙ} is a finite set of variables
- **D** = {D₁, D₂, ..., Dₙ} is a set of domains where Dᵢ is the domain of variable Xᵢ
- **C** = {C₁, C₂, ..., Cₘ} is a set of constraints

Each constraint Cⱼ is a pair (scopeⱼ, relationⱼ) where:
- **scopeⱼ** is a subset of variables {Xᵢ₁, Xᵢ₂, ..., Xᵢₖ}
- **relationⱼ** is a subset of Dᵢ₁ × Dᵢ₂ × ... × Dᵢₖ that specifies the allowed combinations

### Solution Definition

A **solution** to a CSP is an assignment of values to all variables such that all constraints are satisfied. Formally, a solution is a function A: X → ∪D such that:

1. A(Xᵢ) ∈ Dᵢ for all variables Xᵢ
2. For every constraint Cⱼ = (scopeⱼ, relationⱼ), the tuple (A(Xᵢ₁), A(Xᵢ₂), ..., A(Xᵢₖ)) ∈ relationⱼ

## Types of Constraints

### Unary Constraints
Constraints involving only one variable.
```
Example: X₁ ≠ 5 (variable X₁ cannot take value 5)
```

### Binary Constraints
Constraints involving exactly two variables.
```
Example: X₁ ≠ X₂ (variables X₁ and X₂ must have different values)
```

### Global Constraints
Constraints involving an arbitrary number of variables.
```
Example: AllDifferent(X₁, X₂, X₃) (all three variables must have different values)
```

## Problem Classification

### Decision Problems
Determine whether a solution exists.

### Optimization Problems
Find the best solution according to some objective function.

### Counting Problems
Count the number of solutions.

## Basic Python Implementation

Let's implement a basic CSP framework in Python:

```python
from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Constraint:
    """Represents a constraint in a CSP."""
    variables: List[str]
    relation: Set[Tuple[Any, ...]]
    
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        """Check if the constraint is satisfied by the given assignment."""
        if not all(var in assignment for var in self.variables):
            return True  # Partial assignment, constraint is satisfied
        
        values = tuple(assignment[var] for var in self.variables)
        return values in self.relation

class CSP:
    """Basic Constraint Satisfaction Problem implementation."""
    
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]]):
        self.variables = variables
        self.domains = domains
        self.constraints: List[Constraint] = []
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the CSP."""
        self.constraints.append(constraint)
    
    def is_consistent(self, assignment: Dict[str, Any]) -> bool:
        """Check if the current assignment is consistent with all constraints."""
        for constraint in self.constraints:
            if not constraint.is_satisfied(assignment):
                return False
        return True
    
    def is_complete(self, assignment: Dict[str, Any]) -> bool:
        """Check if the assignment assigns values to all variables."""
        return len(assignment) == len(self.variables)
    
    def is_solution(self, assignment: Dict[str, Any]) -> bool:
        """Check if the assignment is a complete solution."""
        return self.is_complete(assignment) and self.is_consistent(assignment)
    
    def get_unassigned_variables(self, assignment: Dict[str, Any]) -> List[str]:
        """Get variables that haven't been assigned values yet."""
        return [var for var in self.variables if var not in assignment]
    
    def get_legal_values(self, var: str, assignment: Dict[str, Any]) -> List[Any]:
        """Get legal values for a variable given the current assignment."""
        legal_values = []
        for value in self.domains[var]:
            test_assignment = assignment.copy()
            test_assignment[var] = value
            if self.is_consistent(test_assignment):
                legal_values.append(value)
        return legal_values

# Example: Map Coloring Problem
def create_map_coloring_csp():
    """Create a CSP for the Australia map coloring problem."""
    variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
    colors = ['red', 'green', 'blue']
    
    # All variables can take any color
    domains = {var: colors for var in variables}
    
    csp = CSP(variables, domains)
    
    # Add constraints: adjacent regions must have different colors
    adjacent_regions = [
        ('WA', 'NT'), ('WA', 'SA'),
        ('NT', 'SA'), ('NT', 'Q'),
        ('SA', 'Q'), ('SA', 'NSW'), ('SA', 'V'),
        ('Q', 'NSW'),
        ('NSW', 'V')
    ]
    
    for region1, region2 in adjacent_regions:
        # Create constraint: region1 ≠ region2
        constraint = Constraint(
            variables=[region1, region2],
            relation={(c1, c2) for c1 in colors for c2 in colors if c1 != c2}
        )
        csp.add_constraint(constraint)
    
    return csp

# Test the implementation
def test_map_coloring():
    """Test the map coloring CSP."""
    csp = create_map_coloring_csp()
    
    # Test a valid assignment
    valid_assignment = {
        'WA': 'red', 'NT': 'green', 'SA': 'blue',
        'Q': 'red', 'NSW': 'green', 'V': 'red', 'T': 'blue'
    }
    
    print("Valid assignment is solution:", csp.is_solution(valid_assignment))
    
    # Test an invalid assignment (adjacent regions with same color)
    invalid_assignment = {
        'WA': 'red', 'NT': 'red', 'SA': 'blue',
        'Q': 'green', 'NSW': 'green', 'V': 'red', 'T': 'blue'
    }
    
    print("Invalid assignment is solution:", csp.is_solution(invalid_assignment))
    
    # Test partial assignment
    partial_assignment = {'WA': 'red', 'NT': 'green'}
    print("Partial assignment is consistent:", csp.is_consistent(partial_assignment))
    print("Unassigned variables:", csp.get_unassigned_variables(partial_assignment))
    print("Legal values for SA:", csp.get_legal_values('SA', partial_assignment))

if __name__ == "__main__":
    test_map_coloring()
```

## Problem Characteristics

### Complexity

CSPs are generally **NP-complete** problems, meaning:
- There is no known polynomial-time algorithm to solve all CSPs
- If a polynomial-time algorithm exists for CSPs, then P = NP
- However, many practical CSPs can be solved efficiently using specialized algorithms

### Tractability Factors

The difficulty of solving a CSP depends on several factors:

1. **Constraint Density**: The ratio of constraints to variables
2. **Domain Size**: The number of possible values for each variable
3. **Constraint Tightness**: How restrictive the constraints are
4. **Problem Structure**: Whether the constraint graph has special properties

### Constraint Graph

The **constraint graph** of a CSP is an undirected graph where:
- Nodes represent variables
- Edges represent binary constraints between variables

Properties of the constraint graph can help determine solution strategies:
- **Tree-structured**: Can be solved in polynomial time using dynamic programming
- **Sparse**: Often easier to solve than dense problems
- **Planar**: May have special properties that aid solution

## Solution Methods Overview

### Systematic Search
- **Backtracking**: Depth-first search with constraint checking
- **Forward Checking**: Maintains arc consistency during search
- **Constraint Propagation**: Uses consistency algorithms to reduce search space

### Local Search
- **Hill Climbing**: Iteratively improve assignments
- **Simulated Annealing**: Use temperature to escape local optima
- **Min-Conflicts**: Focus on reducing constraint violations

### Hybrid Methods
- **Backtracking with Local Search**: Combine systematic and local search
- **Genetic Algorithms**: Use evolutionary approaches
- **Constraint-Based Local Search**: Integrate constraint propagation with local search

## Applications

CSPs have numerous real-world applications:

### Scheduling
- Course scheduling
- Employee scheduling
- Project planning
- Manufacturing scheduling

### Resource Allocation
- Network routing
- Frequency assignment
- Crew scheduling
- Vehicle routing

### Configuration
- Product configuration
- Software configuration
- Hardware configuration
- Network configuration

### Planning
- Automated planning
- Logistics optimization
- Supply chain management
- Production planning

## Summary

Constraint Satisfaction Problems provide a powerful and flexible framework for modeling and solving combinatorial problems. The key to effective CSP solving lies in:

1. **Problem Modeling**: Choosing appropriate variables, domains, and constraints
2. **Algorithm Selection**: Matching solution methods to problem characteristics
3. **Constraint Propagation**: Using consistency algorithms to reduce search space
4. **Heuristics**: Applying domain knowledge to guide search efficiently

Understanding the fundamentals of CSPs is essential for developing effective AI systems that can solve complex real-world problems. 