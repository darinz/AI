# Dynamic Ordering in Constraint Satisfaction Problems

## Introduction

Dynamic ordering heuristics are crucial for efficient CSP solving. They determine the order in which variables are assigned values and which values are tried first. Good ordering can dramatically reduce search time by guiding the search toward promising areas and detecting failures early.

## Variable Ordering Heuristics

### 1. Minimum Remaining Values (MRV)

The MRV heuristic selects the variable with the fewest legal values remaining. This reduces the branching factor early in the search.

**Principle**: Choose the variable that is most constrained (has the fewest options).

**Mathematical Definition**: 
For a partial assignment A, select variable Xᵢ where:
Xᵢ = argmin_{Xⱼ ∈ U} |Dⱼ(A)|

Where U is the set of unassigned variables and Dⱼ(A) is the domain of Xⱼ after applying assignment A.

```python
from typing import List, Dict, Set, Any, Optional
from dataclasses import dataclass
import heapq

class MRVHeuristic:
    """Minimum Remaining Values heuristic for variable ordering."""
    
    def __init__(self, csp):
        self.csp = csp
    
    def select_variable(self, assignment: Dict[str, Any]) -> Optional[str]:
        """Select the variable with minimum remaining values."""
        unassigned = [var for var in self.csp.variables if var not in assignment]
        
        if not unassigned:
            return None
        
        # Calculate remaining values for each unassigned variable
        var_scores = []
        for var in unassigned:
            legal_values = self._count_legal_values(var, assignment)
            var_scores.append((legal_values, var))
        
        # Return variable with minimum remaining values
        if var_scores:
            return min(var_scores, key=lambda x: x[0])[1]
        return None
    
    def _count_legal_values(self, var: str, assignment: Dict[str, Any]) -> int:
        """Count legal values for a variable given current assignment."""
        count = 0
        for value in self.csp.domains[var]:
            test_assignment = assignment.copy()
            test_assignment[var] = value
            if self.csp.is_consistent(test_assignment):
                count += 1
        return count
    
    def get_variable_priority(self, var: str, assignment: Dict[str, Any]) -> int:
        """Get priority score for a variable (lower is better)."""
        return self._count_legal_values(var, assignment)

# Example usage with CSP
def create_mrv_example():
    """Create an example to demonstrate MRV heuristic."""
    from collections import defaultdict
    
    class SimpleCSP:
        def __init__(self, variables, domains, constraints):
            self.variables = variables
            self.domains = domains
            self.constraints = constraints
        
        def is_consistent(self, assignment):
            for var1, var2 in self.constraints:
                if var1 in assignment and var2 in assignment:
                    if assignment[var1] == assignment[var2]:
                        return False
            return True
    
    # Create a simple CSP
    variables = ['A', 'B', 'C', 'D']
    domains = {
        'A': [1, 2, 3],
        'B': [1, 2],
        'C': [1, 2, 3, 4],
        'D': [1]
    }
    constraints = [('A', 'B'), ('B', 'C'), ('C', 'D')]
    
    csp = SimpleCSP(variables, domains, constraints)
    mrv = MRVHeuristic(csp)
    
    # Test MRV selection
    assignment = {'A': 1}
    next_var = mrv.select_variable(assignment)
    print(f"After assigning A=1, MRV selects: {next_var}")
    
    return csp, mrv
```

### 2. Degree Heuristic

The degree heuristic selects the variable with the most constraints on remaining variables. This helps reduce future branching.

**Principle**: Choose the variable that constrains the most other unassigned variables.

**Mathematical Definition**:
For a partial assignment A, select variable Xᵢ where:
Xᵢ = argmax_{Xⱼ ∈ U} |{Xₖ ∈ U : ∃C ∈ C, {Xⱼ, Xₖ} ⊆ scope(C)}|

```python
class DegreeHeuristic:
    """Degree heuristic for variable ordering."""
    
    def __init__(self, csp):
        self.csp = csp
        self.constraint_graph = self._build_constraint_graph()
    
    def _build_constraint_graph(self) -> Dict[str, Set[str]]:
        """Build adjacency list representation of constraint graph."""
        graph = defaultdict(set)
        
        for constraint in self.csp.constraints:
            if len(constraint.scope) == 2:  # Binary constraint
                var1, var2 = constraint.scope
                graph[var1].add(var2)
                graph[var2].add(var1)
        
        return graph
    
    def select_variable(self, assignment: Dict[str, Any]) -> Optional[str]:
        """Select variable with highest degree among unassigned variables."""
        unassigned = [var for var in self.csp.variables if var not in assignment]
        
        if not unassigned:
            return None
        
        # Calculate degree for each unassigned variable
        var_scores = []
        for var in unassigned:
            degree = self._calculate_degree(var, assignment)
            var_scores.append((degree, var))
        
        # Return variable with maximum degree
        if var_scores:
            return max(var_scores, key=lambda x: x[0])[1]
        return None
    
    def _calculate_degree(self, var: str, assignment: Dict[str, Any]) -> int:
        """Calculate degree of variable with respect to unassigned variables."""
        if var not in self.constraint_graph:
            return 0
        
        unassigned_neighbors = 0
        for neighbor in self.constraint_graph[var]:
            if neighbor not in assignment:
                unassigned_neighbors += 1
        
        return unassigned_neighbors
    
    def get_variable_priority(self, var: str, assignment: Dict[str, Any]) -> int:
        """Get priority score for a variable (higher is better for degree)."""
        return self._calculate_degree(var, assignment)
```

### 3. Combined Heuristics

Often, combining multiple heuristics provides better results than using any single heuristic.

```python
class CombinedHeuristic:
    """Combines MRV and Degree heuristics."""
    
    def __init__(self, csp, mrv_weight=1.0, degree_weight=0.1):
        self.csp = csp
        self.mrv_weight = mrv_weight
        self.degree_weight = degree_weight
        self.mrv_heuristic = MRVHeuristic(csp)
        self.degree_heuristic = DegreeHeuristic(csp)
    
    def select_variable(self, assignment: Dict[str, Any]) -> Optional[str]:
        """Select variable using combined heuristic."""
        unassigned = [var for var in self.csp.variables if var not in assignment]
        
        if not unassigned:
            return None
        
        # Calculate combined scores
        var_scores = []
        for var in unassigned:
            mrv_score = self.mrv_heuristic.get_variable_priority(var, assignment)
            degree_score = self.degree_heuristic.get_variable_priority(var, assignment)
            
            # Normalize scores (lower is better for MRV, higher is better for degree)
            combined_score = (self.mrv_weight * mrv_score) - (self.degree_weight * degree_score)
            var_scores.append((combined_score, var))
        
        # Return variable with best combined score
        if var_scores:
            return min(var_scores, key=lambda x: x[0])[1]
        return None

# Tie-breaking strategies
class TieBreakingHeuristic:
    """Handles ties in variable selection."""
    
    @staticmethod
    def break_ties_mrv_first(mrv_scores, degree_scores):
        """Break ties by MRV first, then degree."""
        min_mrv = min(mrv_scores.values())
        candidates = [var for var, score in mrv_scores.items() if score == min_mrv]
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Use degree as tie-breaker
        max_degree = max(degree_scores[var] for var in candidates)
        degree_candidates = [var for var in candidates if degree_scores[var] == max_degree]
        
        return degree_candidates[0] if degree_candidates else candidates[0]
```

## Value Ordering Heuristics

### 1. Least Constraining Value (LCV)

The LCV heuristic chooses values that leave the most options for other variables.

**Principle**: Choose the value that rules out the fewest values in the remaining variables.

```python
class LCVHeuristic:
    """Least Constraining Value heuristic."""
    
    def __init__(self, csp):
        self.csp = csp
    
    def order_values(self, var: str, assignment: Dict[str, Any]) -> List[Any]:
        """Order values by how constraining they are (least constraining first)."""
        values = self.csp.domains[var]
        
        # Calculate how constraining each value is
        value_scores = []
        for value in values:
            constraint_score = self._calculate_constraint_score(var, value, assignment)
            value_scores.append((constraint_score, value))
        
        # Sort by constraint score (lower is better)
        value_scores.sort(key=lambda x: x[0])
        return [value for score, value in value_scores]
    
    def _calculate_constraint_score(self, var: str, value: Any, assignment: Dict[str, Any]) -> int:
        """Calculate how constraining a value is."""
        test_assignment = assignment.copy()
        test_assignment[var] = value
        
        # Count how many values are eliminated from other variables
        total_eliminated = 0
        
        for other_var in self.csp.variables:
            if other_var != var and other_var not in assignment:
                original_count = len(self.csp.domains[other_var])
                remaining_count = 0
                
                for other_value in self.csp.domains[other_var]:
                    test_assignment[other_var] = other_value
                    if self.csp.is_consistent(test_assignment):
                        remaining_count += 1
                    del test_assignment[other_var]
                
                total_eliminated += (original_count - remaining_count)
        
        return total_eliminated
    
    def get_value_priority(self, var: str, value: Any, assignment: Dict[str, Any]) -> int:
        """Get priority score for a value (lower is better)."""
        return self._calculate_constraint_score(var, value, assignment)
```

### 2. Domain-Specific Heuristics

Some problems have domain-specific knowledge that can guide value selection.

```python
class DomainSpecificHeuristic:
    """Domain-specific value ordering heuristics."""
    
    @staticmethod
    def order_by_frequency(values: List[Any], frequency_map: Dict[Any, int]) -> List[Any]:
        """Order values by their frequency in the problem."""
        return sorted(values, key=lambda v: frequency_map.get(v, 0), reverse=True)
    
    @staticmethod
    def order_by_distance(values: List[Any], target: Any, distance_func) -> List[Any]:
        """Order values by distance from a target value."""
        return sorted(values, key=lambda v: distance_func(v, target))
    
    @staticmethod
    def order_by_preference(values: List[Any], preference_order: List[Any]) -> List[Any]:
        """Order values by a predefined preference order."""
        preference_map = {val: idx for idx, val in enumerate(preference_order)}
        return sorted(values, key=lambda v: preference_map.get(v, len(preference_order)))

# Example: Map coloring with color preferences
class MapColoringValueHeuristic:
    """Value ordering for map coloring problems."""
    
    def __init__(self, color_preferences: Dict[str, int] = None):
        self.color_preferences = color_preferences or {}
    
    def order_values(self, var: str, values: List[str], assignment: Dict[str, str]) -> List[str]:
        """Order colors for map coloring."""
        if not self.color_preferences:
            return values
        
        # Order by preference (lower preference number = higher priority)
        return sorted(values, key=lambda c: self.color_preferences.get(c, 999))
```

## Dynamic Ordering in Backtracking

### Implementation with Heuristics

```python
class HeuristicBacktrackingSolver:
    """Backtracking solver with dynamic ordering heuristics."""
    
    def __init__(self, csp, var_heuristic=None, val_heuristic=None):
        self.csp = csp
        self.var_heuristic = var_heuristic or MRVHeuristic(csp)
        self.val_heuristic = val_heuristic or LCVHeuristic(csp)
        self.nodes_explored = 0
    
    def solve(self) -> Optional[Dict[str, Any]]:
        """Solve the CSP using backtracking with heuristics."""
        self.nodes_explored = 0
        return self._backtrack({})
    
    def _backtrack(self, assignment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Backtracking with dynamic ordering."""
        self.nodes_explored += 1
        
        # Check if assignment is complete
        if len(assignment) == len(self.csp.variables):
            return assignment
        
        # Select next variable using heuristic
        var = self.var_heuristic.select_variable(assignment)
        if var is None:
            return None
        
        # Order values using heuristic
        values = self.val_heuristic.order_values(var, assignment)
        
        # Try each value
        for value in values:
            assignment[var] = value
            
            if self.csp.is_consistent(assignment):
                result = self._backtrack(assignment)
                if result is not None:
                    return result
            
            del assignment[var]
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get solving statistics."""
        return {
            'nodes_explored': self.nodes_explored,
            'variables': len(self.csp.variables),
            'constraints': len(self.csp.constraints)
        }

# Performance comparison
def compare_heuristics(csp):
    """Compare different heuristic combinations."""
    heuristics = [
        ("No heuristics", None, None),
        ("MRV only", MRVHeuristic(csp), None),
        ("MRV + LCV", MRVHeuristic(csp), LCVHeuristic(csp)),
        ("Combined", CombinedHeuristic(csp), LCVHeuristic(csp))
    ]
    
    results = {}
    
    for name, var_heuristic, val_heuristic in heuristics:
        solver = HeuristicBacktrackingSolver(csp, var_heuristic, val_heuristic)
        solution = solver.solve()
        stats = solver.get_statistics()
        
        results[name] = {
            'solution_found': solution is not None,
            'nodes_explored': stats['nodes_explored'],
            'solution': solution
        }
    
    return results
```

## Advanced Ordering Techniques

### 1. Look-Ahead Heuristics

Look-ahead heuristics consider the impact of assignments on future variables.

```python
class LookAheadHeuristic:
    """Look-ahead heuristic for variable ordering."""
    
    def __init__(self, csp):
        self.csp = csp
    
    def select_variable(self, assignment: Dict[str, Any]) -> Optional[str]:
        """Select variable using look-ahead information."""
        unassigned = [var for var in self.csp.variables if var not in assignment]
        
        if not unassigned:
            return None
        
        # Calculate look-ahead scores
        var_scores = []
        for var in unassigned:
            score = self._calculate_look_ahead_score(var, assignment)
            var_scores.append((score, var))
        
        return min(var_scores, key=lambda x: x[0])[1]
    
    def _calculate_look_ahead_score(self, var: str, assignment: Dict[str, Any]) -> int:
        """Calculate look-ahead score for a variable."""
        # This is a simplified version - real implementations would be more sophisticated
        total_impact = 0
        
        for value in self.csp.domains[var]:
            test_assignment = assignment.copy()
            test_assignment[var] = value
            
            # Count how many other variables would have reduced domains
            for other_var in self.csp.variables:
                if other_var != var and other_var not in assignment:
                    original_domain_size = len(self.csp.domains[other_var])
                    reduced_domain_size = 0
                    
                    for other_value in self.csp.domains[other_var]:
                        test_assignment[other_var] = other_value
                        if self.csp.is_consistent(test_assignment):
                            reduced_domain_size += 1
                        del test_assignment[other_var]
                    
                    total_impact += (original_domain_size - reduced_domain_size)
        
        return total_impact
```

### 2. Adaptive Ordering

Adaptive ordering changes heuristics based on problem characteristics.

```python
class AdaptiveHeuristic:
    """Adaptive heuristic that changes based on problem state."""
    
    def __init__(self, csp):
        self.csp = csp
        self.mrv_heuristic = MRVHeuristic(csp)
        self.degree_heuristic = DegreeHeuristic(csp)
        self.look_ahead_heuristic = LookAheadHeuristic(csp)
    
    def select_variable(self, assignment: Dict[str, Any]) -> Optional[str]:
        """Select variable using adaptive strategy."""
        # Use different heuristics based on search depth
        depth = len(assignment)
        total_vars = len(self.csp.variables)
        
        if depth < total_vars * 0.3:
            # Early in search: use degree heuristic
            return self.degree_heuristic.select_variable(assignment)
        elif depth < total_vars * 0.7:
            # Middle of search: use MRV
            return self.mrv_heuristic.select_variable(assignment)
        else:
            # Late in search: use look-ahead
            return self.look_ahead_heuristic.select_variable(assignment)
```

## Summary

Dynamic ordering heuristics are essential for efficient CSP solving:

1. **Variable Ordering**: MRV and degree heuristics help reduce branching factor
2. **Value Ordering**: LCV and domain-specific heuristics guide search toward promising areas
3. **Combination**: Using multiple heuristics often provides better results
4. **Adaptation**: Changing heuristics during search can improve performance

Key principles:
- **Fail-Fast**: Choose variables that are most likely to fail quickly
- **Constraint Propagation**: Consider the impact on remaining variables
- **Domain Knowledge**: Use problem-specific information when available
- **Adaptation**: Adjust strategies based on search progress

Understanding and implementing these heuristics is crucial for building efficient CSP solvers that can handle real-world problems. 