# Beam Search in Constraint Satisfaction Problems

## Introduction

Beam search is a heuristic search algorithm that explores a graph by expanding the most promising nodes in a limited set. It's a variation of breadth-first search that uses a heuristic function to evaluate and rank nodes. In the context of CSPs, beam search can be used for both systematic search and local search approaches.

## What is Beam Search?

### Definition

Beam search is a search algorithm that maintains a limited set of the most promising nodes at each level of the search tree. Unlike breadth-first search which explores all nodes at each level, beam search only keeps the top-k nodes based on a heuristic evaluation.

**Mathematical Definition**:
Given a search tree with nodes N and a heuristic function h: N → ℝ, beam search with beam width k:
1. Starts with initial node(s)
2. At each level, generates all successors of current nodes
3. Evaluates all successors using h
4. Keeps only the top-k successors
5. Continues until goal is found or no more nodes to explore

### Key Characteristics

- **Memory Efficient**: Only keeps k nodes at each level
- **Incomplete**: May miss optimal solutions
- **Fast**: Reduces exponential growth of search space
- **Heuristic-Dependent**: Quality depends heavily on heuristic function

## Basic Beam Search Algorithm

### Algorithm Description

```
function beam_search(problem, beam_width, heuristic):
    frontier = [initial_state]
    while frontier is not empty:
        candidates = []
        for each state in frontier:
            for each successor in successors(state):
                candidates.append(successor)
        frontier = select_best(candidates, beam_width, heuristic)
        if goal_test(frontier):
            return solution
    return failure
```

### Python Implementation

```python
from typing import List, Dict, Set, Any, Optional, Callable
from dataclasses import dataclass
import heapq
from collections import deque

@dataclass
class SearchNode:
    """Represents a node in the beam search."""
    state: Any
    parent: Optional['SearchNode'] = None
    action: Any = None
    cost: float = 0.0
    heuristic_value: float = 0.0
    
    def __lt__(self, other):
        return self.heuristic_value < other.heuristic_value

class BeamSearchSolver:
    """Beam search solver for CSPs."""
    
    def __init__(self, csp, beam_width: int, heuristic: Callable):
        self.csp = csp
        self.beam_width = beam_width
        self.heuristic = heuristic
        self.nodes_explored = 0
    
    def solve(self) -> Optional[Dict[str, Any]]:
        """Solve CSP using beam search."""
        self.nodes_explored = 0
        
        # Initialize with empty assignment
        initial_node = SearchNode(
            state={},
            heuristic_value=self.heuristic({}, self.csp)
        )
        
        frontier = [initial_node]
        
        while frontier:
            self.nodes_explored += len(frontier)
            
            # Generate all successors
            candidates = []
            for node in frontier:
                successors = self._generate_successors(node)
                candidates.extend(successors)
            
            if not candidates:
                break
            
            # Select best candidates for next level
            frontier = self._select_best_candidates(candidates)
            
            # Check if any node is a solution
            for node in frontier:
                if self._is_solution(node.state):
                    return node.state
        
        return None
    
    def _generate_successors(self, node: SearchNode) -> List[SearchNode]:
        """Generate successor nodes from current node."""
        successors = []
        assignment = node.state
        
        # Get unassigned variables
        unassigned = [var for var in self.csp.variables if var not in assignment]
        
        if not unassigned:
            return successors
        
        # Select next variable to assign (simple ordering)
        var = unassigned[0]
        
        # Try each value in domain
        for value in self.csp.domains[var]:
            new_assignment = assignment.copy()
            new_assignment[var] = value
            
            # Only add if assignment is consistent
            if self.csp.is_consistent(new_assignment):
                successor = SearchNode(
                    state=new_assignment,
                    parent=node,
                    action=(var, value),
                    cost=node.cost + 1,
                    heuristic_value=self.heuristic(new_assignment, self.csp)
                )
                successors.append(successor)
        
        return successors
    
    def _select_best_candidates(self, candidates: List[SearchNode]) -> List[SearchNode]:
        """Select the best candidates based on heuristic value."""
        if len(candidates) <= self.beam_width:
            return candidates
        
        # Sort by heuristic value (lower is better)
        candidates.sort(key=lambda x: x.heuristic_value)
        return candidates[:self.beam_width]
    
    def _is_solution(self, assignment: Dict[str, Any]) -> bool:
        """Check if assignment is a complete solution."""
        return len(assignment) == len(self.csp.variables) and self.csp.is_consistent(assignment)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get solving statistics."""
        return {
            'nodes_explored': self.nodes_explored,
            'beam_width': self.beam_width,
            'variables': len(self.csp.variables)
        }

# Example heuristic functions
class CSPHeuristics:
    """Collection of heuristic functions for CSP beam search."""
    
    @staticmethod
    def unassigned_variables(assignment: Dict[str, Any], csp) -> float:
        """Heuristic: number of unassigned variables (lower is better)."""
        return len([var for var in csp.variables if var not in assignment])
    
    @staticmethod
    def constraint_violations(assignment: Dict[str, Any], csp) -> float:
        """Heuristic: number of constraint violations (lower is better)."""
        violations = 0
        for constraint in csp.constraints:
            if not constraint.is_satisfied(assignment):
                violations += 1
        return violations
    
    @staticmethod
    def domain_size_sum(assignment: Dict[str, Any], csp) -> float:
        """Heuristic: sum of domain sizes for unassigned variables (lower is better)."""
        total_domain_size = 0
        for var in csp.variables:
            if var not in assignment:
                total_domain_size += len(csp.domains[var])
        return total_domain_size
    
    @staticmethod
    def combined_heuristic(assignment: Dict[str, Any], csp) -> float:
        """Combined heuristic using multiple factors."""
        unassigned = CSPHeuristics.unassigned_variables(assignment, csp)
        violations = CSPHeuristics.constraint_violations(assignment, csp)
        domain_size = CSPHeuristics.domain_size_sum(assignment, csp)
        
        # Weighted combination
        return unassigned + 10 * violations + 0.1 * domain_size

# Test beam search
def test_beam_search():
    """Test beam search with a simple CSP."""
    # Create a simple CSP
    variables = ['A', 'B', 'C']
    domains = {
        'A': [1, 2, 3],
        'B': [1, 2, 3],
        'C': [1, 2, 3]
    }
    
    # Constraints: all variables must be different
    constraints = []
    for i, var1 in enumerate(variables):
        for var2 in variables[i+1:]:
            constraint = Constraint(
                scope=[var1, var2],
                relation={(v1, v2) for v1 in domains[var1] for v2 in domains[var2] if v1 != v2}
            )
            constraints.append(constraint)
    
    csp = SimpleCSP(variables, domains, constraints)
    
    # Test different beam widths
    beam_widths = [1, 2, 5, 10]
    heuristic = CSPHeuristics.combined_heuristic
    
    for beam_width in beam_widths:
        print(f"\nBeam width: {beam_width}")
        solver = BeamSearchSolver(csp, beam_width, heuristic)
        solution = solver.solve()
        stats = solver.get_statistics()
        
        print(f"Solution found: {solution is not None}")
        if solution:
            print(f"Solution: {solution}")
        print(f"Nodes explored: {stats['nodes_explored']}")

if __name__ == "__main__":
    test_beam_search()
```

## Beam Search Variants

### 1. Stochastic Beam Search

Stochastic beam search introduces randomness to avoid getting stuck in local optima.

```python
import random

class StochasticBeamSearchSolver(BeamSearchSolver):
    """Stochastic beam search solver."""
    
    def __init__(self, csp, beam_width: int, heuristic: Callable, temperature: float = 1.0):
        super().__init__(csp, beam_width, heuristic)
        self.temperature = temperature
    
    def _select_best_candidates(self, candidates: List[SearchNode]) -> List[SearchNode]:
        """Select candidates using stochastic sampling."""
        if len(candidates) <= self.beam_width:
            return candidates
        
        # Calculate probabilities based on heuristic values
        scores = [1.0 / (1.0 + node.heuristic_value) for node in candidates]
        total_score = sum(scores)
        probabilities = [score / total_score for score in scores]
        
        # Sample candidates
        selected = []
        for _ in range(self.beam_width):
            # Roulette wheel selection
            r = random.random()
            cumulative = 0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    selected.append(candidates[i])
                    break
        
        return selected
```

### 2. Adaptive Beam Search

Adaptive beam search adjusts the beam width based on search progress.

```python
class AdaptiveBeamSearchSolver(BeamSearchSolver):
    """Adaptive beam search solver."""
    
    def __init__(self, csp, initial_beam_width: int, heuristic: Callable, 
                 min_width: int = 1, max_width: int = 50):
        super().__init__(csp, initial_beam_width, heuristic)
        self.min_width = min_width
        self.max_width = max_width
        self.current_width = initial_beam_width
    
    def solve(self) -> Optional[Dict[str, Any]]:
        """Solve CSP using adaptive beam search."""
        self.nodes_explored = 0
        self.current_width = self.beam_width
        
        initial_node = SearchNode(
            state={},
            heuristic_value=self.heuristic({}, self.csp)
        )
        
        frontier = [initial_node]
        level = 0
        
        while frontier:
            self.nodes_explored += len(frontier)
            
            # Generate successors
            candidates = []
            for node in frontier:
                successors = self._generate_successors(node)
                candidates.extend(successors)
            
            if not candidates:
                break
            
            # Adapt beam width
            self._adapt_beam_width(level, len(candidates))
            
            # Select candidates
            frontier = self._select_best_candidates(candidates)
            
            # Check for solution
            for node in frontier:
                if self._is_solution(node.state):
                    return node.state
            
            level += 1
        
        return None
    
    def _adapt_beam_width(self, level: int, num_candidates: int):
        """Adapt beam width based on search progress."""
        if level < 5:
            # Early in search: increase width to explore more
            self.current_width = min(self.current_width + 2, self.max_width)
        elif num_candidates < self.current_width // 2:
            # Few candidates: decrease width
            self.current_width = max(self.current_width - 1, self.min_width)
        else:
            # Many candidates: increase width slightly
            self.current_width = min(self.current_width + 1, self.max_width)
    
    def _select_best_candidates(self, candidates: List[SearchNode]) -> List[SearchNode]:
        """Select candidates using current beam width."""
        if len(candidates) <= self.current_width:
            return candidates
        
        candidates.sort(key=lambda x: x.heuristic_value)
        return candidates[:self.current_width]
```

## Beam Search for Local Search

### Constraint-Based Local Search

Beam search can be adapted for local search by treating assignments as states and using constraint violation as the heuristic.

```python
class LocalBeamSearchSolver:
    """Beam search for local search in CSPs."""
    
    def __init__(self, csp, beam_width: int, max_iterations: int = 1000):
        self.csp = csp
        self.beam_width = beam_width
        self.max_iterations = max_iterations
    
    def solve(self) -> Optional[Dict[str, Any]]:
        """Solve CSP using local beam search."""
        # Initialize with random assignments
        frontier = []
        for _ in range(self.beam_width):
            assignment = self._generate_random_assignment()
            frontier.append(assignment)
        
        for iteration in range(self.max_iterations):
            # Generate neighbors for all assignments
            candidates = []
            for assignment in frontier:
                neighbors = self._generate_neighbors(assignment)
                candidates.extend(neighbors)
            
            if not candidates:
                break
            
            # Select best candidates
            frontier = self._select_best_local_candidates(candidates)
            
            # Check if any assignment is a solution
            for assignment in frontier:
                if self._is_solution(assignment):
                    return assignment
        
        # Return best assignment found
        if frontier:
            return min(frontier, key=lambda x: self._evaluate_assignment(x))
        
        return None
    
    def _generate_random_assignment(self) -> Dict[str, Any]:
        """Generate a random complete assignment."""
        assignment = {}
        for var in self.csp.variables:
            assignment[var] = random.choice(self.csp.domains[var])
        return assignment
    
    def _generate_neighbors(self, assignment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate neighboring assignments."""
        neighbors = []
        
        for var in self.csp.variables:
            for value in self.csp.domains[var]:
                if assignment[var] != value:
                    neighbor = assignment.copy()
                    neighbor[var] = value
                    neighbors.append(neighbor)
        
        return neighbors
    
    def _select_best_local_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select best candidates based on constraint violations."""
        if len(candidates) <= self.beam_width:
            return candidates
        
        # Sort by number of constraint violations
        candidates.sort(key=lambda x: self._evaluate_assignment(x))
        return candidates[:self.beam_width]
    
    def _evaluate_assignment(self, assignment: Dict[str, Any]) -> int:
        """Evaluate assignment by counting constraint violations."""
        violations = 0
        for constraint in self.csp.constraints:
            if not constraint.is_satisfied(assignment):
                violations += 1
        return violations
    
    def _is_solution(self, assignment: Dict[str, Any]) -> bool:
        """Check if assignment is a solution."""
        return self._evaluate_assignment(assignment) == 0
```

## Hybrid Approaches

### Beam Search with Backtracking

Combining beam search with backtracking can provide better completeness guarantees.

```python
class HybridBeamSearchSolver:
    """Hybrid solver combining beam search with backtracking."""
    
    def __init__(self, csp, beam_width: int, heuristic: Callable, max_depth: int = 10):
        self.csp = csp
        self.beam_width = beam_width
        self.heuristic = heuristic
        self.max_depth = max_depth
    
    def solve(self) -> Optional[Dict[str, Any]]:
        """Solve CSP using hybrid approach."""
        # Start with beam search
        beam_solution = self._beam_search()
        if beam_solution:
            return beam_solution
        
        # If beam search fails, try backtracking from best partial solutions
        return self._backtrack_from_partial()
    
    def _beam_search(self) -> Optional[Dict[str, Any]]:
        """Run beam search."""
        solver = BeamSearchSolver(self.csp, self.beam_width, self.heuristic)
        return solver.solve()
    
    def _backtrack_from_partial(self) -> Optional[Dict[str, Any]]:
        """Run backtracking from partial solutions found by beam search."""
        # This would implement backtracking starting from the best partial solutions
        # found during beam search
        pass
```

## Performance Analysis

### Time Complexity

- **Worst Case**: O(b^d) where b is beam width, d is depth
- **Best Case**: O(b * d) if heuristic is perfect
- **Average Case**: Depends on heuristic quality and problem structure

### Space Complexity

- **O(b * d)**: Stores b nodes at each of d levels

### Comparison with Other Algorithms

```python
def compare_search_algorithms(csp):
    """Compare beam search with other search algorithms."""
    algorithms = [
        ("Beam Search (width=5)", lambda: BeamSearchSolver(csp, 5, CSPHeuristics.combined_heuristic)),
        ("Beam Search (width=10)", lambda: BeamSearchSolver(csp, 10, CSPHeuristics.combined_heuristic)),
        ("Stochastic Beam Search", lambda: StochasticBeamSearchSolver(csp, 5, CSPHeuristics.combined_heuristic)),
        ("Local Beam Search", lambda: LocalBeamSearchSolver(csp, 5))
    ]
    
    results = {}
    
    for name, solver_factory in algorithms:
        solver = solver_factory()
        solution = solver.solve()
        stats = solver.get_statistics()
        
        results[name] = {
            'solution_found': solution is not None,
            'nodes_explored': stats.get('nodes_explored', 0),
            'solution': solution
        }
    
    return results
```

## Applications in CSPs

### 1. Large-Scale Problems

Beam search is particularly useful for large CSPs where exhaustive search is impractical.

### 2. Real-Time Systems

The bounded memory usage makes beam search suitable for real-time applications.

### 3. Initial Solution Generation

Beam search can be used to generate good initial solutions for other algorithms.

### 4. Constraint Optimization

Beam search can be adapted for constraint optimization problems by using objective functions as heuristics.

## Summary

Beam search provides a powerful approach to CSP solving:

1. **Memory Efficiency**: Bounded memory usage regardless of problem size
2. **Speed**: Fast exploration of large search spaces
3. **Flexibility**: Can be adapted for various problem types
4. **Heuristic Integration**: Naturally incorporates domain knowledge

Key considerations:
- **Beam Width**: Critical parameter affecting solution quality vs. speed
- **Heuristic Quality**: Determines effectiveness of the search
- **Completeness**: May miss solutions, especially with narrow beams
- **Hybridization**: Often works best when combined with other techniques

Understanding beam search is essential for solving large-scale CSPs efficiently and for developing hybrid solving approaches. 