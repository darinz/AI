# Local Search in Constraint Satisfaction Problems

## Introduction

Local search algorithms for CSPs start with a complete assignment and iteratively improve it by making small changes to satisfy more constraints. Unlike systematic search methods that guarantee finding a solution if one exists, local search methods are incomplete but can be very effective for large problems where systematic search is impractical.

## What is Local Search?

### Definition

Local search algorithms work by:
1. Starting with a complete assignment (all variables have values)
2. Evaluating the quality of the current assignment
3. Generating neighboring assignments by making small changes
4. Moving to a better neighbor if one exists
5. Repeating until a solution is found or no improvement is possible

**Mathematical Definition**:
Given a CSP (X, D, C) and an objective function f: A → ℝ where A is the set of all assignments:
1. Start with assignment a₀ ∈ A
2. Generate neighborhood N(a) ⊆ A for current assignment a
3. Select a' ∈ N(a) such that f(a') < f(a) (for minimization)
4. Set a = a' and repeat until no improvement is possible

### Key Characteristics

- **Incomplete**: May not find solutions even when they exist
- **Fast**: Often much faster than systematic search for large problems
- **Scalable**: Performance doesn't degrade as badly with problem size
- **Flexible**: Can handle various types of constraints and objectives

## Basic Local Search Framework

### Algorithm Description

```
function local_search(csp, initial_assignment):
    current = initial_assignment
    while not termination_condition():
        neighbors = generate_neighbors(current)
        best_neighbor = select_best_neighbor(neighbors)
        if evaluate(best_neighbor) >= evaluate(current):
            break
        current = best_neighbor
    return current
```

### Python Implementation

```python
from typing import List, Dict, Set, Any, Optional, Callable
from dataclasses import dataclass
import random
import math

@dataclass
class LocalSearchState:
    """Represents a state in local search."""
    assignment: Dict[str, Any]
    violations: int
    score: float

class LocalSearchSolver:
    """Base class for local search solvers."""
    
    def __init__(self, csp, max_iterations: int = 1000, max_steps_without_improvement: int = 100):
        self.csp = csp
        self.max_iterations = max_iterations
        self.max_steps_without_improvement = max_steps_without_improvement
        self.iterations = 0
    
    def solve(self) -> Optional[Dict[str, Any]]:
        """Solve CSP using local search."""
        # Generate initial assignment
        current = self._generate_initial_assignment()
        current_state = self._evaluate_assignment(current)
        
        best_state = current_state
        steps_without_improvement = 0
        
        for iteration in range(self.max_iterations):
            self.iterations = iteration
            
            # Check termination conditions
            if current_state.violations == 0:
                return current_state.assignment
            
            if steps_without_improvement >= self.max_steps_without_improvement:
                break
            
            # Generate and evaluate neighbors
            neighbors = self._generate_neighbors(current_state.assignment)
            best_neighbor = self._select_best_neighbor(neighbors)
            
            # Move to better neighbor
            if best_neighbor.score < current_state.score:
                current_state = best_neighbor
                current = best_neighbor.assignment
                steps_without_improvement = 0
                
                if best_neighbor.score < best_state.score:
                    best_state = best_neighbor
            else:
                steps_without_improvement += 1
        
        # Return best solution found
        if best_state.violations == 0:
            return best_state.assignment
        else:
            return None
    
    def _generate_initial_assignment(self) -> Dict[str, Any]:
        """Generate initial complete assignment."""
        assignment = {}
        for var in self.csp.variables:
            assignment[var] = random.choice(self.csp.domains[var])
        return assignment
    
    def _evaluate_assignment(self, assignment: Dict[str, Any]) -> LocalSearchState:
        """Evaluate an assignment and return its state."""
        violations = self._count_violations(assignment)
        score = self._calculate_score(assignment, violations)
        return LocalSearchState(assignment, violations, score)
    
    def _count_violations(self, assignment: Dict[str, Any]) -> int:
        """Count the number of constraint violations."""
        violations = 0
        for constraint in self.csp.constraints:
            if not constraint.is_satisfied(assignment):
                violations += 1
        return violations
    
    def _calculate_score(self, assignment: Dict[str, Any], violations: int) -> float:
        """Calculate score for an assignment (lower is better)."""
        return violations
    
    def _generate_neighbors(self, assignment: Dict[str, Any]) -> List[LocalSearchState]:
        """Generate neighboring assignments."""
        neighbors = []
        
        for var in self.csp.variables:
            for value in self.csp.domains[var]:
                if assignment[var] != value:
                    neighbor = assignment.copy()
                    neighbor[var] = value
                    neighbors.append(self._evaluate_assignment(neighbor))
        
        return neighbors
    
    def _select_best_neighbor(self, neighbors: List[LocalSearchState]) -> LocalSearchState:
        """Select the best neighbor."""
        return min(neighbors, key=lambda x: x.score)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get solving statistics."""
        return {
            'iterations': self.iterations,
            'max_iterations': self.max_iterations
        }

# Example CSP for testing
class SimpleCSP:
    """Simple CSP implementation for testing local search."""
    
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]], constraints: List[Tuple[str, str, Set[Tuple[Any, Any]]]]):
        self.variables = variables
        self.domains = domains
        self.constraints = []
        
        for var1, var2, relation in constraints:
            constraint = Constraint([var1, var2], relation)
            self.constraints.append(constraint)

@dataclass
class Constraint:
    """Simple constraint representation."""
    scope: List[str]
    relation: Set[Tuple[Any, Any]]
    
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        """Check if constraint is satisfied."""
        if not all(var in assignment for var in self.scope):
            return True
        
        values = tuple(assignment[var] for var in self.scope)
        return values in self.relation

def test_local_search():
    """Test local search with a simple CSP."""
    # Create a simple CSP: X ≠ Y, Y ≠ Z, X ≠ Z
    variables = ['X', 'Y', 'Z']
    domains = {
        'X': [1, 2, 3],
        'Y': [1, 2, 3],
        'Z': [1, 2, 3]
    }
    
    # Constraints: all variables must be different
    constraints = [
        ('X', 'Y', {(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)}),
        ('Y', 'Z', {(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)}),
        ('X', 'Z', {(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)})
    ]
    
    csp = SimpleCSP(variables, domains, constraints)
    
    # Test local search
    solver = LocalSearchSolver(csp)
    solution = solver.solve()
    stats = solver.get_statistics()
    
    print(f"Solution found: {solution is not None}")
    if solution:
        print(f"Solution: {solution}")
    print(f"Iterations: {stats['iterations']}")
    
    return csp, solver

if __name__ == "__main__":
    test_local_search()
```

## Hill Climbing

### Algorithm Description

Hill climbing is the simplest local search algorithm. It always moves to the best neighbor, stopping when no better neighbor exists.

### Python Implementation

```python
class HillClimbingSolver(LocalSearchSolver):
    """Hill climbing solver for CSPs."""
    
    def solve(self) -> Optional[Dict[str, Any]]:
        """Solve CSP using hill climbing."""
        current = self._generate_initial_assignment()
        current_state = self._evaluate_assignment(current)
        
        for iteration in range(self.max_iterations):
            self.iterations = iteration
            
            # Check if we found a solution
            if current_state.violations == 0:
                return current_state.assignment
            
            # Generate neighbors
            neighbors = self._generate_neighbors(current_state.assignment)
            best_neighbor = self._select_best_neighbor(neighbors)
            
            # Move to better neighbor
            if best_neighbor.score < current_state.score:
                current_state = best_neighbor
                current = best_neighbor.assignment
            else:
                # No better neighbor - we're at a local optimum
                break
        
        # Return best solution found
        if current_state.violations == 0:
            return current_state.assignment
        else:
            return None
```

## Min-Conflicts Algorithm

### Algorithm Description

The min-conflicts algorithm is specifically designed for CSPs. It:
1. Starts with a complete assignment
2. Repeatedly selects a conflicted variable
3. Assigns it the value that minimizes the number of conflicts

### Python Implementation

```python
class MinConflictsSolver(LocalSearchSolver):
    """Min-conflicts solver for CSPs."""
    
    def solve(self) -> Optional[Dict[str, Any]]:
        """Solve CSP using min-conflicts algorithm."""
        current = self._generate_initial_assignment()
        
        for iteration in range(self.max_iterations):
            self.iterations = iteration
            
            # Check if we found a solution
            if self._count_violations(current) == 0:
                return current
            
            # Select a conflicted variable
            conflicted_vars = self._get_conflicted_variables(current)
            if not conflicted_vars:
                return current
            
            var = random.choice(conflicted_vars)
            
            # Find value that minimizes conflicts
            best_value = self._find_min_conflicts_value(var, current)
            current[var] = best_value
        
        return None
    
    def _get_conflicted_variables(self, assignment: Dict[str, Any]) -> List[str]:
        """Get list of variables involved in constraint violations."""
        conflicted = set()
        
        for constraint in self.csp.constraints:
            if not constraint.is_satisfied(assignment):
                for var in constraint.scope:
                    conflicted.add(var)
        
        return list(conflicted)
    
    def _find_min_conflicts_value(self, var: str, assignment: Dict[str, Any]) -> Any:
        """Find the value for var that minimizes conflicts."""
        best_value = assignment[var]
        min_conflicts = float('inf')
        
        for value in self.csp.domains[var]:
            test_assignment = assignment.copy()
            test_assignment[var] = value
            conflicts = self._count_violations(test_assignment)
            
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_value = value
        
        return best_value
```

## Simulated Annealing

### Algorithm Description

Simulated annealing uses a temperature parameter to allow "uphill" moves that escape local optima. The temperature decreases over time, making the algorithm more greedy.

### Python Implementation

```python
class SimulatedAnnealingSolver(LocalSearchSolver):
    """Simulated annealing solver for CSPs."""
    
    def __init__(self, csp, initial_temperature: float = 100.0, cooling_rate: float = 0.95, 
                 min_temperature: float = 0.1):
        super().__init__(csp)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
    
    def solve(self) -> Optional[Dict[str, Any]]:
        """Solve CSP using simulated annealing."""
        current = self._generate_initial_assignment()
        current_state = self._evaluate_assignment(current)
        
        best_state = current_state
        temperature = self.initial_temperature
        
        for iteration in range(self.max_iterations):
            self.iterations = iteration
            
            # Check if we found a solution
            if current_state.violations == 0:
                return current_state.assignment
            
            # Generate random neighbor
            neighbor = self._generate_random_neighbor(current_state.assignment)
            
            # Calculate score difference
            delta = neighbor.score - current_state.score
            
            # Accept or reject neighbor
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_state = neighbor
                current = neighbor.assignment
                
                if neighbor.score < best_state.score:
                    best_state = neighbor
            
            # Cool down
            temperature *= self.cooling_rate
            if temperature < self.min_temperature:
                break
        
        # Return best solution found
        if best_state.violations == 0:
            return best_state.assignment
        else:
            return None
    
    def _generate_random_neighbor(self, assignment: Dict[str, Any]) -> LocalSearchState:
        """Generate a random neighbor."""
        var = random.choice(self.csp.variables)
        value = random.choice(self.csp.domains[var])
        
        neighbor = assignment.copy()
        neighbor[var] = value
        
        return self._evaluate_assignment(neighbor)
```

## Tabu Search

### Algorithm Description

Tabu search maintains a list of recently visited solutions to prevent cycling and escape local optima.

### Python Implementation

```python
class TabuSearchSolver(LocalSearchSolver):
    """Tabu search solver for CSPs."""
    
    def __init__(self, csp, tabu_size: int = 10):
        super().__init__(csp)
        self.tabu_size = tabu_size
        self.tabu_list = []
    
    def solve(self) -> Optional[Dict[str, Any]]:
        """Solve CSP using tabu search."""
        current = self._generate_initial_assignment()
        current_state = self._evaluate_assignment(current)
        
        best_state = current_state
        
        for iteration in range(self.max_iterations):
            self.iterations = iteration
            
            # Check if we found a solution
            if current_state.violations == 0:
                return current_state.assignment
            
            # Generate neighbors
            neighbors = self._generate_neighbors(current_state.assignment)
            
            # Find best non-tabu neighbor
            best_neighbor = self._select_best_non_tabu_neighbor(neighbors)
            
            # Move to neighbor
            current_state = best_neighbor
            current = best_neighbor.assignment
            
            # Update tabu list
            self._update_tabu_list(current)
            
            # Update best solution
            if best_neighbor.score < best_state.score:
                best_state = best_neighbor
        
        # Return best solution found
        if best_state.violations == 0:
            return best_state.assignment
        else:
            return None
    
    def _select_best_non_tabu_neighbor(self, neighbors: List[LocalSearchState]) -> LocalSearchState:
        """Select best neighbor that is not in tabu list."""
        non_tabu_neighbors = []
        
        for neighbor in neighbors:
            if not self._is_tabu(neighbor.assignment):
                non_tabu_neighbors.append(neighbor)
        
        if non_tabu_neighbors:
            return min(non_tabu_neighbors, key=lambda x: x.score)
        else:
            # If all neighbors are tabu, select the best one anyway
            return min(neighbors, key=lambda x: x.score)
    
    def _is_tabu(self, assignment: Dict[str, Any]) -> bool:
        """Check if assignment is in tabu list."""
        # Simple hash-based tabu check
        assignment_hash = hash(tuple(sorted(assignment.items())))
        return assignment_hash in self.tabu_list
    
    def _update_tabu_list(self, assignment: Dict[str, Any]):
        """Update tabu list with current assignment."""
        assignment_hash = hash(tuple(sorted(assignment.items())))
        self.tabu_list.append(assignment_hash)
        
        # Keep tabu list size bounded
        if len(self.tabu_list) > self.tabu_size:
            self.tabu_list.pop(0)
```

## Genetic Algorithm

### Algorithm Description

Genetic algorithms use evolutionary principles to solve CSPs:
1. Maintain a population of assignments
2. Select parents based on fitness
3. Create offspring through crossover and mutation
4. Replace population with offspring

### Python Implementation

```python
class GeneticAlgorithmSolver(LocalSearchSolver):
    """Genetic algorithm solver for CSPs."""
    
    def __init__(self, csp, population_size: int = 50, mutation_rate: float = 0.1, 
                 crossover_rate: float = 0.8, tournament_size: int = 3):
        super().__init__(csp)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
    
    def solve(self) -> Optional[Dict[str, Any]]:
        """Solve CSP using genetic algorithm."""
        # Initialize population
        population = [self._generate_initial_assignment() for _ in range(self.population_size)]
        
        for generation in range(self.max_iterations):
            self.iterations = generation
            
            # Evaluate population
            population_states = [self._evaluate_assignment(assignment) for assignment in population]
            
            # Check if we found a solution
            best_state = min(population_states, key=lambda x: x.score)
            if best_state.violations == 0:
                return best_state.assignment
            
            # Create new population
            new_population = []
            for _ in range(self.population_size):
                # Selection
                parent1 = self._tournament_selection(population_states)
                parent2 = self._tournament_selection(population_states)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1.assignment, parent2.assignment)
                else:
                    child = parent1.assignment.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        # Return best solution found
        final_states = [self._evaluate_assignment(assignment) for assignment in population]
        best_state = min(final_states, key=lambda x: x.score)
        
        if best_state.violations == 0:
            return best_state.assignment
        else:
            return None
    
    def _tournament_selection(self, population_states: List[LocalSearchState]) -> LocalSearchState:
        """Select individual using tournament selection."""
        tournament = random.sample(population_states, self.tournament_size)
        return min(tournament, key=lambda x: x.score)
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two parents."""
        child = {}
        
        for var in self.csp.variables:
            if random.random() < 0.5:
                child[var] = parent1[var]
            else:
                child[var] = parent2[var]
        
        return child
    
    def _mutate(self, assignment: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an assignment."""
        mutated = assignment.copy()
        
        # Randomly change one variable
        var = random.choice(self.csp.variables)
        mutated[var] = random.choice(self.csp.domains[var])
        
        return mutated
```

## Performance Comparison

### Comparison Function

```python
def compare_local_search_algorithms(csp):
    """Compare different local search algorithms."""
    algorithms = [
        ("Hill Climbing", HillClimbingSolver(csp)),
        ("Min-Conflicts", MinConflictsSolver(csp)),
        ("Simulated Annealing", SimulatedAnnealingSolver(csp)),
        ("Tabu Search", TabuSearchSolver(csp)),
        ("Genetic Algorithm", GeneticAlgorithmSolver(csp))
    ]
    
    results = {}
    
    for name, solver in algorithms:
        solution = solver.solve()
        stats = solver.get_statistics()
        
        results[name] = {
            'solution_found': solution is not None,
            'iterations': stats['iterations'],
            'solution': solution
        }
    
    return results
```

## Applications

### 1. Large-Scale Problems

Local search is particularly effective for large CSPs where systematic search is impractical.

### 2. Real-Time Systems

The fast convergence makes local search suitable for real-time applications.

### 3. Constraint Optimization

Local search can be adapted for constraint optimization problems.

### 4. Hybrid Methods

Local search can be combined with systematic search methods.

## Summary

Local search provides powerful approaches to CSP solving:

1. **Speed**: Often much faster than systematic search
2. **Scalability**: Performance scales better with problem size
3. **Flexibility**: Can handle various problem types and constraints
4. **Hybridization**: Works well when combined with other methods

Key considerations:
- **Incompleteness**: May not find solutions even when they exist
- **Parameter Tuning**: Performance depends on parameter settings
- **Local Optima**: Can get stuck in suboptimal solutions
- **Randomness**: Results may vary between runs

Understanding local search is essential for solving large-scale CSPs efficiently and for developing hybrid solving approaches. 