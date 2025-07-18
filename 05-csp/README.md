# Constraint Satisfaction Problems (CSP)

This section covers the fundamental concepts and algorithms for solving constraint satisfaction problems, a powerful framework for modeling and solving combinatorial problems.

## Overview

Constraint Satisfaction Problems (CSPs) are mathematical problems defined as a set of objects whose state must satisfy a number of constraints or limitations. CSPs represent the states as a set of variables, the domains of those variables, and a set of constraints that specify the allowable combinations of values for subsets of variables.

## Definitions

### Core Concepts

- **Variables**: A set of variables X = {X₁, X₂, ..., Xₙ}
- **Domains**: Each variable Xᵢ has a domain Dᵢ of possible values
- **Constraints**: A set of constraints C that specify allowable combinations of values
- **Solution**: An assignment of values to all variables that satisfies all constraints
- **Consistency**: A state where all constraints are satisfied

### Formal Definition

A CSP is a triple (X, D, C) where:
- X = {X₁, X₂, ..., Xₙ} is a set of variables
- D = {D₁, D₂, ..., Dₙ} is a set of domains where Dᵢ is the domain of Xᵢ
- C = {C₁, C₂, ..., Cₘ} is a set of constraints

## Examples

### Classic CSP Examples

1. **Map Coloring Problem**
   - Variables: Regions on a map
   - Domains: Available colors
   - Constraints: Adjacent regions must have different colors

2. **N-Queens Problem**
   - Variables: Queen positions on an N×N chessboard
   - Domains: Board positions
   - Constraints: No two queens can attack each other

3. **Sudoku**
   - Variables: Empty cells
   - Domains: Numbers 1-9
   - Constraints: Each row, column, and 3×3 box must contain unique numbers

4. **Scheduling Problems**
   - Variables: Tasks or events
   - Domains: Time slots or resources
   - Constraints: Precedence, resource availability, time windows

## Dynamic Ordering

### Variable Ordering Heuristics

1. **Minimum Remaining Values (MRV)**
   - Choose the variable with the fewest legal values remaining
   - Reduces branching factor early in the search

2. **Degree Heuristic**
   - Choose the variable with the most constraints on remaining variables
   - Helps reduce future branching

3. **Least Constraining Value**
   - Choose the value that rules out the fewest values in the remaining variables
   - Maximizes future choices

### Value Ordering

- **Least Constraining Value**: Choose values that leave the most options for other variables
- **Domain-specific heuristics**: Use problem-specific knowledge to order values

## Arc Consistency

### Definition

Arc consistency (AC-3) ensures that for every pair of variables (X, Y), every value in the domain of X has at least one compatible value in the domain of Y.

### AC-3 Algorithm

```
function AC-3(csp):
    queue = all arcs in csp
    while queue is not empty:
        (X, Y) = queue.pop()
        if revise(csp, X, Y):
            if size of D[X] == 0:
                return false
            for each Z in neighbors(X) - {Y}:
                add (Z, X) to queue
    return true

function revise(csp, X, Y):
    revised = false
    for each x in D[X]:
        if no y in D[Y] satisfies constraint(X, Y):
            delete x from D[X]
            revised = true
    return revised
```

### Benefits

- Reduces search space before backtracking
- Can solve some problems without search
- Provides early failure detection

## Beam Search

### Overview

Beam search is a heuristic search algorithm that explores a graph by expanding the most promising nodes in a limited set. It's a variation of breadth-first search that uses a heuristic function to evaluate and rank nodes.

### Algorithm

```
function beam_search(problem, beam_width):
    frontier = [initial_state]
    while frontier is not empty:
        candidates = []
        for each state in frontier:
            for each successor in successors(state):
                candidates.append(successor)
        frontier = select_best(candidates, beam_width)
        if goal_test(frontier):
            return solution
    return failure
```

### Characteristics

- **Memory efficient**: Only keeps k best nodes at each level
- **Incomplete**: May miss optimal solutions
- **Fast**: Reduces exponential growth of search space
- **Heuristic-dependent**: Quality depends on heuristic function

### Applications in CSP

- Variable ordering in constraint propagation
- Value selection heuristics
- Local search initialization

## Local Search

### Overview

Local search algorithms for CSPs start with a complete assignment and iteratively improve it by making small changes to satisfy more constraints.

### Hill Climbing

```
function hill_climbing(csp):
    current = random_assignment()
    while true:
        neighbor = best_neighbor(current)
        if score(neighbor) <= score(current):
            return current
        current = neighbor
```

### Min-Conflicts Algorithm

```
function min_conflicts(csp, max_steps):
    current = random_assignment()
    for i = 1 to max_steps:
        if current satisfies all constraints:
            return current
        var = randomly_choose_conflicted_variable(current)
        value = value_that_minimizes_conflicts(var, current)
        current[var] = value
    return failure
```

### Simulated Annealing

```
function simulated_annealing(csp, initial_temp, cooling_rate):
    current = random_assignment()
    temp = initial_temp
    while temp > 0:
        neighbor = random_neighbor(current)
        delta = score(neighbor) - score(current)
        if delta > 0 or random() < exp(delta / temp):
            current = neighbor
        temp *= cooling_rate
    return current
```

### Advantages

- **Scalability**: Works well for large problems
- **Flexibility**: Can handle various constraint types
- **Incremental**: Can start from partial solutions
- **Parallelizable**: Multiple searches can run simultaneously

### Limitations

- **Incomplete**: May not find solutions even when they exist
- **Local optima**: Can get stuck in suboptimal solutions
- **Parameter sensitivity**: Performance depends on tuning

## Advanced Topics

### Constraint Propagation

- **Path Consistency**: Generalizes arc consistency to triples of variables
- **k-Consistency**: Generalizes to k-tuples of variables
- **Global Constraints**: Specialized algorithms for common constraint patterns

### Hybrid Methods

- **Backtracking with Local Search**: Use local search to escape local optima
- **Constraint-Based Local Search**: Combine constraint propagation with local search
- **Genetic Algorithms**: Use evolutionary approaches for CSPs

### Real-World Applications

- **Scheduling**: Course scheduling, employee scheduling, project planning
- **Resource Allocation**: Network routing, frequency assignment
- **Configuration**: Product configuration, software configuration
- **Planning**: Automated planning, logistics optimization

## Summary

Constraint Satisfaction Problems provide a powerful framework for modeling and solving a wide variety of combinatorial problems. The key to effective CSP solving lies in:

1. **Problem modeling**: Choosing appropriate variables, domains, and constraints
2. **Search strategy**: Selecting between systematic search and local search
3. **Constraint propagation**: Using consistency algorithms to reduce search space
4. **Heuristics**: Applying domain knowledge to guide search efficiently

The combination of these techniques enables solving complex real-world problems that would be intractable with naive approaches. 