# A* Algorithm

## Overview

A* (A-star) is an informed search algorithm that finds the optimal path from an initial state to a goal state by using a heuristic function to guide the search. It combines the benefits of uniform cost search (optimality) with greedy best-first search (efficiency).

## Algorithm Description

### Core Principle

A* uses an evaluation function $f(n) = g(n) + h(n)$ where:
- $g(n)$ is the cost from the start node to node $n$
- $h(n)$ is the heuristic estimate from node $n$ to the goal
- $f(n)$ is the estimated total cost of the path through node $n$

The algorithm expands the node with the lowest $f(n)$ value, balancing between the actual cost incurred so far and the estimated cost to reach the goal.

### Mathematical Foundation

**Optimality Theorem**: A* is optimal when the heuristic function $h(n)$ is admissible.

**Admissible Heuristic**: $h(n) \leq h^*(n)$ for all nodes $n$, where $h^*(n)$ is the true cost from $n$ to the goal.

**Consistent Heuristic**: $h(n) \leq c(n, n') + h(n')$ for all nodes $n$ and successors $n'$, where $c(n, n')$ is the cost of the edge from $n$ to $n'$.

## Algorithm Implementation

### Basic A* Implementation

```python
import heapq
from collections import defaultdict

def a_star_search(problem, heuristic):
    """
    A* Search implementation
    
    Args:
        problem: Search problem with methods:
            - get_initial_state()
            - is_goal_state(state)
            - get_actions(state)
            - get_next_state(state, action)
            - get_action_cost(state, action, next_state)
        heuristic: Function that estimates cost from state to goal
    
    Returns:
        dict: Solution containing path, path_cost, nodes_expanded
    """
    # Initialize
    initial_state = problem.get_initial_state()
    frontier = [(0, 0, initial_state, [])]  # (f, g, state, path)
    explored = set()
    g_values = {initial_state: 0}  # Track g values for all nodes
    nodes_expanded = 0
    max_frontier_size = 0
    
    while frontier:
        # Get node with minimum f value
        f_value, g_value, current_state, path = heapq.heappop(frontier)
        nodes_expanded += 1
        
        # Check if goal
        if problem.is_goal_state(current_state):
            return {
                'path': path,
                'path_cost': g_value,
                'nodes_expanded': nodes_expanded,
                'frontier_size': len(frontier),
                'explored_size': len(explored),
                'max_frontier_size': max_frontier_size
            }
        
        # Skip if already explored with better g value
        if current_state in explored and g_values[current_state] < g_value:
            continue
        
        explored.add(current_state)
        
        # Expand current state
        actions = problem.get_actions(current_state)
        for action in actions:
            next_state = problem.get_next_state(current_state, action)
            if next_state:
                action_cost = problem.get_action_cost(current_state, action, next_state)
                new_g_value = g_value + action_cost
                
                # Check if this path to next_state is better
                if (next_state not in g_values or 
                    new_g_value < g_values[next_state]):
                    
                    g_values[next_state] = new_g_value
                    h_value = heuristic(next_state)
                    f_value = new_g_value + h_value
                    new_path = path + [action]
                    
                    heapq.heappush(frontier, (f_value, new_g_value, next_state, new_path))
        
        max_frontier_size = max(max_frontier_size, len(frontier))
    
    return None  # No solution found
```

### Enhanced A* with Node Tracking

```python
class AStarNode:
    def __init__(self, state, parent=None, action=None, g_value=0, h_value=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g_value = g_value
        self.h_value = h_value
        self.f_value = g_value + h_value
    
    def __lt__(self, other):
        # For priority queue comparison
        if self.f_value != other.f_value:
            return self.f_value < other.f_value
        # Tie-breaking by g value (prefer higher g for same f)
        return self.g_value > other.g_value
    
    def get_path(self):
        """Reconstruct the path from root to this node"""
        path = []
        current = self
        while current.parent is not None:
            path.append(current.action)
            current = current.parent
        return list(reversed(path))
    
    def get_path_states(self):
        """Get all states in the path from root to this node"""
        states = []
        current = self
        while current is not None:
            states.append(current.state)
            current = current.parent
        return list(reversed(states))

def a_star_search_enhanced(problem, heuristic):
    """
    Enhanced A* with better node management and path tracking
    """
    # Initialize
    initial_state = problem.get_initial_state()
    initial_h = heuristic(initial_state)
    initial_node = AStarNode(initial_state, g_value=0, h_value=initial_h)
    
    frontier = [initial_node]
    explored = set()
    g_values = {initial_state: 0}
    nodes_expanded = 0
    max_frontier_size = 0
    
    while frontier:
        # Get node with minimum f value
        current_node = heapq.heappop(frontier)
        nodes_expanded += 1
        
        # Check if goal
        if problem.is_goal_state(current_node.state):
            return {
                'path': current_node.get_path(),
                'path_cost': current_node.g_value,
                'nodes_expanded': nodes_expanded,
                'frontier_size': len(frontier),
                'explored_size': len(explored),
                'max_frontier_size': max_frontier_size,
                'path_states': current_node.get_path_states()
            }
        
        # Skip if already explored with better g value
        if current_node.state in explored and g_values[current_node.state] < current_node.g_value:
            continue
        
        explored.add(current_node.state)
        
        # Expand current state
        actions = problem.get_actions(current_node.state)
        for action in actions:
            next_state = problem.get_next_state(current_node.state, action)
            if next_state:
                action_cost = problem.get_action_cost(current_node.state, action, next_state)
                new_g_value = current_node.g_value + action_cost
                
                # Check if this path to next_state is better
                if (next_state not in g_values or 
                    new_g_value < g_values[next_state]):
                    
                    g_values[next_state] = new_g_value
                    h_value = heuristic(next_state)
                    
                    # Create new node
                    new_node = AStarNode(
                        next_state, 
                        current_node, 
                        action, 
                        new_g_value, 
                        h_value
                    )
                    heapq.heappush(frontier, new_node)
        
        max_frontier_size = max(max_frontier_size, len(frontier))
    
    return None  # No solution found
```

## Heuristic Functions

### Admissible Heuristics

```python
class HeuristicFunctions:
    """Collection of admissible heuristic functions"""
    
    @staticmethod
    def manhattan_distance(state, goal_state):
        """Manhattan distance for grid-based problems"""
        if isinstance(state, tuple) and isinstance(goal_state, tuple):
            return sum(abs(a - b) for a, b in zip(state, goal_state))
        return 0
    
    @staticmethod
    def euclidean_distance(state, goal_state):
        """Euclidean distance for continuous spaces"""
        if isinstance(state, tuple) and isinstance(goal_state, tuple):
            return sum((a - b) ** 2 for a, b in zip(state, goal_state)) ** 0.5
        return 0
    
    @staticmethod
    def hamming_distance(state, goal_state):
        """Hamming distance for string-based problems"""
        if isinstance(state, str) and isinstance(goal_state, str):
            return sum(1 for a, b in zip(state, goal_state) if a != b)
        return 0
    
    @staticmethod
    def zero_heuristic(state):
        """Zero heuristic (turns A* into UCS)"""
        return 0
    
    @staticmethod
    def random_heuristic(state):
        """Random heuristic (not admissible, for testing)"""
        import random
        return random.randint(0, 10)
    
    @staticmethod
    def pattern_database_heuristic(state, pattern_database):
        """Pattern database heuristic for sliding puzzles"""
        # This would require a pre-computed pattern database
        return pattern_database.get(state, 0)
    
    @staticmethod
    def linear_conflict_heuristic(state, goal_state):
        """Linear conflict heuristic for 8-puzzle"""
        if not isinstance(state, list) or not isinstance(goal_state, list):
            return 0
        
        conflicts = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                # Check if tiles are in linear conflict
                if (state[i] != 0 and state[j] != 0 and
                    state[i] < state[j] and
                    goal_state.index(state[i]) > goal_state.index(state[j])):
                    conflicts += 2
        
        return conflicts
```

### Heuristic Analysis

```python
def analyze_heuristic_quality(problem, heuristic, num_trials=100):
    """
    Analyze the quality of a heuristic function
    """
    def calculate_heuristic_error(problem, heuristic):
        """Calculate average heuristic error"""
        errors = []
        
        # Sample random states
        for _ in range(num_trials):
            # Generate random state (this depends on problem type)
            random_state = generate_random_state(problem)
            
            # Calculate heuristic value
            h_value = heuristic(random_state)
            
            # Calculate actual cost to goal (using UCS)
            actual_cost = calculate_actual_cost(problem, random_state)
            
            if actual_cost > 0:
                error = abs(h_value - actual_cost) / actual_cost
                errors.append(error)
        
        return {
            'average_error': sum(errors) / len(errors) if errors else 0,
            'max_error': max(errors) if errors else 0,
            'min_error': min(errors) if errors else 0,
            'admissible': all(h <= actual for h, actual in zip(heuristic_values, actual_costs))
        }
    
    return calculate_heuristic_error(problem, heuristic)

def test_heuristic_admissibility(problem, heuristic, sample_states):
    """
    Test if a heuristic is admissible
    """
    violations = []
    
    for state in sample_states:
        h_value = heuristic(state)
        actual_cost = calculate_actual_cost(problem, state)
        
        if h_value > actual_cost:
            violations.append({
                'state': state,
                'heuristic_value': h_value,
                'actual_cost': actual_cost,
                'violation': h_value - actual_cost
            })
    
    return {
        'is_admissible': len(violations) == 0,
        'violations': violations,
        'total_states_tested': len(sample_states)
    }
```

## Problem Examples

### 1. 8-Puzzle with A*

```python
class EightPuzzleAStar:
    """8-Puzzle problem with A* search"""
    
    def __init__(self, initial_board, goal_board=None):
        self.initial_board = initial_board
        self.goal_board = goal_board or [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.size = 3
    
    def get_initial_state(self):
        return self.initial_board
    
    def is_goal_state(self, state):
        return state == self.goal_board
    
    def get_actions(self, state):
        actions = []
        i, j = self.get_blank_position(state)
        
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for di, dj in moves:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size:
                actions.append((ni, nj))
        
        return actions
    
    def get_next_state(self, state, action):
        i, j = self.get_blank_position(state)
        ni, nj = action
        
        new_board = [row[:] for row in state]
        new_board[i][j], new_board[ni][nj] = new_board[ni][nj], new_board[i][j]
        
        return new_board
    
    def get_action_cost(self, state, action, next_state):
        return 1
    
    def get_blank_position(self, state):
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] == 0:
                    return i, j
        return None

def manhattan_heuristic_8puzzle(state, goal_state):
    """Manhattan distance heuristic for 8-puzzle"""
    total_distance = 0
    
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0:  # Don't count blank tile
                # Find position of this tile in goal state
                for gi in range(3):
                    for gj in range(3):
                        if goal_state[gi][gj] == state[i][j]:
                            total_distance += abs(i - gi) + abs(j - gj)
                            break
    
    return total_distance

# Example usage
initial_board = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
goal_board = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

problem = EightPuzzleAStar(initial_board, goal_board)
heuristic = lambda state: manhattan_heuristic_8puzzle(state, goal_board)

result = a_star_search(problem, heuristic)
if result:
    print(f"Solution found in {len(result['path'])} moves")
    print(f"Path cost: {result['path_cost']}")
    print(f"Nodes expanded: {result['nodes_expanded']}")
```

### 2. Graph Search with A*

```python
class GraphAStarProblem:
    """Graph search problem with A*"""
    
    def __init__(self, graph, start, goal, coordinates=None):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.coordinates = coordinates or {}
    
    def get_initial_state(self):
        return self.start
    
    def is_goal_state(self, state):
        return state == self.goal
    
    def get_actions(self, state):
        if state in self.graph:
            return [neighbor for neighbor, _ in self.graph[state]]
        return []
    
    def get_next_state(self, state, action):
        return action
    
    def get_action_cost(self, state, action, next_state):
        for neighbor, cost in self.graph[state]:
            if neighbor == action:
                return cost
        return float('inf')

def euclidean_heuristic(problem, state):
    """Euclidean distance heuristic for graph with coordinates"""
    if state not in problem.coordinates or problem.goal not in problem.coordinates:
        return 0
    
    state_pos = problem.coordinates[state]
    goal_pos = problem.coordinates[problem.goal]
    
    return ((state_pos[0] - goal_pos[0]) ** 2 + 
            (state_pos[1] - goal_pos[1]) ** 2) ** 0.5

# Example usage
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('A', 4), ('C', 1), ('D', 5)],
    'C': [('A', 2), ('B', 1), ('D', 8), ('E', 10)],
    'D': [('B', 5), ('C', 8), ('E', 2)],
    'E': [('C', 10), ('D', 2)]
}

coordinates = {
    'A': (0, 0),
    'B': (2, 0),
    'C': (1, 1),
    'D': (3, 1),
    'E': (2, 2)
}

problem = GraphAStarProblem(graph, 'A', 'E', coordinates)
heuristic = lambda state: euclidean_heuristic(problem, state)

result = a_star_search(problem, heuristic)
if result:
    print(f"Optimal path: {' -> '.join(result['path'])}")
    print(f"Path cost: {result['path_cost']}")
    print(f"Nodes expanded: {result['nodes_expanded']}")
```

## Performance Analysis

### Comparison with Other Algorithms

```python
def compare_search_algorithms(problem, heuristic):
    """
    Compare A* with other search algorithms
    """
    algorithms = {
        'A*': lambda: a_star_search(problem, heuristic),
        'UCS': lambda: uniform_cost_search(problem),
        'Greedy': lambda: greedy_best_first_search(problem, heuristic)
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        try:
            result = algorithm()
            if result:
                results[name] = {
                    'path_length': len(result['path']),
                    'path_cost': result['path_cost'],
                    'nodes_expanded': result['nodes_expanded'],
                    'frontier_size': result.get('frontier_size', 0),
                    'explored_size': result.get('explored_size', 0),
                    'success': True
                }
            else:
                results[name] = {'success': False}
        except Exception as e:
            results[name] = {'success': False, 'error': str(e)}
    
    return results

def analyze_a_star_performance(problem, heuristics):
    """
    Analyze A* performance with different heuristics
    """
    results = {}
    
    for name, heuristic in heuristics.items():
        try:
            result = a_star_search(problem, heuristic)
            if result:
                results[name] = {
                    'path_cost': result['path_cost'],
                    'nodes_expanded': result['nodes_expanded'],
                    'frontier_size': result.get('frontier_size', 0),
                    'explored_size': result.get('explored_size', 0),
                    'success': True
                }
            else:
                results[name] = {'success': False}
        except Exception as e:
            results[name] = {'success': False, 'error': str(e)}
    
    return results
```

### Heuristic Quality Impact

```python
def analyze_heuristic_impact(problem, heuristics):
    """
    Analyze how heuristic quality affects A* performance
    """
    results = {}
    
    for name, heuristic in heuristics.items():
        # Test heuristic quality
        quality = analyze_heuristic_quality(problem, heuristic)
        
        # Test A* performance
        performance = a_star_search(problem, heuristic)
        
        results[name] = {
            'heuristic_quality': quality,
            'performance': performance
        }
    
    return results

def plot_heuristic_comparison(results):
    """
    Plot comparison of different heuristics
    """
    import matplotlib.pyplot as plt
    
    names = list(results.keys())
    nodes_expanded = [results[name]['performance']['nodes_expanded'] 
                     for name in names if results[name]['performance']]
    path_costs = [results[name]['performance']['path_cost'] 
                 for name in names if results[name]['performance']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Nodes expanded
    ax1.bar(names, nodes_expanded)
    ax1.set_ylabel('Nodes Expanded')
    ax1.set_title('A* Performance: Nodes Expanded')
    ax1.tick_params(axis='x', rotation=45)
    
    # Path costs
    ax2.bar(names, path_costs)
    ax2.set_ylabel('Path Cost')
    ax2.set_title('A* Performance: Path Cost')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
```

## Advanced A* Techniques

### 1. Weighted A*

```python
def weighted_a_star_search(problem, heuristic, weight=1.5):
    """
    Weighted A* with f(n) = g(n) + w * h(n)
    """
    def weighted_heuristic(state):
        return weight * heuristic(state)
    
    return a_star_search(problem, weighted_heuristic)
```

### 2. Anytime A*

```python
def anytime_a_star_search(problem, heuristic, time_limit=10):
    """
    Anytime A* that improves solution over time
    """
    import time
    
    start_time = time.time()
    best_solution = None
    weight = 2.0
    
    while time.time() - start_time < time_limit:
        # Try with current weight
        solution = weighted_a_star_search(problem, heuristic, weight)
        
        if solution:
            if (best_solution is None or 
                solution['path_cost'] < best_solution['path_cost']):
                best_solution = solution
        
        # Reduce weight for next iteration
        weight = max(1.0, weight * 0.8)
    
    return best_solution
```

### 3. IDA* (Iterative Deepening A*)

```python
def ida_star_search(problem, heuristic):
    """
    Iterative Deepening A* implementation
    """
    def search_with_threshold(node, g_value, threshold, path):
        f_value = g_value + heuristic(node.state)
        
        if f_value > threshold:
            return f_value
        
        if problem.is_goal_state(node.state):
            return {
                'path': path,
                'path_cost': g_value,
                'found': True
            }
        
        min_f = float('inf')
        actions = problem.get_actions(node.state)
        
        for action in actions:
            next_state = problem.get_next_state(node.state, action)
            if next_state:
                action_cost = problem.get_action_cost(node.state, action, next_state)
                new_g = g_value + action_cost
                
                next_node = AStarNode(next_state, node, action, new_g, heuristic(next_state))
                result = search_with_threshold(next_node, new_g, threshold, path + [action])
                
                if isinstance(result, dict) and result.get('found'):
                    return result
                elif isinstance(result, (int, float)):
                    min_f = min(min_f, result)
        
        return min_f
    
    # Start with heuristic value of initial state
    initial_state = problem.get_initial_state()
    threshold = heuristic(initial_state)
    
    while True:
        initial_node = AStarNode(initial_state, g_value=0, h_value=heuristic(initial_state))
        result = search_with_threshold(initial_node, 0, threshold, [])
        
        if isinstance(result, dict) and result.get('found'):
            return result
        
        if result == float('inf'):
            return None  # No solution
        
        threshold = result  # Increase threshold for next iteration
```

## Summary

A* is a powerful search algorithm that combines the optimality of uniform cost search with the efficiency of informed search. Key characteristics include:

1. **Optimality**: Guaranteed when heuristic is admissible
2. **Completeness**: Will find solution if one exists
3. **Efficiency**: Often much faster than uninformed search
4. **Flexibility**: Can be adapted with different heuristics and weights

The choice of heuristic function significantly impacts performance, making A* particularly effective when good domain-specific heuristics are available. 