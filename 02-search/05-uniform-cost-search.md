# Uniform Cost Search (UCS)

## Overview

Uniform Cost Search (UCS) is an uninformed search algorithm that finds the optimal path from the initial state to the goal state by expanding nodes in order of their path cost. It is guaranteed to find the optimal solution when all step costs are positive.

## Algorithm Description

### Core Principle

UCS expands the node with the lowest path cost from the initial state. It uses a priority queue (min-heap) to maintain the frontier, ensuring that the node with the smallest accumulated cost is always selected next.

### Mathematical Foundation

**Optimality Theorem**: UCS is optimal when all step costs are positive.

**Proof**: 
1. UCS expands nodes in order of increasing path cost
2. When a goal node is expanded, its path cost is the minimum among all paths to any goal
3. Therefore, UCS finds the optimal path

**Completeness**: UCS is complete if the step costs are bounded below by a positive constant.

## Algorithm Implementation

### Basic UCS Implementation

```python
import heapq
from collections import defaultdict

def uniform_cost_search(problem):
    """
    Uniform Cost Search implementation
    
    Args:
        problem: Search problem with methods:
            - get_initial_state()
            - is_goal_state(state)
            - get_actions(state)
            - get_next_state(state, action)
            - get_action_cost(state, action, next_state)
    
    Returns:
        dict: Solution containing path, path_cost, nodes_expanded
    """
    # Initialize
    initial_state = problem.get_initial_state()
    frontier = [(0, initial_state, [])]  # (cost, state, path)
    explored = set()
    nodes_expanded = 0
    max_frontier_size = 0
    
    while frontier:
        # Get node with minimum cost
        current_cost, current_state, path = heapq.heappop(frontier)
        nodes_expanded += 1
        
        # Check if goal
        if problem.is_goal_state(current_state):
            return {
                'path': path,
                'path_cost': current_cost,
                'nodes_expanded': nodes_expanded,
                'frontier_size': len(frontier),
                'explored_size': len(explored),
                'max_frontier_size': max_frontier_size
            }
        
        # Skip if already explored
        if current_state in explored:
            continue
        
        explored.add(current_state)
        
        # Expand current state
        actions = problem.get_actions(current_state)
        for action in actions:
            next_state = problem.get_next_state(current_state, action)
            if next_state and next_state not in explored:
                action_cost = problem.get_action_cost(current_state, action, next_state)
                new_cost = current_cost + action_cost
                new_path = path + [action]
                
                heapq.heappush(frontier, (new_cost, next_state, new_path))
        
        max_frontier_size = max(max_frontier_size, len(frontier))
    
    return None  # No solution found
```

### Enhanced UCS with Path Tracking

```python
class UCSNode:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
    
    def __lt__(self, other):
        # For priority queue comparison
        return self.path_cost < other.path_cost
    
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

def uniform_cost_search_enhanced(problem):
    """
    Enhanced UCS with better path tracking and node management
    """
    # Initialize
    initial_node = UCSNode(problem.get_initial_state())
    frontier = [initial_node]
    explored = set()
    nodes_expanded = 0
    max_frontier_size = 0
    
    while frontier:
        # Get node with minimum cost
        current_node = heapq.heappop(frontier)
        nodes_expanded += 1
        
        # Check if goal
        if problem.is_goal_state(current_node.state):
            return {
                'path': current_node.get_path(),
                'path_cost': current_node.path_cost,
                'nodes_expanded': nodes_expanded,
                'frontier_size': len(frontier),
                'explored_size': len(explored),
                'max_frontier_size': max_frontier_size,
                'path_states': current_node.get_path_states()
            }
        
        # Skip if already explored
        if current_node.state in explored:
            continue
        
        explored.add(current_node.state)
        
        # Expand current state
        actions = problem.get_actions(current_node.state)
        for action in actions:
            next_state = problem.get_next_state(current_node.state, action)
            if next_state and next_state not in explored:
                action_cost = problem.get_action_cost(current_node.state, action, next_state)
                new_cost = current_node.path_cost + action_cost
                
                # Create new node
                new_node = UCSNode(next_state, current_node, action, new_cost)
                heapq.heappush(frontier, new_node)
        
        max_frontier_size = max(max_frontier_size, len(frontier))
    
    return None  # No solution found
```

## Problem Examples

### 1. Graph Search Problem

```python
class GraphSearchProblem:
    def __init__(self, graph, start, goal):
        """
        graph: dict mapping nodes to list of (neighbor, cost) tuples
        start: starting node
        goal: goal node
        """
        self.graph = graph
        self.start = start
        self.goal = goal
    
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
        # Find the cost for this edge
        for neighbor, cost in self.graph[state]:
            if neighbor == action:
                return cost
        return float('inf')

# Example usage
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('A', 4), ('C', 1), ('D', 5)],
    'C': [('A', 2), ('B', 1), ('D', 8), ('E', 10)],
    'D': [('B', 5), ('C', 8), ('E', 2)],
    'E': [('C', 10), ('D', 2)]
}

problem = GraphSearchProblem(graph, 'A', 'E')
result = uniform_cost_search(problem)

if result:
    print(f"Optimal path: {' -> '.join(result['path'])}")
    print(f"Path cost: {result['path_cost']}")
    print(f"Nodes expanded: {result['nodes_expanded']}")
else:
    print("No path found")
```

### 2. Grid World Problem

```python
class GridWorldProblem:
    def __init__(self, grid, start, goal):
        """
        grid: 2D list where 0=free, 1=wall
        start: (row, col) tuple
        goal: (row, col) tuple
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = start
        self.goal = goal
    
    def get_initial_state(self):
        return self.start
    
    def is_goal_state(self, state):
        return state == self.goal
    
    def get_actions(self, state):
        row, col = state
        actions = []
        
        # Possible moves: up, down, left, right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in moves:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.rows and 
                0 <= new_col < self.cols and 
                self.grid[new_row][new_col] == 0):
                actions.append((new_row, new_col))
        
        return actions
    
    def get_next_state(self, state, action):
        return action
    
    def get_action_cost(self, state, action, next_state):
        # Manhattan distance cost
        return 1

# Example usage
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
goal = (4, 4)

problem = GridWorldProblem(grid, start, goal)
result = uniform_cost_search(problem)

if result:
    print(f"Optimal path length: {len(result['path'])}")
    print(f"Path cost: {result['path_cost']}")
    print(f"Path: {result['path']}")
else:
    print("No path found")
```

## Performance Analysis

### Time and Space Complexity

```python
def analyze_ucs_performance(problem, max_nodes=10000):
    """
    Analyze UCS performance characteristics
    """
    # Initialize
    initial_state = problem.get_initial_state()
    frontier = [(0, initial_state, [])]
    explored = set()
    nodes_expanded = 0
    frontier_sizes = []
    explored_sizes = []
    
    while frontier and nodes_expanded < max_nodes:
        current_cost, current_state, path = heapq.heappop(frontier)
        nodes_expanded += 1
        
        # Record sizes
        frontier_sizes.append(len(frontier))
        explored_sizes.append(len(explored))
        
        # Check if goal
        if problem.is_goal_state(current_state):
            return {
                'path': path,
                'path_cost': current_cost,
                'nodes_expanded': nodes_expanded,
                'frontier_sizes': frontier_sizes,
                'explored_sizes': explored_sizes,
                'max_frontier_size': max(frontier_sizes),
                'max_explored_size': max(explored_sizes),
                'success': True
            }
        
        # Skip if already explored
        if current_state in explored:
            continue
        
        explored.add(current_state)
        
        # Expand current state
        actions = problem.get_actions(current_state)
        for action in actions:
            next_state = problem.get_next_state(current_state, action)
            if next_state and next_state not in explored:
                action_cost = problem.get_action_cost(current_state, action, next_state)
                new_cost = current_cost + action_cost
                new_path = path + [action]
                
                heapq.heappush(frontier, (new_cost, next_state, new_path))
    
    return {
        'nodes_expanded': nodes_expanded,
        'frontier_sizes': frontier_sizes,
        'explored_sizes': explored_sizes,
        'max_frontier_size': max(frontier_sizes) if frontier_sizes else 0,
        'max_explored_size': max(explored_sizes) if explored_sizes else 0,
        'success': False
    }
```

### Branching Factor Analysis

```python
def calculate_branching_factor(problem, max_depth=10):
    """
    Calculate the effective branching factor of the search tree
    """
    # BFS to count nodes at each level
    initial_state = problem.get_initial_state()
    frontier = [(initial_state, 0)]  # (state, depth)
    explored = set()
    nodes_at_depth = defaultdict(int)
    
    while frontier:
        current_state, depth = frontier.pop(0)
        
        if depth > max_depth:
            break
        
        if current_state in explored:
            continue
        
        explored.add(current_state)
        nodes_at_depth[depth] += 1
        
        # Expand
        actions = problem.get_actions(current_state)
        for action in actions:
            next_state = problem.get_next_state(current_state, action)
            if next_state and next_state not in explored:
                frontier.append((next_state, depth + 1))
    
    # Calculate branching factor
    total_nodes = sum(nodes_at_depth.values())
    if total_nodes > 1:
        # Use the formula: N = (b^(d+1) - 1) / (b - 1)
        # For large d, b ≈ N^(1/d)
        max_depth_found = max(nodes_at_depth.keys())
        if max_depth_found > 0:
            branching_factor = total_nodes ** (1.0 / max_depth_found)
            return branching_factor
    
    return 1.0
```

## UCS Variants and Optimizations

### 1. UCS with Early Goal Test

```python
def uniform_cost_search_early_goal(problem):
    """
    UCS with early goal test - check for goal when adding to frontier
    """
    initial_state = problem.get_initial_state()
    frontier = [(0, initial_state, [])]
    explored = set()
    nodes_expanded = 0
    
    while frontier:
        current_cost, current_state, path = heapq.heappop(frontier)
        nodes_expanded += 1
        
        # Early goal test
        if problem.is_goal_state(current_state):
            return {
                'path': path,
                'path_cost': current_cost,
                'nodes_expanded': nodes_expanded,
                'frontier_size': len(frontier),
                'explored_size': len(explored)
            }
        
        if current_state in explored:
            continue
        
        explored.add(current_state)
        
        # Expand and check for goal immediately
        actions = problem.get_actions(current_state)
        for action in actions:
            next_state = problem.get_next_state(current_state, action)
            if next_state and next_state not in explored:
                action_cost = problem.get_action_cost(current_state, action, next_state)
                new_cost = current_cost + action_cost
                new_path = path + [action]
                
                # Early goal test for next state
                if problem.is_goal_state(next_state):
                    return {
                        'path': new_path,
                        'path_cost': new_cost,
                        'nodes_expanded': nodes_expanded,
                        'frontier_size': len(frontier),
                        'explored_size': len(explored)
                    }
                
                heapq.heappush(frontier, (new_cost, next_state, new_path))
    
    return None
```

### 2. UCS with Cost Threshold

```python
def uniform_cost_search_threshold(problem, cost_threshold=float('inf')):
    """
    UCS with cost threshold - stop if path cost exceeds threshold
    """
    initial_state = problem.get_initial_state()
    frontier = [(0, initial_state, [])]
    explored = set()
    nodes_expanded = 0
    
    while frontier:
        current_cost, current_state, path = heapq.heappop(frontier)
        
        # Check threshold
        if current_cost > cost_threshold:
            continue
        
        nodes_expanded += 1
        
        if problem.is_goal_state(current_state):
            return {
                'path': path,
                'path_cost': current_cost,
                'nodes_expanded': nodes_expanded,
                'frontier_size': len(frontier),
                'explored_size': len(explored)
            }
        
        if current_state in explored:
            continue
        
        explored.add(current_state)
        
        actions = problem.get_actions(current_state)
        for action in actions:
            next_state = problem.get_next_state(current_state, action)
            if next_state and next_state not in explored:
                action_cost = problem.get_action_cost(current_state, action, next_state)
                new_cost = current_cost + action_cost
                
                # Only add if within threshold
                if new_cost <= cost_threshold:
                    new_path = path + [action]
                    heapq.heappush(frontier, (new_cost, next_state, new_path))
    
    return None
```

### 3. Bidirectional UCS

```python
def bidirectional_ucs(problem):
    """
    Bidirectional UCS - search from both start and goal
    """
    # Forward search
    forward_frontier = [(0, problem.get_initial_state(), [])]
    forward_explored = {}
    
    # Backward search (assuming we can reverse the problem)
    goal_states = problem.get_goal_states() if hasattr(problem, 'get_goal_states') else [problem.get_goal_state()]
    backward_frontier = [(0, goal_state, []) for goal_state in goal_states]
    backward_explored = {goal_state: (0, []) for goal_state in goal_states}
    
    while forward_frontier and backward_frontier:
        # Forward step
        if forward_frontier:
            current_cost, current_state, path = heapq.heappop(forward_frontier)
            
            # Check if we've reached backward search
            if current_state in backward_explored:
                backward_cost, backward_path = backward_explored[current_state]
                total_cost = current_cost + backward_cost
                return {
                    'path': path + list(reversed(backward_path)),
                    'path_cost': total_cost,
                    'nodes_expanded': len(forward_explored) + len(backward_explored)
                }
            
            # Expand forward
            if current_state not in forward_explored:
                forward_explored[current_state] = (current_cost, path)
                actions = problem.get_actions(current_state)
                for action in actions:
                    next_state = problem.get_next_state(current_state, action)
                    if next_state and next_state not in forward_explored:
                        action_cost = problem.get_action_cost(current_state, action, next_state)
                        new_cost = current_cost + action_cost
                        new_path = path + [action]
                        heapq.heappush(forward_frontier, (new_cost, next_state, new_path))
        
        # Backward step (simplified - would need inverse problem definition)
        # Implementation depends on problem structure
    
    return None
```

## Comparison with Other Algorithms

```python
def compare_search_algorithms(problem):
    """
    Compare UCS with other search algorithms
    """
    algorithms = {
        'UCS': uniform_cost_search,
        'BFS': breadth_first_search,  # Assuming this exists
        'DFS': depth_first_search,    # Assuming this exists
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        try:
            result = algorithm(problem)
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
```

## Summary

Uniform Cost Search is a fundamental algorithm that guarantees optimality for positive step costs. Key characteristics include:

1. **Optimality**: Guaranteed to find the optimal path
2. **Completeness**: Will find a solution if one exists
3. **Time Complexity**: $O(b^d)$ where $b$ is branching factor, $d$ is solution depth
4. **Space Complexity**: $O(b^d)$ - stores all nodes in frontier
5. **Best Use Cases**: When optimality is required and no heuristic is available

UCS serves as the foundation for more advanced algorithms like A* and is particularly useful when:
- All step costs are positive
- Optimality is required
- No domain-specific heuristics are available
- The state space is manageable in size 