# Tree Search

## Overview

Tree search algorithms explore the state space by systematically traversing a tree structure where each node represents a state and each edge represents an action. The choice of traversal strategy significantly impacts the algorithm's performance and solution quality.

## Search Tree Structure

### Mathematical Foundation

A search tree is a directed graph $T = (V, E)$ where:
- $V$ is the set of nodes (states)
- $E$ is the set of edges (actions)
- Each node has a parent (except the root)
- The root represents the initial state
- Leaf nodes represent unexplored states

### Tree Properties

```python
class SearchTreeNode:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0 if parent is None else parent.depth + 1
        self.children = []
    
    def add_child(self, child):
        self.children.append(child)
    
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

class SearchTree:
    def __init__(self, root_state):
        self.root = SearchTreeNode(root_state)
        self.nodes = {root_state: self.root}
    
    def add_node(self, state, parent_state, action, cost):
        """Add a new node to the tree"""
        if state in self.nodes:
            return self.nodes[state]
        
        parent = self.nodes[parent_state]
        node = SearchTreeNode(state, parent, action, parent.path_cost + cost)
        parent.add_child(node)
        self.nodes[state] = node
        return node
    
    def get_node(self, state):
        """Get a node by state"""
        return self.nodes.get(state)
    
    def size(self):
        """Get the total number of nodes in the tree"""
        return len(self.nodes)
    
    def max_depth(self):
        """Get the maximum depth of the tree"""
        return max(node.depth for node in self.nodes.values())
```

## Depth-First Search (DFS)

### Algorithm Description

DFS explores as far as possible along each branch before backtracking. It uses a stack (LIFO) data structure for the frontier.

### Mathematical Analysis

- **Time Complexity**: $O(b^d)$ where $b$ is branching factor, $d$ is maximum depth
- **Space Complexity**: $O(d)$ - only stores the current path
- **Completeness**: No (may get stuck in infinite loops)
- **Optimality**: No (may find suboptimal solutions)

### Implementation

```python
def depth_first_search(problem):
    """
    Depth-First Search implementation
    
    Args:
        problem: Search problem with get_initial_state(), get_actions(), etc.
    
    Returns:
        dict: Solution containing path, path_cost, nodes_expanded
    """
    # Initialize
    initial_state = problem.get_initial_state()
    frontier = [(initial_state, [])]  # (state, path)
    explored = set()
    nodes_expanded = 0
    
    while frontier:
        # Pop from stack (LIFO)
        current_state, path = frontier.pop()
        nodes_expanded += 1
        
        # Check if goal
        if problem.is_goal_state(current_state):
            return {
                'path': path,
                'path_cost': len(path),
                'nodes_expanded': nodes_expanded,
                'frontier_size': len(frontier),
                'explored_size': len(explored)
            }
        
        # Skip if already explored
        if current_state in explored:
            continue
        
        explored.add(current_state)
        
        # Expand current state
        actions = problem.get_actions(current_state)
        for action in reversed(actions):  # Reverse to maintain order in stack
            next_state = problem.get_next_state(current_state, action)
            if next_state and next_state not in explored:
                new_path = path + [action]
                frontier.append((next_state, new_path))
    
    return None  # No solution found

def depth_first_search_recursive(problem, state=None, path=None, explored=None):
    """
    Recursive implementation of DFS
    """
    if state is None:
        state = problem.get_initial_state()
        path = []
        explored = set()
    
    # Check if goal
    if problem.is_goal_state(state):
        return {
            'path': path,
            'path_cost': len(path),
            'nodes_expanded': len(explored)
        }
    
    # Mark as explored
    explored.add(state)
    
    # Try all actions
    actions = problem.get_actions(state)
    for action in actions:
        next_state = problem.get_next_state(state, action)
        if next_state and next_state not in explored:
            result = depth_first_search_recursive(
                problem, next_state, path + [action], explored
            )
            if result:
                return result
    
    return None
```

### DFS Variants

```python
def depth_limited_search(problem, limit):
    """
    DFS with depth limit to prevent infinite loops
    """
    def dls_recursive(state, path, depth):
        if problem.is_goal_state(state):
            return {
                'path': path,
                'path_cost': len(path),
                'nodes_expanded': 1
            }
        
        if depth == 0:
            return None  # Cutoff
        
        nodes_expanded = 1
        actions = problem.get_actions(state)
        for action in actions:
            next_state = problem.get_next_state(state, action)
            if next_state:
                result = dls_recursive(next_state, path + [action], depth - 1)
                if result:
                    result['nodes_expanded'] += nodes_expanded
                    return result
                nodes_expanded += result.get('nodes_expanded', 0) if result else 0
        
        return {'nodes_expanded': nodes_expanded}
    
    return dls_recursive(problem.get_initial_state(), [], limit)

def iterative_deepening_search(problem):
    """
    Iterative Deepening DFS - combines benefits of DFS and BFS
    """
    depth = 0
    total_nodes_expanded = 0
    
    while True:
        result = depth_limited_search(problem, depth)
        total_nodes_expanded += result.get('nodes_expanded', 0)
        
        if result and 'path' in result:
            result['total_nodes_expanded'] = total_nodes_expanded
            return result
        
        depth += 1
```

## Breadth-First Search (BFS)

### Algorithm Description

BFS explores all nodes at the current depth before moving to nodes at the next depth level. It uses a queue (FIFO) data structure.

### Mathematical Analysis

- **Time Complexity**: $O(b^d)$ where $b$ is branching factor, $d$ is solution depth
- **Space Complexity**: $O(b^d)$ - stores all nodes at current depth
- **Completeness**: Yes (if solution exists)
- **Optimality**: Yes (for unit costs)

### Implementation

```python
from collections import deque

def breadth_first_search(problem):
    """
    Breadth-First Search implementation
    """
    # Initialize
    initial_state = problem.get_initial_state()
    frontier = deque([(initial_state, [])])  # (state, path)
    explored = set()
    nodes_expanded = 0
    
    while frontier:
        # Pop from queue (FIFO)
        current_state, path = frontier.popleft()
        nodes_expanded += 1
        
        # Check if goal
        if problem.is_goal_state(current_state):
            return {
                'path': path,
                'path_cost': len(path),
                'nodes_expanded': nodes_expanded,
                'frontier_size': len(frontier),
                'explored_size': len(explored)
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
                new_path = path + [action]
                frontier.append((next_state, new_path))
    
    return None  # No solution found

def bidirectional_search(problem):
    """
    Bidirectional BFS - search from both start and goal
    """
    # Forward search from initial state
    forward_frontier = deque([(problem.get_initial_state(), [])])
    forward_explored = {problem.get_initial_state(): []}
    
    # Backward search from goal states
    goal_states = problem.get_goal_states() if hasattr(problem, 'get_goal_states') else [problem.get_goal_state()]
    backward_frontier = deque([(goal_state, []) for goal_state in goal_states])
    backward_explored = {goal_state: [] for goal_state in goal_states}
    
    while forward_frontier and backward_frontier:
        # Forward step
        if forward_frontier:
            current_state, path = forward_frontier.popleft()
            
            # Check if we've reached backward search
            if current_state in backward_explored:
                backward_path = backward_explored[current_state]
                return {
                    'path': path + list(reversed(backward_path)),
                    'path_cost': len(path) + len(backward_path),
                    'nodes_expanded': len(forward_explored) + len(backward_explored)
                }
            
            # Expand forward
            if current_state not in forward_explored:
                forward_explored[current_state] = path
                actions = problem.get_actions(current_state)
                for action in actions:
                    next_state = problem.get_next_state(current_state, action)
                    if next_state and next_state not in forward_explored:
                        forward_frontier.append((next_state, path + [action]))
        
        # Backward step
        if backward_frontier:
            current_state, path = backward_frontier.popleft()
            
            # Check if we've reached forward search
            if current_state in forward_explored:
                forward_path = forward_explored[current_state]
                return {
                    'path': forward_path + list(reversed(path)),
                    'path_cost': len(forward_path) + len(path),
                    'nodes_expanded': len(forward_explored) + len(backward_explored)
                }
            
            # Expand backward
            if current_state not in backward_explored:
                backward_explored[current_state] = path
                # For backward search, we need inverse actions
                inverse_actions = problem.get_inverse_actions(current_state) if hasattr(problem, 'get_inverse_actions') else []
                for action in inverse_actions:
                    prev_state = problem.get_previous_state(current_state, action)
                    if prev_state and prev_state not in backward_explored:
                        backward_frontier.append((prev_state, path + [action]))
    
    return None
```

## Iterative Deepening

### Algorithm Description

Iterative Deepening combines the space efficiency of DFS with the completeness and optimality of BFS by running DFS with increasing depth limits.

### Mathematical Analysis

- **Time Complexity**: $O(b^d)$ - same as BFS
- **Space Complexity**: $O(d)$ - same as DFS
- **Completeness**: Yes
- **Optimality**: Yes (for unit costs)

### Implementation

```python
def iterative_deepening_dfs(problem):
    """
    Iterative Deepening Depth-First Search
    """
    depth = 0
    total_nodes_expanded = 0
    
    while True:
        result = depth_limited_dfs(problem, depth)
        total_nodes_expanded += result.get('nodes_expanded', 0)
        
        if result and 'path' in result:
            result['total_nodes_expanded'] = total_nodes_expanded
            result['max_depth'] = depth
            return result
        
        depth += 1

def depth_limited_dfs(problem, limit):
    """
    DFS with depth limit
    """
    def dls_recursive(state, path, depth, explored):
        if problem.is_goal_state(state):
            return {
                'path': path,
                'path_cost': len(path),
                'nodes_expanded': 1
            }
        
        if depth == 0:
            return {'nodes_expanded': 1, 'cutoff': True}
        
        explored.add(state)
        nodes_expanded = 1
        cutoff_occurred = False
        
        actions = problem.get_actions(state)
        for action in actions:
            next_state = problem.get_next_state(state, action)
            if next_state and next_state not in explored:
                result = dls_recursive(next_state, path + [action], depth - 1, explored)
                nodes_expanded += result.get('nodes_expanded', 0)
                
                if result and 'path' in result:
                    return result
                
                if result.get('cutoff', False):
                    cutoff_occurred = True
        
        return {
            'nodes_expanded': nodes_expanded,
            'cutoff': cutoff_occurred
        }
    
    return dls_recursive(problem.get_initial_state(), [], limit, set())
```

## Performance Comparison

```python
def compare_tree_search_algorithms(problem):
    """
    Compare different tree search algorithms
    """
    algorithms = {
        'DFS': depth_first_search,
        'BFS': breadth_first_search,
        'Iterative Deepening': iterative_deepening_dfs
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        try:
            result = algorithm(problem)
            if result:
                results[name] = {
                    'path_length': len(result['path']),
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

def analyze_tree_properties(problem):
    """
    Analyze properties of the search tree
    """
    # Build complete tree up to certain depth
    tree = SearchTree(problem.get_initial_state())
    frontier = [(problem.get_initial_state(), None, 0)]
    max_depth = 5
    
    while frontier and tree.max_depth() < max_depth:
        current_state, action, cost = frontier.pop(0)
        
        if tree.get_node(current_state).depth >= max_depth:
            continue
        
        actions = problem.get_actions(current_state)
        for action in actions:
            next_state = problem.get_next_state(current_state, action)
            if next_state:
                action_cost = problem.get_action_cost(current_state, action, next_state)
                tree.add_node(next_state, current_state, action, action_cost)
                frontier.append((next_state, action, action_cost))
    
    return {
        'total_nodes': tree.size(),
        'max_depth': tree.max_depth(),
        'average_branching_factor': sum(len(node.children) for node in tree.nodes.values()) / tree.size(),
        'leaf_nodes': sum(1 for node in tree.nodes.values() if not node.children)
    }
```

## Summary

Tree search algorithms provide different trade-offs:

1. **DFS**: Space efficient but may not find optimal solutions
2. **BFS**: Guarantees optimality but uses more memory
3. **Iterative Deepening**: Best of both worlds but may be slower

The choice depends on:
- Memory constraints
- Optimality requirements
- Problem characteristics
- Available heuristics 