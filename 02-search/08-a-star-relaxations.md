# A* Relaxations

## Overview

A* relaxations are modifications of the standard A* algorithm that trade optimality for improved performance or additional capabilities. These variants address specific limitations of A* and provide solutions for different problem scenarios.

## Weighted A* (WA*)

### Algorithm Description

Weighted A* uses a modified evaluation function $f(n) = g(n) + w \cdot h(n)$ where $w > 1$ is a weight factor that gives more importance to the heuristic.

**Mathematical Properties**:
- **Suboptimality Bound**: The solution cost is at most $w$ times the optimal cost
- **Efficiency**: Often much faster than standard A* due to stronger heuristic guidance
- **Completeness**: Guaranteed to find a solution if one exists

### Implementation

```python
def weighted_a_star_search(problem, heuristic, weight=1.5):
    """
    Weighted A* Search implementation
    
    Args:
        problem: Search problem
        heuristic: Heuristic function
        weight: Weight factor (w > 1)
    
    Returns:
        dict: Solution with suboptimality bound
    """
    def weighted_heuristic(state):
        return weight * heuristic(state)
    
    # Use standard A* with weighted heuristic
    result = a_star_search(problem, weighted_heuristic)
    
    if result:
        result['suboptimality_bound'] = weight
        result['weight_used'] = weight
    
    return result

def adaptive_weighted_a_star(problem, heuristic, initial_weight=2.0, decay_rate=0.9):
    """
    Adaptive Weighted A* that reduces weight over time
    """
    weight = initial_weight
    best_solution = None
    
    while weight >= 1.0:
        solution = weighted_a_star_search(problem, heuristic, weight)
        
        if solution:
            if (best_solution is None or 
                solution['path_cost'] < best_solution['path_cost']):
                best_solution = solution
        
        weight *= decay_rate
    
    return best_solution

# Example usage
def compare_weighted_a_star(problem, heuristic):
    """Compare different weights for A*"""
    weights = [1.0, 1.2, 1.5, 2.0, 3.0]
    results = {}
    
    for weight in weights:
        result = weighted_a_star_search(problem, heuristic, weight)
        if result:
            results[weight] = {
                'path_cost': result['path_cost'],
                'nodes_expanded': result['nodes_expanded'],
                'suboptimality_bound': result['suboptimality_bound']
            }
    
    return results
```

### Performance Analysis

```python
def analyze_weighted_a_star_performance(problem, heuristic, weights):
    """
    Analyze performance of weighted A* with different weights
    """
    results = {}
    
    for weight in weights:
        start_time = time.time()
        result = weighted_a_star_search(problem, heuristic, weight)
        end_time = time.time()
        
        if result:
            results[weight] = {
                'path_cost': result['path_cost'],
                'nodes_expanded': result['nodes_expanded'],
                'execution_time': end_time - start_time,
                'suboptimality_ratio': result['path_cost'] / optimal_cost if 'optimal_cost' in globals() else None
            }
    
    return results

def plot_weighted_a_star_comparison(results):
    """
    Plot comparison of weighted A* performance
    """
    import matplotlib.pyplot as plt
    
    weights = list(results.keys())
    nodes_expanded = [results[w]['nodes_expanded'] for w in weights]
    path_costs = [results[w]['path_cost'] for w in weights]
    times = [results[w]['execution_time'] for w in weights]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Nodes expanded
    ax1.plot(weights, nodes_expanded, 'bo-')
    ax1.set_xlabel('Weight')
    ax1.set_ylabel('Nodes Expanded')
    ax1.set_title('Weighted A*: Nodes Expanded')
    ax1.grid(True)
    
    # Path cost
    ax2.plot(weights, path_costs, 'ro-')
    ax2.set_xlabel('Weight')
    ax2.set_ylabel('Path Cost')
    ax2.set_title('Weighted A*: Path Cost')
    ax2.grid(True)
    
    # Execution time
    ax3.plot(weights, times, 'go-')
    ax3.set_xlabel('Weight')
    ax3.set_ylabel('Execution Time (s)')
    ax3.set_title('Weighted A*: Execution Time')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
```

## Anytime A*

### Algorithm Description

Anytime A* provides incremental solution improvement over time, making it suitable for real-time applications where quick initial solutions are needed.

**Key Features**:
- **Quick Initial Solution**: Finds a solution quickly using high weight
- **Incremental Improvement**: Gradually improves solution quality
- **Time-Bounded**: Can be stopped at any time with best solution found

### Implementation

```python
def anytime_a_star_search(problem, heuristic, time_limit=10.0, initial_weight=3.0):
    """
    Anytime A* Search implementation
    
    Args:
        problem: Search problem
        heuristic: Heuristic function
        time_limit: Maximum time to search (seconds)
        initial_weight: Initial weight for weighted A*
    
    Returns:
        dict: Best solution found within time limit
    """
    import time
    
    start_time = time.time()
    best_solution = None
    weight = initial_weight
    iteration = 0
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Try with current weight
        current_solution = weighted_a_star_search(problem, heuristic, weight)
        
        if current_solution:
            if (best_solution is None or 
                current_solution['path_cost'] < best_solution['path_cost']):
                best_solution = current_solution.copy()
                best_solution['iteration_found'] = iteration
                best_solution['time_found'] = time.time() - start_time
        
        # Reduce weight for next iteration
        weight = max(1.0, weight * 0.8)
        
        # Early termination if optimal solution found
        if best_solution and weight <= 1.0:
            break
    
    if best_solution:
        best_solution['total_time'] = time.time() - start_time
        best_solution['total_iterations'] = iteration
    
    return best_solution

def real_time_anytime_a_star(problem, heuristic, decision_time=0.1):
    """
    Real-time Anytime A* for time-critical applications
    """
    def make_decision_with_timeout(problem, heuristic, timeout):
        """Make a decision within the given timeout"""
        start_time = time.time()
        best_action = None
        best_cost = float('inf')
        
        # Start with high weight for quick solution
        weight = 2.0
        
        while time.time() - start_time < timeout:
            solution = weighted_a_star_search(problem, heuristic, weight)
            
            if solution and solution['path']:
                action = solution['path'][0]  # First action
                cost = solution['path_cost']
                
                if cost < best_cost:
                    best_action = action
                    best_cost = cost
            
            weight = max(1.0, weight * 0.9)
        
        return best_action, best_cost
    
    return make_decision_with_timeout(problem, heuristic, decision_time)
```

### Performance Monitoring

```python
class AnytimeAStarMonitor:
    """Monitor and analyze Anytime A* performance"""
    
    def __init__(self):
        self.solutions = []
        self.timestamps = []
        self.weights = []
    
    def add_solution(self, solution, timestamp, weight):
        """Add a solution to the monitor"""
        self.solutions.append(solution)
        self.timestamps.append(timestamp)
        self.weights.append(weight)
    
    def get_improvement_curve(self):
        """Get the solution improvement over time"""
        if not self.solutions:
            return [], []
        
        costs = [s['path_cost'] for s in self.solutions]
        return self.timestamps, costs
    
    def get_convergence_analysis(self):
        """Analyze convergence behavior"""
        if len(self.solutions) < 2:
            return {}
        
        costs = [s['path_cost'] for s in self.solutions]
        improvements = [costs[i-1] - costs[i] for i in range(1, len(costs))]
        
        return {
            'total_improvements': len(improvements),
            'average_improvement': sum(improvements) / len(improvements),
            'max_improvement': max(improvements),
            'convergence_rate': improvements[-1] if improvements else 0
        }
    
    def plot_improvement_curve(self):
        """Plot solution improvement over time"""
        import matplotlib.pyplot as plt
        
        timestamps, costs = self.get_improvement_curve()
        
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, costs, 'bo-')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Path Cost')
        plt.title('Anytime A*: Solution Improvement Over Time')
        plt.grid(True)
        plt.show()
```

## IDA* (Iterative Deepening A*)

### Algorithm Description

IDA* combines the space efficiency of iterative deepening with the informed search of A*. It uses a threshold-based approach with the f-value as the threshold.

**Advantages**:
- **Space Efficient**: Uses only O(d) space where d is solution depth
- **Optimal**: Guaranteed to find optimal solution
- **Complete**: Will find solution if one exists

### Implementation

```python
def ida_star_search(problem, heuristic):
    """
    Iterative Deepening A* Search implementation
    """
    def search_with_threshold(node, g_value, threshold, path):
        """Search with given f-value threshold"""
        f_value = g_value + heuristic(node.state)
        
        if f_value > threshold:
            return f_value  # Threshold exceeded
        
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
                
                # Create next node
                next_node = AStarNode(next_state, node, action, new_g, heuristic(next_state))
                
                # Recursive search
                result = search_with_threshold(next_node, new_g, threshold, path + [action])
                
                if isinstance(result, dict) and result.get('found'):
                    return result
                elif isinstance(result, (int, float)):
                    min_f = min(min_f, result)
        
        return min_f
    
    # Start with heuristic value of initial state
    initial_state = problem.get_initial_state()
    threshold = heuristic(initial_state)
    nodes_expanded = 0
    
    while True:
        initial_node = AStarNode(initial_state, g_value=0, h_value=heuristic(initial_state))
        result = search_with_threshold(initial_node, 0, threshold, [])
        
        if isinstance(result, dict) and result.get('found'):
            result['nodes_expanded'] = nodes_expanded
            return result
        
        if result == float('inf'):
            return None  # No solution
        
        threshold = result  # Increase threshold for next iteration
        nodes_expanded += 1

def ida_star_with_transposition_table(problem, heuristic):
    """
    IDA* with transposition table for better performance
    """
    def search_with_tt(node, g_value, threshold, path, tt):
        """Search with transposition table"""
        state_key = str(node.state)
        f_value = g_value + heuristic(node.state)
        
        if f_value > threshold:
            return f_value
        
        # Check transposition table
        if state_key in tt:
            stored_g, stored_threshold = tt[state_key]
            if stored_threshold >= threshold and stored_g <= g_value:
                return float('inf')  # Skip this path
        
        # Update transposition table
        tt[state_key] = (g_value, threshold)
        
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
                result = search_with_tt(next_node, new_g, threshold, path + [action], tt)
                
                if isinstance(result, dict) and result.get('found'):
                    return result
                elif isinstance(result, (int, float)):
                    min_f = min(min_f, result)
        
        return min_f
    
    initial_state = problem.get_initial_state()
    threshold = heuristic(initial_state)
    transposition_table = {}
    
    while True:
        initial_node = AStarNode(initial_state, g_value=0, h_value=heuristic(initial_state))
        result = search_with_tt(initial_node, 0, threshold, [], transposition_table)
        
        if isinstance(result, dict) and result.get('found'):
            return result
        
        if result == float('inf'):
            return None
        
        threshold = result
```

## SMA* (Simplified Memory-Bounded A*)

### Algorithm Description

SMA* is designed for memory-constrained environments. It uses a fixed-size memory buffer and removes the least promising nodes when memory is full.

**Key Features**:
- **Memory Bounded**: Uses fixed amount of memory
- **Complete**: Will find solution if one exists and memory is sufficient
- **Optimal**: Optimal within memory constraints

### Implementation

```python
class SMANode:
    """Node for SMA* algorithm"""
    def __init__(self, state, parent=None, action=None, g_value=0, h_value=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g_value = g_value
        self.h_value = h_value
        self.f_value = g_value + h_value
        self.children = []
        self.is_fully_expanded = False
    
    def __lt__(self, other):
        return self.f_value < other.f_value

def sma_star_search(problem, heuristic, memory_limit=1000):
    """
    Simplified Memory-Bounded A* Search implementation
    """
    def remove_least_promising_node(frontier, explored):
        """Remove the least promising node from memory"""
        if not frontier:
            return
        
        # Find node with highest f-value
        worst_node = max(frontier, key=lambda x: x.f_value)
        frontier.remove(worst_node)
        explored.discard(worst_node.state)
        
        # Update parent's children list
        if worst_node.parent:
            worst_node.parent.children.remove(worst_node)
            worst_node.parent.is_fully_expanded = False
    
    # Initialize
    initial_state = problem.get_initial_state()
    initial_node = SMANode(initial_state, g_value=0, h_value=heuristic(initial_state))
    
    frontier = [initial_node]
    explored = set()
    g_values = {initial_state: 0}
    
    while frontier:
        # Check memory limit
        if len(frontier) + len(explored) > memory_limit:
            remove_least_promising_node(frontier, explored)
        
        # Get best node
        current_node = min(frontier, key=lambda x: x.f_value)
        frontier.remove(current_node)
        
        # Check if goal
        if problem.is_goal_state(current_node.state):
            return {
                'path': current_node.get_path(),
                'path_cost': current_node.g_value,
                'memory_used': len(frontier) + len(explored)
            }
        
        # Skip if already explored with better g value
        if current_node.state in explored and g_values[current_node.state] < current_node.g_value:
            continue
        
        explored.add(current_node.state)
        
        # Expand if not fully expanded
        if not current_node.is_fully_expanded:
            actions = problem.get_actions(current_node.state)
            for action in actions:
                next_state = problem.get_next_state(current_node.state, action)
                if next_state:
                    action_cost = problem.get_action_cost(current_node.state, action, next_state)
                    new_g_value = current_node.g_value + action_cost
                    
                    # Check if this path is better
                    if (next_state not in g_values or 
                        new_g_value < g_values[next_state]):
                        
                        g_values[next_state] = new_g_value
                        h_value = heuristic(next_state)
                        
                        # Create new node
                        new_node = SMANode(
                            next_state, 
                            current_node, 
                            action, 
                            new_g_value, 
                            h_value
                        )
                        
                        current_node.children.append(new_node)
                        frontier.append(new_node)
            
            current_node.is_fully_expanded = True
        
        # Re-add current node to frontier if it has children
        if current_node.children:
            frontier.append(current_node)
    
    return None  # No solution found
```

## D* (Dynamic A*)

### Algorithm Description

D* is designed for dynamic environments where the graph can change during execution. It efficiently replans when changes are detected.

**Key Features**:
- **Dynamic Replanning**: Efficiently updates path when graph changes
- **Incremental**: Only recomputes necessary parts of the path
- **Real-time**: Suitable for robotics and navigation

### Implementation

```python
class DStarNode:
    """Node for D* algorithm"""
    def __init__(self, state, g_value=float('inf'), rhs_value=float('inf')):
        self.state = state
        self.g_value = g_value
        self.rhs_value = rhs_value
        self.key = (min(g_value, rhs_value), min(g_value, rhs_value))
    
    def update_key(self):
        """Update the key for priority queue"""
        self.key = (min(self.g_value, self.rhs_value), min(self.g_value, self.rhs_value))
    
    def __lt__(self, other):
        return self.key < other.key

def d_star_search(problem, heuristic, start_state, goal_state):
    """
    Dynamic A* Search implementation (simplified version)
    """
    def compute_shortest_path():
        """Compute shortest path using D*"""
        while frontier and (frontier[0].key < calculate_key(goal_node) or 
                           goal_node.rhs_value != goal_node.g_value):
            
            current_node = heapq.heappop(frontier)
            
            if current_node.g_value > current_node.rhs_value:
                # Overconsistent
                current_node.g_value = current_node.rhs_value
            else:
                # Underconsistent
                current_node.g_value = float('inf')
                current_node.update_key()
                heapq.heappush(frontier, current_node)
            
            # Update neighbors
            for neighbor_state in get_neighbors(current_node.state):
                update_vertex(neighbor_state)
    
    def update_vertex(state):
        """Update vertex in D* algorithm"""
        if state != start_state:
            # Calculate minimum rhs value
            min_rhs = float('inf')
            for neighbor_state in get_neighbors(state):
                cost = get_edge_cost(state, neighbor_state)
                if neighbor_state in nodes:
                    min_rhs = min(min_rhs, nodes[neighbor_state].g_value + cost)
            
            nodes[state].rhs_value = min_rhs
        
        # Update in priority queue
        if state in nodes:
            node = nodes[state]
            if node.g_value != node.rhs_value:
                node.update_key()
                if state not in [n.state for n in frontier]:
                    heapq.heappush(frontier, node)
    
    def calculate_key(node):
        """Calculate key for node"""
        return (min(node.g_value, node.rhs_value), min(node.g_value, node.rhs_value))
    
    def get_neighbors(state):
        """Get neighboring states"""
        actions = problem.get_actions(state)
        neighbors = []
        for action in actions:
            next_state = problem.get_next_state(state, action)
            if next_state:
                neighbors.append(next_state)
        return neighbors
    
    def get_edge_cost(state1, state2):
        """Get cost of edge between states"""
        actions = problem.get_actions(state1)
        for action in actions:
            next_state = problem.get_next_state(state1, action)
            if next_state == state2:
                return problem.get_action_cost(state1, action, next_state)
        return float('inf')
    
    # Initialize
    nodes = {}
    frontier = []
    
    # Initialize goal node
    goal_node = DStarNode(goal_state, g_value=float('inf'), rhs_value=0)
    nodes[goal_state] = goal_node
    heapq.heappush(frontier, goal_node)
    
    # Initialize start node
    start_node = DStarNode(start_state, g_value=float('inf'), rhs_value=float('inf'))
    nodes[start_state] = start_node
    
    # Compute initial path
    compute_shortest_path()
    
    return {
        'path': reconstruct_path(start_state, goal_state, nodes),
        'nodes': nodes
    }

def reconstruct_path(start_state, goal_state, nodes):
    """Reconstruct path from D* nodes"""
    path = []
    current_state = start_state
    
    while current_state != goal_state:
        if current_state not in nodes:
            return None
        
        # Find best neighbor
        best_neighbor = None
        best_cost = float('inf')
        
        for neighbor_state in get_neighbors(current_state):
            if neighbor_state in nodes:
                cost = get_edge_cost(current_state, neighbor_state)
                total_cost = cost + nodes[neighbor_state].g_value
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_neighbor = neighbor_state
        
        if best_neighbor is None:
            return None
        
        path.append(best_neighbor)
        current_state = best_neighbor
    
    return path
```

## Performance Comparison

```python
def compare_a_star_variants(problem, heuristic):
    """
    Compare different A* variants
    """
    algorithms = {
        'Standard A*': lambda: a_star_search(problem, heuristic),
        'Weighted A* (w=1.5)': lambda: weighted_a_star_search(problem, heuristic, 1.5),
        'Weighted A* (w=2.0)': lambda: weighted_a_star_search(problem, heuristic, 2.0),
        'IDA*': lambda: ida_star_search(problem, heuristic),
        'SMA*': lambda: sma_star_search(problem, heuristic, 1000)
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        try:
            start_time = time.time()
            result = algorithm()
            end_time = time.time()
            
            if result:
                results[name] = {
                    'path_cost': result['path_cost'],
                    'nodes_expanded': result.get('nodes_expanded', 0),
                    'execution_time': end_time - start_time,
                    'memory_used': result.get('memory_used', 0),
                    'success': True
                }
            else:
                results[name] = {'success': False}
        except Exception as e:
            results[name] = {'success': False, 'error': str(e)}
    
    return results
```

## Summary

A* relaxations provide solutions for different problem scenarios:

1. **Weighted A***: Trade optimality for speed
2. **Anytime A***: Provide quick initial solutions with gradual improvement
3. **IDA***: Space-efficient optimal search
4. **SMA***: Memory-bounded search for constrained environments
5. **D***: Dynamic replanning for changing environments

The choice of relaxation depends on:
- **Time constraints**: Use weighted or anytime A*
- **Memory constraints**: Use IDA* or SMA*
- **Dynamic environment**: Use D*
- **Optimality requirements**: Use standard A* or IDA* 