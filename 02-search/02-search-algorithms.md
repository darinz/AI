# Search Algorithms

## Overview

Search algorithms are systematic methods for exploring state spaces to find solutions to problems. They form the foundation of many AI systems and can be classified based on their approach to exploring the search space.

## Algorithm Classification

### 1. Uninformed vs. Informed Search

**Uninformed Search (Blind Search)**: No additional information about goal location
- Examples: BFS, DFS, UCS, Iterative Deepening

**Informed Search (Heuristic Search)**: Uses domain-specific knowledge to guide search
- Examples: A*, Greedy Best-First Search, Hill Climbing

### 2. Complete vs. Incomplete Algorithms

**Complete**: Guaranteed to find a solution if one exists
- Examples: BFS, UCS, A* (with admissible heuristic)

**Incomplete**: May fail to find a solution even if one exists
- Examples: DFS (in infinite spaces), Hill Climbing

### 3. Optimal vs. Suboptimal Algorithms

**Optimal**: Guaranteed to find the best (lowest cost) solution
- Examples: UCS, A* (with admissible heuristic)

**Suboptimal**: May find any solution, not necessarily the best
- Examples: Greedy Best-First Search, Hill Climbing

## Performance Metrics

### Time Complexity

Time complexity measures how the algorithm's runtime grows with problem size:

```python
def analyze_time_complexity(algorithm, problem_sizes):
    """Analyze time complexity of a search algorithm"""
    results = []
    
    for size in problem_sizes:
        # Create problem of given size
        problem = create_problem(size)
        
        # Measure execution time
        start_time = time.time()
        solution = algorithm(problem)
        end_time = time.time()
        
        results.append({
            'size': size,
            'time': end_time - start_time,
            'nodes_expanded': solution.get('nodes_expanded', 0)
        })
    
    return results
```

### Space Complexity

Space complexity measures memory usage:

```python
def analyze_space_complexity(algorithm, problem):
    """Analyze space complexity of a search algorithm"""
    import sys
    
    # Track memory usage before and after
    initial_memory = sys.getsizeof([])
    
    # Run algorithm with memory tracking
    solution = algorithm(problem)
    
    final_memory = sys.getsizeof([])
    memory_used = final_memory - initial_memory
    
    return {
        'memory_used': memory_used,
        'max_frontier_size': solution.get('max_frontier_size', 0),
        'max_explored_size': solution.get('max_explored_size', 0)
    }
```

### Solution Quality

```python
def evaluate_solution_quality(algorithm, problem, num_trials=100):
    """Evaluate solution quality across multiple trials"""
    costs = []
    success_rate = 0
    
    for _ in range(num_trials):
        try:
            solution = algorithm(problem)
            if solution and solution.get('path'):
                costs.append(solution.get('path_cost', float('inf')))
                success_rate += 1
        except:
            pass
    
    success_rate /= num_trials
    
    return {
        'success_rate': success_rate,
        'average_cost': sum(costs) / len(costs) if costs else float('inf'),
        'min_cost': min(costs) if costs else float('inf'),
        'max_cost': max(costs) if costs else float('inf')
    }
```

## Mathematical Analysis

### Branching Factor and Depth

For a tree with branching factor $b$ and solution depth $d$:

- **Time Complexity**: $O(b^d)$ for most uninformed searches
- **Space Complexity**: $O(b^d)$ for BFS, $O(d)$ for DFS

### Optimality Conditions

An algorithm is optimal if it always finds the lowest-cost path to the goal:

**Theorem**: If all step costs are positive and the heuristic is admissible, A* is optimal.

**Proof Sketch**:
1. A* expands nodes in order of increasing $f(n) = g(n) + h(n)$
2. When goal is expanded, $f(goal) = g(goal) + h(goal) = g(goal)$ (since $h(goal) = 0$)
3. Any other path to goal would have $f(n) \geq g(goal)$
4. Therefore, A* finds the optimal path

## Algorithm Comparison Framework

```python
class SearchAlgorithmComparator:
    def __init__(self, problem_generator):
        self.problem_generator = problem_generator
    
    def compare_algorithms(self, algorithms, problem_sizes):
        """Compare multiple algorithms across different problem sizes"""
        results = {}
        
        for algorithm_name, algorithm_func in algorithms.items():
            results[algorithm_name] = {
                'time_complexity': [],
                'space_complexity': [],
                'solution_quality': [],
                'success_rate': []
            }
            
            for size in problem_sizes:
                problem = self.problem_generator(size)
                
                # Time analysis
                start_time = time.time()
                solution = algorithm_func(problem)
                end_time = time.time()
                
                results[algorithm_name]['time_complexity'].append({
                    'size': size,
                    'time': end_time - start_time,
                    'nodes_expanded': solution.get('nodes_expanded', 0)
                })
                
                # Space analysis
                space_analysis = analyze_space_complexity(algorithm_func, problem)
                results[algorithm_name]['space_complexity'].append({
                    'size': size,
                    'memory': space_analysis['memory_used'],
                    'max_frontier': space_analysis['max_frontier_size']
                })
                
                # Quality analysis
                quality_analysis = evaluate_solution_quality(algorithm_func, problem, num_trials=10)
                results[algorithm_name]['solution_quality'].append({
                    'size': size,
                    'avg_cost': quality_analysis['average_cost'],
                    'min_cost': quality_analysis['min_cost']
                })
                results[algorithm_name]['success_rate'].append({
                    'size': size,
                    'rate': quality_analysis['success_rate']
                })
        
        return results
    
    def generate_report(self, results):
        """Generate a comprehensive comparison report"""
        report = []
        report.append("# Search Algorithm Comparison Report\n")
        
        for algorithm_name, data in results.items():
            report.append(f"## {algorithm_name}\n")
            
            # Time complexity summary
            times = [d['time'] for d in data['time_complexity']]
            report.append(f"- Average execution time: {sum(times)/len(times):.4f}s")
            report.append(f"- Nodes expanded: {data['time_complexity'][-1]['nodes_expanded']}")
            
            # Space complexity summary
            memory = [d['memory'] for d in data['space_complexity']]
            report.append(f"- Average memory usage: {sum(memory)/len(memory):.2f} bytes")
            
            # Solution quality summary
            success_rates = [d['rate'] for d in data['success_rate']]
            avg_success = sum(success_rates) / len(success_rates)
            report.append(f"- Success rate: {avg_success:.2%}")
            
            report.append("")
        
        return '\n'.join(report)
```

## Algorithm Selection Guidelines

### When to Use Each Algorithm

```python
def select_algorithm(problem_characteristics):
    """Select the most appropriate search algorithm based on problem characteristics"""
    
    recommendations = []
    
    # Check if optimality is required
    if problem_characteristics.get('require_optimality', False):
        if problem_characteristics.get('has_heuristic', False):
            recommendations.append("A* (with admissible heuristic)")
        else:
            recommendations.append("Uniform Cost Search")
    
    # Check memory constraints
    if problem_characteristics.get('memory_limited', False):
        if problem_characteristics.get('has_heuristic', False):
            recommendations.append("Iterative Deepening A*")
        else:
            recommendations.append("Iterative Deepening DFS")
    
    # Check if real-time performance is needed
    if problem_characteristics.get('real_time', False):
        recommendations.append("Greedy Best-First Search")
        recommendations.append("Weighted A*")
    
    # Check problem size
    if problem_characteristics.get('large_state_space', False):
        recommendations.append("A* (with good heuristic)")
        recommendations.append("Bidirectional Search")
    
    # Default recommendations
    if not recommendations:
        if problem_characteristics.get('has_heuristic', False):
            recommendations.append("A*")
        else:
            recommendations.append("Breadth-First Search")
    
    return recommendations
```

### Problem-Specific Considerations

```python
def analyze_problem_characteristics(problem):
    """Analyze problem characteristics to guide algorithm selection"""
    
    # Analyze state space
    state_space_analysis = analyze_state_space(problem)
    
    # Check for heuristics
    has_heuristic = hasattr(problem, 'heuristic') or hasattr(problem, 'get_heuristic')
    
    # Check goal characteristics
    single_goal = len(problem.get_goal_states()) == 1 if hasattr(problem, 'get_goal_states') else True
    
    # Check cost structure
    uniform_costs = True  # Assume uniform unless proven otherwise
    
    return {
        'state_space_size': state_space_analysis['total_states'],
        'branching_factor': state_space_analysis['average_branching_factor'],
        'has_heuristic': has_heuristic,
        'single_goal': single_goal,
        'uniform_costs': uniform_costs,
        'memory_limited': state_space_analysis['total_states'] > 1000000,
        'large_state_space': state_space_analysis['total_states'] > 100000
    }
```

## Performance Optimization Techniques

### 1. State Representation Optimization

```python
class OptimizedState:
    def __init__(self, data):
        # Use efficient data structures
        self.data = tuple(data)  # Immutable for hashing
        self._hash = None
    
    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.data)
        return self._hash
    
    def __eq__(self, other):
        return self.data == other.data
```

### 2. Frontier Data Structure Selection

```python
from collections import deque
import heapq

class FrontierFactory:
    @staticmethod
    def create_frontier(algorithm_type):
        if algorithm_type == 'bfs':
            return deque()  # O(1) append and popleft
        elif algorithm_type == 'dfs':
            return []  # O(1) append and pop
        elif algorithm_type == 'ucs' or algorithm_type == 'astar':
            return []  # Will be used as heap
        else:
            return deque()
    
    @staticmethod
    def add_to_frontier(frontier, item, algorithm_type):
        if algorithm_type in ['ucs', 'astar']:
            heapq.heappush(frontier, item)
        else:
            frontier.append(item)
    
    @staticmethod
    def remove_from_frontier(frontier, algorithm_type):
        if algorithm_type in ['ucs', 'astar']:
            return heapq.heappop(frontier)
        elif algorithm_type == 'bfs':
            return frontier.popleft()
        else:  # dfs
            return frontier.pop()
```

### 3. Explored Set Optimization

```python
class ExploredSet:
    def __init__(self):
        self._set = set()
        self._count = 0
    
    def add(self, state):
        if state not in self._set:
            self._set.add(state)
            self._count += 1
    
    def __contains__(self, state):
        return state in self._set
    
    def size(self):
        return self._count
    
    def clear(self):
        self._set.clear()
        self._count = 0
```

## Summary

Search algorithm selection depends on multiple factors:

1. **Problem Requirements**: Optimality, completeness, real-time constraints
2. **Problem Characteristics**: State space size, branching factor, availability of heuristics
3. **Resource Constraints**: Memory limitations, time constraints
4. **Solution Quality**: Trade-offs between optimality and efficiency

The key is to match algorithm properties with problem requirements while considering computational resources and performance constraints. 