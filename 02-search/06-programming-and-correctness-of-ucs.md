# Programming and Correctness of UCS

## Overview

This guide focuses on the implementation details, correctness proofs, and testing strategies for Uniform Cost Search (UCS). Understanding these aspects is crucial for building reliable and efficient search algorithms.

## Implementation Details

### Priority Queue Data Structures

The choice of priority queue implementation significantly affects UCS performance:

```python
import heapq
from queue import PriorityQueue
from collections import deque

class PriorityQueueComparison:
    """Compare different priority queue implementations"""
    
    @staticmethod
    def heap_implementation():
        """Using Python's heapq module"""
        frontier = []
        
        def add_node(cost, state, path):
            heapq.heappush(frontier, (cost, state, path))
        
        def get_node():
            return heapq.heappop(frontier)
        
        def is_empty():
            return len(frontier) == 0
        
        return add_node, get_node, is_empty
    
    @staticmethod
    def queue_implementation():
        """Using Python's PriorityQueue"""
        frontier = PriorityQueue()
        
        def add_node(cost, state, path):
            frontier.put((cost, state, path))
        
        def get_node():
            return frontier.get()
        
        def is_empty():
            return frontier.empty()
        
        return add_node, get_node, is_empty
    
    @staticmethod
    def custom_heap_implementation():
        """Custom heap implementation for educational purposes"""
        class CustomHeap:
            def __init__(self):
                self.heap = []
            
            def push(self, item):
                self.heap.append(item)
                self._sift_up(len(self.heap) - 1)
            
            def pop(self):
                if not self.heap:
                    raise IndexError("Heap is empty")
                
                if len(self.heap) == 1:
                    return self.heap.pop()
                
                result = self.heap[0]
                self.heap[0] = self.heap.pop()
                self._sift_down(0)
                return result
            
            def _sift_up(self, index):
                parent = (index - 1) // 2
                if parent >= 0 and self.heap[index] < self.heap[parent]:
                    self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
                    self._sift_up(parent)
            
            def _sift_down(self, index):
                left = 2 * index + 1
                right = 2 * index + 2
                smallest = index
                
                if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
                    smallest = left
                
                if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
                    smallest = right
                
                if smallest != index:
                    self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
                    self._sift_down(smallest)
            
            def is_empty(self):
                return len(self.heap) == 0
        
        frontier = CustomHeap()
        
        def add_node(cost, state, path):
            frontier.push((cost, state, path))
        
        def get_node():
            return frontier.pop()
        
        def is_empty():
            return frontier.is_empty()
        
        return add_node, get_node, is_empty
```

### State Representation

Efficient state representation is crucial for performance:

```python
class StateRepresentation:
    """Different ways to represent states for UCS"""
    
    @staticmethod
    def tuple_representation(state_data):
        """Convert state to immutable tuple for hashing"""
        if isinstance(state_data, list):
            return tuple(tuple(row) if isinstance(row, list) else row for row in state_data)
        elif isinstance(state_data, dict):
            return tuple(sorted(state_data.items()))
        else:
            return state_data
    
    @staticmethod
    def hash_optimized_representation(state_data):
        """Pre-compute hash for better performance"""
        class HashOptimizedState:
            def __init__(self, data):
                self.data = StateRepresentation.tuple_representation(data)
                self._hash = hash(self.data)
            
            def __hash__(self):
                return self._hash
            
            def __eq__(self, other):
                return self.data == other.data
            
            def __str__(self):
                return str(self.data)
        
        return HashOptimizedState(state_data)
    
    @staticmethod
    def compressed_representation(state_data):
        """Compress state representation to save memory"""
        import pickle
        import zlib
        
        serialized = pickle.dumps(state_data)
        compressed = zlib.compress(serialized)
        
        class CompressedState:
            def __init__(self, compressed_data):
                self.compressed_data = compressed_data
                self._hash = hash(compressed_data)
            
            def __hash__(self):
                return self._hash
            
            def __eq__(self, other):
                return self.compressed_data == other.compressed_data
            
            def decompress(self):
                decompressed = zlib.decompress(self.compressed_data)
                return pickle.loads(decompressed)
        
        return CompressedState(compressed)
```

### Path Reconstruction

Efficient path reconstruction is essential for UCS:

```python
class PathReconstruction:
    """Different strategies for path reconstruction"""
    
    @staticmethod
    def parent_pointer_reconstruction():
        """Use parent pointers for path reconstruction"""
        class UCSNodeWithParent:
            def __init__(self, state, parent=None, action=None, path_cost=0):
                self.state = state
                self.parent = parent
                self.action = action
                self.path_cost = path_cost
            
            def get_path(self):
                """Reconstruct path using parent pointers"""
                path = []
                current = self
                while current.parent is not None:
                    path.append(current.action)
                    current = current.parent
                return list(reversed(path))
            
            def get_path_states(self):
                """Get all states in the path"""
                states = []
                current = self
                while current is not None:
                    states.append(current.state)
                    current = current.parent
                return list(reversed(states))
        
        return UCSNodeWithParent
    
    @staticmethod
    def explicit_path_storage():
        """Store complete path with each node"""
        class UCSNodeWithPath:
            def __init__(self, state, path=None, path_cost=0):
                self.state = state
                self.path = path or []
                self.path_cost = path_cost
            
            def add_action(self, action, cost):
                """Add action to path"""
                new_path = self.path + [action]
                new_cost = self.path_cost + cost
                return UCSNodeWithPath(self.state, new_path, new_cost)
        
        return UCSNodeWithPath
    
    @staticmethod
    def hash_table_reconstruction(explored_nodes):
        """Reconstruct path using hash table of explored nodes"""
        def reconstruct_path(goal_state, explored_nodes):
            path = []
            current_state = goal_state
            
            while current_state in explored_nodes:
                parent_info = explored_nodes[current_state]
                if parent_info['action'] is not None:
                    path.append(parent_info['action'])
                current_state = parent_info['parent']
            
            return list(reversed(path))
        
        return reconstruct_path
```

## Correctness Proofs

### Formal Correctness Proof

**Theorem**: UCS is optimal when all step costs are positive.

**Proof by Contradiction**:

```python
def ucs_correctness_proof():
    """
    Formal proof of UCS correctness
    """
    proof_steps = [
        "1. Assume UCS is not optimal",
        "2. Let P* be the optimal path with cost C*",
        "3. Let P be the path found by UCS with cost C > C*",
        "4. Since UCS expands nodes in order of increasing cost,",
        "   all nodes on P* must be expanded before the goal node on P",
        "5. This contradicts the assumption that P was found first",
        "6. Therefore, UCS must be optimal"
    ]
    
    return proof_steps

def formal_ucs_proof():
    """
    More detailed formal proof
    """
    proof = {
        'assumptions': [
            'All step costs are positive',
            'State space is finite',
            'Goal state is reachable'
        ],
        'definitions': [
            'g(n) = cost from start to node n',
            'f(n) = g(n) for UCS (no heuristic)',
            'Optimal path = path with minimum total cost'
        ],
        'lemmas': [
            'Lemma 1: UCS expands nodes in order of increasing g(n)',
            'Lemma 2: If node n is expanded, all nodes with g(m) < g(n) have been expanded',
            'Lemma 3: When goal is expanded, its g value is optimal'
        ],
        'main_proof': [
            '1. Let G be the goal state found by UCS',
            '2. Let g(G) be the cost of the path to G',
            '3. By Lemma 1, all nodes with g(n) < g(G) were expanded before G',
            '4. If there exists a better path to goal with cost g*(G) < g(G),',
            '   then some node on that path would have g(n) < g(G)',
            '5. By Lemma 2, that node would have been expanded before G',
            '6. This contradicts the assumption that G was the first goal found',
            '7. Therefore, g(G) must be optimal'
        ]
    }
    
    return proof
```

### Loop Invariant Analysis

```python
def ucs_loop_invariant():
    """
    Loop invariant for UCS algorithm
    """
    invariant = {
        'precondition': [
            'Frontier contains nodes reachable from start',
            'Explored set contains nodes that have been expanded',
            'All nodes in frontier have finite path costs'
        ],
        'invariant': [
            'For every node n in frontier, g(n) is the cost of the optimal path from start to n',
            'For every node n in explored, g(n) is the cost of the optimal path from start to n',
            'If goal state is in explored, then g(goal) is optimal'
        ],
        'postcondition': [
            'If solution found: g(goal) is optimal',
            'If no solution: goal is unreachable'
        ],
        'maintenance': [
            'When node n is expanded:',
            '  - All its neighbors are added to frontier with correct g values',
            '  - Node n is moved to explored set',
            '  - Invariant is maintained for all remaining nodes'
        ]
    }
    
    return invariant

def verify_loop_invariant(ucs_implementation, problem):
    """
    Verify loop invariant during execution
    """
    def invariant_checker(frontier, explored, current_node):
        """Check if loop invariant holds"""
        violations = []
        
        # Check frontier invariant
        for cost, state, path in frontier:
            if cost != len(path):  # Assuming unit costs
                violations.append(f"Frontier invariant violated: {state}")
        
        # Check explored invariant
        for state in explored:
            # This would need more sophisticated checking
            pass
        
        return violations
    
    # Modify UCS to include invariant checking
    def ucs_with_invariant_checking(problem):
        initial_state = problem.get_initial_state()
        frontier = [(0, initial_state, [])]
        explored = set()
        
        while frontier:
            current_cost, current_state, path = heapq.heappop(frontier)
            
            # Check invariant
            violations = invariant_checker(frontier, explored, current_state)
            if violations:
                print(f"Invariant violations: {violations}")
            
            if problem.is_goal_state(current_state):
                return {'path': path, 'path_cost': current_cost}
            
            if current_state in explored:
                continue
            
            explored.add(current_state)
            
            # Expand and maintain invariant
            actions = problem.get_actions(current_state)
            for action in actions:
                next_state = problem.get_next_state(current_state, action)
                if next_state and next_state not in explored:
                    action_cost = problem.get_action_cost(current_state, action, next_state)
                    new_cost = current_cost + action_cost
                    new_path = path + [action]
                    heapq.heappush(frontier, (new_cost, next_state, new_path))
        
        return None
    
    return ucs_with_invariant_checking(problem)
```

### Termination Guarantees

```python
def ucs_termination_analysis():
    """
    Analyze termination conditions for UCS
    """
    termination_conditions = {
        'finite_state_space': {
            'condition': 'State space is finite',
            'guarantee': 'Algorithm will terminate',
            'proof': 'Finite number of nodes to explore'
        },
        'positive_costs': {
            'condition': 'All step costs are positive',
            'guarantee': 'No infinite loops',
            'proof': 'Path cost strictly increases with each step'
        },
        'goal_reachable': {
            'condition': 'Goal state is reachable',
            'guarantee': 'Solution will be found',
            'proof': 'Goal will eventually be added to frontier'
        },
        'goal_unreachable': {
            'condition': 'Goal state is unreachable',
            'guarantee': 'Algorithm will terminate with no solution',
            'proof': 'All reachable states will be explored'
        }
    }
    
    return termination_conditions

def detect_infinite_loops(ucs_implementation, problem):
    """
    Detect potential infinite loops in UCS
    """
    def ucs_with_loop_detection(problem):
        initial_state = problem.get_initial_state()
        frontier = [(0, initial_state, [])]
        explored = set()
        iteration_count = 0
        max_iterations = 10000  # Safety limit
        
        while frontier and iteration_count < max_iterations:
            iteration_count += 1
            
            current_cost, current_state, path = heapq.heappop(frontier)
            
            # Check for suspicious patterns
            if len(path) > 1000:  # Unusually long path
                print(f"Warning: Very long path detected: {len(path)} steps")
            
            if current_cost > 10000:  # Unusually high cost
                print(f"Warning: Very high cost detected: {current_cost}")
            
            if problem.is_goal_state(current_state):
                return {
                    'path': path,
                    'path_cost': current_cost,
                    'iterations': iteration_count
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
                    new_path = path + [action]
                    heapq.heappush(frontier, (new_cost, next_state, new_path))
        
        if iteration_count >= max_iterations:
            print("Warning: Maximum iterations reached - possible infinite loop")
        
        return None
    
    return ucs_with_loop_detection(problem)
```

## Testing and Validation

### Unit Test Design

```python
import unittest
import random

class UCSTestCase(unittest.TestCase):
    """Comprehensive test suite for UCS implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.simple_graph = {
            'A': [('B', 1), ('C', 2)],
            'B': [('A', 1), ('D', 3)],
            'C': [('A', 2), ('D', 1)],
            'D': [('B', 3), ('C', 1)]
        }
        
        self.simple_problem = GraphSearchProblem(self.simple_graph, 'A', 'D')
    
    def test_optimality(self):
        """Test that UCS finds optimal path"""
        result = uniform_cost_search(self.simple_problem)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['path_cost'], 2)  # A -> C -> D
        self.assertEqual(result['path'], ['C', 'D'])
    
    def test_completeness(self):
        """Test that UCS finds solution when one exists"""
        # Test with reachable goal
        result = uniform_cost_search(self.simple_problem)
        self.assertIsNotNone(result)
        
        # Test with unreachable goal
        unreachable_problem = GraphSearchProblem(self.simple_graph, 'A', 'Z')
        result = uniform_cost_search(unreachable_problem)
        self.assertIsNone(result)
    
    def test_positive_costs(self):
        """Test UCS with positive costs"""
        # This should work normally
        result = uniform_cost_search(self.simple_problem)
        self.assertIsNotNone(result)
    
    def test_zero_costs(self):
        """Test UCS with zero costs (edge case)"""
        zero_cost_graph = {
            'A': [('B', 0), ('C', 1)],
            'B': [('A', 0), ('D', 1)],
            'C': [('A', 1), ('D', 0)],
            'D': [('B', 1), ('C', 0)]
        }
        
        problem = GraphSearchProblem(zero_cost_graph, 'A', 'D')
        result = uniform_cost_search(problem)
        
        # Should still find a path, but may not be optimal
        self.assertIsNotNone(result)
    
    def test_large_graph(self):
        """Test UCS with larger graph"""
        # Generate random graph
        n_nodes = 100
        graph = {}
        
        for i in range(n_nodes):
            graph[f'node_{i}'] = []
            for j in range(random.randint(1, 5)):  # 1-5 neighbors
                neighbor = f'node_{random.randint(0, n_nodes-1)}'
                cost = random.randint(1, 10)
                if neighbor != f'node_{i}':
                    graph[f'node_{i}'].append((neighbor, cost))
        
        problem = GraphSearchProblem(graph, 'node_0', 'node_99')
        result = uniform_cost_search(problem)
        
        # Should either find solution or determine it's impossible
        if result:
            self.assertGreater(result['path_cost'], 0)
            self.assertGreater(len(result['path']), 0)
    
    def test_path_reconstruction(self):
        """Test that path reconstruction is correct"""
        result = uniform_cost_search(self.simple_problem)
        
        if result:
            # Verify path leads from start to goal
            current_state = self.simple_problem.get_initial_state()
            total_cost = 0
            
            for action in result['path']:
                # Find the action in current state's neighbors
                found = False
                for neighbor, cost in self.simple_graph[current_state]:
                    if neighbor == action:
                        total_cost += cost
                        current_state = neighbor
                        found = True
                        break
                
                self.assertTrue(found, f"Invalid action {action} from state {current_state}")
            
            self.assertEqual(current_state, self.simple_problem.goal)
            self.assertEqual(total_cost, result['path_cost'])
    
    def test_frontier_management(self):
        """Test that frontier is managed correctly"""
        result = uniform_cost_search(self.simple_problem)
        
        if result:
            # Check that frontier size is reasonable
            self.assertGreaterEqual(result['frontier_size'], 0)
            self.assertLess(result['max_frontier_size'], 1000)  # Reasonable limit
    
    def test_explored_set(self):
        """Test that explored set prevents cycles"""
        # Create graph with cycles
        cyclic_graph = {
            'A': [('B', 1)],
            'B': [('C', 1)],
            'C': [('A', 1), ('D', 1)],
            'D': []
        }
        
        problem = GraphSearchProblem(cyclic_graph, 'A', 'D')
        result = uniform_cost_search(problem)
        
        # Should find solution without infinite loop
        self.assertIsNotNone(result)
        self.assertEqual(result['path_cost'], 3)  # A -> B -> C -> D

def run_ucs_tests():
    """Run all UCS tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)
```

### Edge Case Handling

```python
def test_edge_cases():
    """Test UCS with various edge cases"""
    
    def test_empty_graph():
        """Test with empty graph"""
        empty_graph = {}
        problem = GraphSearchProblem(empty_graph, 'A', 'B')
        result = uniform_cost_search(problem)
        assert result is None
    
    def test_single_node():
        """Test with single node"""
        single_node_graph = {'A': []}
        problem = GraphSearchProblem(single_node_graph, 'A', 'A')
        result = uniform_cost_search(problem)
        assert result is not None
        assert result['path_cost'] == 0
        assert result['path'] == []
    
    def test_disconnected_graph():
        """Test with disconnected graph"""
        disconnected_graph = {
            'A': [('B', 1)],
            'B': [('A', 1)],
            'C': [('D', 1)],
            'D': [('C', 1)]
        }
        problem = GraphSearchProblem(disconnected_graph, 'A', 'C')
        result = uniform_cost_search(problem)
        assert result is None
    
    def test_very_large_costs():
        """Test with very large costs"""
        large_cost_graph = {
            'A': [('B', 1000000)],
            'B': [('C', 1000000)],
            'C': []
        }
        problem = GraphSearchProblem(large_cost_graph, 'A', 'C')
        result = uniform_cost_search(problem)
        assert result is not None
        assert result['path_cost'] == 2000000
    
    # Run all edge case tests
    test_empty_graph()
    test_single_node()
    test_disconnected_graph()
    test_very_large_costs()
    print("All edge case tests passed!")

test_edge_cases()
```

### Performance Benchmarking

```python
def benchmark_ucs_performance():
    """Benchmark UCS performance across different problem sizes"""
    import time
    import matplotlib.pyplot as plt
    
    def generate_test_problems():
        """Generate test problems of various sizes"""
        problems = []
        
        for n_nodes in [10, 50, 100, 500, 1000]:
            # Generate random graph
            graph = {}
            for i in range(n_nodes):
                graph[f'node_{i}'] = []
                for j in range(random.randint(1, 3)):  # 1-3 neighbors
                    neighbor = f'node_{random.randint(0, n_nodes-1)}'
                    cost = random.randint(1, 10)
                    if neighbor != f'node_{i}':
                        graph[f'node_{i}'].append((neighbor, cost))
            
            problem = GraphSearchProblem(graph, 'node_0', f'node_{n_nodes-1}')
            problems.append((n_nodes, problem))
        
        return problems
    
    def run_benchmarks():
        """Run performance benchmarks"""
        problems = generate_test_problems()
        results = []
        
        for n_nodes, problem in problems:
            start_time = time.time()
            result = uniform_cost_search(problem)
            end_time = time.time()
            
            results.append({
                'nodes': n_nodes,
                'time': end_time - start_time,
                'nodes_expanded': result['nodes_expanded'] if result else 0,
                'path_length': len(result['path']) if result else 0
            })
        
        return results
    
    def plot_results(results):
        """Plot benchmark results"""
        nodes = [r['nodes'] for r in results]
        times = [r['time'] for r in results]
        expanded = [r['nodes_expanded'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Time complexity
        ax1.plot(nodes, times, 'bo-')
        ax1.set_xlabel('Number of Nodes')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('UCS Time Complexity')
        ax1.grid(True)
        
        # Nodes expanded
        ax2.plot(nodes, expanded, 'ro-')
        ax2.set_xlabel('Number of Nodes')
        ax2.set_ylabel('Nodes Expanded')
        ax2.set_title('UCS Nodes Expanded')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    # Run benchmarks
    results = run_benchmarks()
    
    # Print summary
    print("UCS Performance Benchmark Results:")
    print("Nodes\tTime(s)\tExpanded\tPath Length")
    print("-" * 40)
    for r in results:
        print(f"{r['nodes']}\t{r['time']:.4f}\t{r['nodes_expanded']}\t\t{r['path_length']}")
    
    # Plot results (if matplotlib is available)
    try:
        plot_results(results)
    except ImportError:
        print("Matplotlib not available for plotting")

# Run performance benchmark
benchmark_ucs_performance()
```

## Summary

The correctness and reliability of UCS implementation depend on:

1. **Proper Data Structures**: Efficient priority queue implementation
2. **Correct State Management**: Proper handling of explored set and frontier
3. **Path Reconstruction**: Accurate tracking of paths from start to goal
4. **Edge Case Handling**: Robust handling of unusual problem instances
5. **Comprehensive Testing**: Thorough validation of algorithm behavior

Key principles for implementing correct UCS:

- **Invariant Maintenance**: Ensure loop invariants are preserved
- **Termination Guarantees**: Verify algorithm always terminates
- **Optimality Preservation**: Maintain optimality guarantees
- **Performance Monitoring**: Track and optimize performance characteristics
- **Error Handling**: Gracefully handle edge cases and errors 