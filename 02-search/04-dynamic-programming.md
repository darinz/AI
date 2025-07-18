# Dynamic Programming

## Overview

Dynamic Programming (DP) is a method for solving complex problems by breaking them down into simpler subproblems. It is particularly effective for optimization problems where the same subproblems are solved multiple times.

## Core Principles

### 1. Optimal Substructure

A problem has optimal substructure if an optimal solution to the problem contains optimal solutions to its subproblems.

**Mathematical Definition**: If $S^*$ is an optimal solution to problem $P$, then any subproblem $S_i^*$ within $S^*$ must be optimal for its corresponding subproblem $P_i$.

### 2. Overlapping Subproblems

The same subproblems are solved multiple times during the computation.

**Example**: In computing Fibonacci numbers, $F(n-2)$ is computed multiple times when calculating $F(n)$.

### 3. Memoization vs. Tabulation

**Memoization (Top-down)**: Store results of subproblems as they are computed
**Tabulation (Bottom-up)**: Build solutions to subproblems iteratively

## Mathematical Foundation

### Bellman's Principle of Optimality

For any initial state and initial decision, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision.

**Formal Statement**: If $V^*(s)$ is the optimal value function, then:

$$V^*(s) = \max_{a \in A(s)} \left\{ R(s, a) + \gamma \sum_{s'} P(s'|s, a) V^*(s') \right\}$$

Where:
- $V^*(s)$ is the optimal value at state $s$
- $A(s)$ is the set of available actions at state $s$
- $R(s, a)$ is the immediate reward for taking action $a$ in state $s$
- $\gamma$ is the discount factor
- $P(s'|s, a)$ is the transition probability

## Classic Dynamic Programming Problems

### 1. Fibonacci Numbers

**Problem**: Compute the $n$-th Fibonacci number efficiently.

**Recursive Definition**: $F(n) = F(n-1) + F(n-2)$ with $F(0) = 0$ and $F(1) = 1$

```python
def fibonacci_naive(n):
    """Naive recursive implementation - O(2^n)"""
    if n <= 1:
        return n
    return fibonacci_naive(n-1) + fibonacci_naive(n-2)

def fibonacci_memoization(n, memo=None):
    """Top-down DP with memoization - O(n) time, O(n) space"""
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memoization(n-1, memo) + fibonacci_memoization(n-2, memo)
    return memo[n]

def fibonacci_tabulation(n):
    """Bottom-up DP with tabulation - O(n) time, O(n) space"""
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

def fibonacci_optimized(n):
    """Space-optimized DP - O(n) time, O(1) space"""
    if n <= 1:
        return n
    
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr

# Example usage
n = 10
print(f"F({n}) = {fibonacci_optimized(n)}")

# Performance comparison
import time

def benchmark_fibonacci():
    n = 35
    algorithms = {
        'Memoization': fibonacci_memoization,
        'Tabulation': fibonacci_tabulation,
        'Optimized': fibonacci_optimized
    }
    
    for name, func in algorithms.items():
        start_time = time.time()
        result = func(n)
        end_time = time.time()
        print(f"{name}: F({n}) = {result}, Time: {end_time - start_time:.6f}s")

benchmark_fibonacci()
```

### 2. Longest Common Subsequence (LCS)

**Problem**: Find the length of the longest subsequence present in both strings.

**Mathematical Definition**: 
$$LCS(i, j) = \begin{cases} 
0 & \text{if } i = 0 \text{ or } j = 0 \\
LCS(i-1, j-1) + 1 & \text{if } X[i-1] = Y[j-1] \\
\max(LCS(i-1, j), LCS(i, j-1)) & \text{otherwise}
\end{cases}$$

```python
def lcs_recursive(X, Y, m, n):
    """Recursive LCS - O(2^(m+n))"""
    if m == 0 or n == 0:
        return 0
    elif X[m-1] == Y[n-1]:
        return 1 + lcs_recursive(X, Y, m-1, n-1)
    else:
        return max(lcs_recursive(X, Y, m-1, n), lcs_recursive(X, Y, m, n-1))

def lcs_dp(X, Y):
    """Dynamic Programming LCS - O(mn) time and space"""
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Build DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct the subsequence
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if X[i-1] == Y[j-1]:
            lcs.append(X[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return {
        'length': dp[m][n],
        'subsequence': ''.join(reversed(lcs)),
        'dp_table': dp
    }

def lcs_space_optimized(X, Y):
    """Space-optimized LCS - O(mn) time, O(min(m,n)) space"""
    m, n = len(X), len(Y)
    
    # Use the shorter string for the DP array
    if m < n:
        X, Y = Y, X
        m, n = n, m
    
    # Use only two rows
    dp = [[0] * (n + 1) for _ in range(2)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                dp[i % 2][j] = dp[(i-1) % 2][j-1] + 1
            else:
                dp[i % 2][j] = max(dp[(i-1) % 2][j], dp[i % 2][j-1])
    
    return dp[m % 2][n]

# Example usage
X = "ABCDGH"
Y = "AEDFHR"
result = lcs_dp(X, Y)
print(f"LCS of '{X}' and '{Y}': {result['subsequence']} (length: {result['length']})")
```

### 3. Knapsack Problem

**Problem**: Given items with weights and values, find the maximum value that can be obtained with a weight limit.

**Mathematical Definition**:
$$dp[i][w] = \max(dp[i-1][w], dp[i-1][w-w_i] + v_i)$$

```python
def knapsack_01_naive(weights, values, capacity):
    """Naive recursive 0/1 knapsack - O(2^n)"""
    def knapsack_recursive(n, w):
        if n == 0 or w == 0:
            return 0
        
        if weights[n-1] > w:
            return knapsack_recursive(n-1, w)
        
        return max(
            knapsack_recursive(n-1, w),
            knapsack_recursive(n-1, w - weights[n-1]) + values[n-1]
        )
    
    return knapsack_recursive(len(weights), capacity)

def knapsack_01_dp(weights, values, capacity):
    """Dynamic Programming 0/1 knapsack - O(nW) time and space"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Build DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w - weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    
    # Reconstruct solution
    selected_items = []
    i, w = n, capacity
    while i > 0 and w > 0:
        if dp[i][w] != dp[i-1][w]:
            selected_items.append(i-1)
            w -= weights[i-1]
        i -= 1
    
    return {
        'max_value': dp[n][capacity],
        'selected_items': list(reversed(selected_items)),
        'dp_table': dp
    }

def knapsack_01_optimized(weights, values, capacity):
    """Space-optimized 0/1 knapsack - O(nW) time, O(W) space"""
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

# Example usage
weights = [2, 1, 3, 2]
values = [12, 10, 20, 15]
capacity = 5

result = knapsack_01_dp(weights, values, capacity)
print(f"Maximum value: {result['max_value']}")
print(f"Selected items: {result['selected_items']}")
```

### 4. Edit Distance

**Problem**: Find the minimum number of operations (insert, delete, replace) to transform one string into another.

**Mathematical Definition**:
$$ED(i, j) = \begin{cases} 
j & \text{if } i = 0 \\
i & \text{if } j = 0 \\
ED(i-1, j-1) & \text{if } X[i-1] = Y[j-1] \\
1 + \min(ED(i-1, j), ED(i, j-1), ED(i-1, j-1)) & \text{otherwise}
\end{cases}$$

```python
def edit_distance_recursive(X, Y, m, n):
    """Recursive edit distance - O(3^(m+n))"""
    if m == 0:
        return n
    if n == 0:
        return m
    
    if X[m-1] == Y[n-1]:
        return edit_distance_recursive(X, Y, m-1, n-1)
    
    return 1 + min(
        edit_distance_recursive(X, Y, m-1, n),    # Delete
        edit_distance_recursive(X, Y, m, n-1),    # Insert
        edit_distance_recursive(X, Y, m-1, n-1)   # Replace
    )

def edit_distance_dp(X, Y):
    """Dynamic Programming edit distance - O(mn) time and space"""
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # Reconstruct operations
    operations = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and X[i-1] == Y[j-1]:
            operations.append(f"Keep '{X[i-1]}'")
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + 1):
            operations.append(f"Delete '{X[i-1]}'")
            i -= 1
        elif j > 0 and (i == 0 or dp[i][j] == dp[i][j-1] + 1):
            operations.append(f"Insert '{Y[j-1]}'")
            j -= 1
        else:
            operations.append(f"Replace '{X[i-1]}' with '{Y[j-1]}'")
            i -= 1
            j -= 1
    
    return {
        'distance': dp[m][n],
        'operations': list(reversed(operations)),
        'dp_table': dp
    }

# Example usage
X = "kitten"
Y = "sitting"
result = edit_distance_dp(X, Y)
print(f"Edit distance between '{X}' and '{Y}': {result['distance']}")
print("Operations:", result['operations'])
```

## Advanced Dynamic Programming Techniques

### 1. State Compression

For problems with small state spaces, use bit manipulation:

```python
def tsp_dp_bitmask(distances):
    """Traveling Salesman Problem with bitmask DP"""
    n = len(distances)
    dp = {}
    
    def solve(pos, mask):
        if mask == (1 << n) - 1:  # All cities visited
            return distances[pos][0]  # Return to start
        
        state = (pos, mask)
        if state in dp:
            return dp[state]
        
        ans = float('inf')
        for next_city in range(n):
            if mask & (1 << next_city) == 0:  # City not visited
                ans = min(ans, distances[pos][next_city] + 
                         solve(next_city, mask | (1 << next_city)))
        
        dp[state] = ans
        return ans
    
    return solve(0, 1)  # Start from city 0, only city 0 visited

# Example usage
distances = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]
print(f"TSP minimum cost: {tsp_dp_bitmask(distances)}")
```

### 2. Digit DP

For problems involving digits or numbers:

```python
def count_numbers_with_digit_sum(n, target_sum):
    """Count numbers from 1 to n with digit sum equal to target_sum"""
    digits = list(map(int, str(n)))
    length = len(digits)
    
    def solve(pos, tight, sum_so_far):
        if pos == length:
            return 1 if sum_so_far == target_sum else 0
        
        state = (pos, tight, sum_so_far)
        if state in dp:
            return dp[state]
        
        limit = digits[pos] if tight else 9
        ans = 0
        
        for d in range(limit + 1):
            new_tight = tight and (d == limit)
            new_sum = sum_so_far + d
            if new_sum <= target_sum:
                ans += solve(pos + 1, new_tight, new_sum)
        
        dp[state] = ans
        return ans
    
    dp = {}
    return solve(0, True, 0)

# Example usage
n = 100
target = 10
print(f"Numbers from 1 to {n} with digit sum {target}: {count_numbers_with_digit_sum(n, target)}")
```

### 3. Convex Hull Trick

For optimizing functions of the form $f(x) = \min_i(a_i x + b_i)$:

```python
class ConvexHullTrick:
    def __init__(self):
        self.lines = []  # (slope, intercept)
    
    def add_line(self, slope, intercept):
        """Add line y = slope * x + intercept"""
        while len(self.lines) >= 2:
            # Check if new line is better than the last two
            a1, b1 = self.lines[-2]
            a2, b2 = self.lines[-1]
            a3, b3 = slope, intercept
            
            # Intersection of lines 1 and 2
            x12 = (b2 - b1) / (a1 - a2)
            # Intersection of lines 2 and 3
            x23 = (b3 - b2) / (a2 - a3)
            
            if x12 <= x23:
                self.lines.pop()
            else:
                break
        
        self.lines.append((slope, intercept))
    
    def query(self, x):
        """Find minimum value at x"""
        if not self.lines:
            return float('inf')
        
        # Binary search for the best line
        left, right = 0, len(self.lines) - 1
        while left < right:
            mid = (left + right) // 2
            if self.get_value(mid, x) <= self.get_value(mid + 1, x):
                right = mid
            else:
                left = mid + 1
        
        return self.get_value(left, x)
    
    def get_value(self, i, x):
        slope, intercept = self.lines[i]
        return slope * x + intercept

# Example usage
cht = ConvexHullTrick()
cht.add_line(2, 1)   # y = 2x + 1
cht.add_line(-1, 5)  # y = -x + 5
cht.add_line(0, 3)   # y = 3

print(f"Minimum at x=2: {cht.query(2)}")
```

## Performance Analysis and Optimization

```python
def analyze_dp_performance(problem_sizes, algorithms):
    """Analyze performance of different DP approaches"""
    results = {}
    
    for size in problem_sizes:
        results[size] = {}
        
        for name, algorithm in algorithms.items():
            # Generate problem instance
            problem = generate_problem_instance(size)
            
            # Measure time and space
            start_time = time.time()
            solution = algorithm(problem)
            end_time = time.time()
            
            # Estimate space usage (rough approximation)
            import sys
            space_usage = sys.getsizeof(solution) if solution else 0
            
            results[size][name] = {
                'time': end_time - start_time,
                'space': space_usage,
                'solution_quality': evaluate_solution(solution)
            }
    
    return results

def optimize_dp_parameters(problem, param_ranges):
    """Find optimal parameters for DP algorithm"""
    best_params = None
    best_performance = float('inf')
    
    for params in itertools.product(*param_ranges.values()):
        param_dict = dict(zip(param_ranges.keys(), params))
        
        # Run algorithm with these parameters
        start_time = time.time()
        solution = run_dp_with_params(problem, param_dict)
        end_time = time.time()
        
        performance = end_time - start_time
        if performance < best_performance:
            best_performance = performance
            best_params = param_dict
    
    return best_params, best_performance
```

## Summary

Dynamic Programming is a powerful technique for solving optimization problems with overlapping subproblems. Key considerations include:

1. **Problem Analysis**: Identify optimal substructure and overlapping subproblems
2. **State Definition**: Choose appropriate state representation
3. **Implementation Strategy**: Decide between memoization and tabulation
4. **Space Optimization**: Reduce memory usage when possible
5. **Performance Tuning**: Optimize for specific problem characteristics

The choice of approach depends on the problem structure, memory constraints, and performance requirements. 