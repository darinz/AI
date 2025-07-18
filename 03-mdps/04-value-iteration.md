# Value Iteration

## Introduction

Value iteration is a dynamic programming algorithm for finding the optimal policy in a Markov Decision Process (MDP). It combines policy evaluation and policy improvement into a single algorithm that iteratively improves the value function until it converges to the optimal value function.

## Mathematical Foundation

### Optimal Value Function

The optimal value function V*(s) is defined as:

```
V*(s) = max_π V^π(s)
```

It represents the maximum expected cumulative reward achievable from state s under any policy.

### Bellman Optimality Equation

The optimal value function satisfies the Bellman optimality equation:

```
V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV*(s')]
```

This equation expresses the optimal value as the maximum over all actions of the expected immediate reward plus the discounted optimal value of the next state.

### Optimal Policy

Once we have the optimal value function, the optimal policy can be extracted using:

```
π*(s) = argmax_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV*(s')]
```

## Value Iteration Algorithm

### Algorithm Description

Value iteration iteratively applies the Bellman optimality equation as an update rule:

```
Initialize V(s) = 0 for all s
Repeat until convergence:
    For each state s:
        V(s) ← max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV(s')]
```

### Implementation

```python
import numpy as np
from typing import Dict, List, Tuple

def value_iteration(mdp, theta=1e-6, max_iterations=1000):
    """
    Value iteration algorithm.
    
    Args:
        mdp: MDP model
        theta: Convergence threshold
        max_iterations: Maximum number of iterations
    
    Returns:
        Optimal value function V*(s) and optimal policy π*(s)
    """
    # Initialize value function
    V = {state: 0.0 for state in mdp.state_space.states}
    
    for iteration in range(max_iterations):
        delta = 0
        
        for state in mdp.state_space.states:
            v = V[state]
            
            # Find the maximum value over all actions
            max_value = float('-inf')
            for action in mdp.action_space.actions:
                value = 0
                next_states = mdp.get_next_states(state, action)
                
                for next_state, prob in next_states.items():
                    reward = mdp.get_reward(state, action, next_state)
                    value += prob * (reward + mdp.gamma * V[next_state])
                
                max_value = max(max_value, value)
            
            V[state] = max_value
            delta = max(delta, abs(v - max_value))
        
        # Check convergence
        if delta < theta:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    # Extract optimal policy
    policy = {}
    for state in mdp.state_space.states:
        best_action = None
        best_value = float('-inf')
        
        for action in mdp.action_space.actions:
            value = 0
            next_states = mdp.get_next_states(state, action)
            
            for next_state, prob in next_states.items():
                reward = mdp.get_reward(state, action, next_state)
                value += prob * (reward + mdp.gamma * V[next_state])
            
            if value > best_value:
                best_value = value
                best_action = action
        
        policy[state] = best_action
    
    return V, policy
```

## Advanced Value Iteration Variants

### 1. Asynchronous Value Iteration

Instead of updating all states simultaneously, asynchronous value iteration updates states one at a time.

```python
def asynchronous_value_iteration(mdp, theta=1e-6, max_iterations=1000):
    """
    Asynchronous value iteration.
    
    Args:
        mdp: MDP model
        theta: Convergence threshold
        max_iterations: Maximum number of iterations
    
    Returns:
        Optimal value function V*(s) and optimal policy π*(s)
    """
    V = {state: 0.0 for state in mdp.state_space.states}
    
    for iteration in range(max_iterations):
        delta = 0
        
        # Update states in random order
        states = list(mdp.state_space.states)
        np.random.shuffle(states)
        
        for state in states:
            v = V[state]
            
            # Find the maximum value over all actions
            max_value = float('-inf')
            for action in mdp.action_space.actions:
                value = 0
                next_states = mdp.get_next_states(state, action)
                
                for next_state, prob in next_states.items():
                    reward = mdp.get_reward(state, action, next_state)
                    value += prob * (reward + mdp.gamma * V[next_state])
                
                max_value = max(max_value, value)
            
            V[state] = max_value
            delta = max(delta, abs(v - max_value))
        
        if delta < theta:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    # Extract optimal policy
    policy = extract_policy(mdp, V)
    
    return V, policy
```

### 2. Prioritized Value Iteration

Prioritized value iteration updates states based on their importance, focusing on states that are likely to have changed significantly.

```python
def prioritized_value_iteration(mdp, theta=1e-6, max_iterations=1000):
    """
    Prioritized value iteration.
    
    Args:
        mdp: MDP model
        theta: Convergence threshold
        max_iterations: Maximum number of iterations
    
    Returns:
        Optimal value function V*(s) and optimal policy π*(s)
    """
    V = {state: 0.0 for state in mdp.state_space.states}
    priority_queue = []
    
    # Build predecessor mapping
    predecessors = {state: set() for state in mdp.state_space.states}
    for state in mdp.state_space.states:
        for action in mdp.action_space.actions:
            next_states = mdp.get_next_states(state, action)
            for next_state in next_states:
                predecessors[next_state].add(state)
    
    for iteration in range(max_iterations):
        if not priority_queue:
            # Initialize priority queue
            for state in mdp.state_space.states:
                priority = compute_priority(mdp, V, state)
                if priority > theta:
                    priority_queue.append((priority, state))
            
            priority_queue.sort(reverse=True)
        
        if not priority_queue:
            break
        
        # Update highest priority state
        priority, state = priority_queue.pop(0)
        old_v = V[state]
        
        # Update value function
        max_value = float('-inf')
        for action in mdp.action_space.actions:
            value = 0
            next_states = mdp.get_next_states(state, action)
            
            for next_state, prob in next_states.items():
                reward = mdp.get_reward(state, action, next_state)
                value += prob * (reward + mdp.gamma * V[next_state])
            
            max_value = max(max_value, value)
        
        V[state] = max_value
        
        # Update priorities of predecessors
        for pred_state in predecessors[state]:
            pred_priority = compute_priority(mdp, V, pred_state)
            if pred_priority > theta:
                # Remove old entry if exists
                priority_queue = [(p, s) for p, s in priority_queue if s != pred_state]
                priority_queue.append((pred_priority, pred_state))
                priority_queue.sort(reverse=True)
    
    # Extract optimal policy
    policy = extract_policy(mdp, V)
    
    return V, policy

def compute_priority(mdp, V, state):
    """Compute priority for a state."""
    max_value = float('-inf')
    for action in mdp.action_space.actions:
        value = 0
        next_states = mdp.get_next_states(state, action)
        
        for next_state, prob in next_states.items():
            reward = mdp.get_reward(state, action, next_state)
            value += prob * (reward + mdp.gamma * V[next_state])
        
        max_value = max(max_value, value)
    
    return abs(V[state] - max_value)
```

### 3. Modified Policy Iteration

Modified policy iteration combines value iteration with policy iteration for better efficiency.

```python
def modified_policy_iteration(mdp, theta=1e-6, max_iterations=1000, k=10):
    """
    Modified policy iteration.
    
    Args:
        mdp: MDP model
        theta: Convergence threshold
        max_iterations: Maximum number of iterations
        k: Number of policy evaluation steps
    
    Returns:
        Optimal value function V*(s) and optimal policy π*(s)
    """
    # Initialize random policy
    policy = {}
    for state in mdp.state_space.states:
        policy[state] = np.random.choice(mdp.action_space.actions)
    
    for iteration in range(max_iterations):
        # Policy evaluation (k steps)
        V = {state: 0.0 for state in mdp.state_space.states}
        
        for _ in range(k):
            for state in mdp.state_space.states:
                action = policy[state]
                value = 0
                next_states = mdp.get_next_states(state, action)
                
                for next_state, prob in next_states.items():
                    reward = mdp.get_reward(state, action, next_state)
                    value += prob * (reward + mdp.gamma * V[next_state])
                
                V[state] = value
        
        # Policy improvement
        stable = True
        for state in mdp.state_space.states:
            old_action = policy[state]
            
            # Find best action
            best_action = None
            best_value = float('-inf')
            
            for action in mdp.action_space.actions:
                value = 0
                next_states = mdp.get_next_states(state, action)
                
                for next_state, prob in next_states.items():
                    reward = mdp.get_reward(state, action, next_state)
                    value += prob * (reward + mdp.gamma * V[next_state])
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            policy[state] = best_action
            
            if old_action != best_action:
                stable = False
        
        if stable:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    return V, policy
```

## Helper Functions

```python
def extract_policy(mdp, V):
    """Extract optimal policy from value function."""
    policy = {}
    
    for state in mdp.state_space.states:
        best_action = None
        best_value = float('-inf')
        
        for action in mdp.action_space.actions:
            value = 0
            next_states = mdp.get_next_states(state, action)
            
            for next_state, prob in next_states.items():
                reward = mdp.get_reward(state, action, next_state)
                value += prob * (reward + mdp.gamma * V[next_state])
            
            if value > best_value:
                best_value = value
                best_action = action
        
        policy[state] = best_action
    
    return policy

def compute_q_values(mdp, V):
    """Compute Q-values from value function."""
    Q = {}
    
    for state in mdp.state_space.states:
        Q[state] = {}
        for action in mdp.action_space.actions:
            value = 0
            next_states = mdp.get_next_states(state, action)
            
            for next_state, prob in next_states.items():
                reward = mdp.get_reward(state, action, next_state)
                value += prob * (reward + mdp.gamma * V[next_state])
            
            Q[state][action] = value
    
    return Q
```

## Example: Grid World Value Iteration

```python
# Create a grid world MDP
def create_grid_world_mdp():
    """Create a 3x3 grid world MDP."""
    mdp = MDPModel()
    
    # Add states (3x3 grid)
    for x in range(3):
        for y in range(3):
            mdp.add_state((x, y))
    
    # Add actions
    actions = ['up', 'down', 'left', 'right']
    for action in actions:
        mdp.add_action(action)
    
    # Add transitions
    for state in mdp.state_space.states:
        x, y = state
        for action in actions:
            if action == 'up':
                next_state = (x, y + 1)
            elif action == 'down':
                next_state = (x, y - 1)
            elif action == 'left':
                next_state = (x - 1, y)
            elif action == 'right':
                next_state = (x + 1, y)
            
            # Check boundaries
            if 0 <= next_state[0] < 3 and 0 <= next_state[1] < 3:
                mdp.set_transition(state, action, next_state, 1.0)
            else:
                mdp.set_transition(state, action, state, 1.0)
    
    # Add rewards
    goal_state = (2, 2)
    for state in mdp.state_space.states:
        for action in actions:
            next_states = mdp.get_next_states(state, action)
            for next_state in next_states:
                if next_state == goal_state:
                    mdp.set_reward(state, action, next_state, 10.0)
                else:
                    mdp.set_reward(state, action, next_state, -1.0)
    
    return mdp

# Test value iteration
mdp = create_grid_world_mdp()

print("Value Iteration Results:")
print("=" * 50)

# Standard value iteration
V_optimal, policy_optimal = value_iteration(mdp)

print("\nOptimal Value Function:")
for state in sorted(mdp.state_space.states):
    print(f"  V*({state}) = {V_optimal[state]:.3f}")

print("\nOptimal Policy:")
for state in sorted(mdp.state_space.states):
    print(f"  π*({state}) = {policy_optimal[state]}")

# Compute Q-values
Q_optimal = compute_q_values(mdp, V_optimal)

print("\nOptimal Q-Values:")
for state in sorted(mdp.state_space.states):
    print(f"  State {state}:")
    for action in mdp.action_space.actions:
        print(f"    Q*({state}, {action}) = {Q_optimal[state][action]:.3f}")

# Test different algorithms
print("\nComparing Algorithms:")
print("=" * 50)

# Asynchronous value iteration
V_async, policy_async = asynchronous_value_iteration(mdp)
print("\nAsynchronous Value Iteration:")
for state in sorted(mdp.state_space.states):
    print(f"  V*({state}) = {V_async[state]:.3f}")

# Modified policy iteration
V_modified, policy_modified = modified_policy_iteration(mdp, k=5)
print("\nModified Policy Iteration:")
for state in sorted(mdp.state_space.states):
    print(f"  V*({state}) = {V_modified[state]:.3f}")

# Compare policies
print("\nPolicy Comparison:")
print("State\tStandard\tAsync\t\tModified")
print("-" * 50)
for state in sorted(mdp.state_space.states):
    print(f"{state}\t{policy_optimal[state]}\t\t{policy_async[state]}\t\t{policy_modified[state]}")
```

## Convergence Analysis

### Convergence Guarantees

Value iteration is guaranteed to converge to the optimal value function for finite MDPs with discount factor γ < 1.

### Convergence Rate

The convergence rate depends on the discount factor γ:
- **Linear convergence**: ||V_{k+1} - V*||_∞ ≤ γ ||V_k - V*||_∞
- **Faster convergence** with smaller γ
- **Slower convergence** with γ closer to 1

### Error Bounds

For any iteration k:

```
||V_k - V*||_∞ ≤ γ^k ||V_0 - V*||_∞
```

Where V_k is the value function after k iterations.

### Stopping Criteria

Common stopping criteria:
1. **Fixed number of iterations**: Stop after k iterations
2. **Convergence threshold**: Stop when max change < θ
3. **Relative convergence**: Stop when max change / max value < θ

## Complexity Analysis

### Time Complexity

- **Standard value iteration**: O(|S|²|A|) per iteration
- **Asynchronous**: O(|S||A|) per iteration
- **Prioritized**: O(log|S|) per state update

### Space Complexity

- **Value function**: O(|S|)
- **Policy**: O(|S|)
- **Q-values**: O(|S||A|)

## Applications

Value iteration is used in various domains:

1. **Robotics**: Path planning, navigation
2. **Game Playing**: Strategy optimization
3. **Operations Research**: Resource allocation
4. **Finance**: Portfolio optimization
5. **Healthcare**: Treatment planning

## Summary

Value iteration is a fundamental algorithm for solving MDPs that finds the optimal policy by iteratively improving the value function. Key points:

1. **Convergence**: Guaranteed to converge to optimal solution
2. **Efficiency**: Multiple variants for different scenarios
3. **Flexibility**: Can handle various MDP structures
4. **Foundation**: Basis for many other RL algorithms

The Python implementation demonstrates the standard algorithm and several variants, showing how to extract optimal policies and analyze convergence properties. 