# Policy Evaluation

## Introduction

Policy evaluation is the process of computing the value function for a given policy. The value function V^π(s) represents the expected cumulative reward starting from state s and following policy π. This is a fundamental step in many reinforcement learning algorithms.

## Mathematical Foundation

### Value Function Definition

The value function V^π(s) for policy π is defined as:

```
V^π(s) = E[Σ_{t=0}^∞ γ^t R(s_t, a_t, s_{t+1}) | s_0 = s, π]
```

Where:
- γ is the discount factor
- R(s_t, a_t, s_{t+1}) is the reward at time t
- The expectation is taken over all possible trajectories following policy π

### Bellman Equation for Policy Evaluation

The value function satisfies the Bellman equation:

```
V^π(s) = Σ_{s'} P(s'|s,π(s)) [R(s,π(s),s') + γV^π(s')]
```

This equation expresses the value of a state as the expected immediate reward plus the discounted value of the next state.

### Matrix Form

For finite state spaces, the Bellman equation can be written in matrix form:

```
V^π = R^π + γP^π V^π
```

Where:
- V^π is the value function vector
- R^π is the expected reward vector
- P^π is the transition probability matrix under policy π

Solving for V^π:

```
V^π = (I - γP^π)^(-1) R^π
```

## Algorithms for Policy Evaluation

### 1. Iterative Policy Evaluation

The most common method for policy evaluation is iterative policy evaluation, which uses the Bellman equation as an update rule.

#### Algorithm

```
Initialize V(s) = 0 for all s
Repeat until convergence:
    For each state s:
        V(s) ← Σ_{s'} P(s'|s,π(s)) [R(s,π(s),s') + γV(s')]
```

#### Implementation

```python
import numpy as np
from typing import Dict, List, Tuple

def iterative_policy_evaluation(mdp, policy, theta=1e-6, max_iterations=1000):
    """
    Iterative policy evaluation algorithm.
    
    Args:
        mdp: MDP model
        policy: Policy to evaluate
        theta: Convergence threshold
        max_iterations: Maximum number of iterations
    
    Returns:
        Value function V(s) for all states
    """
    # Initialize value function
    V = {state: 0.0 for state in mdp.state_space.states}
    
    for iteration in range(max_iterations):
        delta = 0
        
        for state in mdp.state_space.states:
            v = V[state]
            
            # Get action from policy
            action = policy.get_action(state)
            
            # Compute new value using Bellman equation
            new_v = 0
            next_states = mdp.get_next_states(state, action)
            
            for next_state, prob in next_states.items():
                reward = mdp.get_reward(state, action, next_state)
                new_v += prob * (reward + mdp.gamma * V[next_state])
            
            V[state] = new_v
            delta = max(delta, abs(v - new_v))
        
        # Check convergence
        if delta < theta:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    return V
```

### 2. Matrix Inversion Method

For small MDPs, we can solve the Bellman equation directly using matrix inversion.

```python
def matrix_policy_evaluation(mdp, policy):
    """
    Policy evaluation using matrix inversion.
    
    Args:
        mdp: MDP model
        policy: Policy to evaluate
    
    Returns:
        Value function V(s) for all states
    """
    n_states = len(mdp.state_space.states)
    
    # Create transition matrix P^π and reward vector R^π
    P = np.zeros((n_states, n_states))
    R = np.zeros(n_states)
    
    state_to_idx = {state: i for i, state in enumerate(mdp.state_space.states)}
    
    for i, state in enumerate(mdp.state_space.states):
        action = policy.get_action(state)
        next_states = mdp.get_next_states(state, action)
        
        for next_state, prob in next_states.items():
            j = state_to_idx[next_state]
            P[i, j] = prob
            
            # Expected reward for this transition
            reward = mdp.get_reward(state, action, next_state)
            R[i] += prob * reward
    
    # Solve V^π = (I - γP^π)^(-1) R^π
    I = np.eye(n_states)
    V_vector = np.linalg.solve(I - mdp.gamma * P, R)
    
    # Convert back to dictionary
    V = {state: V_vector[i] for i, state in enumerate(mdp.state_space.states)}
    
    return V
```

### 3. Monte Carlo Policy Evaluation

Monte Carlo methods estimate the value function by averaging returns from sample trajectories.

```python
def monte_carlo_policy_evaluation(mdp, policy, num_episodes=1000, max_steps=100):
    """
    Monte Carlo policy evaluation.
    
    Args:
        mdp: MDP model
        policy: Policy to evaluate
        num_episodes: Number of episodes to simulate
        max_steps: Maximum steps per episode
    
    Returns:
        Value function V(s) for all states
    """
    V = {state: 0.0 for state in mdp.state_space.states}
    returns = {state: [] for state in mdp.state_space.states}
    
    for episode in range(num_episodes):
        # Generate episode
        episode_data = []
        state = np.random.choice(mdp.state_space.states)
        
        for step in range(max_steps):
            action = policy.get_action(state)
            next_states = mdp.get_next_states(state, action)
            
            if not next_states:
                break
                
            # Sample next state
            next_state = np.random.choice(
                list(next_states.keys()), 
                p=list(next_states.values())
            )
            
            reward = mdp.get_reward(state, action, next_state)
            episode_data.append((state, action, reward))
            
            state = next_state
            
            # Check if episode should end
            if state == (mdp.width-1, mdp.height-1) if hasattr(mdp, 'width') else False:
                break
        
        # Calculate returns and update value function
        G = 0
        for t in reversed(range(len(episode_data))):
            state, action, reward = episode_data[t]
            G = reward + mdp.gamma * G
            returns[state].append(G)
            V[state] = np.mean(returns[state])
    
    return V
```

## Advanced Policy Evaluation Methods

### 1. Asynchronous Policy Evaluation

Instead of updating all states simultaneously, asynchronous methods update states one at a time.

```python
def asynchronous_policy_evaluation(mdp, policy, theta=1e-6, max_iterations=1000):
    """
    Asynchronous policy evaluation.
    
    Args:
        mdp: MDP model
        policy: Policy to evaluate
        theta: Convergence threshold
        max_iterations: Maximum number of iterations
    
    Returns:
        Value function V(s) for all states
    """
    V = {state: 0.0 for state in mdp.state_space.states}
    
    for iteration in range(max_iterations):
        delta = 0
        
        # Update states in random order
        states = list(mdp.state_space.states)
        np.random.shuffle(states)
        
        for state in states:
            v = V[state]
            
            action = policy.get_action(state)
            new_v = 0
            next_states = mdp.get_next_states(state, action)
            
            for next_state, prob in next_states.items():
                reward = mdp.get_reward(state, action, next_state)
                new_v += prob * (reward + mdp.gamma * V[next_state])
            
            V[state] = new_v
            delta = max(delta, abs(v - new_v))
        
        if delta < theta:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    return V
```

### 2. Prioritized Sweeping

Prioritized sweeping updates states based on their importance, focusing on states that are likely to have changed significantly.

```python
def prioritized_sweeping_policy_evaluation(mdp, policy, theta=1e-6, max_iterations=1000):
    """
    Prioritized sweeping policy evaluation.
    
    Args:
        mdp: MDP model
        policy: Policy to evaluate
        theta: Convergence threshold
        max_iterations: Maximum number of iterations
    
    Returns:
        Value function V(s) for all states
    """
    V = {state: 0.0 for state in mdp.state_space.states}
    priority_queue = []
    
    # Build predecessor mapping
    predecessors = {state: set() for state in mdp.state_space.states}
    for state in mdp.state_space.states:
        action = policy.get_action(state)
        next_states = mdp.get_next_states(state, action)
        for next_state in next_states:
            predecessors[next_state].add(state)
    
    for iteration in range(max_iterations):
        if not priority_queue:
            # Initialize priority queue
            for state in mdp.state_space.states:
                action = policy.get_action(state)
                new_v = 0
                next_states = mdp.get_next_states(state, action)
                
                for next_state, prob in next_states.items():
                    reward = mdp.get_reward(state, action, next_state)
                    new_v += prob * (reward + mdp.gamma * V[next_state])
                
                priority = abs(V[state] - new_v)
                if priority > theta:
                    priority_queue.append((priority, state))
            
            priority_queue.sort(reverse=True)
        
        if not priority_queue:
            break
        
        # Update highest priority state
        priority, state = priority_queue.pop(0)
        old_v = V[state]
        
        action = policy.get_action(state)
        new_v = 0
        next_states = mdp.get_next_states(state, action)
        
        for next_state, prob in next_states.items():
            reward = mdp.get_reward(state, action, next_state)
            new_v += prob * (reward + mdp.gamma * V[next_state])
        
        V[state] = new_v
        
        # Update priorities of predecessors
        for pred_state in predecessors[state]:
            action = policy.get_action(pred_state)
            pred_new_v = 0
            pred_next_states = mdp.get_next_states(pred_state, action)
            
            for next_state, prob in pred_next_states.items():
                reward = mdp.get_reward(pred_state, action, next_state)
                pred_new_v += prob * (reward + mdp.gamma * V[next_state])
            
            pred_priority = abs(V[pred_state] - pred_new_v)
            if pred_priority > theta:
                # Remove old entry if exists
                priority_queue = [(p, s) for p, s in priority_queue if s != pred_state]
                priority_queue.append((pred_priority, pred_state))
                priority_queue.sort(reverse=True)
    
    return V
```

## Example: Grid World Policy Evaluation

```python
# Create a simple grid world MDP
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

# Create a random policy
class RandomPolicy:
    def __init__(self, mdp):
        self.mdp = mdp
    
    def get_action(self, state):
        return np.random.choice(self.mdp.action_space.actions)

# Test policy evaluation
mdp = create_grid_world_mdp()
policy = RandomPolicy(mdp)

print("Testing Policy Evaluation Methods:")
print("=" * 50)

# Method 1: Iterative Policy Evaluation
print("\n1. Iterative Policy Evaluation:")
V_iterative = iterative_policy_evaluation(mdp, policy)
print("Value function:")
for state in sorted(mdp.state_space.states):
    print(f"  V({state}) = {V_iterative[state]:.3f}")

# Method 2: Matrix Inversion
print("\n2. Matrix Inversion Method:")
V_matrix = matrix_policy_evaluation(mdp, policy)
print("Value function:")
for state in sorted(mdp.state_space.states):
    print(f"  V({state}) = {V_matrix[state]:.3f}")

# Method 3: Monte Carlo
print("\n3. Monte Carlo Policy Evaluation:")
V_mc = monte_carlo_policy_evaluation(mdp, policy, num_episodes=1000)
print("Value function:")
for state in sorted(mdp.state_space.states):
    print(f"  V({state}) = {V_mc[state]:.3f}")

# Compare methods
print("\nComparison:")
print("State\tIterative\tMatrix\t\tMonte Carlo")
print("-" * 50)
for state in sorted(mdp.state_space.states):
    print(f"{state}\t{V_iterative[state]:.3f}\t\t{V_matrix[state]:.3f}\t\t{V_mc[state]:.3f}")
```

## Convergence Properties

### Convergence Guarantees

1. **Iterative Policy Evaluation**: Guaranteed to converge for finite MDPs
2. **Matrix Inversion**: Exact solution (when matrix is invertible)
3. **Monte Carlo**: Converges to true value function as number of episodes → ∞

### Convergence Rate

- **Iterative**: Linear convergence rate
- **Matrix Inversion**: O(n³) for n states
- **Monte Carlo**: Slower convergence, but works with unknown dynamics

### Error Bounds

For iterative policy evaluation with discount factor γ:

```
||V_k - V^π||_∞ ≤ γ^k ||V_0 - V^π||_∞
```

Where V_k is the value function after k iterations.

## Summary

Policy evaluation is a fundamental algorithm in reinforcement learning that computes the value function for a given policy. The iterative policy evaluation method is the most commonly used approach, but other methods like matrix inversion and Monte Carlo estimation have their own advantages.

Key points:
1. Policy evaluation computes V^π(s) for all states
2. Uses the Bellman equation as an update rule
3. Guaranteed to converge for finite MDPs
4. Foundation for policy improvement algorithms
5. Multiple implementation methods available

The Python implementation demonstrates different approaches to policy evaluation, from the standard iterative method to more advanced techniques like prioritized sweeping. 