# Model-free Monte Carlo

## Introduction

Model-free Monte Carlo methods learn directly from experience without building an explicit model of the environment. These methods estimate value functions by averaging returns from sample trajectories, making them particularly useful when the environment dynamics are unknown or complex.

## Key Concepts

### Episode-based Learning

Monte Carlo methods learn from complete episodes:
- **Episode**: A sequence of states, actions, and rewards from start to finish
- **Return**: The cumulative discounted reward from a state to the end of an episode
- **First-visit vs Every-visit**: How to handle multiple visits to the same state

### Value Function Estimation

The value function is estimated as the average of returns:

```
V^π(s) = average of all returns starting from state s
```

## Mathematical Foundation

### Return Definition

The return G_t from time t is defined as:

```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... + γ^{T-t-1}R_T
```

Where:
- γ is the discount factor
- T is the final time step of the episode

### Monte Carlo Estimation

For a policy π, the value function is estimated as:

```
V^π(s) = (1/N(s)) Σ_{i=1}^{N(s)} G_i(s)
```

Where:
- N(s) is the number of visits to state s
- G_i(s) is the return from the i-th visit to state s

## Implementation

### Basic Monte Carlo Agent

```python
import numpy as np
from typing import Dict, List, Tuple
import random

class MonteCarloAgent:
    def __init__(self, states, actions, gamma=0.9, epsilon=0.1, 
                 first_visit=True):
        """
        Monte Carlo agent.
        
        Args:
            states: List of possible states
            actions: List of possible actions
            gamma: Discount factor
            epsilon: Exploration rate
            first_visit: If True, use first-visit MC, else every-visit MC
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.first_visit = first_visit
        
        # Initialize Q-values
        self.Q = {}
        for state in states:
            self.Q[state] = {}
            for action in actions:
                self.Q[state][action] = 0.0
        
        # Initialize returns for each state-action pair
        self.returns = {}
        for state in states:
            self.returns[state] = {}
            for action in actions:
                self.returns[state][action] = []
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.Q[state].items(), key=lambda x: x[1])[0]
    
    def compute_returns(self, episode):
        """
        Compute returns for an episode.
        
        Args:
            episode: List of (state, action, reward) tuples
        
        Returns:
            List of returns corresponding to each step
        """
        returns = []
        G = 0
        
        # Process episode in reverse order
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def update(self, episode):
        """
        Update Q-values using Monte Carlo method.
        
        Args:
            episode: List of (state, action, reward) tuples
        """
        returns = self.compute_returns(episode)
        
        # Track visited state-action pairs
        visited_pairs = set()
        
        for t, (state, action, reward) in enumerate(episode):
            state_action = (state, action)
            
            if self.first_visit:
                # First-visit MC: only update on first occurrence
                if state_action in visited_pairs:
                    continue
                visited_pairs.add(state_action)
            
            # Add return to the list
            self.returns[state][action].append(returns[t])
            
            # Update Q-value as average of all returns
            self.Q[state][action] = np.mean(self.returns[state][action])
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train the agent."""
        for episode_num in range(num_episodes):
            state = env.reset()
            episode_data = []
            
            # Generate episode
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_data.append((state, action, reward))
                
                state = next_state
                if done:
                    break
            
            # Update Q-values using Monte Carlo
            self.update(episode_data)
            
            if episode_num % 100 == 0:
                avg_q = np.mean([np.mean(list(q.values())) for q in self.Q.values()])
                print(f"Episode {episode_num}: Average Q-value = {avg_q:.3f}")
```

### Monte Carlo Control

Monte Carlo control combines policy evaluation with policy improvement.

```python
class MonteCarloControlAgent:
    def __init__(self, states, actions, gamma=0.9, epsilon=0.1, 
                 first_visit=True):
        """
        Monte Carlo control agent.
        
        Args:
            states: List of possible states
            actions: List of possible actions
            gamma: Discount factor
            epsilon: Exploration rate
            first_visit: If True, use first-visit MC, else every-visit MC
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.first_visit = first_visit
        
        # Initialize Q-values
        self.Q = {}
        for state in states:
            self.Q[state] = {}
            for action in actions:
                self.Q[state][action] = 0.0
        
        # Initialize returns for each state-action pair
        self.returns = {}
        for state in states:
            self.returns[state] = {}
            for action in actions:
                self.returns[state][action] = []
        
        # Initialize policy (epsilon-greedy)
        self.policy = {}
        for state in states:
            self.policy[state] = {}
            for action in actions:
                self.policy[state][action] = 1.0 / len(actions)
    
    def get_action(self, state):
        """Choose action using current policy."""
        action_probs = list(self.policy[state].values())
        return np.random.choice(self.actions, p=action_probs)
    
    def update_policy(self, state):
        """Update policy to be epsilon-greedy with respect to Q-values."""
        best_action = max(self.Q[state].items(), key=lambda x: x[1])[0]
        
        for action in self.actions:
            if action == best_action:
                self.policy[state][action] = 1 - self.epsilon + self.epsilon / len(self.actions)
            else:
                self.policy[state][action] = self.epsilon / len(self.actions)
    
    def compute_returns(self, episode):
        """Compute returns for an episode."""
        returns = []
        G = 0
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def update(self, episode):
        """Update Q-values and policy using Monte Carlo control."""
        returns = self.compute_returns(episode)
        
        visited_pairs = set()
        
        for t, (state, action, reward) in enumerate(episode):
            state_action = (state, action)
            
            if self.first_visit:
                if state_action in visited_pairs:
                    continue
                visited_pairs.add(state_action)
            
            # Update Q-value
            self.returns[state][action].append(returns[t])
            self.Q[state][action] = np.mean(self.returns[state][action])
            
            # Update policy
            self.update_policy(state)
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train the agent."""
        for episode_num in range(num_episodes):
            state = env.reset()
            episode_data = []
            
            # Generate episode
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_data.append((state, action, reward))
                
                state = next_state
                if done:
                    break
            
            # Update Q-values and policy
            self.update(episode_data)
            
            if episode_num % 100 == 0:
                avg_q = np.mean([np.mean(list(q.values())) for q in self.Q.values()])
                print(f"Episode {episode_num}: Average Q-value = {avg_q:.3f}")
```

### Off-Policy Monte Carlo

Off-policy Monte Carlo learns about a target policy while following a different behavior policy.

```python
class OffPolicyMonteCarloAgent:
    def __init__(self, states, actions, gamma=0.9, epsilon=0.1):
        """
        Off-policy Monte Carlo agent.
        
        Args:
            states: List of possible states
            actions: List of possible actions
            gamma: Discount factor
            epsilon: Exploration rate for behavior policy
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-values
        self.Q = {}
        for state in states:
            self.Q[state] = {}
            for action in actions:
                self.Q[state][action] = 0.0
        
        # Initialize cumulative weights
        self.C = {}
        for state in states:
            self.C[state] = {}
            for action in actions:
                self.C[state][action] = 0.0
    
    def get_behavior_action(self, state):
        """Choose action using behavior policy (epsilon-greedy)."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.Q[state].items(), key=lambda x: x[1])[0]
    
    def get_target_action(self, state):
        """Choose action using target policy (greedy)."""
        return max(self.Q[state].items(), key=lambda x: x[1])[0]
    
    def compute_importance_sampling_ratio(self, episode):
        """
        Compute importance sampling ratios for an episode.
        
        Returns:
            List of importance sampling ratios
        """
        ratios = []
        
        for state, action, reward in episode:
            # Target policy is greedy
            target_action = self.get_target_action(state)
            
            if action == target_action:
                # Target policy probability = 1
                target_prob = 1.0
            else:
                # Target policy probability = 0
                target_prob = 0.0
            
            # Behavior policy is epsilon-greedy
            if action == max(self.Q[state].items(), key=lambda x: x[1])[0]:
                behavior_prob = 1 - self.epsilon + self.epsilon / len(self.actions)
            else:
                behavior_prob = self.epsilon / len(self.actions)
            
            # Importance sampling ratio
            if behavior_prob > 0:
                ratio = target_prob / behavior_prob
            else:
                ratio = 0.0
            
            ratios.append(ratio)
        
        return ratios
    
    def update(self, episode):
        """Update Q-values using off-policy Monte Carlo."""
        returns = self.compute_returns(episode)
        ratios = self.compute_importance_sampling_ratio(episode)
        
        W = 1.0  # Importance sampling weight
        
        for t, (state, action, reward) in enumerate(episode):
            # Update cumulative weight
            self.C[state][action] += W
            
            # Incremental update
            self.Q[state][action] += (W / self.C[state][action]) * (returns[t] - self.Q[state][action])
            
            # Update importance sampling weight
            W *= ratios[t]
            
            # If W becomes 0, we can stop (no more updates will be made)
            if W == 0:
                break
    
    def compute_returns(self, episode):
        """Compute returns for an episode."""
        returns = []
        G = 0
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train the agent."""
        for episode_num in range(num_episodes):
            state = env.reset()
            episode_data = []
            
            # Generate episode using behavior policy
            for step in range(max_steps):
                action = self.get_behavior_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_data.append((state, action, reward))
                
                state = next_state
                if done:
                    break
            
            # Update Q-values using off-policy Monte Carlo
            self.update(episode_data)
            
            if episode_num % 100 == 0:
                avg_q = np.mean([np.mean(list(q.values())) for q in self.Q.values()])
                print(f"Episode {episode_num}: Average Q-value = {avg_q:.3f}")
```

## Advanced Monte Carlo Methods

### Weighted Importance Sampling

Weighted importance sampling provides more stable estimates than ordinary importance sampling.

```python
class WeightedImportanceSamplingAgent:
    def __init__(self, states, actions, gamma=0.9, epsilon=0.1):
        """
        Weighted importance sampling Monte Carlo agent.
        
        Args:
            states: List of possible states
            actions: List of possible actions
            gamma: Discount factor
            epsilon: Exploration rate for behavior policy
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-values
        self.Q = {}
        for state in states:
            self.Q[state] = {}
            for action in actions:
                self.Q[state][action] = 0.0
        
        # Initialize weighted sums and counts
        self.weighted_sums = {}
        self.counts = {}
        for state in states:
            self.weighted_sums[state] = {}
            self.counts[state] = {}
            for action in actions:
                self.weighted_sums[state][action] = 0.0
                self.counts[state][action] = 0.0
    
    def get_behavior_action(self, state):
        """Choose action using behavior policy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.Q[state].items(), key=lambda x: x[1])[0]
    
    def get_target_action(self, state):
        """Choose action using target policy."""
        return max(self.Q[state].items(), key=lambda x: x[1])[0]
    
    def compute_importance_sampling_ratio(self, episode):
        """Compute importance sampling ratios."""
        ratios = []
        
        for state, action, reward in episode:
            target_action = self.get_target_action(state)
            
            if action == target_action:
                target_prob = 1.0
            else:
                target_prob = 0.0
            
            if action == max(self.Q[state].items(), key=lambda x: x[1])[0]:
                behavior_prob = 1 - self.epsilon + self.epsilon / len(self.actions)
            else:
                behavior_prob = self.epsilon / len(self.actions)
            
            if behavior_prob > 0:
                ratio = target_prob / behavior_prob
            else:
                ratio = 0.0
            
            ratios.append(ratio)
        
        return ratios
    
    def update(self, episode):
        """Update Q-values using weighted importance sampling."""
        returns = self.compute_returns(episode)
        ratios = self.compute_importance_sampling_ratio(episode)
        
        W = 1.0
        
        for t, (state, action, reward) in enumerate(episode):
            # Update weighted sum and count
            self.weighted_sums[state][action] += W * returns[t]
            self.counts[state][action] += W
            
            # Update Q-value
            if self.counts[state][action] > 0:
                self.Q[state][action] = self.weighted_sums[state][action] / self.counts[state][action]
            
            # Update importance sampling weight
            W *= ratios[t]
            
            if W == 0:
                break
    
    def compute_returns(self, episode):
        """Compute returns for an episode."""
        returns = []
        G = 0
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        return returns
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train the agent."""
        for episode_num in range(num_episodes):
            state = env.reset()
            episode_data = []
            
            for step in range(max_steps):
                action = self.get_behavior_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_data.append((state, action, reward))
                
                state = next_state
                if done:
                    break
            
            self.update(episode_data)
            
            if episode_num % 100 == 0:
                avg_q = np.mean([np.mean(list(q.values())) for q in self.Q.values()])
                print(f"Episode {episode_num}: Average Q-value = {avg_q:.3f}")
```

## Example: Grid World Environment

```python
class GridWorldEnv:
    def __init__(self, width=4, height=4):
        """Simple grid world environment."""
        self.width = width
        self.height = height
        self.start_state = (0, 0)
        self.goal_state = (width-1, height-1)
        self.current_state = self.start_state
        
    def reset(self):
        """Reset environment to start state."""
        self.current_state = self.start_state
        return self.current_state
    
    def step(self, action):
        """Take a step in the environment."""
        x, y = self.current_state
        
        if action == 'up':
            next_state = (x, y + 1)
        elif action == 'down':
            next_state = (x, y - 1)
        elif action == 'left':
            next_state = (x - 1, y)
        elif action == 'right':
            next_state = (x + 1, y)
        
        # Check boundaries
        if (0 <= next_state[0] < self.width and 
            0 <= next_state[1] < self.height):
            self.current_state = next_state
        else:
            next_state = self.current_state
        
        # Determine reward and done
        if next_state == self.goal_state:
            reward = 1.0
            done = True
        else:
            reward = -0.01
            done = False
        
        return next_state, reward, done, {}

# Test different Monte Carlo methods
def test_monte_carlo_methods():
    """Test different Monte Carlo methods."""
    env = GridWorldEnv(4, 4)
    states = [(x, y) for x in range(4) for y in range(4)]
    actions = ['up', 'down', 'left', 'right']
    
    print("Testing Model-Free Monte Carlo Methods:")
    print("=" * 60)
    
    # On-policy Monte Carlo
    print("\n1. On-Policy Monte Carlo:")
    mc_agent = MonteCarloAgent(states, actions, first_visit=True)
    mc_agent.train(env, num_episodes=500)
    
    # Monte Carlo Control
    print("\n2. Monte Carlo Control:")
    mc_control_agent = MonteCarloControlAgent(states, actions)
    mc_control_agent.train(env, num_episodes=500)
    
    # Off-policy Monte Carlo
    print("\n3. Off-Policy Monte Carlo:")
    off_policy_agent = OffPolicyMonteCarloAgent(states, actions)
    off_policy_agent.train(env, num_episodes=500)
    
    # Weighted Importance Sampling
    print("\n4. Weighted Importance Sampling:")
    weighted_agent = WeightedImportanceSamplingAgent(states, actions)
    weighted_agent.train(env, num_episodes=500)
    
    # Compare final Q-values
    print("\nFinal Q-Values Comparison (State (0,0)):")
    print("Action\tOn-Policy\tMC Control\tOff-Policy\tWeighted")
    print("-" * 70)
    
    for action in actions:
        on_policy_q = mc_agent.Q[(0, 0)][action]
        mc_control_q = mc_control_agent.Q[(0, 0)][action]
        off_policy_q = off_policy_agent.Q[(0, 0)][action]
        weighted_q = weighted_agent.Q[(0, 0)][action]
        
        print(f"{action}\t{on_policy_q:.3f}\t\t{mc_control_q:.3f}\t\t{off_policy_q:.3f}\t\t{weighted_q:.3f}")

if __name__ == "__main__":
    test_monte_carlo_methods()
```

## Advantages and Disadvantages

### Advantages

1. **Model-free**: No need to learn environment dynamics
2. **Episode-based**: Natural for episodic tasks
3. **Convergence**: Guaranteed to converge to optimal policy
4. **Simplicity**: Conceptually simple and easy to implement

### Disadvantages

1. **Episode requirement**: Needs complete episodes
2. **Sample efficiency**: May require many episodes
3. **Variance**: High variance in estimates
4. **Exploration**: Requires good exploration strategy

## Summary

Model-free Monte Carlo methods provide a powerful approach to reinforcement learning by learning directly from experience. Key points:

1. **Episode-based learning**: Learn from complete trajectories
2. **Return averaging**: Estimate value functions by averaging returns
3. **Policy improvement**: Combine evaluation with control
4. **Importance sampling**: Handle off-policy learning

The Python implementation demonstrates various Monte Carlo methods and shows how they can be applied to solve sequential decision problems without requiring a model of the environment. 