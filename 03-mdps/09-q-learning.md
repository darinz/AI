# Q-Learning

## Introduction

Q-learning is an off-policy temporal difference learning algorithm for reinforcement learning. It learns the Q-function by updating estimates based on the maximum Q-value of the next state, making it an off-policy method that can learn about the optimal policy while following a different behavior policy.

## Key Concepts

### Off-Policy Learning

Q-learning is an **off-policy** algorithm, meaning:
- It learns about the optimal policy while following a different behavior policy
- The target policy is greedy (optimal), but the behavior policy can be exploratory
- Updates are based on the maximum Q-value of the next state

### Temporal Difference Learning

Q-learning uses temporal difference (TD) learning:
- Learns from single steps of experience
- Updates estimates based on the difference between predicted and actual values
- Combines bootstrapping with sampling

## Mathematical Foundation

### Q-Learning Update Rule

The Q-learning update rule is:

```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

Where:
- Q(s,a) is the current Q-value estimate
- α is the learning rate
- r is the immediate reward
- γ is the discount factor
- max_a' Q(s',a') is the maximum Q-value of the next state
- The term [r + γ max_a' Q(s',a') - Q(s,a)] is the TD error

### Convergence Guarantees

Q-learning is guaranteed to converge to the optimal Q-function under certain conditions:
- All state-action pairs are visited infinitely often
- The learning rate satisfies the Robbins-Monro conditions
- The environment is stationary

## Implementation

### Basic Q-Learning Agent

```python
import numpy as np
from typing import Dict, List, Tuple
import random

class QLearningAgent:
    def __init__(self, states, actions, gamma=0.9, alpha=0.1, epsilon=0.1):
        """
        Q-learning agent.
        
        Args:
            states: List of possible states
            actions: List of possible actions
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        # Initialize Q-values
        self.Q = {}
        for state in states:
            self.Q[state] = {}
            for action in actions:
                self.Q[state][action] = 0.0
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.Q[state].items(), key=lambda x: x[1])[0]
    
    def update(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.Q[state][action]
        max_next_q = max(self.Q[next_state].values())
        target = reward + self.gamma * max_next_q
        self.Q[state][action] = current_q + self.alpha * (target - current_q)
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train the agent using Q-learning."""
        for episode in range(num_episodes):
            state = env.reset()
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Q-learning update
                self.update(state, action, reward, next_state)
                
                state = next_state
                if done:
                    break
            
            if episode % 100 == 0:
                avg_q = np.mean([np.mean(list(q.values())) for q in self.Q.values()])
                print(f"Episode {episode}: Average Q-value = {avg_q:.3f}")
```

### Double Q-Learning

Double Q-learning addresses the overestimation bias in Q-learning by using two Q-functions.

```python
class DoubleQLearningAgent:
    def __init__(self, states, actions, gamma=0.9, alpha=0.1, epsilon=0.1):
        """
        Double Q-learning agent.
        
        Args:
            states: List of possible states
            actions: List of possible actions
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        # Initialize two Q-functions
        self.Q1 = {}
        self.Q2 = {}
        for state in states:
            self.Q1[state] = {}
            self.Q2[state] = {}
            for action in actions:
                self.Q1[state][action] = 0.0
                self.Q2[state][action] = 0.0
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy based on Q1 + Q2."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            # Use sum of Q1 and Q2 for action selection
            q_sum = {}
            for action in self.actions:
                q_sum[action] = self.Q1[state][action] + self.Q2[state][action]
            return max(q_sum.items(), key=lambda x: x[1])[0]
    
    def update(self, state, action, reward, next_state):
        """
        Update Q-values using Double Q-learning.
        
        Randomly choose which Q-function to update.
        """
        if random.random() < 0.5:
            # Update Q1
            current_q = self.Q1[state][action]
            # Use Q2 to select action, Q1 to evaluate
            best_action = max(self.Q2[next_state].items(), key=lambda x: x[1])[0]
            target = reward + self.gamma * self.Q1[next_state][best_action]
            self.Q1[state][action] = current_q + self.alpha * (target - current_q)
        else:
            # Update Q2
            current_q = self.Q2[state][action]
            # Use Q1 to select action, Q2 to evaluate
            best_action = max(self.Q1[next_state].items(), key=lambda x: x[1])[0]
            target = reward + self.gamma * self.Q2[next_state][best_action]
            self.Q2[state][action] = current_q + self.alpha * (target - current_q)
    
    def get_q_value(self, state, action):
        """Get Q-value as average of Q1 and Q2."""
        return (self.Q1[state][action] + self.Q2[state][action]) / 2
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train the agent using Double Q-learning."""
        for episode in range(num_episodes):
            state = env.reset()
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Double Q-learning update
                self.update(state, action, reward, next_state)
                
                state = next_state
                if done:
                    break
            
            if episode % 100 == 0:
                avg_q = np.mean([np.mean([self.get_q_value(s, a) for a in self.actions]) 
                               for s in self.states])
                print(f"Episode {episode}: Average Q-value = {avg_q:.3f}")
```

### Dueling Q-Learning

Dueling Q-learning separates the value function and advantage function for better learning.

```python
class DuelingQLearningAgent:
    def __init__(self, states, actions, gamma=0.9, alpha=0.1, epsilon=0.1):
        """
        Dueling Q-learning agent.
        
        Args:
            states: List of possible states
            actions: List of possible actions
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        # Initialize value function V(s) and advantage function A(s,a)
        self.V = {}
        self.A = {}
        for state in states:
            self.V[state] = 0.0
            self.A[state] = {}
            for action in actions:
                self.A[state][action] = 0.0
    
    def get_q_value(self, state, action):
        """Get Q-value as V(s) + A(s,a)."""
        return self.V[state] + self.A[state][action]
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = {action: self.get_q_value(state, action) for action in self.actions}
            return max(q_values.items(), key=lambda x: x[1])[0]
    
    def update(self, state, action, reward, next_state):
        """
        Update value and advantage functions using dueling Q-learning.
        
        Q(s,a) = V(s) + A(s,a)
        """
        current_q = self.get_q_value(state, action)
        max_next_q = max(self.get_q_value(next_state, a) for a in self.actions)
        target = reward + self.gamma * max_next_q
        td_error = target - current_q
        
        # Update value function
        self.V[state] += self.alpha * td_error
        
        # Update advantage function
        self.A[state][action] += self.alpha * td_error
        
        # Normalize advantage function (optional)
        mean_advantage = np.mean(list(self.A[state].values()))
        for a in self.actions:
            self.A[state][a] -= mean_advantage
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train the agent using Dueling Q-learning."""
        for episode in range(num_episodes):
            state = env.reset()
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Dueling Q-learning update
                self.update(state, action, reward, next_state)
                
                state = next_state
                if done:
                    break
            
            if episode % 100 == 0:
                avg_q = np.mean([np.mean([self.get_q_value(s, a) for a in self.actions]) 
                               for s in self.states])
                print(f"Episode {episode}: Average Q-value = {avg_q:.3f}")
```

### Prioritized Experience Replay Q-Learning

Prioritized experience replay focuses learning on experiences with high TD error.

```python
import heapq

class PrioritizedQLearningAgent:
    def __init__(self, states, actions, gamma=0.9, alpha=0.1, epsilon=0.1, 
                 buffer_size=10000, beta=0.4):
        """
        Prioritized experience replay Q-learning agent.
        
        Args:
            states: List of possible states
            actions: List of possible actions
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate
            buffer_size: Size of experience replay buffer
            beta: Importance sampling exponent
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.beta = beta
        
        # Initialize Q-values
        self.Q = {}
        for state in states:
            self.Q[state] = {}
            for action in actions:
                self.Q[state][action] = 0.0
        
        # Experience replay buffer
        self.experience_buffer = []
        self.priorities = []
        self.max_priority = 1.0
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.Q[state].items(), key=lambda x: x[1])[0]
    
    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer."""
        experience = (state, action, reward, next_state, done)
        
        if len(self.experience_buffer) >= self.buffer_size:
            # Remove oldest experience
            self.experience_buffer.pop(0)
            self.priorities.pop(0)
        
        self.experience_buffer.append(experience)
        self.priorities.append(self.max_priority)
    
    def sample_experience(self, batch_size=32):
        """Sample experiences based on priorities."""
        if len(self.experience_buffer) < batch_size:
            return self.experience_buffer
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / np.sum(priorities)
        
        # Sample experiences
        indices = np.random.choice(len(self.experience_buffer), batch_size, p=probabilities)
        experiences = [self.experience_buffer[i] for i in indices]
        
        # Compute importance sampling weights
        weights = []
        for i in indices:
            weight = (len(self.experience_buffer) * probabilities[i]) ** (-self.beta)
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights) / np.max(weights)
        
        return experiences, weights, indices
    
    def update(self, state, action, reward, next_state, done, weight=1.0):
        """
        Update Q-value using Q-learning with importance sampling weight.
        """
        current_q = self.Q[state][action]
        
        if done:
            target = reward
        else:
            max_next_q = max(self.Q[next_state].values())
            target = reward + self.gamma * max_next_q
        
        td_error = abs(target - current_q)
        self.Q[state][action] = current_q + self.alpha * weight * (target - current_q)
        
        return td_error
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors."""
        for i, td_error in zip(indices, td_errors):
            priority = td_error + 1e-6  # Small constant to avoid zero priority
            self.priorities[i] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def train(self, env, num_episodes=1000, max_steps=100, updates_per_step=4):
        """Train the agent using prioritized experience replay."""
        for episode in range(num_episodes):
            state = env.reset()
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Add experience to buffer
                self.add_experience(state, action, reward, next_state, done)
                
                # Sample and update from replay buffer
                if len(self.experience_buffer) > 32:
                    experiences, weights, indices = self.sample_experience(32)
                    td_errors = []
                    
                    for (s, a, r, ns, d), weight in zip(experiences, weights):
                        td_error = self.update(s, a, r, ns, d, weight)
                        td_errors.append(td_error)
                    
                    # Update priorities
                    self.update_priorities(indices, td_errors)
                
                state = next_state
                if done:
                    break
            
            if episode % 100 == 0:
                avg_q = np.mean([np.mean(list(q.values())) for q in self.Q.values()])
                print(f"Episode {episode}: Average Q-value = {avg_q:.3f}")
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

# Test different Q-learning variants
def test_q_learning_variants():
    """Test different Q-learning variants."""
    env = GridWorldEnv(4, 4)
    states = [(x, y) for x in range(4) for y in range(4)]
    actions = ['up', 'down', 'left', 'right']
    
    print("Testing Q-Learning Variants:")
    print("=" * 50)
    
    # Basic Q-learning
    print("\n1. Basic Q-Learning:")
    q_agent = QLearningAgent(states, actions)
    q_agent.train(env, num_episodes=500)
    
    # Double Q-learning
    print("\n2. Double Q-Learning:")
    double_q_agent = DoubleQLearningAgent(states, actions)
    double_q_agent.train(env, num_episodes=500)
    
    # Dueling Q-learning
    print("\n3. Dueling Q-Learning:")
    dueling_q_agent = DuelingQLearningAgent(states, actions)
    dueling_q_agent.train(env, num_episodes=500)
    
    # Prioritized experience replay Q-learning
    print("\n4. Prioritized Experience Replay Q-Learning:")
    prioritized_q_agent = PrioritizedQLearningAgent(states, actions)
    prioritized_q_agent.train(env, num_episodes=500)
    
    # Compare final Q-values
    print("\nFinal Q-Values Comparison (State (0,0)):")
    print("Action\tQ-Learning\tDouble Q\tDueling Q\tPrioritized Q")
    print("-" * 70)
    
    for action in actions:
        q_val = q_agent.Q[(0, 0)][action]
        double_q_val = double_q_agent.get_q_value((0, 0), action)
        dueling_q_val = dueling_q_agent.get_q_value((0, 0), action)
        prioritized_q_val = prioritized_q_agent.Q[(0, 0)][action]
        
        print(f"{action}\t{q_val:.3f}\t\t{double_q_val:.3f}\t\t{dueling_q_val:.3f}\t\t{prioritized_q_val:.3f}")

if __name__ == "__main__":
    test_q_learning_variants()
```

## Advantages and Disadvantages

### Advantages

1. **Off-policy**: Can learn about optimal policy while following different behavior
2. **Convergence**: Guaranteed to converge to optimal Q-function
3. **Efficiency**: Learns from single steps of experience
4. **Flexibility**: Can use any exploration strategy

### Disadvantages

1. **Overestimation bias**: May overestimate Q-values due to max operator
2. **Exploration dependency**: Still requires good exploration strategy
3. **Sample efficiency**: May require many samples to converge

## Summary

Q-learning is a fundamental off-policy temporal difference learning algorithm. Key points:

1. **Off-policy learning**: Learns about optimal policy while following different behavior
2. **Temporal difference**: Updates based on single steps of experience
3. **Variants**: Double Q-learning, Dueling Q-learning, Prioritized experience replay
4. **Convergence**: Guaranteed to converge to optimal Q-function

The Python implementation demonstrates various Q-learning variants and shows how they can be applied to solve sequential decision problems. 