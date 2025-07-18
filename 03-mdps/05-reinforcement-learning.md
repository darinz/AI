# Reinforcement Learning

## Introduction

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties for its actions and learns to maximize cumulative rewards over time.

## Core Concepts

### Agent-Environment Interaction

The RL framework consists of:
- **Agent**: The learner/decision maker
- **Environment**: The world the agent interacts with
- **State**: Current situation of the environment
- **Action**: What the agent can do
- **Reward**: Feedback from the environment
- **Policy**: Strategy for choosing actions

### Mathematical Framework

An RL problem is typically modeled as a Markov Decision Process (MDP) with:
- State space S
- Action space A
- Transition function P(s'|s,a)
- Reward function R(s,a,s')
- Discount factor γ

## Types of Reinforcement Learning

### 1. Model-Based vs Model-Free

**Model-Based RL:**
- Learns a model of the environment
- Uses the model for planning
- Examples: Dyna-Q, Model-based Monte Carlo

**Model-Free RL:**
- Learns directly from experience
- No explicit model of the environment
- Examples: Q-learning, SARSA, Monte Carlo

### 2. On-Policy vs Off-Policy

**On-Policy:**
- Learns about the policy being used for behavior
- Examples: SARSA, Monte Carlo Control

**Off-Policy:**
- Learns about a different policy than the one used for behavior
- Examples: Q-learning, Expected SARSA

## Key Algorithms

### 1. Monte Carlo Methods

Monte Carlo methods learn from complete episodes of experience.

```python
import numpy as np
from typing import Dict, List, Tuple
import random

class MonteCarloAgent:
    def __init__(self, states, actions, gamma=0.9, epsilon=0.1):
        """
        Monte Carlo agent.
        
        Args:
            states: List of possible states
            actions: List of possible actions
            gamma: Discount factor
            epsilon: Exploration rate
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
        
        # Initialize returns
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
    
    def update(self, episode):
        """
        Update Q-values using Monte Carlo method.
        
        Args:
            episode: List of (state, action, reward) tuples
        """
        G = 0  # Return
        
        # Process episode in reverse order
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + self.gamma * G
            
            # Check if this state-action pair appears for the first time
            if (state, action) not in [(s, a) for s, a, _ in episode[:t]]:
                self.returns[state][action].append(G)
                self.Q[state][action] = np.mean(self.returns[state][action])
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train the agent."""
        for episode in range(num_episodes):
            state = env.reset()
            episode_data = []
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_data.append((state, action, reward))
                
                state = next_state
                if done:
                    break
            
            # Update Q-values
            self.update(episode_data)
            
            if episode % 100 == 0:
                print(f"Episode {episode}: Average Q-value = {np.mean([np.mean(list(q.values())) for q in self.Q.values()]):.3f}")
```

### 2. Temporal Difference Learning

Temporal Difference (TD) methods learn from single steps of experience.

#### SARSA (On-Policy TD)

```python
class SARSAAgent:
    def __init__(self, states, actions, gamma=0.9, alpha=0.1, epsilon=0.1):
        """
        SARSA agent.
        
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
    
    def update(self, state, action, reward, next_state, next_action):
        """
        Update Q-value using SARSA.
        
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        """
        current_q = self.Q[state][action]
        next_q = self.Q[next_state][next_action]
        target = reward + self.gamma * next_q
        self.Q[state][action] = current_q + self.alpha * (target - current_q)
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train the agent using SARSA."""
        for episode in range(num_episodes):
            state = env.reset()
            action = self.get_action(state)
            
            for step in range(max_steps):
                next_state, reward, done, _ = env.step(action)
                next_action = self.get_action(next_state)
                
                # SARSA update
                self.update(state, action, reward, next_state, next_action)
                
                state = next_state
                action = next_action
                
                if done:
                    break
            
            if episode % 100 == 0:
                print(f"Episode {episode}: Average Q-value = {np.mean([np.mean(list(q.values())) for q in self.Q.values()]):.3f}")
```

#### Q-Learning (Off-Policy TD)

```python
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
                print(f"Episode {episode}: Average Q-value = {np.mean([np.mean(list(q.values())) for q in self.Q.values()]):.3f}")
```

### 3. Expected SARSA

Expected SARSA uses the expected value of the next action instead of the actual next action.

```python
class ExpectedSARSAAgent:
    def __init__(self, states, actions, gamma=0.9, alpha=0.1, epsilon=0.1):
        """
        Expected SARSA agent.
        
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
    
    def get_expected_q_value(self, state):
        """Get expected Q-value for a state under current policy."""
        q_values = list(self.Q[state].values())
        max_q = max(q_values)
        
        # Count actions with maximum Q-value
        num_max_actions = sum(1 for q in q_values if q == max_q)
        
        # Calculate expected Q-value
        expected_q = 0
        for action in self.actions:
            if self.Q[state][action] == max_q:
                # Greedy action
                prob = (1 - self.epsilon) / num_max_actions + self.epsilon / len(self.actions)
            else:
                # Non-greedy action
                prob = self.epsilon / len(self.actions)
            
            expected_q += prob * self.Q[state][action]
        
        return expected_q
    
    def update(self, state, action, reward, next_state):
        """
        Update Q-value using Expected SARSA.
        
        Q(s,a) ← Q(s,a) + α[r + γ E[Q(s',a')] - Q(s,a)]
        """
        current_q = self.Q[state][action]
        expected_next_q = self.get_expected_q_value(next_state)
        target = reward + self.gamma * expected_next_q
        self.Q[state][action] = current_q + self.alpha * (target - current_q)
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train the agent using Expected SARSA."""
        for episode in range(num_episodes):
            state = env.reset()
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Expected SARSA update
                self.update(state, action, reward, next_state)
                
                state = next_state
                if done:
                    break
            
            if episode % 100 == 0:
                print(f"Episode {episode}: Average Q-value = {np.mean([np.mean(list(q.values())) for q in self.Q.values()]):.3f}")
```

## Exploration Strategies

### 1. Epsilon-Greedy

```python
def epsilon_greedy_policy(Q, state, actions, epsilon):
    """Epsilon-greedy policy."""
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        return max(Q[state].items(), key=lambda x: x[1])[0]
```

### 2. Softmax (Boltzmann)

```python
def softmax_policy(Q, state, actions, temperature=1.0):
    """Softmax policy."""
    q_values = [Q[state][action] for action in actions]
    exp_q = np.exp(np.array(q_values) / temperature)
    probs = exp_q / np.sum(exp_q)
    return np.random.choice(actions, p=probs)
```

### 3. Upper Confidence Bound (UCB)

```python
class UCBAgent:
    def __init__(self, states, actions, gamma=0.9, alpha=0.1, c=2.0):
        """
        UCB agent.
        
        Args:
            states: List of possible states
            actions: List of possible actions
            gamma: Discount factor
            alpha: Learning rate
            c: Exploration constant
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.c = c
        
        # Initialize Q-values and visit counts
        self.Q = {}
        self.N = {}
        self.t = 0
        
        for state in states:
            self.Q[state] = {}
            self.N[state] = {}
            for action in actions:
                self.Q[state][action] = 0.0
                self.N[state][action] = 0
    
    def get_action(self, state):
        """Choose action using UCB."""
        if self.t < len(self.actions):
            # Initial exploration
            return self.actions[self.t]
        
        # UCB selection
        ucb_values = {}
        for action in self.actions:
            if self.N[state][action] == 0:
                ucb_values[action] = float('inf')
            else:
                ucb_values[action] = (self.Q[state][action] + 
                                    self.c * np.sqrt(np.log(self.t) / self.N[state][action]))
        
        return max(ucb_values.items(), key=lambda x: x[1])[0]
    
    def update(self, state, action, reward, next_state):
        """Update Q-value and visit count."""
        self.N[state][action] += 1
        current_q = self.Q[state][action]
        max_next_q = max(self.Q[next_state].values())
        target = reward + self.gamma * max_next_q
        self.Q[state][action] = current_q + self.alpha * (target - current_q)
        self.t += 1
```

## Function Approximation

For large state spaces, we need function approximation to represent Q-values.

### Linear Function Approximation

```python
class LinearQAgent:
    def __init__(self, feature_dim, actions, gamma=0.9, alpha=0.1, epsilon=0.1):
        """
        Linear Q-learning agent.
        
        Args:
            feature_dim: Dimension of feature vector
            actions: List of possible actions
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate
        """
        self.feature_dim = feature_dim
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        # Initialize weights for each action
        self.weights = {}
        for action in actions:
            self.weights[action] = np.zeros(feature_dim)
    
    def get_features(self, state):
        """Extract features from state."""
        # This is a simple example - in practice, you'd use more sophisticated features
        if isinstance(state, tuple):
            return np.array([state[0], state[1], state[0] * state[1]])
        else:
            return np.array([state])
    
    def get_q_value(self, state, action):
        """Get Q-value using linear function approximation."""
        features = self.get_features(state)
        return np.dot(self.weights[action], features)
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = {action: self.get_q_value(state, action) for action in self.actions}
            return max(q_values.items(), key=lambda x: x[1])[0]
    
    def update(self, state, action, reward, next_state):
        """Update weights using linear Q-learning."""
        features = self.get_features(state)
        current_q = self.get_q_value(state, action)
        max_next_q = max(self.get_q_value(next_state, a) for a in self.actions)
        target = reward + self.gamma * max_next_q
        
        # Update weights
        td_error = target - current_q
        self.weights[action] += self.alpha * td_error * features
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

# Test different agents
def test_agents():
    """Test different RL agents on grid world."""
    env = GridWorldEnv(4, 4)
    states = [(x, y) for x in range(4) for y in range(4)]
    actions = ['up', 'down', 'left', 'right']
    
    print("Testing Reinforcement Learning Agents:")
    print("=" * 50)
    
    # Test Q-learning
    print("\n1. Q-Learning Agent:")
    q_agent = QLearningAgent(states, actions)
    q_agent.train(env, num_episodes=500)
    
    # Test SARSA
    print("\n2. SARSA Agent:")
    sarsa_agent = SARSAAgent(states, actions)
    sarsa_agent.train(env, num_episodes=500)
    
    # Test Expected SARSA
    print("\n3. Expected SARSA Agent:")
    expected_sarsa_agent = ExpectedSARSAAgent(states, actions)
    expected_sarsa_agent.train(env, num_episodes=500)
    
    # Compare final policies
    print("\nFinal Policies Comparison:")
    print("State\tQ-Learning\tSARSA\t\tExpected SARSA")
    print("-" * 60)
    
    for state in sorted(states):
        q_action = q_agent.get_action(state)
        sarsa_action = sarsa_agent.get_action(state)
        expected_action = expected_sarsa_agent.get_action(state)
        print(f"{state}\t{q_action}\t\t{sarsa_action}\t\t{expected_action}")

if __name__ == "__main__":
    test_agents()
```

## Summary

Reinforcement learning provides a framework for learning optimal behavior through interaction with an environment. Key concepts include:

1. **Agent-Environment Interaction**: The core loop of RL
2. **Exploration vs Exploitation**: Balancing trying new actions vs using known good actions
3. **Temporal Difference Learning**: Learning from single steps
4. **Monte Carlo Methods**: Learning from complete episodes
5. **Function Approximation**: Scaling to large state spaces

The Python implementation demonstrates various RL algorithms and exploration strategies, showing how they can be applied to solve sequential decision problems. 