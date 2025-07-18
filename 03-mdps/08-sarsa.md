# SARSA

## Introduction

SARSA (State-Action-Reward-State-Action) is an on-policy temporal difference learning algorithm for reinforcement learning. It learns the Q-function by updating estimates based on the actual next action taken by the current policy, making it an on-policy method.

## Key Concepts

### On-Policy Learning

SARSA is an **on-policy** algorithm, meaning:
- It learns about the policy being used for behavior
- The target policy and behavior policy are the same
- Updates are based on the actual next action taken

### Temporal Difference Learning

SARSA uses temporal difference (TD) learning:
- Learns from single steps of experience
- Updates estimates based on the difference between predicted and actual values
- Combines bootstrapping with sampling

## Mathematical Foundation

### SARSA Update Rule

The SARSA update rule is:

```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
```

Where:
- Q(s,a) is the current Q-value estimate
- α is the learning rate
- r is the immediate reward
- γ is the discount factor
- Q(s',a') is the Q-value of the next state-action pair
- The term [r + γQ(s',a') - Q(s,a)] is the TD error

### Expected SARSA

Expected SARSA uses the expected value of the next action:

```
Q(s,a) ← Q(s,a) + α[r + γE[Q(s',a')] - Q(s,a)]
```

Where E[Q(s',a')] is the expected Q-value under the current policy.

## Implementation

### Basic SARSA Agent

```python
import numpy as np
from typing import Dict, List, Tuple
import random

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
                # Take action and observe result
                next_state, reward, done, _ = env.step(action)
                
                # Choose next action (on-policy)
                next_action = self.get_action(next_state)
                
                # SARSA update
                self.update(state, action, reward, next_state, next_action)
                
                # Move to next state and action
                state = next_state
                action = next_action
                
                if done:
                    break
            
            if episode % 100 == 0:
                avg_q = np.mean([np.mean(list(q.values())) for q in self.Q.values()])
                print(f"Episode {episode}: Average Q-value = {avg_q:.3f}")
```

### Expected SARSA Agent

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
        
        Q(s,a) ← Q(s,a) + α[r + γE[Q(s',a')] - Q(s,a)]
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
                avg_q = np.mean([np.mean(list(q.values())) for q in self.Q.values()])
                print(f"Episode {episode}: Average Q-value = {avg_q:.3f}")
```

### N-Step SARSA

N-step SARSA uses returns over n steps instead of just one step.

```python
class NStepSARSAAgent:
    def __init__(self, states, actions, gamma=0.9, alpha=0.1, epsilon=0.1, n=3):
        """
        N-step SARSA agent.
        
        Args:
            states: List of possible states
            actions: List of possible actions
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate
            n: Number of steps for n-step return
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.n = n
        
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
    
    def compute_n_step_return(self, rewards, states, actions, t, T):
        """
        Compute n-step return.
        
        Args:
            rewards: List of rewards
            states: List of states
            actions: List of actions
            t: Current time step
            T: Final time step
        
        Returns:
            n-step return
        """
        if t + self.n >= T:
            # If we don't have n steps left, use all remaining steps
            n_steps = T - t
        else:
            n_steps = self.n
        
        # Compute n-step return
        G = 0
        for i in range(n_steps):
            G += (self.gamma ** i) * rewards[t + i]
        
        # Add bootstrapping term if we have n steps
        if t + self.n < T:
            G += (self.gamma ** self.n) * self.Q[states[t + self.n]][actions[t + self.n]]
        
        return G
    
    def update(self, state, action, n_step_return):
        """
        Update Q-value using n-step return.
        
        Q(s,a) ← Q(s,a) + α[G - Q(s,a)]
        """
        current_q = self.Q[state][action]
        self.Q[state][action] = current_q + self.alpha * (n_step_return - current_q)
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train the agent using n-step SARSA."""
        for episode in range(num_episodes):
            state = env.reset()
            action = self.get_action(state)
            
            # Store episode data
            states = [state]
            actions = [action]
            rewards = []
            
            for step in range(max_steps):
                # Take action and observe result
                next_state, reward, done, _ = env.step(action)
                
                # Store experience
                states.append(next_state)
                rewards.append(reward)
                
                if not done:
                    # Choose next action
                    next_action = self.get_action(next_state)
                    actions.append(next_action)
                else:
                    # Terminal state
                    actions.append(None)
                
                # Update Q-values for states that are n steps behind
                if step >= self.n - 1:
                    t = step - self.n + 1
                    n_step_return = self.compute_n_step_return(
                        rewards, states, actions, t, step + 1
                    )
                    self.update(states[t], actions[t], n_step_return)
                
                # Move to next state and action
                state = next_state
                if not done:
                    action = next_action
                
                if done:
                    # Update remaining states
                    for t in range(max(0, step - self.n + 2), step + 1):
                        n_step_return = self.compute_n_step_return(
                            rewards, states, actions, t, step + 1
                        )
                        self.update(states[t], actions[t], n_step_return)
                    break
            
            if episode % 100 == 0:
                avg_q = np.mean([np.mean(list(q.values())) for q in self.Q.values()])
                print(f"Episode {episode}: Average Q-value = {avg_q:.3f}")
```

### SARSA with Eligibility Traces

SARSA(λ) uses eligibility traces to combine TD and Monte Carlo methods.

```python
class SARSALambdaAgent:
    def __init__(self, states, actions, gamma=0.9, alpha=0.1, epsilon=0.1, lambda_val=0.9):
        """
        SARSA(λ) agent.
        
        Args:
            states: List of possible states
            actions: List of possible actions
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate
            lambda_val: Eligibility trace decay parameter
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.lambda_val = lambda_val
        
        # Initialize Q-values
        self.Q = {}
        for state in states:
            self.Q[state] = {}
            for action in actions:
                self.Q[state][action] = 0.0
        
        # Initialize eligibility traces
        self.eligibility_traces = {}
        for state in states:
            self.eligibility_traces[state] = {}
            for action in actions:
                self.eligibility_traces[state][action] = 0.0
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.Q[state].items(), key=lambda x: x[1])[0]
    
    def update_eligibility_traces(self, state, action):
        """Update eligibility traces."""
        # Increment eligibility trace for current state-action pair
        self.eligibility_traces[state][action] += 1
    
    def decay_eligibility_traces(self):
        """Decay all eligibility traces."""
        for state in self.states:
            for action in self.actions:
                self.eligibility_traces[state][action] *= self.gamma * self.lambda_val
    
    def update_q_values(self, td_error):
        """Update Q-values using eligibility traces."""
        for state in self.states:
            for action in self.actions:
                if self.eligibility_traces[state][action] > 0:
                    self.Q[state][action] += self.alpha * td_error * self.eligibility_traces[state][action]
    
    def reset_eligibility_traces(self):
        """Reset all eligibility traces to zero."""
        for state in self.states:
            for action in self.actions:
                self.eligibility_traces[state][action] = 0.0
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train the agent using SARSA(λ)."""
        for episode in range(num_episodes):
            state = env.reset()
            action = self.get_action(state)
            
            # Reset eligibility traces at the start of each episode
            self.reset_eligibility_traces()
            
            for step in range(max_steps):
                # Take action and observe result
                next_state, reward, done, _ = env.step(action)
                
                # Choose next action
                if not done:
                    next_action = self.get_action(next_state)
                else:
                    next_action = None
                
                # Update eligibility traces
                self.update_eligibility_traces(state, action)
                
                # Compute TD error
                current_q = self.Q[state][action]
                if next_action is not None:
                    next_q = self.Q[next_state][next_action]
                    target = reward + self.gamma * next_q
                else:
                    target = reward
                
                td_error = target - current_q
                
                # Update Q-values using eligibility traces
                self.update_q_values(td_error)
                
                # Decay eligibility traces
                self.decay_eligibility_traces()
                
                # Move to next state and action
                state = next_state
                action = next_action
                
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

# Test different SARSA variants
def test_sarsa_variants():
    """Test different SARSA variants."""
    env = GridWorldEnv(4, 4)
    states = [(x, y) for x in range(4) for y in range(4)]
    actions = ['up', 'down', 'left', 'right']
    
    print("Testing SARSA Variants:")
    print("=" * 50)
    
    # Basic SARSA
    print("\n1. Basic SARSA:")
    sarsa_agent = SARSAAgent(states, actions)
    sarsa_agent.train(env, num_episodes=500)
    
    # Expected SARSA
    print("\n2. Expected SARSA:")
    expected_sarsa_agent = ExpectedSARSAAgent(states, actions)
    expected_sarsa_agent.train(env, num_episodes=500)
    
    # N-step SARSA
    print("\n3. N-step SARSA (n=3):")
    n_step_sarsa_agent = NStepSARSAAgent(states, actions, n=3)
    n_step_sarsa_agent.train(env, num_episodes=500)
    
    # SARSA(λ)
    print("\n4. SARSA(λ):")
    sarsa_lambda_agent = SARSALambdaAgent(states, actions, lambda_val=0.9)
    sarsa_lambda_agent.train(env, num_episodes=500)
    
    # Compare final Q-values
    print("\nFinal Q-Values Comparison (State (0,0)):")
    print("Action\tSARSA\t\tExpected\tN-Step\t\tSARSA(λ)")
    print("-" * 60)
    
    for action in actions:
        sarsa_q = sarsa_agent.Q[(0, 0)][action]
        expected_q = expected_sarsa_agent.Q[(0, 0)][action]
        n_step_q = n_step_sarsa_agent.Q[(0, 0)][action]
        lambda_q = sarsa_lambda_agent.Q[(0, 0)][action]
        
        print(f"{action}\t{sarsa_q:.3f}\t\t{expected_q:.3f}\t\t{n_step_q:.3f}\t\t{lambda_q:.3f}")

if __name__ == "__main__":
    test_sarsa_variants()
```

## Advantages and Disadvantages

### Advantages

1. **On-policy**: Learns about the policy being used
2. **Conservative**: Tends to be more conservative than off-policy methods
3. **Stable**: Generally more stable than Q-learning
4. **Temporal difference**: Efficient learning from single steps

### Disadvantages

1. **On-policy limitation**: Cannot learn about optimal policy while following different behavior
2. **Exploration dependency**: Performance depends on exploration strategy
3. **Slower convergence**: May converge more slowly than off-policy methods

## Summary

SARSA is a fundamental on-policy temporal difference learning algorithm. Key points:

1. **On-policy learning**: Learns about the policy being used for behavior
2. **Temporal difference**: Updates based on single steps of experience
3. **Variants**: Expected SARSA, n-step SARSA, SARSA(λ)
4. **Conservative**: Generally more conservative than off-policy methods

The Python implementation demonstrates various SARSA variants and shows how they can be applied to solve sequential decision problems. 