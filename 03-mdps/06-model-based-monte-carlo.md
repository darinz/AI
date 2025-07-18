# Model-based Monte Carlo

## Introduction

Model-based Monte Carlo methods learn a model of the environment and use it for planning. These methods combine the benefits of model-free learning (learning from experience) with the advantages of having an explicit model (efficient planning and exploration).

## Key Concepts

### Model Learning

The agent learns estimates of:
- **Transition probabilities**: P̂(s'|s,a)
- **Reward function**: R̂(s,a,s')

### Planning

Once the model is learned, the agent can:
- **Simulate** future trajectories
- **Plan** optimal actions
- **Explore** efficiently

### Certainty Equivalence

The learned model is treated as if it were the true model of the environment.

## Mathematical Foundation

### Model Learning

For each state-action pair (s,a), we maintain:
- **Visit count**: N(s,a)
- **Transition counts**: N(s,a,s')
- **Reward sums**: R_sum(s,a,s')

### Model Estimates

```
P̂(s'|s,a) = N(s,a,s') / N(s,a)
R̂(s,a,s') = R_sum(s,a,s') / N(s,a,s')
```

### Planning with Learned Model

Once we have the model, we can use any planning algorithm:
- Value iteration
- Policy iteration
- Monte Carlo tree search

## Implementation

### Model Class

```python
import numpy as np
from typing import Dict, List, Tuple, Optional
import random

class LearnedModel:
    def __init__(self, states, actions):
        """
        Learned model of the environment.
        
        Args:
            states: List of possible states
            actions: List of possible actions
        """
        self.states = states
        self.actions = actions
        
        # Initialize counters
        self.visit_counts = {}
        self.transition_counts = {}
        self.reward_sums = {}
        
        for state in states:
            self.visit_counts[state] = {}
            self.transition_counts[state] = {}
            self.reward_sums[state] = {}
            
            for action in actions:
                self.visit_counts[state][action] = 0
                self.transition_counts[state][action] = {}
                self.reward_sums[state][action] = {}
    
    def update(self, state, action, reward, next_state):
        """
        Update model with new experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Update visit count
        self.visit_counts[state][action] += 1
        
        # Update transition count
        if next_state not in self.transition_counts[state][action]:
            self.transition_counts[state][action][next_state] = 0
        self.transition_counts[state][action][next_state] += 1
        
        # Update reward sum
        if next_state not in self.reward_sums[state][action]:
            self.reward_sums[state][action][next_state] = 0
        self.reward_sums[state][action][next_state] += reward
    
    def get_transition_prob(self, state, action, next_state):
        """Get estimated transition probability."""
        if self.visit_counts[state][action] == 0:
            return 0.0
        
        count = self.transition_counts[state][action].get(next_state, 0)
        return count / self.visit_counts[state][action]
    
    def get_expected_reward(self, state, action, next_state):
        """Get estimated expected reward."""
        count = self.transition_counts[state][action].get(next_state, 0)
        if count == 0:
            return 0.0
        
        return self.reward_sums[state][action][next_state] / count
    
    def get_next_states(self, state, action):
        """Get possible next states and their probabilities."""
        if self.visit_counts[state][action] == 0:
            return {}
        
        next_states = {}
        for next_state in self.transition_counts[state][action]:
            prob = self.get_transition_prob(state, action, next_state)
            if prob > 0:
                next_states[next_state] = prob
        
        return next_states
    
    def is_known(self, state, action):
        """Check if state-action pair has been visited."""
        return self.visit_counts[state][action] > 0
    
    def get_unknown_actions(self, state):
        """Get actions that haven't been tried in this state."""
        return [action for action in self.actions 
                if not self.is_known(state, action)]
```

### Model-Based Monte Carlo Agent

```python
class ModelBasedMonteCarloAgent:
    def __init__(self, states, actions, gamma=0.9, alpha=0.1, epsilon=0.1, 
                 planning_steps=5):
        """
        Model-based Monte Carlo agent.
        
        Args:
            states: List of possible states
            actions: List of possible actions
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate
            planning_steps: Number of planning steps per real step
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        
        # Initialize Q-values
        self.Q = {}
        for state in states:
            self.Q[state] = {}
            for action in actions:
                self.Q[state][action] = 0.0
        
        # Initialize learned model
        self.model = LearnedModel(states, actions)
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.Q[state].items(), key=lambda x: x[1])[0]
    
    def update_model(self, state, action, reward, next_state):
        """Update the learned model."""
        self.model.update(state, action, reward, next_state)
    
    def simulate_step(self, state, action):
        """Simulate one step using the learned model."""
        next_states = self.model.get_next_states(state, action)
        
        if not next_states:
            # If unknown, assume no transition
            return state, 0.0
        
        # Sample next state
        next_state = random.choices(
            list(next_states.keys()), 
            weights=list(next_states.values())
        )[0]
        
        # Get expected reward
        reward = self.model.get_expected_reward(state, action, next_state)
        
        return next_state, reward
    
    def planning_update(self, state, action):
        """Update Q-value using simulated experience."""
        next_state, reward = self.simulate_step(state, action)
        
        # Q-learning update
        current_q = self.Q[state][action]
        max_next_q = max(self.Q[next_state].values())
        target = reward + self.gamma * max_next_q
        self.Q[state][action] = current_q + self.alpha * (target - current_q)
    
    def plan(self, state):
        """Perform planning steps."""
        for _ in range(self.planning_steps):
            # Random state-action pair for planning
            random_state = random.choice(self.states)
            random_action = random.choice(self.actions)
            
            if self.model.is_known(random_state, random_action):
                self.planning_update(random_state, random_action)
    
    def train_step(self, state, action, reward, next_state):
        """Train the agent on one step."""
        # Update model
        self.update_model(state, action, reward, next_state)
        
        # Q-learning update
        current_q = self.Q[state][action]
        max_next_q = max(self.Q[next_state].values())
        target = reward + self.gamma * max_next_q
        self.Q[state][action] = current_q + self.alpha * (target - current_q)
        
        # Planning
        self.plan(state)
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train the agent."""
        for episode in range(num_episodes):
            state = env.reset()
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Train on real experience
                self.train_step(state, action, reward, next_state)
                
                state = next_state
                if done:
                    break
            
            if episode % 100 == 0:
                avg_q = np.mean([np.mean(list(q.values())) for q in self.Q.values()])
                print(f"Episode {episode}: Average Q-value = {avg_q:.3f}")
```

## Dyna-Q Algorithm

Dyna-Q is a specific model-based Monte Carlo algorithm that combines Q-learning with model-based planning.

```python
class DynaQAgent:
    def __init__(self, states, actions, gamma=0.9, alpha=0.1, epsilon=0.1, 
                 planning_steps=5):
        """
        Dyna-Q agent.
        
        Args:
            states: List of possible states
            actions: List of possible actions
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate
            planning_steps: Number of planning steps per real step
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        
        # Initialize Q-values
        self.Q = {}
        for state in states:
            self.Q[state] = {}
            for action in actions:
                self.Q[state][action] = 0.0
        
        # Initialize learned model
        self.model = LearnedModel(states, actions)
        
        # Keep track of visited state-action pairs for planning
        self.visited_pairs = []
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.Q[state].items(), key=lambda x: x[1])[0]
    
    def update_model(self, state, action, reward, next_state):
        """Update the learned model."""
        self.model.update(state, action, reward, next_state)
        
        # Add to visited pairs if not already there
        if (state, action) not in self.visited_pairs:
            self.visited_pairs.append((state, action))
    
    def planning_step(self):
        """Perform one planning step."""
        if not self.visited_pairs:
            return
        
        # Random state-action pair from visited pairs
        state, action = random.choice(self.visited_pairs)
        
        # Simulate experience
        next_state, reward = self.simulate_step(state, action)
        
        # Q-learning update
        current_q = self.Q[state][action]
        max_next_q = max(self.Q[next_state].values())
        target = reward + self.gamma * max_next_q
        self.Q[state][action] = current_q + self.alpha * (target - current_q)
    
    def simulate_step(self, state, action):
        """Simulate one step using the learned model."""
        next_states = self.model.get_next_states(state, action)
        
        if not next_states:
            return state, 0.0
        
        # Sample next state
        next_state = random.choices(
            list(next_states.keys()), 
            weights=list(next_states.values())
        )[0]
        
        # Get expected reward
        reward = self.model.get_expected_reward(state, action, next_state)
        
        return next_state, reward
    
    def train_step(self, state, action, reward, next_state):
        """Train the agent on one step."""
        # Update model
        self.update_model(state, action, reward, next_state)
        
        # Q-learning update
        current_q = self.Q[state][action]
        max_next_q = max(self.Q[next_state].values())
        target = reward + self.gamma * max_next_q
        self.Q[state][action] = current_q + self.alpha * (target - current_q)
        
        # Planning steps
        for _ in range(self.planning_steps):
            self.planning_step()
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train the agent."""
        for episode in range(num_episodes):
            state = env.reset()
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Train on real experience
                self.train_step(state, action, reward, next_state)
                
                state = next_state
                if done:
                    break
            
            if episode % 100 == 0:
                avg_q = np.mean([np.mean(list(q.values())) for q in self.Q.values()])
                print(f"Episode {episode}: Average Q-value = {avg_q:.3f}")
```

## Prioritized Sweeping

Prioritized sweeping is an advanced model-based method that focuses planning on states that are likely to have changed significantly.

```python
class PrioritizedSweepingAgent:
    def __init__(self, states, actions, gamma=0.9, alpha=0.1, epsilon=0.1, 
                 theta=0.01, max_queue_size=1000):
        """
        Prioritized sweeping agent.
        
        Args:
            states: List of possible states
            actions: List of possible actions
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Exploration rate
            theta: Priority threshold
            max_queue_size: Maximum size of priority queue
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.theta = theta
        self.max_queue_size = max_queue_size
        
        # Initialize Q-values
        self.Q = {}
        for state in states:
            self.Q[state] = {}
            for action in actions:
                self.Q[state][action] = 0.0
        
        # Initialize learned model
        self.model = LearnedModel(states, actions)
        
        # Priority queue for planning
        self.priority_queue = []
        
        # Predecessor mapping
        self.predecessors = {state: set() for state in states}
    
    def get_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.Q[state].items(), key=lambda x: x[1])[0]
    
    def update_predecessors(self, state, action, next_state):
        """Update predecessor mapping."""
        self.predecessors[next_state].add((state, action))
    
    def compute_priority(self, state, action):
        """Compute priority for a state-action pair."""
        next_states = self.model.get_next_states(state, action)
        
        if not next_states:
            return 0.0
        
        # Compute current Q-value
        current_q = self.Q[state][action]
        
        # Compute target Q-value
        target_q = 0
        for next_state, prob in next_states.items():
            reward = self.model.get_expected_reward(state, action, next_state)
            max_next_q = max(self.Q[next_state].values())
            target_q += prob * (reward + self.gamma * max_next_q)
        
        return abs(target_q - current_q)
    
    def add_to_queue(self, state, action, priority):
        """Add state-action pair to priority queue."""
        # Remove if already in queue
        self.priority_queue = [(p, s, a) for p, s, a in self.priority_queue 
                              if (s, a) != (state, action)]
        
        # Add with new priority
        self.priority_queue.append((priority, state, action))
        
        # Sort by priority (highest first)
        self.priority_queue.sort(reverse=True)
        
        # Limit queue size
        if len(self.priority_queue) > self.max_queue_size:
            self.priority_queue = self.priority_queue[:self.max_queue_size]
    
    def planning_step(self):
        """Perform one planning step."""
        if not self.priority_queue:
            return
        
        # Get highest priority state-action pair
        priority, state, action = self.priority_queue.pop(0)
        
        if priority < self.theta:
            return
        
        # Update Q-value
        next_states = self.model.get_next_states(state, action)
        
        if not next_states:
            return
        
        current_q = self.Q[state][action]
        target_q = 0
        
        for next_state, prob in next_states.items():
            reward = self.model.get_expected_reward(state, action, next_state)
            max_next_q = max(self.Q[next_state].values())
            target_q += prob * (reward + self.gamma * max_next_q)
        
        self.Q[state][action] = current_q + self.alpha * (target_q - current_q)
        
        # Update priorities of predecessors
        for pred_state, pred_action in self.predecessors[state]:
            if self.model.is_known(pred_state, pred_action):
                priority = self.compute_priority(pred_state, pred_action)
                if priority > self.theta:
                    self.add_to_queue(pred_state, pred_action, priority)
    
    def train_step(self, state, action, reward, next_state):
        """Train the agent on one step."""
        # Update model
        self.model.update(state, action, reward, next_state)
        self.update_predecessors(state, action, next_state)
        
        # Q-learning update
        current_q = self.Q[state][action]
        max_next_q = max(self.Q[next_state].values())
        target = reward + self.gamma * max_next_q
        self.Q[state][action] = current_q + self.alpha * (target - current_q)
        
        # Add to priority queue
        priority = self.compute_priority(state, action)
        if priority > self.theta:
            self.add_to_queue(state, action, priority)
        
        # Planning steps
        for _ in range(10):  # Fixed number of planning steps
            self.planning_step()
    
    def train(self, env, num_episodes=1000, max_steps=100):
        """Train the agent."""
        for episode in range(num_episodes):
            state = env.reset()
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Train on real experience
                self.train_step(state, action, reward, next_state)
                
                state = next_state
                if done:
                    break
            
            if episode % 100 == 0:
                avg_q = np.mean([np.mean(list(q.values())) for q in self.Q.values()])
                print(f"Episode {episode}: Average Q-value = {avg_q:.3f}")
```

## Example: Comparison with Model-Free Methods

```python
# Create a simple environment
class SimpleEnv:
    def __init__(self):
        self.states = [0, 1, 2, 3, 4]
        self.current_state = 0
        
    def reset(self):
        self.current_state = 0
        return self.current_state
    
    def step(self, action):
        # Simple deterministic environment
        if action == 'right':
            next_state = min(self.current_state + 1, 4)
        else:  # left
            next_state = max(self.current_state - 1, 0)
        
        # Reward: +1 for reaching state 4, -0.01 otherwise
        if next_state == 4:
            reward = 1.0
            done = True
        else:
            reward = -0.01
            done = False
        
        self.current_state = next_state
        return next_state, reward, done, {}

# Test different agents
def compare_agents():
    """Compare model-based and model-free agents."""
    env = SimpleEnv()
    states = env.states
    actions = ['left', 'right']
    
    print("Comparing Model-Based vs Model-Free Methods:")
    print("=" * 60)
    
    # Q-learning (model-free)
    print("\n1. Q-Learning (Model-Free):")
    q_agent = QLearningAgent(states, actions)
    q_agent.train(env, num_episodes=200)
    
    # Dyna-Q (model-based)
    print("\n2. Dyna-Q (Model-Based):")
    dyna_agent = DynaQAgent(states, actions, planning_steps=5)
    dyna_agent.train(env, num_episodes=200)
    
    # Prioritized sweeping (model-based)
    print("\n3. Prioritized Sweeping (Model-Based):")
    ps_agent = PrioritizedSweepingAgent(states, actions)
    ps_agent.train(env, num_episodes=200)
    
    # Compare final Q-values
    print("\nFinal Q-Values Comparison:")
    print("State\tQ-Learning\tDyna-Q\t\tPrioritized Sweeping")
    print("-" * 70)
    
    for state in states:
        q_left = q_agent.Q[state]['left']
        q_right = q_agent.Q[state]['right']
        q_best = max(q_left, q_right)
        
        dyna_left = dyna_agent.Q[state]['left']
        dyna_right = dyna_agent.Q[state]['right']
        dyna_best = max(dyna_left, dyna_right)
        
        ps_left = ps_agent.Q[state]['left']
        ps_right = ps_agent.Q[state]['right']
        ps_best = max(ps_left, ps_right)
        
        print(f"{state}\t{q_best:.3f}\t\t{dyna_best:.3f}\t\t{ps_best:.3f}")

if __name__ == "__main__":
    compare_agents()
```

## Advantages and Disadvantages

### Advantages

1. **Sample Efficiency**: Can learn from fewer real experiences
2. **Planning**: Can simulate future scenarios
3. **Exploration**: Can plan exploration strategies
4. **Transfer**: Model can be reused for similar environments

### Disadvantages

1. **Model Error**: Learned model may be inaccurate
2. **Computational Cost**: Planning adds computational overhead
3. **Memory**: Need to store transition and reward estimates
4. **Complexity**: More complex than model-free methods

## Summary

Model-based Monte Carlo methods combine the benefits of learning from experience with the advantages of having an explicit model. Key points:

1. **Model Learning**: Estimate transition probabilities and rewards
2. **Planning**: Use learned model for efficient planning
3. **Sample Efficiency**: Learn from fewer real experiences
4. **Algorithms**: Dyna-Q, prioritized sweeping, etc.

The Python implementation demonstrates various model-based methods and shows how they compare to model-free approaches in terms of learning efficiency and performance. 