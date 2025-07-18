# Overview of Markov Decision Processes (MDPs)

## Introduction

Markov Decision Processes (MDPs) are a mathematical framework for modeling sequential decision-making problems under uncertainty. They provide a formal way to describe how an agent should act in an environment where outcomes are partly random and partly under the agent's control.

## Core Components

### 1. States (S)
States represent the possible situations the agent can be in. The state space can be:
- **Finite**: A limited number of discrete states
- **Infinite**: Countable or uncountable states
- **Continuous**: Real-valued state variables

**Example**: In a grid world, states could be the agent's position (x, y coordinates).

### 2. Actions (A)
Actions are the choices available to the agent in each state. Actions can be:
- **State-dependent**: Different actions available in different states
- **State-independent**: Same actions available everywhere
- **Continuous**: Real-valued action parameters

**Example**: In a grid world, actions might be {up, down, left, right}.

### 3. Transitions (P)
The transition function P(s'|s,a) describes how the environment changes:
- **Deterministic**: P(s'|s,a) = 1 for one state s', 0 for others
- **Stochastic**: P(s'|s,a) gives probability distribution over next states
- **Markov Property**: Future depends only on current state and action

### 4. Rewards (R)
The reward function R(s,a,s') provides immediate feedback:
- **Positive**: Rewards for desired outcomes
- **Negative**: Penalties for undesired outcomes
- **Sparse**: Rewards only at specific states/actions
- **Dense**: Rewards for every transition

### 5. Policy (π)
A policy π(s) maps states to actions:
- **Deterministic**: π(s) = a (single action per state)
- **Stochastic**: π(a|s) (probability distribution over actions)
- **Optimal**: Maximizes expected cumulative reward

## Mathematical Formulation

### MDP Definition
An MDP is a 5-tuple (S, A, P, R, γ) where:
- S: State space
- A: Action space  
- P: Transition function P(s'|s,a)
- R: Reward function R(s,a,s')
- γ: Discount factor γ ∈ [0,1]

### Value Function
The value function V^π(s) represents the expected cumulative reward starting from state s and following policy π:

```
V^π(s) = E[Σ_{t=0}^∞ γ^t R(s_t, a_t, s_{t+1}) | s_0 = s, π]
```

### Q-Function
The Q-function Q^π(s,a) represents the expected cumulative reward starting from state s, taking action a, then following policy π:

```
Q^π(s,a) = E[Σ_{t=0}^∞ γ^t R(s_t, a_t, s_{t+1}) | s_0 = s, a_0 = a, π]
```

### Bellman Equations

**Value Function Bellman Equation:**
```
V^π(s) = Σ_{s'} P(s'|s,π(s)) [R(s,π(s),s') + γV^π(s')]
```

**Q-Function Bellman Equation:**
```
Q^π(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γQ^π(s',π(s'))]
```

**Optimal Value Function:**
```
V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV*(s')]
```

**Optimal Q-Function:**
```
Q*(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ max_{a'} Q*(s',a')]
```

## Python Implementation

### Basic MDP Class

```python
import numpy as np
from typing import Dict, List, Tuple, Optional

class MDP:
    def __init__(self, states: List, actions: List, transitions: Dict, 
                 rewards: Dict, gamma: float = 0.9):
        """
        Initialize an MDP.
        
        Args:
            states: List of possible states
            actions: List of possible actions
            transitions: Dict mapping (state, action) -> Dict[state, probability]
            rewards: Dict mapping (state, action, next_state) -> reward
            gamma: Discount factor
        """
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma
        
    def get_transition_prob(self, state, action, next_state):
        """Get transition probability P(next_state | state, action)."""
        return self.transitions.get((state, action), {}).get(next_state, 0.0)
    
    def get_reward(self, state, action, next_state):
        """Get reward R(state, action, next_state)."""
        return self.rewards.get((state, action, next_state), 0.0)
    
    def get_next_states(self, state, action):
        """Get possible next states and their probabilities."""
        return self.transitions.get((state, action), {})
```

### Simple Grid World Example

```python
class GridWorldMDP(MDP):
    def __init__(self, width: int, height: int, obstacles: List[Tuple] = None):
        """
        Create a grid world MDP.
        
        Args:
            width: Grid width
            height: Grid height
            obstacles: List of (x, y) obstacle positions
        """
        self.width = width
        self.height = height
        self.obstacles = obstacles or []
        
        # Create states (all grid positions)
        states = [(x, y) for x in range(width) for y in range(height) 
                 if (x, y) not in obstacles]
        
        # Actions: up, down, left, right
        actions = ['up', 'down', 'left', 'right']
        
        # Build transitions and rewards
        transitions = {}
        rewards = {}
        
        for state in states:
            for action in actions:
                next_state = self._get_next_state(state, action)
                if next_state in states:
                    transitions[(state, action)] = {next_state: 1.0}
                    
                    # Reward: -1 for each step, +10 for goal
                    reward = -1.0
                    if next_state == (width-1, height-1):  # Goal state
                        reward = 10.0
                    rewards[(state, action, next_state)] = reward
        
        super().__init__(states, actions, transitions, rewards)
    
    def _get_next_state(self, state, action):
        """Get next state given current state and action."""
        x, y = state
        
        if action == 'up':
            return (x, y + 1)
        elif action == 'down':
            return (x, y - 1)
        elif action == 'left':
            return (x - 1, y)
        elif action == 'right':
            return (x + 1, y)
        
        return state  # Stay in place for invalid actions
```

### Policy Class

```python
class Policy:
    def __init__(self, mdp: MDP):
        self.mdp = mdp
        self.policy = {}
        
    def get_action(self, state):
        """Get action for a given state (deterministic policy)."""
        return self.policy.get(state, self.mdp.actions[0])
    
    def set_action(self, state, action):
        """Set action for a given state."""
        self.policy[state] = action
    
    def get_stochastic_action(self, state):
        """Get action for a given state (stochastic policy)."""
        if state in self.policy:
            return np.random.choice(self.mdp.actions, p=self.policy[state])
        return np.random.choice(self.mdp.actions)
    
    def set_stochastic_policy(self, state, action_probs):
        """Set stochastic policy for a state."""
        self.policy[state] = action_probs
```

### Example Usage

```python
# Create a 3x3 grid world
mdp = GridWorldMDP(3, 3, obstacles=[(1, 1)])

# Create a random policy
policy = Policy(mdp)
for state in mdp.states:
    if state != (2, 2):  # Not goal state
        policy.set_action(state, np.random.choice(mdp.actions))

# Test the MDP
print("States:", mdp.states)
print("Actions:", mdp.actions)
print("Sample transition:", mdp.get_transition_prob((0, 0), 'right', (1, 0)))
print("Sample reward:", mdp.get_reward((0, 0), 'right', (1, 0)))

# Simulate a trajectory
def simulate_trajectory(mdp, policy, start_state, max_steps=100):
    """Simulate a trajectory following a policy."""
    trajectory = []
    current_state = start_state
    
    for step in range(max_steps):
        action = policy.get_action(current_state)
        next_states = mdp.get_next_states(current_state, action)
        
        if not next_states:
            break
            
        next_state = list(next_states.keys())[0]  # Deterministic
        reward = mdp.get_reward(current_state, action, next_state)
        
        trajectory.append((current_state, action, reward, next_state))
        current_state = next_state
        
        if current_state == (mdp.width-1, mdp.height-1):  # Goal reached
            break
    
    return trajectory

# Run simulation
trajectory = simulate_trajectory(mdp, policy, (0, 0))
print("\nTrajectory:")
for state, action, reward, next_state in trajectory:
    print(f"State: {state}, Action: {action}, Reward: {reward}, Next: {next_state}")
```

## Key Properties

### Markov Property
The future depends only on the current state and action, not the history:
```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
```

### Stationarity
The transition and reward functions don't change over time:
```
P(s'|s,a) and R(s,a,s') are time-invariant
```

### Discounting
Future rewards are discounted by factor γ:
- γ = 0: Only immediate rewards matter
- γ = 1: Future rewards matter equally
- γ ∈ (0,1): Balance between immediate and future rewards

## Applications

MDPs are used in various domains:

1. **Robotics**: Navigation, manipulation, autonomous vehicles
2. **Game Playing**: Chess, Go, video games
3. **Operations Research**: Inventory management, resource allocation
4. **Finance**: Portfolio optimization, trading strategies
5. **Healthcare**: Treatment planning, medical diagnosis
6. **Natural Language Processing**: Dialogue systems, text generation

## Summary

MDPs provide a powerful framework for modeling sequential decision problems. The key components (states, actions, transitions, rewards, policies) and mathematical formulations (value functions, Q-functions, Bellman equations) form the foundation for reinforcement learning algorithms.

The Python implementation demonstrates how to create and work with MDPs, including a practical grid world example that can be extended for more complex scenarios. 