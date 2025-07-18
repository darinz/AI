# Modeling Markov Decision Processes

## Introduction

Modeling is the process of defining the core components of an MDP: states, actions, transitions, and rewards. A well-designed model is crucial for effective learning and planning algorithms.

## Core Components

### State Space (S)

The state space defines all possible situations the agent can encounter.

#### Types of State Spaces

**1. Discrete State Space**
- Finite number of states
- Easy to represent and compute
- Example: Grid world positions, game board states

**2. Continuous State Space**
- Infinite number of states
- Requires function approximation
- Example: Robot joint angles, sensor readings

**3. Mixed State Space**
- Combination of discrete and continuous variables
- Common in real-world applications
- Example: Robot position (continuous) + battery level (discrete)

#### State Representation

```python
class StateSpace:
    def __init__(self, state_type="discrete"):
        self.state_type = state_type
        self.states = []
        
    def add_state(self, state):
        """Add a state to the state space."""
        if state not in self.states:
            self.states.append(state)
    
    def get_state_index(self, state):
        """Get index of a state in the state space."""
        return self.states.index(state)
    
    def get_state_by_index(self, index):
        """Get state by its index."""
        return self.states[index]
    
    def size(self):
        """Get the size of the state space."""
        return len(self.states)
```

### Action Space (A)

The action space defines all possible actions the agent can take.

#### Types of Action Spaces

**1. Discrete Action Space**
- Finite number of actions
- Example: {up, down, left, right}, {buy, sell, hold}

**2. Continuous Action Space**
- Infinite number of actions
- Example: Robot joint velocities, steering angles

**3. Hierarchical Action Space**
- Actions composed of sub-actions
- Example: Navigation with high-level (go to room) and low-level (move) actions

#### Action Representation

```python
class ActionSpace:
    def __init__(self, action_type="discrete"):
        self.action_type = action_type
        self.actions = []
        
    def add_action(self, action):
        """Add an action to the action space."""
        if action not in self.actions:
            self.actions.append(action)
    
    def get_action_index(self, action):
        """Get index of an action in the action space."""
        return self.actions.index(action)
    
    def get_action_by_index(self, index):
        """Get action by its index."""
        return self.actions[index]
    
    def size(self):
        """Get the size of the action space."""
        return len(self.actions)
```

### Transition Function P(s'|s,a)

The transition function describes how the environment changes based on the agent's actions.

#### Mathematical Definition

```
P(s'|s,a) = Probability of reaching state s' from state s after taking action a
```

#### Properties

1. **Probability Distribution**: Σ_s' P(s'|s,a) = 1 for all s, a
2. **Non-negative**: P(s'|s,a) ≥ 0 for all s, s', a
3. **Markov Property**: P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1}|s_t, a_t)

#### Types of Transitions

**1. Deterministic Transitions**
```
P(s'|s,a) = 1 for one state s', 0 for others
```

**2. Stochastic Transitions**
```
P(s'|s,a) gives probability distribution over next states
```

**3. Noisy Transitions**
- Intended action succeeds with probability p
- Other actions occur with probability (1-p)/|A|

#### Implementation

```python
class TransitionFunction:
    def __init__(self):
        self.transitions = {}
        
    def set_transition(self, state, action, next_state, probability):
        """Set transition probability P(next_state | state, action)."""
        if (state, action) not in self.transitions:
            self.transitions[(state, action)] = {}
        self.transitions[(state, action)][next_state] = probability
    
    def get_transition_prob(self, state, action, next_state):
        """Get transition probability P(next_state | state, action)."""
        return self.transitions.get((state, action), {}).get(next_state, 0.0)
    
    def get_next_states(self, state, action):
        """Get possible next states and their probabilities."""
        return self.transitions.get((state, action), {})
    
    def is_deterministic(self, state, action):
        """Check if transition is deterministic for given state and action."""
        next_states = self.get_next_states(state, action)
        return len(next_states) == 1 and list(next_states.values())[0] == 1.0
```

### Reward Function R(s,a,s')

The reward function provides immediate feedback for taking actions.

#### Mathematical Definition

```
R(s,a,s') = Expected reward for transitioning from s to s' via action a
```

#### Types of Rewards

**1. State-based Rewards**
```
R(s) = Reward for being in state s
```

**2. Action-based Rewards**
```
R(a) = Cost/reward for taking action a
```

**3. Transition-based Rewards**
```
R(s,a,s') = Reward for transitioning from s to s' via action a
```

**4. Sparse Rewards**
- Rewards only at specific states/actions
- Common in goal-oriented problems

**5. Dense Rewards**
- Rewards for every transition
- Provides more learning signal

#### Implementation

```python
class RewardFunction:
    def __init__(self):
        self.rewards = {}
        
    def set_reward(self, state, action, next_state, reward):
        """Set reward R(state, action, next_state)."""
        self.rewards[(state, action, next_state)] = reward
    
    def get_reward(self, state, action, next_state):
        """Get reward R(state, action, next_state)."""
        return self.rewards.get((state, action, next_state), 0.0)
    
    def set_state_reward(self, state, reward):
        """Set reward for being in a state (for all actions)."""
        for action in ['up', 'down', 'left', 'right']:  # Assuming standard actions
            for next_state in [(state[0]+1, state[1]), (state[0]-1, state[1]), 
                              (state[0], state[1]+1), (state[0], state[1]-1)]:
                self.set_reward(state, action, next_state, reward)
```

## Complete MDP Model

### Enhanced MDP Class

```python
import numpy as np
from typing import Dict, List, Tuple, Optional

class MDPModel:
    def __init__(self, gamma: float = 0.9):
        """
        Initialize an MDP model.
        
        Args:
            gamma: Discount factor
        """
        self.gamma = gamma
        self.state_space = StateSpace()
        self.action_space = ActionSpace()
        self.transitions = TransitionFunction()
        self.rewards = RewardFunction()
        
    def add_state(self, state):
        """Add a state to the model."""
        self.state_space.add_state(state)
    
    def add_action(self, action):
        """Add an action to the model."""
        self.action_space.add_action(action)
    
    def set_transition(self, state, action, next_state, probability):
        """Set transition probability."""
        self.transitions.set_transition(state, action, next_state, probability)
    
    def set_reward(self, state, action, next_state, reward):
        """Set reward."""
        self.rewards.set_reward(state, action, next_state, reward)
    
    def get_transition_prob(self, state, action, next_state):
        """Get transition probability."""
        return self.transitions.get_transition_prob(state, action, next_state)
    
    def get_reward(self, state, action, next_state):
        """Get reward."""
        return self.rewards.get_reward(state, action, next_state)
    
    def get_next_states(self, state, action):
        """Get possible next states."""
        return self.transitions.get_next_states(state, action)
    
    def validate_model(self):
        """Validate the MDP model."""
        errors = []
        
        # Check transition probabilities sum to 1
        for state in self.state_space.states:
            for action in self.action_space.actions:
                next_states = self.get_next_states(state, action)
                prob_sum = sum(next_states.values())
                if abs(prob_sum - 1.0) > 1e-6 and prob_sum > 0:
                    errors.append(f"Transition probabilities don't sum to 1 for {state}, {action}: {prob_sum}")
        
        return errors
```

## Example Models

### 1. Simple Grid World

```python
def create_simple_grid_world(width: int, height: int, obstacles: List[Tuple] = None):
    """Create a simple grid world MDP."""
    mdp = MDPModel()
    obstacles = obstacles or []
    
    # Add states
    for x in range(width):
        for y in range(height):
            if (x, y) not in obstacles:
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
            
            # Check if next state is valid
            if next_state in mdp.state_space.states:
                mdp.set_transition(state, action, next_state, 1.0)
            else:
                # Stay in place if hitting boundary
                mdp.set_transition(state, action, state, 1.0)
    
    # Add rewards
    goal_state = (width - 1, height - 1)
    for state in mdp.state_space.states:
        for action in actions:
            next_states = mdp.get_next_states(state, action)
            for next_state in next_states:
                if next_state == goal_state:
                    mdp.set_reward(state, action, next_state, 10.0)
                else:
                    mdp.set_reward(state, action, next_state, -1.0)
    
    return mdp
```

### 2. Stochastic Grid World

```python
def create_stochastic_grid_world(width: int, height: int, noise: float = 0.1):
    """Create a stochastic grid world with action noise."""
    mdp = MDPModel()
    
    # Add states and actions
    for x in range(width):
        for y in range(height):
            mdp.add_state((x, y))
    
    actions = ['up', 'down', 'left', 'right']
    for action in actions:
        mdp.add_action(action)
    
    # Add stochastic transitions
    for state in mdp.state_space.states:
        x, y = state
        for intended_action in actions:
            # Intended next state
            if intended_action == 'up':
                intended_next = (x, y + 1)
            elif intended_action == 'down':
                intended_next = (x, y - 1)
            elif intended_action == 'left':
                intended_next = (x - 1, y)
            elif intended_action == 'right':
                intended_next = (x + 1, y)
            
            # Check if intended next state is valid
            if intended_next not in mdp.state_space.states:
                intended_next = state
            
            # Set transition probabilities
            mdp.set_transition(state, intended_action, intended_next, 1.0 - noise)
            
            # Add noise transitions
            for noise_action in actions:
                if noise_action != intended_action:
                    if noise_action == 'up':
                        noise_next = (x, y + 1)
                    elif noise_action == 'down':
                        noise_next = (x, y - 1)
                    elif noise_action == 'left':
                        noise_next = (x - 1, y)
                    elif noise_action == 'right':
                        noise_next = (x + 1, y)
                    
                    if noise_next not in mdp.state_space.states:
                        noise_next = state
                    
                    current_prob = mdp.get_transition_prob(state, intended_action, noise_next)
                    mdp.set_transition(state, intended_action, noise_next, 
                                     current_prob + noise / (len(actions) - 1))
    
    # Add rewards
    goal_state = (width - 1, height - 1)
    for state in mdp.state_space.states:
        for action in actions:
            next_states = mdp.get_next_states(state, action)
            for next_state in next_states:
                if next_state == goal_state:
                    mdp.set_reward(state, action, next_state, 10.0)
                else:
                    mdp.set_reward(state, action, next_state, -1.0)
    
    return mdp
```

### 3. Inventory Management Model

```python
def create_inventory_mdp(max_inventory: int, max_order: int, demand_probs: List[float]):
    """Create an inventory management MDP."""
    mdp = MDPModel()
    
    # States: current inventory level
    for inventory in range(max_inventory + 1):
        mdp.add_state(inventory)
    
    # Actions: order quantity
    for order in range(max_order + 1):
        mdp.add_action(order)
    
    # Transitions: inventory + order - demand
    for inventory in range(max_inventory + 1):
        for order in range(max_order + 1):
            for demand, prob in enumerate(demand_probs):
                if prob > 0:
                    new_inventory = max(0, inventory + order - demand)
                    if new_inventory <= max_inventory:
                        current_prob = mdp.get_transition_prob(inventory, order, new_inventory)
                        mdp.set_transition(inventory, order, new_inventory, current_prob + prob)
    
    # Rewards: -holding_cost * inventory - order_cost * order
    holding_cost = 1.0
    order_cost = 2.0
    for inventory in range(max_inventory + 1):
        for order in range(max_order + 1):
            next_states = mdp.get_next_states(inventory, order)
            for next_inventory in next_states:
                reward = -holding_cost * next_inventory - order_cost * order
                mdp.set_reward(inventory, order, next_inventory, reward)
    
    return mdp
```

## Model Validation

### Validation Functions

```python
def validate_transition_probabilities(mdp):
    """Validate that transition probabilities sum to 1."""
    errors = []
    for state in mdp.state_space.states:
        for action in mdp.action_space.actions:
            next_states = mdp.get_next_states(state, action)
            prob_sum = sum(next_states.values())
            if abs(prob_sum - 1.0) > 1e-6 and prob_sum > 0:
                errors.append(f"State {state}, Action {action}: probabilities sum to {prob_sum}")
    return errors

def check_reachability(mdp):
    """Check if all states are reachable."""
    reachable = set()
    queue = [mdp.state_space.states[0]]  # Start from first state
    
    while queue:
        state = queue.pop(0)
        if state not in reachable:
            reachable.add(state)
            for action in mdp.action_space.actions:
                next_states = mdp.get_next_states(state, action)
                for next_state in next_states:
                    if next_state not in reachable:
                        queue.append(next_state)
    
    unreachable = set(mdp.state_space.states) - reachable
    return list(unreachable)

def analyze_model(mdp):
    """Analyze the MDP model."""
    print(f"States: {mdp.state_space.size()}")
    print(f"Actions: {mdp.action_space.size()}")
    print(f"Discount factor: {mdp.gamma}")
    
    # Check for deterministic vs stochastic transitions
    deterministic_count = 0
    stochastic_count = 0
    
    for state in mdp.state_space.states:
        for action in mdp.action_space.actions:
            next_states = mdp.get_next_states(state, action)
            if len(next_states) == 1:
                deterministic_count += 1
            else:
                stochastic_count += 1
    
    print(f"Deterministic transitions: {deterministic_count}")
    print(f"Stochastic transitions: {stochastic_count}")
    
    # Validate model
    errors = mdp.validate_model()
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Model is valid!")
```

## Example Usage

```python
# Create and test a simple grid world
mdp = create_simple_grid_world(3, 3, obstacles=[(1, 1)])

# Analyze the model
analyze_model(mdp)

# Test transitions
print("\nSample transitions:")
state = (0, 0)
action = 'right'
next_states = mdp.get_next_states(state, action)
print(f"From {state}, action {action}: {next_states}")

# Test rewards
print("\nSample rewards:")
reward = mdp.get_reward((0, 0), 'right', (1, 0))
print(f"Reward for (0,0) -> right -> (1,0): {reward}")

# Create stochastic grid world
stochastic_mdp = create_stochastic_grid_world(3, 3, noise=0.1)
print("\nStochastic grid world analysis:")
analyze_model(stochastic_mdp)
```

## Summary

MDP modeling involves carefully defining the state space, action space, transition function, and reward function. The model should accurately represent the problem while being computationally tractable. Key considerations include:

1. **State representation**: Choose appropriate state variables
2. **Action space**: Define meaningful actions for the agent
3. **Transitions**: Model environment dynamics accurately
4. **Rewards**: Provide appropriate learning signal
5. **Validation**: Ensure model consistency and correctness

The Python implementation provides a flexible framework for creating and validating MDP models, with examples ranging from simple grid worlds to more complex inventory management problems. 