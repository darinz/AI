# TD Learning: Temporal Difference Learning in Games

## Introduction

Temporal Difference (TD) learning is a reinforcement learning method that updates value estimates based on the difference between predicted and actual outcomes. It's particularly useful for game evaluation and has been instrumental in breakthroughs like AlphaGo and AlphaZero.

## What is TD Learning?

TD learning is a method that:
- **Updates value estimates incrementally** based on observed differences
- **Uses bootstrapping** for efficient learning
- **Balances exploration and exploitation**
- **Handles delayed rewards** and long-term planning

**Key Insight:**
Instead of waiting for the final outcome, TD learning updates estimates based on the difference between current prediction and the next prediction.

## Mathematical Foundation

### TD Error

The TD error is the difference between the current value estimate and the target value:

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

Where:
- $\delta_t$ is the TD error at time $t$
- $R_{t+1}$ is the immediate reward
- $\gamma$ is the discount factor
- $V(S_{t+1})$ is the value estimate of the next state
- $V(S_t)$ is the value estimate of the current state

### TD Update Rule

The value function is updated using the TD error:

$$V(S_t) \leftarrow V(S_t) + \alpha \delta_t$$

Where $\alpha$ is the learning rate.

### TD(λ) Learning

TD(λ) combines TD learning with eligibility traces:

$$V(s) \leftarrow V(s) + \alpha \delta_t e_t(s)$$

Where $e_t(s)$ is the eligibility trace for state $s$ at time $t$.

## Algorithm Structure

### Core Principles

1. **Bootstrapping**: Use current estimates to update other estimates
2. **Temporal differences**: Learn from differences between consecutive predictions
3. **Eligibility traces**: Track recently visited states for credit assignment
4. **Exploration vs exploitation**: Balance between trying new actions and using learned values

### Algorithm Steps

1. **Initialize** value function $V(s)$ for all states
2. **Observe** current state $S_t$
3. **Choose action** using exploration policy
4. **Execute action** and observe reward $R_{t+1}$ and next state $S_{t+1}$
5. **Compute TD error**: $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$
6. **Update value function**: $V(S_t) \leftarrow V(S_t) + \alpha \delta_t$
7. **Repeat** until convergence

## Python Implementation

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
import random
import time
from collections import defaultdict

class GameState:
    """Base game state class."""
    
    def __init__(self, state_id: str, features: Dict[str, float], is_terminal: bool = False):
        self.state_id = state_id
        self.features = features
        self.is_terminal = is_terminal
        self.utility = 0.0 if not is_terminal else self._compute_terminal_utility()
    
    def _compute_terminal_utility(self) -> float:
        """Compute utility for terminal states."""
        # Simplified terminal utility computation
        return self.features.get('terminal_value', 0.0)
    
    def __hash__(self):
        return hash(self.state_id)
    
    def __eq__(self, other):
        return self.state_id == other.state_id

class ValueFunction:
    """Value function representation."""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.values = defaultdict(float)  # State -> value mapping
        self.eligibility_traces = defaultdict(float)  # For TD(λ)
    
    def get_value(self, state: GameState) -> float:
        """Get value estimate for a state."""
        return self.values[state.state_id]
    
    def set_value(self, state: GameState, value: float):
        """Set value estimate for a state."""
        self.values[state.state_id] = value
    
    def update_value(self, state: GameState, td_error: float):
        """Update value using TD error."""
        current_value = self.get_value(state)
        new_value = current_value + self.learning_rate * td_error
        self.set_value(state, new_value)
    
    def update_with_eligibility(self, state: GameState, td_error: float, lambda_param: float = 0.9):
        """Update value using eligibility traces (TD(λ))."""
        # Update eligibility trace
        self.eligibility_traces[state.state_id] += 1.0
        
        # Update all states with non-zero eligibility
        for state_id, eligibility in self.eligibility_traces.items():
            if eligibility > 0:
                self.values[state_id] += self.learning_rate * td_error * eligibility
                self.eligibility_traces[state_id] *= self.discount_factor * lambda_param

class TDAgent:
    """TD learning agent."""
    
    def __init__(self, value_function: ValueFunction, exploration_rate: float = 0.1):
        self.value_function = value_function
        self.exploration_rate = exploration_rate
        self.episode_history = []
    
    def choose_action(self, state: GameState, available_actions: List[str]) -> str:
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.exploration_rate:
            # Exploration: choose random action
            return random.choice(available_actions)
        else:
            # Exploitation: choose action with highest expected value
            return self._choose_best_action(state, available_actions)
    
    def _choose_best_action(self, state: GameState, available_actions: List[str]) -> str:
        """Choose the action with highest expected value."""
        best_action = None
        best_value = float('-inf')
        
        for action in available_actions:
            # Simulate action outcome (in practice, this would use the environment)
            next_state = self._simulate_action(state, action)
            expected_value = self.value_function.get_value(next_state)
            
            if expected_value > best_value:
                best_value = expected_value
                best_action = action
        
        return best_action or random.choice(available_actions)
    
    def _simulate_action(self, state: GameState, action: str) -> GameState:
        """Simulate the outcome of an action."""
        # Simplified simulation - in practice, this would use the actual environment
        new_features = state.features.copy()
        
        if action == 'good_move':
            new_features['position_value'] = new_features.get('position_value', 0.0) + 0.1
        elif action == 'bad_move':
            new_features['position_value'] = new_features.get('position_value', 0.0) - 0.1
        elif action == 'neutral_move':
            pass  # No change
        
        return GameState(
            state_id=f"{state.state_id}_{action}",
            features=new_features
        )
    
    def learn_from_episode(self, episode: List[Tuple[GameState, str, float, GameState]]):
        """Learn from a complete episode using TD learning."""
        for t, (state, action, reward, next_state) in enumerate(episode):
            # Compute TD error
            current_value = self.value_function.get_value(state)
            next_value = self.value_function.get_value(next_state)
            
            td_error = reward + self.value_function.discount_factor * next_value - current_value
            
            # Update value function
            self.value_function.update_value(state, td_error)
    
    def learn_from_episode_td_lambda(self, episode: List[Tuple[GameState, str, float, GameState]], lambda_param: float = 0.9):
        """Learn from episode using TD(λ)."""
        # Reset eligibility traces
        self.value_function.eligibility_traces.clear()
        
        for t, (state, action, reward, next_state) in enumerate(episode):
            # Compute TD error
            current_value = self.value_function.get_value(state)
            next_value = self.value_function.get_value(next_state)
            
            td_error = reward + self.value_function.discount_factor * next_value - current_value
            
            # Update with eligibility traces
            self.value_function.update_with_eligibility(state, td_error, lambda_param)

class GameEnvironment:
    """Game environment for TD learning."""
    
    def __init__(self, initial_state: GameState):
        self.initial_state = initial_state
        self.current_state = initial_state
        self.step_count = 0
        self.max_steps = 50
    
    def reset(self) -> GameState:
        """Reset the environment."""
        self.current_state = self.initial_state
        self.step_count = 0
        return self.current_state
    
    def get_available_actions(self) -> List[str]:
        """Get available actions."""
        return ['good_move', 'bad_move', 'neutral_move']
    
    def step(self, action: str) -> Tuple[GameState, float, bool]:
        """Take a step in the environment."""
        self.step_count += 1
        
        # Compute reward based on action
        reward = self._compute_reward(action)
        
        # Update state
        new_features = self.current_state.features.copy()
        
        if action == 'good_move':
            new_features['position_value'] = new_features.get('position_value', 0.0) + 0.1
        elif action == 'bad_move':
            new_features['position_value'] = new_features.get('position_value', 0.0) - 0.1
        
        # Check if episode should end
        done = self.step_count >= self.max_steps or new_features.get('position_value', 0.0) >= 1.0
        
        if done:
            # Terminal state
            new_features['terminal_value'] = new_features.get('position_value', 0.0)
            next_state = GameState(f"terminal_{self.step_count}", new_features, is_terminal=True)
        else:
            next_state = GameState(f"step_{self.step_count}", new_features)
        
        self.current_state = next_state
        return next_state, reward, done
    
    def _compute_reward(self, action: str) -> float:
        """Compute reward for an action."""
        if action == 'good_move':
            return 0.1
        elif action == 'bad_move':
            return -0.1
        else:
            return 0.0

class SelfPlayTrainer:
    """Trainer for self-play learning."""
    
    def __init__(self, agent: TDAgent, environment: GameEnvironment):
        self.agent = agent
        self.environment = environment
        self.training_history = []
    
    def train_episode(self) -> Dict[str, float]:
        """Train for one episode."""
        state = self.environment.reset()
        episode = []
        total_reward = 0.0
        
        while True:
            # Choose action
            available_actions = self.environment.get_available_actions()
            action = self.agent.choose_action(state, available_actions)
            
            # Take action
            next_state, reward, done = self.environment.step(action)
            
            # Record experience
            episode.append((state, action, reward, next_state))
            total_reward += reward
            
            if done:
                break
            
            state = next_state
        
        # Learn from episode
        self.agent.learn_from_episode(episode)
        
        # Record training statistics
        stats = {
            'episode_length': len(episode),
            'total_reward': total_reward,
            'final_position_value': next_state.features.get('position_value', 0.0)
        }
        self.training_history.append(stats)
        
        return stats
    
    def train_multiple_episodes(self, num_episodes: int) -> List[Dict[str, float]]:
        """Train for multiple episodes."""
        results = []
        
        for episode in range(num_episodes):
            stats = self.train_episode()
            results.append(stats)
            
            if episode % 100 == 0:
                avg_reward = np.mean([r['total_reward'] for r in results[-100:]])
                print(f"Episode {episode}: Average reward = {avg_reward:.3f}")
        
        return results

class NeuralNetworkValueFunction:
    """Neural network-based value function."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, learning_rate: float = 0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Simple neural network weights (in practice, use a proper framework like PyTorch)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros(1)
    
    def get_value(self, state: GameState) -> float:
        """Get value estimate using neural network."""
        features = self._extract_features(state)
        return self._forward_pass(features)[0, 0]
    
    def update_value(self, state: GameState, td_error: float):
        """Update neural network using TD error."""
        features = self._extract_features(state)
        
        # Simplified backpropagation (in practice, use automatic differentiation)
        # This is a placeholder for the actual neural network update
        pass
    
    def _extract_features(self, state: GameState) -> np.ndarray:
        """Extract features from state for neural network input."""
        # Convert state features to fixed-size vector
        features = np.zeros(self.input_size)
        
        # Map state features to feature vector
        feature_mapping = {
            'position_value': 0,
            'normal_behavior': 1,
            'exploitable': 2,
            'unusual_pattern': 3
        }
        
        for feature_name, value in state.features.items():
            if feature_name in feature_mapping:
                idx = feature_mapping[feature_name]
                features[idx] = value
        
        return features.reshape(1, -1)
    
    def _forward_pass(self, features: np.ndarray) -> np.ndarray:
        """Forward pass through neural network."""
        # Hidden layer
        hidden = np.tanh(features @ self.W1 + self.b1)
        
        # Output layer
        output = hidden @ self.W2 + self.b2
        
        return output

# Example: Chess-like game with TD learning
class ChessLikeState(GameState):
    """State for a simplified chess-like game."""
    
    def __init__(self, board: np.ndarray, player_turn: int, move_count: int = 0):
        self.board = board.copy()
        self.player_turn = player_turn  # 1 for white, -1 for black
        self.move_count = move_count
        
        # Extract features for value function
        features = self._extract_features()
        is_terminal = self._is_terminal()
        
        super().__init__(f"chess_{move_count}", features, is_terminal)
    
    def _extract_features(self) -> Dict[str, float]:
        """Extract features for value function."""
        features = {}
        
        # Material balance
        white_material = np.sum(self.board > 0)
        black_material = np.sum(self.board < 0)
        features['material_balance'] = (white_material - black_material) / 10.0
        
        # Position control
        center_control = np.sum(self.board[1:3, 1:3])
        features['center_control'] = center_control / 4.0
        
        # Mobility (simplified)
        features['mobility'] = np.sum(self.board != 0) / 16.0
        
        # Game phase
        features['game_phase'] = min(self.move_count / 20.0, 1.0)
        
        return features
    
    def _is_terminal(self) -> bool:
        """Check if the game is terminal."""
        # Simplified: game ends after certain number of moves
        return self.move_count >= 30
    
    def get_legal_actions(self) -> List[Tuple[int, int, int, int]]:
        """Get legal moves (from_row, from_col, to_row, to_col)."""
        if self.is_terminal:
            return []
        
        legal_moves = []
        for from_row in range(4):
            for from_col in range(4):
                if self.board[from_row, from_col] * self.player_turn > 0:
                    for to_row in range(4):
                        for to_col in range(4):
                            if self.board[to_row, to_col] * self.player_turn <= 0:
                                legal_moves.append((from_row, from_col, to_row, to_col))
        
        return legal_moves

class ChessLikeEnvironment:
    """Environment for chess-like game."""
    
    def __init__(self):
        self.board_size = 4
        self.reset()
    
    def reset(self) -> ChessLikeState:
        """Reset the game."""
        # Initialize board
        board = np.zeros((4, 4), dtype=int)
        board[0, :] = 1   # White pieces
        board[3, :] = -1  # Black pieces
        
        self.current_state = ChessLikeState(board, 1, 0)
        return self.current_state
    
    def step(self, action: Tuple[int, int, int, int]) -> Tuple[ChessLikeState, float, bool]:
        """Execute a move."""
        from_row, from_col, to_row, to_col = action
        
        # Make the move
        new_board = self.current_state.board.copy()
        piece = new_board[from_row, from_col]
        new_board[from_row, from_col] = 0
        new_board[to_row, to_col] = piece
        
        # Switch turns
        next_player = -self.current_state.player_turn
        next_move_count = self.current_state.move_count + 1
        
        # Create new state
        next_state = ChessLikeState(new_board, next_player, next_move_count)
        
        # Compute reward
        reward = self._compute_reward(self.current_state, next_state)
        
        self.current_state = next_state
        return next_state, reward, next_state.is_terminal
    
    def _compute_reward(self, old_state: ChessLikeState, new_state: ChessLikeState) -> float:
        """Compute reward for the move."""
        # Reward based on material gain/loss
        old_material = old_state.features['material_balance']
        new_material = new_state.features['material_balance']
        
        material_change = new_material - old_material
        return material_change * 10.0  # Scale up the reward

# Example usage
def demonstrate_td_learning():
    """Demonstrate TD learning in games."""
    
    print("=== TD Learning Demonstration ===")
    
    # Create value function and agent
    value_function = ValueFunction(learning_rate=0.1, discount_factor=0.9)
    agent = TDAgent(value_function, exploration_rate=0.1)
    
    # Create environment
    initial_state = GameState("start", {'position_value': 0.0})
    environment = GameEnvironment(initial_state)
    
    # Create trainer
    trainer = SelfPlayTrainer(agent, environment)
    
    # Train for multiple episodes
    print("\nTraining TD agent...")
    results = trainer.train_multiple_episodes(1000)
    
    # Analyze results
    print("\nTraining Results:")
    print(f"Average episode length: {np.mean([r['episode_length'] for r in results]):.1f}")
    print(f"Average total reward: {np.mean([r['total_reward'] for r in results]):.3f}")
    print(f"Final average position value: {np.mean([r['final_position_value'] for r in results]):.3f}")
    
    # Show learning progress
    print("\nLearning Progress:")
    window_size = 100
    for i in range(0, len(results), window_size):
        window = results[i:i+window_size]
        avg_reward = np.mean([r['total_reward'] for r in window])
        print(f"Episodes {i}-{i+len(window)-1}: Average reward = {avg_reward:.3f}")

def compare_td_methods():
    """Compare different TD learning methods."""
    
    print("\n=== TD Learning Methods Comparison ===")
    
    # Create environments
    initial_state = GameState("start", {'position_value': 0.0})
    environment = GameEnvironment(initial_state)
    
    # Test different methods
    methods = [
        ("TD(0)", 0.0),
        ("TD(0.5)", 0.5),
        ("TD(0.9)", 0.9),
        ("TD(1.0)", 1.0)
    ]
    
    results = {}
    
    for method_name, lambda_param in methods:
        print(f"\nTesting {method_name}...")
        
        value_function = ValueFunction(learning_rate=0.1, discount_factor=0.9)
        agent = TDAgent(value_function, exploration_rate=0.1)
        trainer = SelfPlayTrainer(agent, environment)
        
        # Train for fewer episodes for comparison
        episode_results = trainer.train_multiple_episodes(500)
        
        # Compute final performance
        final_rewards = [r['total_reward'] for r in episode_results[-100:]]
        results[method_name] = np.mean(final_rewards)
    
    print("\nComparison Results:")
    for method_name, avg_reward in results.items():
        print(f"{method_name}: Average reward = {avg_reward:.3f}")

def demonstrate_chess_td():
    """Demonstrate TD learning in chess-like game."""
    
    print("\n=== Chess-like Game TD Learning ===")
    
    # Create chess-like environment
    chess_env = ChessLikeEnvironment()
    
    # Create value function and agent
    value_function = ValueFunction(learning_rate=0.1, discount_factor=0.95)
    agent = TDAgent(value_function, exploration_rate=0.2)
    
    # Train agent
    print("Training chess agent...")
    
    for episode in range(100):
        state = chess_env.reset()
        episode_data = []
        total_reward = 0.0
        
        while not state.is_terminal:
            # Get legal actions
            legal_actions = state.get_legal_actions()
            if not legal_actions:
                break
            
            # Choose action (simplified - just pick random legal move)
            action = random.choice(legal_actions)
            
            # Execute action
            next_state, reward, done = chess_env.step(action)
            
            # Record experience
            episode_data.append((state, action, reward, next_state))
            total_reward += reward
            
            state = next_state
        
        # Learn from episode
        agent.learn_from_episode(episode_data)
        
        if episode % 20 == 0:
            print(f"Episode {episode}: Total reward = {total_reward:.2f}")

if __name__ == "__main__":
    demonstrate_td_learning()
    compare_td_methods()
    demonstrate_chess_td()
```

## Mathematical Analysis

### 1. Convergence Properties

**TD(0) convergence:**
- Converges to optimal value function under certain conditions
- Requires sufficient exploration and appropriate learning rate
- May converge to suboptimal solutions in some cases

**TD(λ) convergence:**
- Generally converges faster than TD(0)
- λ = 1 is equivalent to Monte Carlo learning
- λ = 0 is equivalent to TD(0)

### 2. Learning Rate Selection

**Fixed learning rate:**
$$\alpha_t = \alpha$$

**Decaying learning rate:**
$$\alpha_t = \frac{\alpha_0}{1 + t}$$

**Adaptive learning rate:**
$$\alpha_t = \frac{\alpha_0}{\sqrt{\sum_{i=1}^{t} \delta_i^2}}$$

### 3. Exploration vs Exploitation

**Epsilon-greedy:**
$$\pi(s) = \begin{cases}
\text{random action} & \text{with probability } \epsilon \\
\arg\max_a Q(s, a) & \text{with probability } 1 - \epsilon
\end{cases}$$

**Softmax:**
$$\pi(a|s) = \frac{e^{Q(s, a)/\tau}}{\sum_b e^{Q(s, b)/\tau}}$$

## Applications in Games

### 1. Self-Play Learning

**AlphaGo/AlphaZero:**
- TD learning for position evaluation
- Self-play training data generation
- Neural network value function training

**Chess engines:**
- Position evaluation improvement
- Endgame tablebase generation
- Opening book optimization

### 2. Game Evaluation

**Position assessment:**
- Learn to evaluate non-terminal positions
- Improve search algorithm performance
- Handle complex strategic situations

**Move prediction:**
- Predict opponent's likely moves
- Improve move ordering in search
- Enhance tactical analysis

### 3. Multi-Agent Learning

**Opponent modeling:**
- Learn opponent strategies
- Adapt to different playing styles
- Improve counter-strategy development

**Coordination:**
- Learn cooperative strategies
- Handle team-based games
- Develop communication protocols

## Optimizations

### 1. Function Approximation

**Linear approximation:**
$$V(s) = \sum_{i} w_i \phi_i(s)$$

**Neural networks:**
$$V(s) = f_\theta(s)$$

**Tile coding:**
- Discretize continuous state spaces
- Enable efficient learning
- Handle high-dimensional inputs

### 2. Experience Replay

**Buffer management:**
- Store experiences in replay buffer
- Sample randomly for learning
- Improve sample efficiency

**Prioritized replay:**
- Prioritize experiences with high TD error
- Focus learning on important transitions
- Accelerate convergence

### 3. Target Networks

**Stable learning:**
- Use separate target network for TD targets
- Update target network periodically
- Reduce learning instability

**Double Q-learning:**
- Use two networks for action selection and evaluation
- Reduce overestimation bias
- Improve learning stability

## Limitations and Challenges

### 1. Sample Efficiency

**Problem:**
- TD learning requires many samples
- Exploration can be expensive
- Convergence may be slow

**Solutions:**
- Experience replay
- Prioritized sampling
- Function approximation

### 2. Exploration Strategy

**Problem:**
- Balancing exploration and exploitation
- Avoiding local optima
- Efficient exploration in large state spaces

**Solutions:**
- Adaptive exploration rates
- Intrinsic motivation
- Hierarchical exploration

### 3. Function Approximation

**Problem:**
- Approximation error
- Generalization issues
- Overfitting to training data

**Solutions:**
- Regularization techniques
- Cross-validation
- Ensemble methods

## Summary

TD learning is a powerful method for game AI that:

1. **Enables incremental learning** from experience
2. **Uses bootstrapping** for efficient updates
3. **Handles delayed rewards** through temporal differences
4. **Supports function approximation** for large state spaces
5. **Enables self-play learning** for game improvement
6. **Provides theoretical guarantees** under certain conditions

The method has been instrumental in recent breakthroughs in game AI and continues to be an active area of research.

## Key Takeaways

1. **TD learning updates value estimates** based on temporal differences
2. **Bootstrapping enables efficient learning** without waiting for final outcomes
3. **TD(λ) combines TD learning** with eligibility traces for better credit assignment
4. **Function approximation** is essential for large state spaces
5. **Self-play learning** has enabled major breakthroughs in game AI
6. **Exploration strategies** are crucial for effective learning
7. **Experience replay and target networks** improve learning stability
8. **Continuous research** addresses limitations and improves performance 