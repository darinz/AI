# Overview: Games in Artificial Intelligence

## Introduction

Game theory provides a mathematical framework for analyzing strategic interactions between rational agents. In artificial intelligence, games serve as excellent testbeds for developing intelligent decision-making algorithms, from simple board games to complex multi-agent systems.

## What are Games in AI?

A game in AI is a formal model of strategic interaction where:
- Multiple agents (players) make decisions
- Each agent has preferences over outcomes
- The outcome depends on the actions of all agents
- Agents act rationally to achieve their goals

## Key Concepts

### Strategic Decision Making Under Uncertainty

In games, agents must make decisions without complete information about:
- Opponent's current state
- Opponent's future actions
- Environmental factors
- Random events

**Example**: In chess, you don't know your opponent's strategy, but you can model their likely responses.

### Multi-Agent Interactions and Competition

Games involve multiple agents with potentially conflicting objectives:
- **Zero-sum games**: One player's gain equals another's loss
- **Non-zero-sum games**: All players can benefit or lose simultaneously
- **Cooperative games**: Players can form alliances

### Optimal Policy Computation

An optimal policy is a strategy that maximizes expected utility:
- **Pure strategy**: Deterministic action selection
- **Mixed strategy**: Probabilistic action selection
- **Nash equilibrium**: No player can unilaterally improve their outcome

### Game Tree Search and Evaluation

Game trees represent all possible game states and transitions:
- **Nodes**: Game states
- **Edges**: Actions/moves
- **Leaves**: Terminal states with payoffs
- **Evaluation**: Function to assess non-terminal states

## Mathematical Foundations

### Game Representation

A game can be formally defined as a tuple:

$$G = (N, S, A, T, R, \gamma)$$

Where:
- $N$: Set of players
- $S$: Set of states
- $A$: Set of actions for each player
- $T$: Transition function $T(s, a) \rightarrow s'$
- $R$: Reward function $R(s, a, s') \rightarrow \mathbb{R}$
- $\gamma$: Discount factor

### Utility Function

The utility function $U_i(s)$ represents player $i$'s preference for state $s$:

$$U_i(s) = \sum_{t=0}^{\infty} \gamma^t R_i(s_t, a_t, s_{t+1})$$

### Nash Equilibrium

A strategy profile $\sigma^*$ is a Nash equilibrium if:

$$U_i(\sigma_i^*, \sigma_{-i}^*) \geq U_i(\sigma_i, \sigma_{-i}^*)$$

For all players $i$ and alternative strategies $\sigma_i$.

## Types of Games

### 1. Sequential vs. Simultaneous

**Sequential Games (Turn-based):**
- Players take turns
- Each player knows previous actions
- Example: Chess, Tic-tac-toe

**Simultaneous Games:**
- Players act at the same time
- No knowledge of others' current actions
- Example: Rock-paper-scissors, Prisoner's dilemma

### 2. Perfect vs. Imperfect Information

**Perfect Information:**
- All players know the complete game state
- Example: Chess, Checkers

**Imperfect Information:**
- Some information is hidden
- Example: Poker, Bridge

### 3. Deterministic vs. Stochastic

**Deterministic Games:**
- No random elements
- Actions have predictable outcomes
- Example: Chess, Go

**Stochastic Games:**
- Random events influence outcomes
- Example: Backgammon, Monopoly

## Python Implementation: Basic Game Framework

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np

class Game(ABC):
    """Abstract base class for games."""
    
    def __init__(self, num_players: int):
        self.num_players = num_players
        self.current_player = 0
        self.game_over = False
        
    @abstractmethod
    def get_initial_state(self) -> Any:
        """Return the initial game state."""
        pass
    
    @abstractmethod
    def get_legal_actions(self, state: Any, player: int) -> List[Any]:
        """Return list of legal actions for a player in a given state."""
        pass
    
    @abstractmethod
    def apply_action(self, state: Any, action: Any, player: int) -> Any:
        """Apply an action and return the new state."""
        pass
    
    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        """Check if the state is terminal (game over)."""
        pass
    
    @abstractmethod
    def get_utility(self, state: Any, player: int) -> float:
        """Return the utility value for a player in a terminal state."""
        pass
    
    @abstractmethod
    def get_current_player(self, state: Any) -> int:
        """Return the player whose turn it is."""
        pass

class TicTacToe(Game):
    """Implementation of Tic-tac-toe game."""
    
    def __init__(self):
        super().__init__(num_players=2)
        self.board_size = 3
        
    def get_initial_state(self) -> np.ndarray:
        """Return empty 3x3 board."""
        return np.zeros((3, 3), dtype=int)
    
    def get_legal_actions(self, state: np.ndarray, player: int) -> List[Tuple[int, int]]:
        """Return list of empty positions."""
        empty_positions = []
        for i in range(3):
            for j in range(3):
                if state[i, j] == 0:
                    empty_positions.append((i, j))
        return empty_positions
    
    def apply_action(self, state: np.ndarray, action: Tuple[int, int], player: int) -> np.ndarray:
        """Place player's mark on the board."""
        new_state = state.copy()
        i, j = action
        new_state[i, j] = player
        return new_state
    
    def is_terminal(self, state: np.ndarray) -> bool:
        """Check if game is over (win or draw)."""
        # Check rows, columns, and diagonals for win
        for player in [1, 2]:
            # Rows
            for i in range(3):
                if np.all(state[i, :] == player):
                    return True
            # Columns
            for j in range(3):
                if np.all(state[:, j] == player):
                    return True
            # Diagonals
            if np.all(np.diag(state) == player):
                return True
            if np.all(np.diag(np.fliplr(state)) == player):
                return True
        
        # Check for draw
        return np.all(state != 0)
    
    def get_utility(self, state: np.ndarray, player: int) -> float:
        """Return utility: 1 for win, -1 for loss, 0 for draw."""
        if not self.is_terminal(state):
            return 0
        
        # Check if current player won
        for p in [1, 2]:
            # Rows
            for i in range(3):
                if np.all(state[i, :] == p):
                    return 1 if p == player else -1
            # Columns
            for j in range(3):
                if np.all(state[:, j] == p):
                    return 1 if p == player else -1
            # Diagonals
            if np.all(np.diag(state) == p):
                return 1 if p == player else -1
            if np.all(np.diag(np.fliplr(state)) == p):
                return 1 if p == player else -1
        
        return 0  # Draw
    
    def get_current_player(self, state: np.ndarray) -> int:
        """Return player whose turn it is based on number of moves made."""
        moves_made = np.sum(state != 0)
        return 1 if moves_made % 2 == 0 else 2
    
    def display_board(self, state: np.ndarray):
        """Display the current board state."""
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        for i in range(3):
            row = '|'.join([symbols[state[i, j]] for j in range(3)])
            print(f" {row} ")
            if i < 2:
                print("-----------")

# Example usage
def play_tic_tac_toe():
    """Interactive Tic-tac-toe game."""
    game = TicTacToe()
    state = game.get_initial_state()
    
    print("Tic-tac-toe Game")
    print("Player 1: X, Player 2: O")
    print("Enter moves as row,column (e.g., 1,2)")
    
    while not game.is_terminal(state):
        current_player = game.get_current_player(state)
        game.display_board(state)
        print(f"\nPlayer {current_player}'s turn")
        
        legal_actions = game.get_legal_actions(state, current_player)
        print(f"Legal moves: {legal_actions}")
        
        # Get player move
        while True:
            try:
                move_input = input("Enter your move (row,col): ")
                row, col = map(int, move_input.split(','))
                if (row, col) in legal_actions:
                    break
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Invalid input. Use format: row,col")
        
        state = game.apply_action(state, (row, col), current_player)
    
    game.display_board(state)
    utility = game.get_utility(state, 1)
    if utility == 1:
        print("Player 1 wins!")
    elif utility == -1:
        print("Player 2 wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    play_tic_tac_toe()
```

## Game Complexity Analysis

### State Space Complexity

The number of possible game states:
- **Tic-tac-toe**: $3^9 = 19,683$ states
- **Chess**: Approximately $10^{47}$ states
- **Go**: Approximately $10^{170}$ states

### Game Tree Complexity

The branching factor and depth determine search complexity:
- **Branching factor**: Average number of legal moves per position
- **Depth**: Maximum number of moves to reach terminal state
- **Total nodes**: $O(b^d)$ where $b$ is branching factor, $d$ is depth

### Computational Challenges

1. **Combinatorial explosion**: Exponential growth of search space
2. **Imperfect information**: Hidden information increases complexity
3. **Multiple equilibria**: Finding optimal strategies in complex games
4. **Real-time constraints**: Limited time for decision making

## Applications in AI

### 1. Game Playing

- **Classical games**: Chess, Go, Checkers
- **Modern games**: Video games, board games
- **Educational games**: Learning environments

### 2. Multi-Agent Systems

- **Robotics**: Coordinated robot teams
- **Economics**: Market simulations
- **Social systems**: Crowd behavior modeling

### 3. Strategic Planning

- **Military**: Battle planning and logistics
- **Business**: Competitive strategy
- **Sports**: Team tactics and individual play

### 4. AI Safety

- **Alignment**: Ensuring AI systems pursue human objectives
- **Robustness**: Handling unexpected situations
- **Transparency**: Understanding AI decision-making

## Summary

Games in AI provide a rich framework for understanding:
- Strategic decision making under uncertainty
- Multi-agent interactions and competition
- Optimal policy computation
- Game tree search and evaluation

The mathematical foundations of game theory, combined with computational algorithms, enable AI systems to play games optimally and apply these principles to real-world strategic problems.

## Key Takeaways

1. **Games are formal models** of strategic interactions between rational agents
2. **Game theory provides mathematical tools** for analyzing optimal strategies
3. **Different game types** require different analytical approaches
4. **Computational complexity** is a major challenge in game playing
5. **Games serve as testbeds** for developing intelligent decision-making algorithms
6. **Game principles apply** to many real-world strategic problems 