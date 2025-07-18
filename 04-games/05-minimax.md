# Minimax: Optimal Decision Making in Zero-Sum Games

## Introduction

Minimax is a recursive algorithm for finding optimal moves in two-player zero-sum games by exploring the game tree to a specified depth. It's the foundation for many game-playing algorithms and provides a principled approach to adversarial decision making.

## What is Minimax?

Minimax is an algorithm that:
- **Maximizes** the minimum gain for the maximizing player
- **Minimizes** the maximum loss for the minimizing player
- **Assumes** both players play optimally
- **Explores** the game tree to find the best move

## Mathematical Foundation

### Minimax Value Function

The minimax value of a node is defined recursively:

$$V(s) = \begin{cases}
\text{Utility}(s) & \text{if } s \text{ is terminal} \\
\max_{a \in A(s)} V(\text{Result}(s, a)) & \text{if } s \text{ is a max node} \\
\min_{a \in A(s)} V(\text{Result}(s, a)) & \text{if } s \text{ is a min node}
\end{cases}$$

Where:
- $s$ is the current state
- $A(s)$ is the set of available actions
- $\text{Result}(s, a)$ is the state resulting from action $a$ in state $s$

### Optimal Strategy

The optimal strategy for each player is:

**Maximizing player:**
$$\pi^*(s) = \arg\max_{a \in A(s)} V(\text{Result}(s, a))$$

**Minimizing player:**
$$\pi^*(s) = \arg\min_{a \in A(s)} V(\text{Result}(s, a))$$

## Algorithm Structure

### Core Principles

1. **Maximizing player**: Chooses moves to maximize own utility
2. **Minimizing player**: Chooses moves to minimize opponent's utility
3. **Alternating turns**: Players take turns making decisions
4. **Perfect information**: All players know the complete game state

### Algorithm Steps

1. **Generate game tree** to specified depth
2. **Evaluate leaf nodes** using evaluation function
3. **Propagate values up** using minimax rule:
   - Max nodes: $\max(\text{children})$
   - Min nodes: $\min(\text{children})$
4. **Choose root action** with optimal value

## Python Implementation

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum
import time

class Player(Enum):
    """Player enumeration."""
    MAX = "max"
    MIN = "min"

@dataclass
class GameState:
    """Base game state class."""
    player: Player
    is_terminal: bool = False
    utility: float = 0.0

class MinimaxNode:
    """Node in the minimax search tree."""
    
    def __init__(self, state: GameState, action: Any = None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.value = None
        self.depth = 0
        
    def add_child(self, child):
        """Add a child node."""
        self.children.append(child)
        child.parent = self
        child.depth = self.depth + 1
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (terminal or no children)."""
        return self.state.is_terminal or len(self.children) == 0

class MinimaxSearch:
    """Minimax search algorithm implementation."""
    
    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.nodes_evaluated = 0
        self.transposition_table = {}
        
    def search(self, root_state: GameState, game_rules) -> Tuple[Any, float]:
        """
        Perform minimax search and return best action and value.
        
        Args:
            root_state: Initial game state
            game_rules: Game rules object
            
        Returns:
            Tuple of (best_action, minimax_value)
        """
        root = MinimaxNode(root_state)
        value = self._minimax(root, 0, game_rules)
        
        # Find the best action
        best_action = None
        if root.state.player == Player.MAX:
            best_value = float('-inf')
            for child in root.children:
                if child.value > best_value:
                    best_value = child.value
                    best_action = child.action
        else:
            best_value = float('inf')
            for child in root.children:
                if child.value < best_value:
                    best_value = child.value
                    best_action = child.action
        
        return best_action, value
    
    def _minimax(self, node: MinimaxNode, depth: int, game_rules) -> float:
        """
        Recursive minimax algorithm.
        
        Args:
            node: Current node
            depth: Current depth
            game_rules: Game rules object
            
        Returns:
            Minimax value of the node
        """
        self.nodes_evaluated += 1
        
        # Check transposition table
        state_hash = self._hash_state(node.state)
        if state_hash in self.transposition_table:
            stored_depth, value = self.transposition_table[state_hash]
            if stored_depth >= depth:
                return value
        
        # Terminal node or maximum depth reached
        if node.is_leaf() or depth >= self.max_depth:
            if node.state.is_terminal:
                value = node.state.utility
            else:
                # Use evaluation function for non-terminal leaves
                value = game_rules.evaluate(node.state)
            
            # Store in transposition table
            self.transposition_table[state_hash] = (depth, value)
            return value
        
        # Generate children if not already done
        if len(node.children) == 0:
            self._generate_children(node, game_rules)
        
        # Apply minimax rule based on player
        if node.state.player == Player.MAX:
            value = float('-inf')
            for child in node.children:
                child_value = self._minimax(child, depth + 1, game_rules)
                child.value = child_value
                value = max(value, child_value)
        else:  # MIN player
            value = float('inf')
            for child in node.children:
                child_value = self._minimax(child, depth + 1, game_rules)
                child.value = child_value
                value = min(value, child_value)
        
        # Store in transposition table
        self.transposition_table[state_hash] = (depth, value)
        return value
    
    def _generate_children(self, node: MinimaxNode, game_rules):
        """Generate child nodes for the given node."""
        actions = game_rules.get_legal_actions(node.state)
        for action in actions:
            child_state = game_rules.apply_action(node.state, action)
            child = MinimaxNode(child_state, action, node)
            node.add_child(child)
    
    def _hash_state(self, state: GameState) -> str:
        """Create a hash of the game state for transposition table."""
        # This is a simplified hash - in practice, use a proper hash function
        return str(hash(str(state)))

# Example: Tic-tac-toe Implementation
class TicTacToeState(GameState):
    """State for Tic-tac-toe game."""
    
    def __init__(self, board: np.ndarray, player: Player):
        super().__init__(player)
        self.board = board.copy()
        self.is_terminal = self._check_terminal()
        if self.is_terminal:
            self.utility = self._calculate_utility()
    
    def _check_terminal(self) -> bool:
        """Check if the game is over."""
        # Check rows, columns, and diagonals for win
        for player in [1, -1]:
            # Rows
            for i in range(3):
                if np.all(self.board[i, :] == player):
                    return True
            # Columns
            for j in range(3):
                if np.all(self.board[:, j] == player):
                    return True
            # Diagonals
            if np.all(np.diag(self.board) == player):
                return True
            if np.all(np.diag(np.fliplr(self.board)) == player):
                return True
        
        # Check for draw
        return np.all(self.board != 0)
    
    def _calculate_utility(self) -> float:
        """Calculate utility for terminal states."""
        # Check if MAX player (X) won
        for player in [1, -1]:
            # Rows
            for i in range(3):
                if np.all(self.board[i, :] == player):
                    return 1.0 if player == 1 else -1.0
            # Columns
            for j in range(3):
                if np.all(self.board[:, j] == player):
                    return 1.0 if player == 1 else -1.0
            # Diagonals
            if np.all(np.diag(self.board) == player):
                return 1.0 if player == 1 else -1.0
            if np.all(np.diag(np.fliplr(self.board)) == player):
                return 1.0 if player == 1 else -1.0
        
        return 0.0  # Draw

class TicTacToeRules:
    """Rules for Tic-tac-toe game."""
    
    def get_legal_actions(self, state: TicTacToeState) -> List[Tuple[int, int]]:
        """Get legal moves for the current state."""
        if state.is_terminal:
            return []
        
        empty_positions = []
        for i in range(3):
            for j in range(3):
                if state.board[i, j] == 0:
                    empty_positions.append((i, j))
        return empty_positions
    
    def apply_action(self, state: TicTacToeState, action: Tuple[int, int]) -> TicTacToeState:
        """Apply an action to the current state."""
        new_board = state.board.copy()
        i, j = action
        
        # Place the piece
        if state.player == Player.MAX:
            new_board[i, j] = 1  # X
        else:
            new_board[i, j] = -1  # O
        
        # Switch players
        next_player = Player.MIN if state.player == Player.MAX else Player.MAX
        
        return TicTacToeState(new_board, next_player)
    
    def evaluate(self, state: TicTacToeState) -> float:
        """Evaluate non-terminal Tic-tac-toe positions."""
        if state.is_terminal:
            return state.utility
        
        # Simple evaluation based on potential winning lines
        score = 0
        
        # Check rows, columns, and diagonals
        for player in [1, -1]:
            # Rows
            for i in range(3):
                line = state.board[i, :]
                score += self._evaluate_line(line, player)
            
            # Columns
            for j in range(3):
                line = state.board[:, j]
                score += self._evaluate_line(line, player)
            
            # Diagonals
            diag1 = np.diag(state.board)
            diag2 = np.diag(np.fliplr(state.board))
            score += self._evaluate_line(diag1, player)
            score += self._evaluate_line(diag2, player)
        
        return score / 100.0  # Normalize
    
    def _evaluate_line(self, line: np.ndarray, player: int) -> float:
        """Evaluate a line (row, column, or diagonal) for a player."""
        player_count = np.sum(line == player)
        opponent_count = np.sum(line == -player)
        empty_count = np.sum(line == 0)
        
        if opponent_count > 0:
            return 0  # Line blocked by opponent
        
        if player_count == 2 and empty_count == 1:
            return 10  # Near win
        elif player_count == 1 and empty_count == 2:
            return 1   # Potential
        else:
            return 0

# Example: Connect Four Implementation
class ConnectFourState(GameState):
    """State for Connect Four game."""
    
    def __init__(self, board: np.ndarray, player: Player):
        super().__init__(player)
        self.board = board.copy()
        self.rows, self.cols = board.shape
        self.is_terminal = self._check_terminal()
        if self.is_terminal:
            self.utility = self._calculate_utility()
    
    def _check_terminal(self) -> bool:
        """Check if the game is over."""
        # Check for win
        for player in [1, -1]:
            if self._has_win(player):
                return True
        
        # Check for draw (board full)
        return np.all(self.board != 0)
    
    def _has_win(self, player: int) -> bool:
        """Check if a player has won."""
        # Check horizontal
        for i in range(self.rows):
            for j in range(self.cols - 3):
                if np.all(self.board[i, j:j+4] == player):
                    return True
        
        # Check vertical
        for i in range(self.rows - 3):
            for j in range(self.cols):
                if np.all(self.board[i:i+4, j] == player):
                    return True
        
        # Check diagonal (positive slope)
        for i in range(self.rows - 3):
            for j in range(self.cols - 3):
                if np.all([self.board[i+k, j+k] == player for k in range(4)]):
                    return True
        
        # Check diagonal (negative slope)
        for i in range(3, self.rows):
            for j in range(self.cols - 3):
                if np.all([self.board[i-k, j+k] == player for k in range(4)]):
                    return True
        
        return False
    
    def _calculate_utility(self) -> float:
        """Calculate utility for terminal states."""
        if self._has_win(1):  # MAX player won
            return 1.0
        elif self._has_win(-1):  # MIN player won
            return -1.0
        else:
            return 0.0  # Draw

class ConnectFourRules:
    """Rules for Connect Four game."""
    
    def __init__(self, rows: int = 6, cols: int = 7):
        self.rows = rows
        self.cols = cols
    
    def get_legal_actions(self, state: ConnectFourState) -> List[int]:
        """Get legal column moves for the current state."""
        if state.is_terminal:
            return []
        
        legal_columns = []
        for col in range(self.cols):
            if state.board[0, col] == 0:  # Top row is empty
                legal_columns.append(col)
        
        return legal_columns
    
    def apply_action(self, state: ConnectFourState, action: int) -> ConnectFourState:
        """Apply a column move to the current state."""
        new_board = state.board.copy()
        col = action
        
        # Find the lowest empty position in the column
        for row in range(self.rows - 1, -1, -1):
            if new_board[row, col] == 0:
                if state.player == Player.MAX:
                    new_board[row, col] = 1  # X
                else:
                    new_board[row, col] = -1  # O
                break
        
        # Switch players
        next_player = Player.MIN if state.player == Player.MAX else Player.MAX
        
        return ConnectFourState(new_board, next_player)
    
    def evaluate(self, state: ConnectFourState) -> float:
        """Evaluate non-terminal Connect Four positions."""
        if state.is_terminal:
            return state.utility
        
        # Evaluate based on potential winning sequences
        score = 0
        
        for player in [1, -1]:
            player_score = self._evaluate_player(state.board, player)
            score += player_score if player == 1 else -player_score
        
        return score / 1000.0  # Normalize
    
    def _evaluate_player(self, board: np.ndarray, player: int) -> float:
        """Evaluate board for a specific player."""
        score = 0
        
        # Check all possible 4-in-a-row sequences
        for i in range(self.rows):
            for j in range(self.cols - 3):
                # Horizontal
                sequence = board[i, j:j+4]
                score += self._evaluate_sequence(sequence, player)
        
        for i in range(self.rows - 3):
            for j in range(self.cols):
                # Vertical
                sequence = board[i:i+4, j]
                score += self._evaluate_sequence(sequence, player)
        
        for i in range(self.rows - 3):
            for j in range(self.cols - 3):
                # Diagonal (positive slope)
                sequence = [board[i+k, j+k] for k in range(4)]
                score += self._evaluate_sequence(sequence, player)
                
                # Diagonal (negative slope)
                sequence = [board[i+3-k, j+k] for k in range(4)]
                score += self._evaluate_sequence(sequence, player)
        
        return score
    
    def _evaluate_sequence(self, sequence: List[int], player: int) -> float:
        """Evaluate a 4-position sequence for a player."""
        player_count = sum(1 for x in sequence if x == player)
        opponent_count = sum(1 for x in sequence if x == -player)
        empty_count = sum(1 for x in sequence if x == 0)
        
        if opponent_count > 0:
            return 0  # Sequence blocked by opponent
        
        if player_count == 3 and empty_count == 1:
            return 100  # Near win
        elif player_count == 2 and empty_count == 2:
            return 10   # Potential
        elif player_count == 1 and empty_count == 3:
            return 1    # Weak potential
        else:
            return 0

# Example usage
def demonstrate_minimax():
    """Demonstrate minimax algorithm."""
    
    print("=== Minimax Search Demonstration ===")
    
    # Tic-tac-toe example
    print("\n1. Tic-tac-toe Example:")
    board = np.zeros((3, 3), dtype=int)
    initial_state = TicTacToeState(board, Player.MAX)
    game_rules = TicTacToeRules()
    
    search = MinimaxSearch(max_depth=9)
    start_time = time.time()
    best_action, value = search.search(initial_state, game_rules)
    end_time = time.time()
    
    print(f"Best action: {best_action}")
    print(f"Minimax value: {value:.3f}")
    print(f"Nodes evaluated: {search.nodes_evaluated}")
    print(f"Time taken: {end_time - start_time:.3f} seconds")
    
    # Connect Four example
    print("\n2. Connect Four Example:")
    board = np.zeros((6, 7), dtype=int)
    initial_c4_state = ConnectFourState(board, Player.MAX)
    c4_rules = ConnectFourRules()
    
    c4_search = MinimaxSearch(max_depth=4)  # Smaller depth due to complexity
    start_time = time.time()
    best_action, value = c4_search.search(initial_c4_state, c4_rules)
    end_time = time.time()
    
    print(f"Best action (column): {best_action}")
    print(f"Minimax value: {value:.3f}")
    print(f"Nodes evaluated: {c4_search.nodes_evaluated}")
    print(f"Time taken: {end_time - start_time:.3f} seconds")

def compare_depths():
    """Compare minimax performance at different depths."""
    
    print("\n=== Depth Comparison ===")
    
    board = np.zeros((3, 3), dtype=int)
    initial_state = TicTacToeState(board, Player.MAX)
    game_rules = TicTacToeRules()
    
    depths = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    print("Depth | Nodes | Time (s) | Best Action")
    print("-" * 40)
    
    for depth in depths:
        search = MinimaxSearch(max_depth=depth)
        start_time = time.time()
        best_action, value = search.search(initial_state, game_rules)
        end_time = time.time()
        
        print(f"{depth:5d} | {search.nodes_evaluated:5d} | {end_time - start_time:7.3f} | {best_action}")

if __name__ == "__main__":
    demonstrate_minimax()
    compare_depths()
```

## Mathematical Analysis

### 1. Time Complexity

**Without pruning:**
- Time: $O(b^d)$ where $b$ is branching factor, $d$ is depth
- Space: $O(b \cdot d)$ for recursive calls

**With alpha-beta pruning:**
- Time: $O(b^{d/2})$ in best case
- Space: $O(b \cdot d)$

### 2. Space Complexity

**Tree representation:**
- Nodes: $O(b^d)$
- Edges: $O(b^d)$
- Total: $O(b^d)$

**Recursive implementation:**
- Call stack: $O(d)$
- Node storage: $O(b^d)$

### 3. Optimality

Minimax is optimal for:
- **Perfect information**: All players know the game state
- **Zero-sum games**: One player's gain equals another's loss
- **Rational opponents**: Opponents play optimally
- **Finite games**: Games with a finite number of states

## Applications

### 1. Classic Board Games

**Chess:**
- Complex branching factor (~35)
- Deep search requirements
- Sophisticated evaluation functions

**Checkers:**
- Moderate branching factor (~10)
- Solved for perfect play
- Good testbed for algorithms

**Go:**
- Very high branching factor (~250)
- Requires advanced techniques
- AlphaGo breakthrough

### 2. Modern Games

**Video Games:**
- Strategy games
- Turn-based RPGs
- Puzzle games

**Card Games:**
- Perfect information variants
- Simplified rule sets
- Educational implementations

### 3. AI Research

**Algorithm Development:**
- Alpha-beta pruning
- Transposition tables
- Move ordering

**Evaluation Function Design:**
- Feature engineering
- Machine learning integration
- Performance optimization

## Optimizations

### 1. Alpha-Beta Pruning

```python
def minimax_alpha_beta(node, depth, alpha, beta, game_rules):
    if node.is_leaf() or depth >= max_depth:
        return game_rules.evaluate(node.state)
    
    if node.state.player == Player.MAX:
        value = float('-inf')
        for child in node.children:
            value = max(value, minimax_alpha_beta(child, depth + 1, alpha, beta, game_rules))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Beta cutoff
        return value
    
    else:  # MIN player
        value = float('inf')
        for child in node.children:
            value = min(value, minimax_alpha_beta(child, depth + 1, alpha, beta, game_rules))
            beta = min(beta, value)
            if alpha >= beta:
                break  # Alpha cutoff
        return value
```

### 2. Transposition Tables

```python
class TranspositionTable:
    def __init__(self):
        self.table = {}
    
    def get(self, state_hash, depth):
        if state_hash in self.table:
            stored_depth, value, node_type = self.table[state_hash]
            if stored_depth >= depth:
                return value
        return None
    
    def store(self, state_hash, depth, value, node_type):
        self.table[state_hash] = (depth, value, node_type)
```

### 3. Iterative Deepening

```python
def iterative_deepening_search(root_state, game_rules, max_depth):
    best_action = None
    
    for depth in range(1, max_depth + 1):
        search = MinimaxSearch(max_depth=depth)
        action, value = search.search(root_state, game_rules)
        best_action = action
        
        # Can add time limit check here
        if time.time() - start_time > time_limit:
            break
    
    return best_action
```

## Limitations and Challenges

### 1. Computational Complexity

**Exponential growth:**
- Large branching factors
- Deep search requirements
- Memory constraints

**Solutions:**
- Pruning techniques
- Transposition tables
- Move ordering
- Iterative deepening

### 2. Evaluation Function Quality

**Heuristic nature:**
- Imperfect evaluation
- Domain expertise required
- Performance impact

**Solutions:**
- Machine learning
- Self-play training
- Feature engineering
- Ensemble methods

### 3. Game Complexity

**Large state spaces:**
- Chess: ~10^47 states
- Go: ~10^170 states
- Memory limitations

**Solutions:**
- Selective search
- Monte Carlo methods
- Neural networks
- Distributed computing

## Summary

Minimax is a fundamental algorithm for game AI that:

1. **Provides optimal play** in perfect information zero-sum games
2. **Uses recursive search** to explore game trees
3. **Assumes rational opponents** playing optimally
4. **Requires evaluation functions** for non-terminal positions
5. **Can be optimized** with various techniques
6. **Serves as foundation** for more advanced algorithms

The algorithm's simplicity and effectiveness make it a cornerstone of game AI research and development.

## Key Takeaways

1. **Minimax finds optimal moves** by exploring the game tree recursively
2. **Two player types** require different value computation strategies
3. **Evaluation functions** are crucial for non-terminal positions
4. **Optimization techniques** can dramatically improve performance
5. **Computational complexity** is the main limiting factor
6. **Applications span** many game types and AI research areas
7. **Extensions and improvements** continue to advance the field 