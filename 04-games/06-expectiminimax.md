# Expectiminimax: Handling Adversarial and Stochastic Games

## Introduction

Expectiminimax extends minimax to handle games with three types of nodes: max, min, and chance nodes. This makes it suitable for games with both adversarial and stochastic elements, such as backgammon, poker, and many modern board games.

## What is Expectiminimax?

Expectiminimax is an algorithm that combines:
- **Minimax**: For adversarial decision making
- **Expectimax**: For probabilistic outcomes
- **Three node types**: Max, min, and chance nodes

**Node Types:**
- **Max nodes**: Player 1's turn (maximize utility)
- **Min nodes**: Player 2's turn (minimize utility)
- **Chance nodes**: Random events (expected value)

## Mathematical Foundation

### Expectiminimax Value Function

The expectiminimax value of a node is defined recursively:

$$V(s) = \begin{cases}
\text{Utility}(s) & \text{if } s \text{ is terminal} \\
\max_{a \in A(s)} V(\text{Result}(s, a)) & \text{if } s \text{ is a max node} \\
\min_{a \in A(s)} V(\text{Result}(s, a)) & \text{if } s \text{ is a min node} \\
\sum_{a \in A(s)} P(a) \cdot V(\text{Result}(s, a)) & \text{if } s \text{ is a chance node}
\end{cases}$$

Where:
- $s$ is the current state
- $A(s)$ is the set of available actions
- $\text{Result}(s, a)$ is the state resulting from action $a$ in state $s$
- $P(a)$ is the probability of action $a$ occurring

### Expected Value for Chance Nodes

For chance nodes, the expected value is:

$$E[V] = \sum_{i=1}^{n} P_i \cdot V_i$$

Where:
- $P_i$ is the probability of outcome $i$
- $V_i$ is the value of outcome $i$
- $n$ is the number of possible outcomes

## Algorithm Structure

### Node Types and Decision Rules

1. **Max Nodes**: Player 1's decision points
   - Choose action with maximum expected value
   - Represent strategic choices for maximizing player

2. **Min Nodes**: Player 2's decision points
   - Choose action with minimum expected value
   - Represent adversarial responses

3. **Chance Nodes**: Random event points
   - Compute weighted average of child values
   - Represent probabilistic outcomes (dice, cards, etc.)

### Algorithm Steps

1. **Generate game tree** to specified depth
2. **Identify node types** based on game state
3. **Evaluate leaf nodes** using evaluation function
4. **Propagate values up** using expectiminimax rule:
   - Max nodes: $\max(\text{children})$
   - Min nodes: $\min(\text{children})$
   - Chance nodes: $\sum P_i \cdot V_i$
5. **Choose root action** with optimal expected value

## Python Implementation

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum
import random
import time

class NodeType(Enum):
    """Types of nodes in the expectiminimax tree."""
    MAX = "max"
    MIN = "min"
    CHANCE = "chance"
    TERMINAL = "terminal"

@dataclass
class GameState:
    """Base game state class."""
    node_type: NodeType
    is_terminal: bool = False
    utility: float = 0.0

class ExpectiminimaxNode:
    """Node in the expectiminimax search tree."""
    
    def __init__(self, state: GameState, action: Any = None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.value = None
        self.probability = 1.0  # For chance nodes
        self.depth = 0
        
    def add_child(self, child):
        """Add a child node."""
        self.children.append(child)
        child.parent = self
        child.depth = self.depth + 1
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (terminal or no children)."""
        return self.state.is_terminal or len(self.children) == 0

class ExpectiminimaxSearch:
    """Expectiminimax search algorithm implementation."""
    
    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.nodes_evaluated = 0
        self.transposition_table = {}
        
    def search(self, root_state: GameState, game_rules) -> Tuple[Any, float]:
        """
        Perform expectiminimax search and return best action and value.
        
        Args:
            root_state: Initial game state
            game_rules: Game rules object
            
        Returns:
            Tuple of (best_action, expected_value)
        """
        root = ExpectiminimaxNode(root_state)
        value = self._expectiminimax(root, 0, game_rules)
        
        # Find the best action
        best_action = None
        if root.state.node_type == NodeType.MAX:
            best_value = float('-inf')
            for child in root.children:
                if child.value > best_value:
                    best_value = child.value
                    best_action = child.action
        elif root.state.node_type == NodeType.MIN:
            best_value = float('inf')
            for child in root.children:
                if child.value < best_value:
                    best_value = child.value
                    best_action = child.action
        else:  # Chance node at root (unusual but possible)
            best_action = None  # No action to choose
        
        return best_action, value
    
    def _expectiminimax(self, node: ExpectiminimaxNode, depth: int, game_rules) -> float:
        """
        Recursive expectiminimax algorithm.
        
        Args:
            node: Current node
            depth: Current depth
            game_rules: Game rules object
            
        Returns:
            Expected value of the node
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
        
        # Apply expectiminimax rule based on node type
        if node.state.node_type == NodeType.MAX:
            value = float('-inf')
            for child in node.children:
                child_value = self._expectiminimax(child, depth + 1, game_rules)
                child.value = child_value
                value = max(value, child_value)
                
        elif node.state.node_type == NodeType.MIN:
            value = float('inf')
            for child in node.children:
                child_value = self._expectiminimax(child, depth + 1, game_rules)
                child.value = child_value
                value = min(value, child_value)
                
        elif node.state.node_type == NodeType.CHANCE:
            value = 0.0
            for child in node.children:
                child_value = self._expectiminimax(child, depth + 1, game_rules)
                child.value = child_value
                value += child.probability * child_value
        
        else:
            raise ValueError(f"Unknown node type: {node.state.node_type}")
        
        # Store in transposition table
        self.transposition_table[state_hash] = (depth, value)
        return value
    
    def _generate_children(self, node: ExpectiminimaxNode, game_rules):
        """Generate child nodes for the given node."""
        if node.state.node_type == NodeType.CHANCE:
            # Generate chance outcomes
            outcomes = game_rules.get_chance_outcomes(node.state)
            for outcome, probability in outcomes:
                child_state = game_rules.apply_chance_outcome(node.state, outcome)
                child = ExpectiminimaxNode(child_state, outcome, node)
                child.probability = probability
                node.add_child(child)
        else:
            # Generate action outcomes
            actions = game_rules.get_legal_actions(node.state)
            for action in actions:
                child_state = game_rules.apply_action(node.state, action)
                child = ExpectiminimaxNode(child_state, action, node)
                node.add_child(child)
    
    def _hash_state(self, state: GameState) -> str:
        """Create a hash of the game state for transposition table."""
        # This is a simplified hash - in practice, use a proper hash function
        return str(hash(str(state)))

# Example: Backgammon Implementation
class BackgammonState(GameState):
    """State for a simplified backgammon game."""
    
    def __init__(self, board: np.ndarray, player_turn: bool, dice: List[int] = None):
        super().__init__(NodeType.MAX if player_turn else NodeType.MIN)
        self.board = board.copy()  # 24 points + 2 home areas
        self.player_turn = player_turn
        self.dice = dice or []
        
        # Check for terminal state
        if self._is_game_over():
            self.is_terminal = True
            self.node_type = NodeType.TERMINAL
            self.utility = self._calculate_final_utility()
    
    def _is_game_over(self) -> bool:
        """Check if the game is over."""
        # Simplified: game ends when all pieces are home
        player_pieces = np.sum(self.board[:24] > 0)
        opponent_pieces = np.sum(self.board[:24] < 0)
        return player_pieces == 0 or opponent_pieces == 0
    
    def _calculate_final_utility(self) -> float:
        """Calculate utility for terminal states."""
        player_pieces = np.sum(self.board[:24] > 0)
        opponent_pieces = np.sum(self.board[:24] < 0)
        
        if player_pieces == 0 and opponent_pieces > 0:
            return 1.0  # Player wins
        elif opponent_pieces == 0 and player_pieces > 0:
            return 0.0  # Opponent wins
        else:
            return 0.5  # Draw

class BackgammonRules:
    """Rules for simplified backgammon."""
    
    def __init__(self):
        self.board_size = 26  # 24 points + 2 home areas
        
    def get_legal_actions(self, state: BackgammonState) -> List[Tuple[int, int]]:
        """Get legal moves for the current state."""
        if not state.dice:
            return []  # Need to roll dice first
        
        legal_moves = []
        player_sign = 1 if state.player_turn else -1
        
        # Find pieces that can move
        for from_pos in range(24):
            if self.board[from_pos] * player_sign > 0:  # Player's piece
                for die in state.dice:
                    to_pos = from_pos + die * player_sign
                    if 0 <= to_pos < 24:
                        # Check if move is legal
                        if self.board[to_pos] * player_sign >= 0:  # Empty or friendly
                            legal_moves.append((from_pos, to_pos))
        
        return legal_moves
    
    def apply_action(self, state: BackgammonState, action: Tuple[int, int]) -> BackgammonState:
        """Apply a move action."""
        from_pos, to_pos = action
        new_state = BackgammonState(state.board, state.player_turn, state.dice.copy())
        
        # Make the move
        player_sign = 1 if state.player_turn else -1
        new_state.board[from_pos] -= player_sign
        new_state.board[to_pos] += player_sign
        
        # Remove used die
        die_used = abs(to_pos - from_pos)
        if die_used in new_state.dice:
            new_state.dice.remove(die_used)
        
        # Switch turns if no dice left
        if not new_state.dice:
            new_state.player_turn = not new_state.player_turn
            new_state.node_type = NodeType.MIN if new_state.player_turn else NodeType.MAX
        
        return new_state
    
    def get_chance_outcomes(self, state: BackgammonState) -> List[Tuple[List[int], float]]:
        """Get possible dice roll outcomes."""
        # Simplified: roll 2 dice
        outcomes = []
        for die1 in range(1, 7):
            for die2 in range(1, 7):
                dice = [die1, die2]
                if die1 == die2:  # Doubles
                    dice.extend([die1, die2])
                outcomes.append((dice, 1.0 / 36))
        
        return outcomes
    
    def apply_chance_outcome(self, state: BackgammonState, outcome: List[int]) -> BackgammonState:
        """Apply dice roll outcome."""
        new_state = BackgammonState(state.board, state.player_turn, outcome)
        return new_state
    
    def evaluate(self, state: BackgammonState) -> float:
        """Evaluate non-terminal backgammon positions."""
        if state.is_terminal:
            return state.utility
        
        # Simple evaluation based on piece positions
        player_score = 0
        opponent_score = 0
        
        for i in range(24):
            if state.board[i] > 0:  # Player pieces
                player_score += state.board[i] * (i + 1)  # Closer to home is better
            elif state.board[i] < 0:  # Opponent pieces
                opponent_score += abs(state.board[i]) * (24 - i)  # Further from home is better
        
        return (player_score - opponent_score) / 1000.0

# Example: Simplified Poker Implementation
class PokerState(GameState):
    """State for a simplified poker game."""
    
    def __init__(self, player_hand: List[int], opponent_hand: List[int], 
                 community_cards: List[int], player_turn: bool, pot: int = 0):
        super().__init__(NodeType.MAX if player_turn else NodeType.MIN)
        self.player_hand = player_hand.copy()
        self.opponent_hand = opponent_hand.copy()
        self.community_cards = community_cards.copy()
        self.player_turn = player_turn
        self.pot = pot
        
        # Check for terminal state
        if self._is_game_over():
            self.is_terminal = True
            self.node_type = NodeType.TERMINAL
            self.utility = self._calculate_final_utility()
    
    def _is_game_over(self) -> bool:
        """Check if the game is over."""
        # Game ends when all community cards are dealt and betting is complete
        return len(self.community_cards) >= 5
    
    def _calculate_final_utility(self) -> float:
        """Calculate utility for terminal states."""
        player_score = self._evaluate_hand(self.player_hand + self.community_cards)
        opponent_score = self._evaluate_hand(self.opponent_hand + self.community_cards)
        
        if player_score > opponent_score:
            return self.pot / 100.0  # Normalize
        elif player_score < opponent_score:
            return -self.pot / 100.0
        else:
            return 0.0  # Split pot
    
    def _evaluate_hand(self, cards: List[int]) -> int:
        """Simple hand evaluation (higher is better)."""
        # Simplified: just sum card values
        return sum(cards)

class PokerRules:
    """Rules for simplified poker game."""
    
    def __init__(self):
        self.deck = list(range(1, 53))  # 52 cards
        
    def get_legal_actions(self, state: PokerState) -> List[str]:
        """Get legal actions for the current state."""
        if state.is_terminal:
            return []
        
        if len(state.community_cards) < 5:
            return ["deal_card"]  # Deal next community card
        else:
            return ["fold", "call", "raise"]  # Betting actions
    
    def apply_action(self, state: PokerState, action: str) -> PokerState:
        """Apply an action to the current state."""
        if action == "deal_card":
            # Deal next community card (chance node)
            new_state = PokerState(
                state.player_hand, state.opponent_hand,
                state.community_cards, state.player_turn, state.pot
            )
            new_state.node_type = NodeType.CHANCE
            return new_state
        else:
            # Betting action
            new_state = PokerState(
                state.player_hand, state.opponent_hand,
                state.community_cards, not state.player_turn, state.pot
            )
            # Simplified betting logic
            if action == "raise":
                new_state.pot += 10
            elif action == "call":
                new_state.pot += 5
            # fold doesn't change pot
            
            return new_state
    
    def get_chance_outcomes(self, state: PokerState) -> List[Tuple[int, float]]:
        """Get possible card deals."""
        # Available cards (simplified)
        used_cards = set(state.player_hand + state.opponent_hand + state.community_cards)
        available_cards = [c for c in self.deck if c not in used_cards]
        
        outcomes = []
        probability = 1.0 / len(available_cards)
        
        for card in available_cards:
            outcomes.append((card, probability))
        
        return outcomes
    
    def apply_chance_outcome(self, state: PokerState, outcome: int) -> PokerState:
        """Apply card deal outcome."""
        new_community_cards = state.community_cards + [outcome]
        new_state = PokerState(
            state.player_hand, state.opponent_hand,
            new_community_cards, state.player_turn, state.pot
        )
        return new_state
    
    def evaluate(self, state: PokerState) -> float:
        """Evaluate non-terminal poker positions."""
        if state.is_terminal:
            return state.utility
        
        # Simple evaluation based on current hand strength
        player_score = self._evaluate_hand(state.player_hand + state.community_cards)
        opponent_score = self._evaluate_hand(state.opponent_hand + state.community_cards)
        
        return (player_score - opponent_score) / 100.0
    
    def _evaluate_hand(self, cards: List[int]) -> int:
        """Simple hand evaluation."""
        return sum(cards)

# Example usage
def demonstrate_expectiminimax():
    """Demonstrate expectiminimax algorithm."""
    
    print("=== Expectiminimax Search Demonstration ===")
    
    # Backgammon example
    print("\n1. Backgammon Example:")
    board = np.zeros(26)
    board[0] = 2   # Player pieces
    board[23] = -2 # Opponent pieces
    initial_bg_state = BackgammonState(board, True)
    bg_rules = BackgammonRules()
    
    search = ExpectiminimaxSearch(max_depth=3)
    start_time = time.time()
    best_action, expected_value = search.search(initial_bg_state, bg_rules)
    end_time = time.time()
    
    print(f"Best action: {best_action}")
    print(f"Expected value: {expected_value:.3f}")
    print(f"Nodes evaluated: {search.nodes_evaluated}")
    print(f"Time taken: {end_time - start_time:.3f} seconds")
    
    # Poker example
    print("\n2. Poker Example:")
    player_hand = [1, 2]  # Ace, 2
    opponent_hand = [3, 4]  # 3, 4
    community_cards = [5, 6, 7]  # Flop
    initial_poker_state = PokerState(player_hand, opponent_hand, community_cards, True)
    poker_rules = PokerRules()
    
    poker_search = ExpectiminimaxSearch(max_depth=2)
    start_time = time.time()
    best_action, expected_value = poker_search.search(initial_poker_state, poker_rules)
    end_time = time.time()
    
    print(f"Best action: {best_action}")
    print(f"Expected value: {expected_value:.3f}")
    print(f"Nodes evaluated: {poker_search.nodes_evaluated}")
    print(f"Time taken: {end_time - start_time:.3f} seconds")

def compare_algorithms():
    """Compare minimax, expectimax, and expectiminimax."""
    
    print("\n=== Algorithm Comparison ===")
    
    # Create a simple game state that can be used with all three algorithms
    # This would require implementing a game that can be represented
    # with different node types for fair comparison
    
    print("Algorithm comparison would show:")
    print("- Minimax: Only max/min nodes")
    print("- Expectimax: Max/min + chance nodes")
    print("- Expectiminimax: Max/min + chance nodes with proper structure")

if __name__ == "__main__":
    demonstrate_expectiminimax()
    compare_algorithms()
```

## Mathematical Analysis

### 1. Time Complexity

**Without pruning:**
- Time: $O(b^d)$ where $b$ is branching factor, $d$ is depth
- Space: $O(b \cdot d)$ for recursive calls

**With pruning:**
- Time: $O(b^{d/2})$ in best case (alpha-beta pruning)
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

Expectiminimax is optimal for:
- **Perfect information**: All players know the game state
- **Known probabilities**: Chance event probabilities are known
- **Rational opponents**: Opponents play optimally
- **Finite games**: Games with a finite number of states

## Applications

### 1. Board Games with Random Elements

**Backgammon:**
- Dice rolls determine available moves
- Chance nodes represent dice outcomes
- Optimal play requires probability consideration

**Monopoly:**
- Dice rolls determine movement
- Chance and Community Chest cards
- Property acquisition and rent collection

### 2. Card Games

**Poker:**
- Hidden information (opponent cards)
- Probability of different hands
- Betting decisions under uncertainty

**Bridge:**
- Bidding with incomplete information
- Play of the hand with known/unknown cards
- Partnership coordination

### 3. Video Games

**Strategy Games:**
- Random events (weather, terrain)
- Unit combat with random damage
- Resource generation with variability

**RPGs:**
- Combat with random damage/effects
- Item drops and rewards
- Character development choices

## Optimizations

### 1. Alpha-Beta Pruning

Alpha-beta pruning can be adapted for expectiminimax:

```python
def expectiminimax_alpha_beta(node, depth, alpha, beta, game_rules):
    if node.is_leaf() or depth >= max_depth:
        return game_rules.evaluate(node.state)
    
    if node.state.node_type == NodeType.MAX:
        value = float('-inf')
        for child in node.children:
            value = max(value, expectiminimax_alpha_beta(child, depth + 1, alpha, beta, game_rules))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Beta cutoff
        return value
    
    elif node.state.node_type == NodeType.MIN:
        value = float('inf')
        for child in node.children:
            value = min(value, expectiminimax_alpha_beta(child, depth + 1, alpha, beta, game_rules))
            beta = min(beta, value)
            if alpha >= beta:
                break  # Alpha cutoff
        return value
    
    else:  # Chance node
        value = 0.0
        for child in node.children:
            value += child.probability * expectiminimax_alpha_beta(child, depth + 1, alpha, beta, game_rules)
        return value
```

### 2. Transposition Tables

Store previously computed values:

```python
class TranspositionTable:
    def __init__(self):
        self.table = {}
    
    def get(self, state_hash, depth):
        if state_hash in self.table:
            stored_depth, value = self.table[state_hash]
            if stored_depth >= depth:
                return value
        return None
    
    def store(self, state_hash, depth, value):
        self.table[state_hash] = (depth, value)
```

### 3. Move Ordering

Order moves to improve pruning efficiency:

```python
def order_moves(node, game_rules):
    """Order moves to improve alpha-beta pruning."""
    if node.state.node_type == NodeType.MAX:
        # Order by expected value (highest first)
        return sorted(node.children, key=lambda c: c.value or 0, reverse=True)
    elif node.state.node_type == NodeType.MIN:
        # Order by expected value (lowest first)
        return sorted(node.children, key=lambda c: c.value or 0)
    else:
        # Chance nodes: order by probability (highest first)
        return sorted(node.children, key=lambda c: c.probability, reverse=True)
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

### 2. Probability Estimation

**Unknown probabilities:**
- Opponent behavior modeling
- Hidden information
- Incomplete knowledge

**Solutions:**
- Bayesian inference
- Opponent modeling
- Learning from experience

### 3. Imperfect Information

**Hidden information:**
- Opponent's cards
- Hidden game state
- Unknown strategies

**Solutions:**
- Information set modeling
- Bayesian games
- Monte Carlo methods

## Summary

Expectiminimax is a powerful algorithm for decision-making in games with both adversarial and stochastic elements that:

1. **Extends minimax** to handle probabilistic outcomes
2. **Considers three node types**: max, min, and chance
3. **Computes expected values** for optimal decision making
4. **Applies to many games** with random elements
5. **Can be optimized** with pruning and caching techniques

The algorithm provides a principled approach to handling uncertainty in games and serves as a foundation for more advanced techniques in game AI.

## Key Takeaways

1. **Expectiminimax handles both adversarial and stochastic elements** through three node types
2. **Max and min nodes** require different value computation strategies
3. **Chance nodes** compute expected values using probabilities
4. **Expected value calculation** is fundamental to the algorithm
5. **Optimization techniques** can significantly improve performance
6. **Applications span** many game types with random elements
7. **Limitations exist** in computational complexity and information requirements
8. **Extensions and improvements** continue to advance the field 