# Modeling: Game Representation and Structure

## Introduction

Game modeling involves representing strategic interactions as formal mathematical structures that AI algorithms can analyze and solve. The quality of the model directly impacts the effectiveness of game-playing algorithms.

## Components of Game Models

### Players

Players are rational agents with defined objectives and preferences. Each player:
- Has a set of possible actions
- Aims to maximize their utility
- May have different information sets
- Can be human or AI agents

**Mathematical Representation:**
$$N = \{1, 2, \ldots, n\}$$

Where $N$ is the set of players and $n$ is the number of players.

### States

States represent complete descriptions of the game situation at any point in time.

**State Space:**
$$S = \{s_1, s_2, \ldots, s_m\}$$

**State Properties:**
- **Complete information**: All players know the current state
- **Incomplete information**: Some information is hidden
- **Observable**: Players can observe state transitions
- **Partially observable**: Players see only partial state information

### Actions

Actions are the moves or decisions available to players in each state.

**Action Space:**
$$A_i(s) = \{a_1, a_2, \ldots, a_k\}$$

Where $A_i(s)$ is the set of legal actions for player $i$ in state $s$.

### Transitions

Transitions describe how states change based on actions taken by players.

**Transition Function:**
$$T: S \times A_1 \times A_2 \times \ldots \times A_n \rightarrow S$$

For deterministic games:
$$s' = T(s, a_1, a_2, \ldots, a_n)$$

For stochastic games:
$$P(s'|s, a_1, a_2, \ldots, a_n) = T(s, a_1, a_2, \ldots, a_n, s')$$

### Payoffs

Payoffs represent the outcomes and rewards for different scenarios.

**Utility Function:**
$$U_i: S \rightarrow \mathbb{R}$$

Where $U_i(s)$ is the utility of player $i$ in state $s$.

## Model Types

### 1. Extensive Form (Game Trees)

Extensive form represents games as trees with sequential moves.

**Structure:**
- **Nodes**: Game states
- **Edges**: Actions/moves
- **Information sets**: Groups of nodes where a player cannot distinguish between them
- **Payoffs**: Terminal node values

**Mathematical Representation:**
$$G = (N, V, E, P, I, u)$$

Where:
- $N$: Set of players
- $V$: Set of nodes (states)
- $E$: Set of edges (actions)
- $P$: Player function (who moves at each node)
- $I$: Information partition
- $u$: Utility functions

### 2. Strategic Form (Normal Form)

Strategic form represents games as payoff matrices for simultaneous decisions.

**Payoff Matrix:**
$$U = [u_{ij}]$$

Where $u_{ij}$ is the payoff when player 1 chooses strategy $i$ and player 2 chooses strategy $j$.

### 3. Normal Form (Simplified)

Normal form provides a simplified representation focusing on strategies and outcomes.

## Python Implementation: Game Modeling Framework

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

class Player(Enum):
    """Player enumeration."""
    PLAYER_1 = 1
    PLAYER_2 = 2
    CHANCE = 0  # For random events

@dataclass
class GameState:
    """Base class for game states."""
    player: Player
    is_terminal: bool = False
    
class GameModel(ABC):
    """Abstract base class for game models."""
    
    def __init__(self, num_players: int):
        self.num_players = num_players
        self.players = [Player.PLAYER_1, Player.PLAYER_2][:num_players]
        
    @abstractmethod
    def get_initial_state(self) -> GameState:
        """Return the initial game state."""
        pass
    
    @abstractmethod
    def get_legal_actions(self, state: GameState) -> List[Any]:
        """Return list of legal actions in a given state."""
        pass
    
    @abstractmethod
    def apply_action(self, state: GameState, action: Any) -> GameState:
        """Apply an action and return the new state."""
        pass
    
    @abstractmethod
    def get_utility(self, state: GameState, player: Player) -> float:
        """Return the utility value for a player in a terminal state."""
        pass
    
    @abstractmethod
    def is_terminal(self, state: GameState) -> bool:
        """Check if the state is terminal."""
        pass

class ExtensiveFormGame(GameModel):
    """Extensive form game representation using game trees."""
    
    def __init__(self):
        super().__init__(num_players=2)
        self.nodes = {}  # state_id -> GameState
        self.edges = defaultdict(list)  # state_id -> [(action, next_state_id)]
        self.information_sets = defaultdict(set)  # player -> set of state_ids
        
    def add_node(self, state_id: str, state: GameState):
        """Add a node to the game tree."""
        self.nodes[state_id] = state
        
    def add_edge(self, from_state_id: str, action: Any, to_state_id: str):
        """Add an edge to the game tree."""
        self.edges[from_state_id].append((action, to_state_id))
        
    def add_information_set(self, player: Player, state_ids: Set[str]):
        """Add an information set for a player."""
        self.information_sets[player].update(state_ids)
        
    def get_successors(self, state_id: str) -> List[Tuple[Any, str]]:
        """Get successor states and actions."""
        return self.edges.get(state_id, [])
    
    def build_tree(self, max_depth: int = 10):
        """Build the complete game tree up to max_depth."""
        # Implementation depends on specific game
        pass

class StrategicFormGame(GameModel):
    """Strategic form game representation using payoff matrices."""
    
    def __init__(self, num_strategies_player1: int, num_strategies_player2: int):
        super().__init__(num_players=2)
        self.num_strategies_p1 = num_strategies_player1
        self.num_strategies_p2 = num_strategies_player2
        self.payoff_matrix_p1 = np.zeros((num_strategies_player1, num_strategies_player2))
        self.payoff_matrix_p2 = np.zeros((num_strategies_player1, num_strategies_player2))
        
    def set_payoff(self, strategy_p1: int, strategy_p2: int, payoff_p1: float, payoff_p2: float):
        """Set payoff for a strategy combination."""
        self.payoff_matrix_p1[strategy_p1, strategy_p2] = payoff_p1
        self.payoff_matrix_p2[strategy_p1, strategy_p2] = payoff_p2
        
    def get_payoff(self, strategy_p1: int, strategy_p2: int, player: Player) -> float:
        """Get payoff for a strategy combination."""
        if player == Player.PLAYER_1:
            return self.payoff_matrix_p1[strategy_p1, strategy_p2]
        else:
            return self.payoff_matrix_p2[strategy_p1, strategy_p2]
    
    def find_nash_equilibrium(self) -> List[Tuple[int, int]]:
        """Find pure strategy Nash equilibria."""
        equilibria = []
        
        for i in range(self.num_strategies_p1):
            for j in range(self.num_strategies_p2):
                # Check if (i, j) is a Nash equilibrium
                is_equilibrium = True
                
                # Check if player 1 can improve by deviating
                for k in range(self.num_strategies_p1):
                    if k != i and self.payoff_matrix_p1[k, j] > self.payoff_matrix_p1[i, j]:
                        is_equilibrium = False
                        break
                
                if not is_equilibrium:
                    continue
                
                # Check if player 2 can improve by deviating
                for k in range(self.num_strategies_p2):
                    if k != j and self.payoff_matrix_p2[i, k] > self.payoff_matrix_p2[i, j]:
                        is_equilibrium = False
                        break
                
                if is_equilibrium:
                    equilibria.append((i, j))
        
        return equilibria

# Example: Prisoner's Dilemma
class PrisonersDilemma(StrategicFormGame):
    """Implementation of the Prisoner's Dilemma game."""
    
    def __init__(self):
        super().__init__(num_strategies_player1=2, num_strategies_player2=2)
        self.setup_payoffs()
        
    def setup_payoffs(self):
        """Set up the classic Prisoner's Dilemma payoffs."""
        # Strategies: 0 = Cooperate, 1 = Defect
        # Payoffs: (Player 1, Player 2)
        
        # Both cooperate: (-1, -1) - both get light sentence
        self.set_payoff(0, 0, -1, -1)
        
        # Player 1 cooperates, Player 2 defects: (-3, 0) - P1 gets heavy sentence
        self.set_payoff(0, 1, -3, 0)
        
        # Player 1 defects, Player 2 cooperates: (0, -3) - P2 gets heavy sentence
        self.set_payoff(1, 0, 0, -3)
        
        # Both defect: (-2, -2) - both get medium sentence
        self.set_payoff(1, 1, -2, -2)
    
    def get_strategy_name(self, strategy: int) -> str:
        """Get human-readable strategy name."""
        return "Cooperate" if strategy == 0 else "Defect"
    
    def display_payoffs(self):
        """Display the payoff matrices."""
        print("Prisoner's Dilemma Payoff Matrix:")
        print("Player 1 (rows) vs Player 2 (columns)")
        print("Format: (Player 1 payoff, Player 2 payoff)")
        print()
        
        strategies = ["Cooperate", "Defect"]
        
        print(" " * 15 + "|" + " " * 15 + "|" + " " * 15)
        print(" " * 15 + "|" + "Cooperate".center(15) + "|" + "Defect".center(15))
        print("-" * 50)
        
        for i, strategy1 in enumerate(strategies):
            row = strategy1.center(15) + "|"
            for j in range(2):
                payoff1 = self.payoff_matrix_p1[i, j]
                payoff2 = self.payoff_matrix_p2[i, j]
                cell = f"({payoff1:2.0f}, {payoff2:2.0f})"
                row += cell.center(15) + "|"
            print(row)
            print("-" * 50)

# Example: Tic-tac-toe in Extensive Form
class TicTacToeState(GameState):
    """State representation for Tic-tac-toe."""
    
    def __init__(self, board: np.ndarray, player: Player):
        super().__init__(player)
        self.board = board.copy()
        self.is_terminal = self._check_terminal()
        
    def _check_terminal(self) -> bool:
        """Check if the game is over."""
        # Check rows, columns, and diagonals for win
        for player in [1, 2]:
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
    
    def get_utility(self, player: Player) -> float:
        """Get utility for a player."""
        if not self.is_terminal:
            return 0
        
        player_id = player.value
        # Check if current player won
        for p in [1, 2]:
            # Rows
            for i in range(3):
                if np.all(self.board[i, :] == p):
                    return 1 if p == player_id else -1
            # Columns
            for j in range(3):
                if np.all(self.board[:, j] == p):
                    return 1 if p == player_id else -1
            # Diagonals
            if np.all(np.diag(self.board) == p):
                return 1 if p == player_id else -1
            if np.all(np.diag(np.fliplr(self.board)) == p):
                return 1 if p == player_id else -1
        
        return 0  # Draw
    
    def __hash__(self):
        """Make state hashable for use in dictionaries."""
        return hash(self.board.tobytes())
    
    def __eq__(self, other):
        """Check equality of states."""
        return np.array_equal(self.board, other.board)

class TicTacToeExtensiveForm(ExtensiveFormGame):
    """Tic-tac-toe in extensive form."""
    
    def __init__(self):
        super().__init__()
        self.state_counter = 0
        
    def get_initial_state(self) -> TicTacToeState:
        """Return the initial game state."""
        board = np.zeros((3, 3), dtype=int)
        return TicTacToeState(board, Player.PLAYER_1)
    
    def get_legal_actions(self, state: TicTacToeState) -> List[Tuple[int, int]]:
        """Return list of legal actions."""
        if state.is_terminal:
            return []
        
        empty_positions = []
        for i in range(3):
            for j in range(3):
                if state.board[i, j] == 0:
                    empty_positions.append((i, j))
        return empty_positions
    
    def apply_action(self, state: TicTacToeState, action: Tuple[int, int]) -> TicTacToeState:
        """Apply an action and return the new state."""
        new_board = state.board.copy()
        i, j = action
        new_board[i, j] = state.player.value
        
        # Determine next player
        next_player = Player.PLAYER_2 if state.player == Player.PLAYER_1 else Player.PLAYER_1
        
        return TicTacToeState(new_board, next_player)
    
    def get_utility(self, state: TicTacToeState, player: Player) -> float:
        """Return the utility value for a player."""
        return state.get_utility(player)
    
    def is_terminal(self, state: TicTacToeState) -> bool:
        """Check if the state is terminal."""
        return state.is_terminal
    
    def build_game_tree(self, max_depth: int = 9):
        """Build the complete game tree."""
        initial_state = self.get_initial_state()
        self._build_tree_recursive(initial_state, "", 0, max_depth)
    
    def _build_tree_recursive(self, state: TicTacToeState, path: str, depth: int, max_depth: int):
        """Recursively build the game tree."""
        if depth >= max_depth or state.is_terminal:
            return
        
        state_id = path if path else "root"
        self.add_node(state_id, state)
        
        legal_actions = self.get_legal_actions(state)
        for action in legal_actions:
            next_state = self.apply_action(state, action)
            next_path = f"{path}_{action[0]}{action[1]}" if path else f"{action[0]}{action[1]}"
            
            self.add_edge(state_id, action, next_path)
            self._build_tree_recursive(next_state, next_path, depth + 1, max_depth)

# Example usage
def demonstrate_game_modeling():
    """Demonstrate different game modeling approaches."""
    
    print("=== Strategic Form Game: Prisoner's Dilemma ===")
    pd = PrisonersDilemma()
    pd.display_payoffs()
    
    equilibria = pd.find_nash_equilibrium()
    print(f"\nNash Equilibria: {equilibria}")
    for eq in equilibria:
        strategy1 = pd.get_strategy_name(eq[0])
        strategy2 = pd.get_strategy_name(eq[1])
        print(f"  - Player 1: {strategy1}, Player 2: {strategy2}")
    
    print("\n=== Extensive Form Game: Tic-tac-toe ===")
    ttt = TicTacToeExtensiveForm()
    initial_state = ttt.get_initial_state()
    print(f"Initial state player: {initial_state.player}")
    print(f"Initial board:\n{initial_state.board}")
    
    legal_actions = ttt.get_legal_actions(initial_state)
    print(f"Legal actions: {legal_actions}")
    
    # Apply first action
    if legal_actions:
        action = legal_actions[0]
        next_state = ttt.apply_action(initial_state, action)
        print(f"\nAfter action {action}:")
        print(f"Board:\n{next_state.board}")
        print(f"Current player: {next_state.player}")
        print(f"Is terminal: {next_state.is_terminal}")

if __name__ == "__main__":
    demonstrate_game_modeling()
```

## Mathematical Modeling Techniques

### 1. State Space Modeling

**State Representation:**
$$s = (s_1, s_2, \ldots, s_n)$$

Where each $s_i$ represents a component of the state.

**State Transition:**
$$s' = f(s, a_1, a_2, \ldots, a_n)$$

### 2. Information Set Modeling

For imperfect information games, information sets group states that are indistinguishable to a player:

$$I_i = \{s \in S : \text{player } i \text{ cannot distinguish between states in } I_i\}$$

### 3. Strategy Representation

**Pure Strategy:**
$$\sigma_i: S_i \rightarrow A_i$$

**Mixed Strategy:**
$$\sigma_i: S_i \rightarrow \Delta(A_i)$$

Where $\Delta(A_i)$ is the set of probability distributions over $A_i$.

### 4. Utility Modeling

**Expected Utility:**
$$EU_i(\sigma) = \sum_{s \in S} P(s|\sigma) \cdot U_i(s)$$

Where $P(s|\sigma)$ is the probability of reaching state $s$ under strategy profile $\sigma$.

## Model Validation and Verification

### 1. Completeness Check

Ensure all possible game scenarios are represented:
- All legal actions are included
- All state transitions are defined
- All terminal conditions are specified

### 2. Consistency Check

Verify that the model is internally consistent:
- No contradictory rules
- Proper payoff assignments
- Valid state transitions

### 3. Equilibrium Analysis

Check for the existence and properties of equilibria:
- Nash equilibrium existence
- Pareto optimality
- Stability analysis

## Advanced Modeling Techniques

### 1. Stochastic Games

For games with random elements:

**Transition Probability:**
$$P(s'|s, a_1, a_2, \ldots, a_n)$$

**Expected Payoff:**
$$E[U_i] = \sum_{s'} P(s'|s, a) \cdot U_i(s')$$

### 2. Multi-Stage Games

For games with multiple phases:

**Stage Transition:**
$$s_{t+1} = T_t(s_t, a_t)$$

**Cumulative Utility:**
$$U_i = \sum_{t=0}^{T} \gamma^t \cdot U_i^t(s_t)$$

### 3. Bayesian Games

For games with incomplete information:

**Type Space:**
$$\Theta_i = \{\theta_i^1, \theta_i^2, \ldots, \theta_i^k\}$$

**Belief Function:**
$$\mu_i: \Theta_{-i} \rightarrow [0, 1]$$

## Summary

Game modeling is a fundamental step in AI game playing that involves:

1. **Defining game components**: Players, states, actions, transitions, payoffs
2. **Choosing representation**: Extensive form, strategic form, or normal form
3. **Implementing algorithms**: Tree search, matrix operations, equilibrium finding
4. **Validating models**: Completeness, consistency, and equilibrium analysis

The quality of the game model directly impacts the effectiveness of game-playing algorithms and the insights gained from game-theoretic analysis.

## Key Takeaways

1. **Game models are mathematical abstractions** of strategic interactions
2. **Different model types** are suitable for different game characteristics
3. **State space representation** is crucial for computational efficiency
4. **Information sets** handle imperfect information in games
5. **Strategy representation** can be pure or mixed
6. **Model validation** ensures correctness and completeness
7. **Advanced techniques** handle stochastic, multi-stage, and Bayesian games 