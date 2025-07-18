# Non-zero-sum Games: Cooperative and Mixed-Motive Interactions

## Introduction

Non-zero-sum games allow for cooperative outcomes where multiple players can benefit simultaneously, expanding beyond pure competition. These games model real-world scenarios where cooperation, negotiation, and collective action are possible and often beneficial.

## What are Non-zero-sum Games?

Non-zero-sum games are characterized by:
- **Cooperative possibilities**: Players can benefit simultaneously
- **Mixed motives**: Both competitive and cooperative elements
- **Pareto optimality**: Outcomes where no player can improve without harming others
- **Social dilemmas**: Individual vs. collective rationality conflicts

**Key Concepts:**
- **Cooperative games**: Players can form coalitions and share payoffs
- **Bargaining theory**: Negotiation and agreement processes
- **Mechanism design**: Creating incentives for desired behavior
- **Social welfare**: Maximizing collective benefit

## Mathematical Foundation

### Game Representation

**Normal form (strategic form):**
$$G = (N, S, u)$$

Where:
- $N = \{1, 2, \ldots, n\}$ is the set of players
- $S = S_1 \times S_2 \times \ldots \times S_n$ is the strategy space
- $u_i: S \rightarrow \mathbb{R}$ is player $i$'s utility function

**Characteristic function form (cooperative games):**
$$v: 2^N \rightarrow \mathbb{R}$$

Where $v(C)$ is the value that coalition $C$ can achieve.

### Pareto Optimality

A strategy profile $s^*$ is Pareto optimal if there exists no other profile $s$ such that:
$$u_i(s) \geq u_i(s^*) \text{ for all } i \text{ and } u_j(s) > u_j(s^*) \text{ for some } j$$

### Nash Bargaining Solution

For two-player bargaining games, the Nash bargaining solution maximizes:
$$\max_{(u_1, u_2) \in U} (u_1 - d_1)(u_2 - d_2)$$

Where $d_i$ is player $i$'s disagreement point.

### Shapley Value

For cooperative games, the Shapley value for player $i$ is:
$$\phi_i(v) = \sum_{C \subseteq N \setminus \{i\}} \frac{|C|!(n-|C|-1)!}{n!} [v(C \cup \{i\}) - v(C)]$$

## Algorithm Structure

### Core Principles

1. **Cooperation possibilities**: Players can form coalitions and share benefits
2. **Bargaining processes**: Negotiation and agreement mechanisms
3. **Social welfare**: Maximizing collective benefit
4. **Incentive compatibility**: Designing mechanisms that align individual and collective interests

### Solution Concepts

1. **Nash equilibrium**: No player can unilaterally improve
2. **Pareto optimality**: No player can improve without harming others
3. **Core**: No coalition can improve by deviating
4. **Shapley value**: Fair division of coalitional value

## Python Implementation

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Set, FrozenSet
import numpy as np
from dataclasses import dataclass
from enum import Enum
import random
import time
from itertools import combinations, permutations

class Player(Enum):
    """Player enumeration."""
    PLAYER_1 = 1
    PLAYER_2 = 2
    PLAYER_3 = 3

@dataclass
class Strategy:
    """Strategy representation."""
    name: str
    player: Player
    probability: float = 1.0

class NonZeroSumGame:
    """Base class for non-zero-sum games."""
    
    def __init__(self, num_players: int):
        self.num_players = num_players
        self.players = [Player.PLAYER_1, Player.PLAYER_2, Player.PLAYER_3][:num_players]
        self.strategies = {player: [] for player in self.players}
        self.payoff_matrices = {}
        self.characteristic_function = {}  # For cooperative games
    
    def add_strategy(self, player: Player, strategy: Strategy):
        """Add a strategy for a player."""
        self.strategies[player].append(strategy)
    
    def set_payoff(self, strategy_profile: Tuple[str, ...], payoffs: List[float]):
        """Set payoffs for a strategy profile."""
        self.payoff_matrices[strategy_profile] = payoffs
    
    def set_coalition_value(self, coalition: FrozenSet[Player], value: float):
        """Set value for a coalition (cooperative games)."""
        self.characteristic_function[coalition] = value
    
    def get_payoff(self, strategy_profile: Tuple[str, ...], player: Player) -> float:
        """Get payoff for a player given a strategy profile."""
        if strategy_profile in self.payoff_matrices:
            player_index = self.players.index(player)
            return self.payoff_matrices[strategy_profile][player_index]
        return 0.0
    
    def get_coalition_value(self, coalition: FrozenSet[Player]) -> float:
        """Get value for a coalition."""
        return self.characteristic_function.get(coalition, 0.0)
    
    def get_all_strategy_profiles(self) -> List[Tuple[str, ...]]:
        """Get all possible strategy profiles."""
        strategy_lists = [self.strategies[player] for player in self.players]
        return list(product(*strategy_lists))
    
    def find_pareto_optimal_profiles(self) -> List[Tuple[str, ...]]:
        """Find Pareto optimal strategy profiles."""
        profiles = self.get_all_strategy_profiles()
        pareto_optimal = []
        
        for profile in profiles:
            is_pareto_optimal = True
            profile_payoffs = [self.get_payoff(profile, player) for player in self.players]
            
            for other_profile in profiles:
                if profile == other_profile:
                    continue
                
                other_payoffs = [self.get_payoff(other_profile, player) for player in self.players]
                
                # Check if other profile Pareto dominates this one
                dominates = True
                for i in range(len(self.players)):
                    if other_payoffs[i] < profile_payoffs[i]:
                        dominates = False
                        break
                
                if dominates and any(other_payoffs[i] > profile_payoffs[i] for i in range(len(self.players))):
                    is_pareto_optimal = False
                    break
            
            if is_pareto_optimal:
                pareto_optimal.append(profile)
        
        return pareto_optimal
    
    def find_nash_equilibria(self) -> List[Tuple[str, ...]]:
        """Find pure strategy Nash equilibria."""
        equilibria = []
        strategy_profiles = self.get_all_strategy_profiles()
        
        for profile in strategy_profiles:
            is_equilibrium = True
            
            for player in self.players:
                player_index = self.players.index(player)
                current_strategy = profile[player_index]
                
                # Check if player can improve by deviating
                for strategy in self.strategies[player]:
                    if strategy.name == current_strategy:
                        continue
                    
                    # Create new profile with deviation
                    new_profile = list(profile)
                    new_profile[player_index] = strategy.name
                    new_profile = tuple(new_profile)
                    
                    current_payoff = self.get_payoff(profile, player)
                    new_payoff = self.get_payoff(new_profile, player)
                    
                    if new_payoff > current_payoff:
                        is_equilibrium = False
                        break
                
                if not is_equilibrium:
                    break
            
            if is_equilibrium:
                equilibria.append(profile)
        
        return equilibria
    
    def compute_shapley_values(self) -> Dict[Player, float]:
        """Compute Shapley values for all players."""
        shapley_values = {player: 0.0 for player in self.players}
        
        for player in self.players:
            for coalition_size in range(self.num_players):
                for coalition in combinations([p for p in self.players if p != player], coalition_size):
                    coalition_set = frozenset(coalition)
                    coalition_with_player = frozenset(coalition + (player,))
                    
                    # Compute marginal contribution
                    marginal_contribution = (self.get_coalition_value(coalition_with_player) - 
                                           self.get_coalition_value(coalition_set))
                    
                    # Weight by probability of this coalition size
                    weight = (np.math.factorial(coalition_size) * 
                             np.math.factorial(self.num_players - coalition_size - 1) / 
                             np.math.factorial(self.num_players))
                    
                    shapley_values[player] += weight * marginal_contribution
        
        return shapley_values
    
    def find_core(self) -> List[Dict[Player, float]]:
        """Find core allocations (simplified version)."""
        # This is a simplified implementation
        # In practice, finding the core requires solving linear programming problems
        
        core_allocations = []
        
        # Check if grand coalition is stable
        grand_coalition = frozenset(self.players)
        grand_coalition_value = self.get_coalition_value(grand_coalition)
        
        if grand_coalition_value > 0:
            # Simple equal division
            equal_division = {player: grand_coalition_value / self.num_players 
                            for player in self.players}
            core_allocations.append(equal_division)
        
        return core_allocations

# Example: Stag Hunt Game
class StagHuntGame(NonZeroSumGame):
    """Implementation of the Stag Hunt game."""
    
    def __init__(self):
        super().__init__(num_players=2)
        self.setup_game()
    
    def setup_game(self):
        """Set up the Stag Hunt game."""
        # Add strategies
        hunt_stag = Strategy("Hunt Stag", Player.PLAYER_1)
        hunt_hare = Strategy("Hunt Hare", Player.PLAYER_1)
        self.add_strategy(Player.PLAYER_1, hunt_stag)
        self.add_strategy(Player.PLAYER_1, hunt_hare)
        
        hunt_stag2 = Strategy("Hunt Stag", Player.PLAYER_2)
        hunt_hare2 = Strategy("Hunt Hare", Player.PLAYER_2)
        self.add_strategy(Player.PLAYER_2, hunt_stag2)
        self.add_strategy(Player.PLAYER_2, hunt_hare2)
        
        # Set payoffs
        # Both hunt stag: (3, 3) - both get large payoff
        self.set_payoff(("Hunt Stag", "Hunt Stag"), [3, 3])
        
        # One hunts stag, one hunts hare: (0, 1) or (1, 0)
        self.set_payoff(("Hunt Stag", "Hunt Hare"), [0, 1])
        self.set_payoff(("Hunt Hare", "Hunt Stag"), [1, 0])
        
        # Both hunt hare: (1, 1) - both get small payoff
        self.set_payoff(("Hunt Hare", "Hunt Hare"), [1, 1])
    
    def display_payoffs(self):
        """Display the payoff matrix."""
        print("Stag Hunt Game Payoff Matrix:")
        print("Player 1 (rows) vs Player 2 (columns)")
        print("Format: (Player 1 payoff, Player 2 payoff)")
        print()
        
        strategies = ["Hunt Stag", "Hunt Hare"]
        
        print(" " * 15 + "|" + " " * 15 + "|" + " " * 15)
        print(" " * 15 + "|" + "Hunt Stag".center(15) + "|" + "Hunt Hare".center(15))
        print("-" * 50)
        
        for i, strategy1 in enumerate(strategies):
            row = strategy1.center(15) + "|"
            for j in range(2):
                strategy2 = strategies[j]
                payoff1 = self.get_payoff((strategy1, strategy2), Player.PLAYER_1)
                payoff2 = self.get_payoff((strategy1, strategy2), Player.PLAYER_2)
                cell = f"({payoff1:2.0f}, {payoff2:2.0f})"
                row += cell.center(15) + "|"
            print(row)
            print("-" * 50)

# Example: Public Goods Game
class PublicGoodsGame(NonZeroSumGame):
    """Implementation of the Public Goods game."""
    
    def __init__(self, num_players: int = 3):
        super().__init__(num_players=num_players)
        self.setup_game()
    
    def setup_game(self):
        """Set up the Public Goods game."""
        # Add strategies
        for player in self.players:
            contribute = Strategy("Contribute", player)
            free_ride = Strategy("Free Ride", player)
            self.add_strategy(player, contribute)
            self.add_strategy(player, free_ride)
        
        # Set up characteristic function for cooperative analysis
        for coalition_size in range(1, self.num_players + 1):
            for coalition in combinations(self.players, coalition_size):
                coalition_set = frozenset(coalition)
                # Value is proportional to coalition size
                self.set_coalition_value(coalition_set, len(coalition) * 2)
        
        # Set up strategic form payoffs
        self._setup_strategic_payoffs()
    
    def _setup_strategic_payoffs(self):
        """Set up strategic form payoffs."""
        # This is a simplified version
        # In practice, payoffs would depend on the specific public goods mechanism
        
        for profile in self.get_all_strategy_profiles():
            payoffs = []
            contributors = sum(1 for i, strategy in enumerate(profile) 
                             if strategy == "Contribute")
            
            for player in self.players:
                player_index = self.players.index(player)
                player_strategy = profile[player_index]
                
                if player_strategy == "Contribute":
                    # Cost of contribution + benefit from public good
                    payoff = -1 + (contributors * 0.8)
                else:
                    # Only benefit from public good
                    payoff = contributors * 0.8
                
                payoffs.append(payoff)
            
            self.set_payoff(profile, payoffs)

# Example: Bargaining Game
class BargainingGame(NonZeroSumGame):
    """Implementation of a bargaining game."""
    
    def __init__(self):
        super().__init__(num_players=2)
        self.setup_game()
    
    def setup_game(self):
        """Set up the bargaining game."""
        # Add strategies (different division proposals)
        for player in self.players:
            for i in range(11):  # 0%, 10%, ..., 100%
                strategy = Strategy(f"Propose {i*10}%", player)
                self.add_strategy(player, strategy)
        
        # Set payoffs based on agreement/disagreement
        for profile in self.get_all_strategy_profiles():
            proposal1 = int(profile[0].split()[1].replace('%', ''))
            proposal2 = int(profile[1].split()[1].replace('%', ''))
            
            if proposal1 + proposal2 <= 100:
                # Agreement
                payoffs = [proposal1 / 100.0, proposal2 / 100.0]
            else:
                # Disagreement
                payoffs = [0.0, 0.0]
            
            self.set_payoff(profile, payoffs)
    
    def find_nash_bargaining_solution(self) -> Tuple[float, float]:
        """Find Nash bargaining solution."""
        # Simplified implementation
        # In practice, this would solve the optimization problem
        
        # Find Pareto optimal outcomes
        pareto_optimal = self.find_pareto_optimal_profiles()
        
        best_solution = None
        best_product = -1
        
        for profile in pareto_optimal:
            payoff1 = self.get_payoff(profile, Player.PLAYER_1)
            payoff2 = self.get_payoff(profile, Player.PLAYER_2)
            
            # Nash product (assuming disagreement point is (0, 0))
            product = payoff1 * payoff2
            
            if product > best_product:
                best_product = product
                best_solution = (payoff1, payoff2)
        
        return best_solution

class GameAnalyzer:
    """Analyzer for non-zero-sum games."""
    
    def __init__(self, game: NonZeroSumGame):
        self.game = game
    
    def analyze_game(self):
        """Perform comprehensive game analysis."""
        print(f"=== Analysis of {self.game.__class__.__name__} ===")
        
        # Display payoffs
        if hasattr(self.game, 'display_payoffs'):
            self.game.display_payoffs()
        
        # Find Nash equilibria
        print("\n1. Nash Equilibria:")
        equilibria = self.game.find_nash_equilibria()
        if equilibria:
            for i, equilibrium in enumerate(equilibria, 1):
                print(f"   Equilibrium {i}: {equilibrium}")
                payoffs = [self.game.get_payoff(equilibrium, player) for player in self.game.players]
                print(f"   Payoffs: {payoffs}")
        else:
            print("   No pure strategy Nash equilibria")
        
        # Find Pareto optimal profiles
        print("\n2. Pareto Optimal Profiles:")
        pareto_optimal = self.game.find_pareto_optimal_profiles()
        for profile in pareto_optimal:
            payoffs = [self.game.get_payoff(profile, player) for player in self.game.players]
            print(f"   {profile}: {payoffs}")
        
        # Cooperative analysis (if applicable)
        if hasattr(self.game, 'characteristic_function') and self.game.characteristic_function:
            print("\n3. Cooperative Analysis:")
            
            # Shapley values
            shapley_values = self.game.compute_shapley_values()
            print("   Shapley Values:")
            for player, value in shapley_values.items():
                print(f"     {player.name}: {value:.3f}")
            
            # Core
            core = self.game.find_core()
            if core:
                print("   Core Allocations:")
                for i, allocation in enumerate(core, 1):
                    print(f"     Allocation {i}: {allocation}")
            else:
                print("   Core is empty")
        
        # Social welfare analysis
        print("\n4. Social Welfare Analysis:")
        profiles = self.game.get_all_strategy_profiles()
        best_welfare = -float('inf')
        best_profile = None
        
        for profile in profiles:
            payoffs = [self.game.get_payoff(profile, player) for player in self.game.players]
            welfare = sum(payoffs)
            
            if welfare > best_welfare:
                best_welfare = welfare
                best_profile = profile
        
        print(f"   Maximum social welfare: {best_welfare:.3f}")
        print(f"   Achieved by: {best_profile}")
        payoffs = [self.game.get_payoff(best_profile, player) for player in self.game.players]
        print(f"   Payoffs: {payoffs}")

class BargainingAnalyzer:
    """Analyzer for bargaining games."""
    
    def __init__(self, game: BargainingGame):
        self.game = game
    
    def analyze_bargaining(self):
        """Analyze bargaining game."""
        print("=== Bargaining Game Analysis ===")
        
        # Find Nash bargaining solution
        nash_solution = self.game.find_nash_bargaining_solution()
        print(f"Nash bargaining solution: {nash_solution}")
        
        # Find efficient outcomes
        pareto_optimal = self.game.find_pareto_optimal_profiles()
        efficient_outcomes = []
        
        for profile in pareto_optimal:
            payoff1 = self.game.get_payoff(profile, Player.PLAYER_1)
            payoff2 = self.game.get_payoff(profile, Player.PLAYER_2)
            if payoff1 + payoff2 == 1.0:  # Efficient outcomes
                efficient_outcomes.append((profile, payoff1, payoff2))
        
        print(f"Efficient outcomes: {len(efficient_outcomes)}")
        for profile, p1, p2 in efficient_outcomes:
            print(f"  {profile}: ({p1:.2f}, {p2:.2f})")

class SocialDilemmaAnalyzer:
    """Analyzer for social dilemmas."""
    
    def __init__(self, game: NonZeroSumGame):
        self.game = game
    
    def analyze_social_dilemma(self):
        """Analyze social dilemma aspects."""
        print("=== Social Dilemma Analysis ===")
        
        # Find individual vs collective rationality
        profiles = self.game.get_all_strategy_profiles()
        
        # Find best individual outcomes
        best_individual = {player: -float('inf') for player in self.game.players}
        best_individual_profile = {player: None for player in self.game.players}
        
        for profile in profiles:
            for player in self.game.players:
                payoff = self.game.get_payoff(profile, player)
                if payoff > best_individual[player]:
                    best_individual[player] = payoff
                    best_individual_profile[player] = profile
        
        # Find best collective outcome
        best_collective = -float('inf')
        best_collective_profile = None
        
        for profile in profiles:
            total_payoff = sum(self.game.get_payoff(profile, player) for player in self.game.players)
            if total_payoff > best_collective:
                best_collective = total_payoff
                best_collective_profile = profile
        
        print("Individual rationality:")
        for player in self.game.players:
            print(f"  {player.name}: {best_individual[player]:.3f} at {best_individual_profile[player]}")
        
        print(f"Collective rationality: {best_collective:.3f} at {best_collective_profile}")
        
        # Check for social dilemma
        is_dilemma = False
        for player in self.game.players:
            if best_individual_profile[player] != best_collective_profile:
                is_dilemma = True
                break
        
        if is_dilemma:
            print("This game exhibits a social dilemma!")
        else:
            print("Individual and collective rationality align.")

# Example usage
def demonstrate_non_zero_sum_games():
    """Demonstrate non-zero-sum games analysis."""
    
    print("=== Non-zero-sum Games Demonstration ===")
    
    # Stag Hunt Game
    print("\n1. Stag Hunt Game:")
    stag_hunt = StagHuntGame()
    analyzer = GameAnalyzer(stag_hunt)
    analyzer.analyze_game()
    
    # Public Goods Game
    print("\n2. Public Goods Game:")
    public_goods = PublicGoodsGame(num_players=3)
    analyzer = GameAnalyzer(public_goods)
    analyzer.analyze_game()
    
    # Bargaining Game
    print("\n3. Bargaining Game:")
    bargaining = BargainingGame()
    bargaining_analyzer = BargainingAnalyzer(bargaining)
    bargaining_analyzer.analyze_bargaining()
    
    # Social Dilemma Analysis
    print("\n4. Social Dilemma Analysis:")
    dilemma_analyzer = SocialDilemmaAnalyzer(public_goods)
    dilemma_analyzer.analyze_social_dilemma()

def demonstrate_cooperation_mechanisms():
    """Demonstrate mechanisms for promoting cooperation."""
    
    print("\n=== Cooperation Mechanisms ===")
    
    # Repeated interaction
    print("\n1. Repeated Interaction:")
    print("   Cooperation can emerge through repeated play")
    print("   Players can use strategies like Tit-for-Tat")
    print("   Future consequences encourage cooperation")
    
    # Communication
    print("\n2. Communication:")
    print("   Pre-play communication can coordinate expectations")
    print("   Cheap talk can influence behavior")
    print("   Signaling can reveal intentions")
    
    # Institutions
    print("\n3. Institutions:")
    print("   Formal rules and enforcement mechanisms")
    print("   Penalties for defection")
    print("   Rewards for cooperation")
    
    # Reputation
    print("\n4. Reputation:")
    print("   Past behavior affects future opportunities")
    print("   Reputation systems can sustain cooperation")
    print("   Social sanctions for defection")

def compare_solution_concepts():
    """Compare different solution concepts for non-zero-sum games."""
    
    print("\n=== Solution Concepts Comparison ===")
    
    # Create a simple coordination game with conflict
    class CoordinationConflictGame(NonZeroSumGame):
        def __init__(self):
            super().__init__(num_players=2)
            self.setup_game()
        
        def setup_game(self):
            # Add strategies
            a = Strategy("A", Player.PLAYER_1)
            b = Strategy("B", Player.PLAYER_1)
            self.add_strategy(Player.PLAYER_1, a)
            self.add_strategy(Player.PLAYER_1, b)
            
            a2 = Strategy("A", Player.PLAYER_2)
            b2 = Strategy("B", Player.PLAYER_2)
            self.add_strategy(Player.PLAYER_2, a2)
            self.add_strategy(Player.PLAYER_2, b2)
            
            # Set payoffs
            self.set_payoff(("A", "A"), [3, 2])  # Player 1 prefers A
            self.set_payoff(("B", "B"), [2, 3])  # Player 2 prefers B
            self.set_payoff(("A", "B"), [0, 0])  # Mismatch
            self.set_payoff(("B", "A"), [0, 0])  # Mismatch
    
    game = CoordinationConflictGame()
    analyzer = GameAnalyzer(game)
    analyzer.analyze_game()

if __name__ == "__main__":
    demonstrate_non_zero_sum_games()
    demonstrate_cooperation_mechanisms()
    compare_solution_concepts()
```

## Mathematical Analysis

### 1. Cooperative Game Theory

**Core:**
The core is the set of allocations where no coalition can improve by deviating:
$$x_i \geq v(\{i\}) \text{ for all } i$$
$$\sum_{i \in C} x_i \geq v(C) \text{ for all coalitions } C$$

**Shapley value properties:**
- Efficiency: $\sum_i \phi_i(v) = v(N)$
- Symmetry: $\phi_i(v) = \phi_j(v)$ if $i$ and $j$ are symmetric
- Additivity: $\phi_i(v + w) = \phi_i(v) + \phi_i(w)$

### 2. Bargaining Theory

**Nash bargaining solution:**
$$\max_{(u_1, u_2) \in U} (u_1 - d_1)(u_2 - d_2)$$

**Kalai-Smorodinsky solution:**
Maximizes the ratio of gains relative to ideal points.

### 3. Mechanism Design

**Incentive compatibility:**
$$\sum_{s_{-i}} p(s_{-i}) u_i(f(s_i, s_{-i}), s_i) \geq \sum_{s_{-i}} p(s_{-i}) u_i(f(s_i', s_{-i}), s_i)$$

**Individual rationality:**
$$u_i(f(s), s_i) \geq u_i(d_i, s_i)$$

## Applications

### 1. Economics

**Market design:**
- Auction design
- Matching markets
- Resource allocation

**Industrial organization:**
- Cartel formation
- Joint ventures
- Strategic alliances

### 2. Political Science

**Coalition formation:**
- Government formation
- Legislative bargaining
- International alliances

**Voting systems:**
- Proportional representation
- Coalition voting
- Strategic voting

### 3. Computer Science

**Distributed systems:**
- Consensus protocols
- Load balancing
- Resource sharing

**Multi-agent systems:**
- Coalition formation
- Task allocation
- Resource negotiation

## Solution Methods

### 1. Cooperative Solution Methods

**Core computation:**
```python
def compute_core(game):
    # Solve linear programming problem
    # Find allocations satisfying core constraints
    pass
```

**Shapley value computation:**
```python
def compute_shapley_value(game, player):
    value = 0
    for coalition in all_coalitions_without_player:
        marginal_contribution = (game.value(coalition + player) - 
                               game.value(coalition))
        weight = compute_weight(coalition, game.num_players)
        value += weight * marginal_contribution
    return value
```

### 2. Bargaining Solution Methods

**Nash bargaining:**
```python
def nash_bargaining_solution(game, disagreement_point):
    # Solve optimization problem
    # Maximize product of gains
    pass
```

**Alternating offers:**
```python
def alternating_offers_bargaining(game, discount_factor):
    # Implement Rubinstein bargaining model
    pass
```

### 3. Mechanism Design

**Vickrey-Clarke-Groves mechanism:**
```python
def vcg_mechanism(agents, alternatives):
    # Compute efficient allocation
    # Charge agents their externality
    pass
```

## Limitations and Challenges

### 1. Computational Complexity

**Problem:**
- Computing core is NP-hard
- Shapley value requires exponential time
- Mechanism design is complex

**Solutions:**
- Approximation algorithms
- Special case analysis
- Heuristic methods

### 2. Information Requirements

**Problem:**
- Players may have private information
- Preferences may be uncertain
- Communication may be limited

**Solutions:**
- Bayesian games
- Mechanism design
- Communication protocols

### 3. Enforcement Issues

**Problem:**
- Agreements may not be enforceable
- Players may renege on commitments
- External enforcement may be costly

**Solutions:**
- Reputation mechanisms
- Repeated interaction
- Institutional design

## Summary

Non-zero-sum games provide a rich framework for analyzing cooperative and mixed-motive interactions that:

1. **Model real-world scenarios** where cooperation is possible and beneficial
2. **Support multiple solution concepts** including Nash equilibrium, Pareto optimality, and cooperative solutions
3. **Enable analysis of social dilemmas** and collective action problems
4. **Provide tools for mechanism design** and institutional analysis
5. **Apply across many disciplines** including economics, political science, and computer science
6. **Support both theoretical analysis** and practical applications

Understanding non-zero-sum games is essential for analyzing complex social and economic interactions.

## Key Takeaways

1. **Non-zero-sum games allow** for cooperative outcomes and mutual benefit
2. **Multiple solution concepts** are relevant including Nash equilibrium, Pareto optimality, and cooperative solutions
3. **Social dilemmas** arise when individual and collective rationality conflict
4. **Cooperation mechanisms** include repeated interaction, communication, and institutions
5. **Bargaining theory** provides tools for analyzing negotiation and agreement
6. **Mechanism design** creates incentives for desired behavior
7. **Computational challenges** exist but can be addressed through various methods
8. **Applications span** economics, politics, and computer science 