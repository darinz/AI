# Modeling Search Problems

## Overview

Search problems are fundamental to artificial intelligence, representing the process of finding a sequence of actions that transforms an initial state into a goal state. Effective problem modeling is crucial for applying search algorithms successfully.

## Key Concepts

### State Space Representation

A search problem is defined by:
- **States**: Complete descriptions of the world at any point in time
- **Actions**: Operations that can transform one state into another
- **Transitions**: Rules that define how actions change states
- **Initial State**: The starting configuration
- **Goal State(s)**: Desired final configuration(s)
- **Path Cost**: Cost associated with taking a sequence of actions

### Mathematical Foundation

A search problem can be formally defined as a tuple $(S, A, T, s_0, G, c)$ where:

- $S$ is the set of all possible states
- $A$ is the set of all possible actions
- $T: S \times A \rightarrow S$ is the transition function
- $s_0 \in S$ is the initial state
- $G \subseteq S$ is the set of goal states
- $c: S \times A \times S \rightarrow \mathbb{R}^+$ is the cost function

## Problem Types

### 1. Deterministic vs. Stochastic

**Deterministic Problems**: Each action has exactly one outcome
```python
def deterministic_transition(state, action):
    # Always returns the same next state for given state and action
    return next_state
```

**Stochastic Problems**: Actions may have multiple possible outcomes
```python
def stochastic_transition(state, action):
    # Returns a probability distribution over possible next states
    return [(next_state1, prob1), (next_state2, prob2), ...]
```

### 2. Fully Observable vs. Partially Observable

**Fully Observable**: Complete information about the current state
**Partially Observable**: Only partial information is available

### 3. Single Agent vs. Multi-Agent

**Single Agent**: One decision maker
**Multi-Agent**: Multiple decision makers with potentially conflicting goals

## Problem Formulation Examples

### Example 1: 8-Puzzle Problem

The 8-puzzle is a classic sliding tile puzzle with 8 numbered tiles and one empty space.

```python
class EightPuzzleState:
    def __init__(self, board):
        """
        board: 3x3 list representing the puzzle state
        Empty space is represented by 0
        """
        self.board = board
        self.size = 3
    
    def __hash__(self):
        return hash(str(self.board))
    
    def __eq__(self, other):
        return self.board == other.board
    
    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.board])
    
    def get_blank_position(self):
        """Find the position of the empty space (0)"""
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return i, j
        return None
    
    def is_goal(self):
        """Check if the current state is the goal state"""
        goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        return self.board == goal
    
    def get_actions(self):
        """Get all possible actions from current state"""
        actions = []
        i, j = self.get_blank_position()
        
        # Possible moves: up, down, left, right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for di, dj in moves:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size:
                actions.append((ni, nj))
        
        return actions
    
    def apply_action(self, action):
        """Apply an action to get the next state"""
        i, j = self.get_blank_position()
        ni, nj = action
        
        # Create new board
        new_board = [row[:] for row in self.board]
        new_board[i][j], new_board[ni][nj] = new_board[ni][nj], new_board[i][j]
        
        return EightPuzzleState(new_board)

class EightPuzzleProblem:
    def __init__(self, initial_state):
        self.initial_state = initial_state
    
    def get_initial_state(self):
        return self.initial_state
    
    def is_goal_state(self, state):
        return state.is_goal()
    
    def get_actions(self, state):
        return state.get_actions()
    
    def get_next_state(self, state, action):
        return state.apply_action(action)
    
    def get_action_cost(self, state, action, next_state):
        return 1  # Unit cost for each move

# Example usage
initial_board = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
initial_state = EightPuzzleState(initial_board)
problem = EightPuzzleProblem(initial_state)

print("Initial State:")
print(initial_state)
print(f"Is goal: {problem.is_goal_state(initial_state)}")
print(f"Possible actions: {problem.get_actions(initial_state)}")
```

### Example 2: Missionaries and Cannibals

Three missionaries and three cannibals must cross a river using a boat that can hold at most two people.

```python
class MissionariesCannibalsState:
    def __init__(self, missionaries_left, cannibals_left, boat_left):
        """
        missionaries_left: number of missionaries on left bank
        cannibals_left: number of cannibals on left bank
        boat_left: True if boat is on left bank, False otherwise
        """
        self.missionaries_left = missionaries_left
        self.cannibals_left = cannibals_left
        self.boat_left = boat_left
        
        # Calculate right bank
        self.missionaries_right = 3 - missionaries_left
        self.cannibals_right = 3 - cannibals_left
    
    def __hash__(self):
        return hash((self.missionaries_left, self.cannibals_left, self.boat_left))
    
    def __eq__(self, other):
        return (self.missionaries_left == other.missionaries_left and
                self.cannibals_left == other.cannibals_left and
                self.boat_left == other.boat_left)
    
    def __str__(self):
        return f"Left: M={self.missionaries_left}, C={self.cannibals_left}, Boat={'Yes' if self.boat_left else 'No'}"
    
    def is_valid(self):
        """Check if state is valid (no missionaries eaten)"""
        # Missionaries cannot be outnumbered by cannibals on either bank
        if self.missionaries_left > 0 and self.cannibals_left > self.missionaries_left:
            return False
        if self.missionaries_right > 0 and self.cannibals_right > self.missionaries_right:
            return False
        return True
    
    def is_goal(self):
        """Check if all missionaries and cannibals have crossed"""
        return (self.missionaries_left == 0 and 
                self.cannibals_left == 0 and 
                not self.boat_left)
    
    def get_actions(self):
        """Get all possible boat crossings"""
        actions = []
        if self.boat_left:
            # Boat is on left bank, can take people to right
            for m in range(self.missionaries_left + 1):
                for c in range(self.cannibals_left + 1):
                    if m + c >= 1 and m + c <= 2:  # Boat capacity constraint
                        actions.append((m, c))
        else:
            # Boat is on right bank, can take people to left
            for m in range(self.missionaries_right + 1):
                for c in range(self.cannibals_right + 1):
                    if m + c >= 1 and m + c <= 2:  # Boat capacity constraint
                        actions.append((m, c))
        return actions
    
    def apply_action(self, action):
        """Apply a boat crossing action"""
        m, c = action
        if self.boat_left:
            # Moving from left to right
            new_missionaries_left = self.missionaries_left - m
            new_cannibals_left = self.cannibals_left - c
            new_boat_left = False
        else:
            # Moving from right to left
            new_missionaries_left = self.missionaries_left + m
            new_cannibals_left = self.cannibals_left + c
            new_boat_left = True
        
        return MissionariesCannibalsState(new_missionaries_left, new_cannibals_left, new_boat_left)

class MissionariesCannibalsProblem:
    def __init__(self):
        self.initial_state = MissionariesCannibalsState(3, 3, True)
    
    def get_initial_state(self):
        return self.initial_state
    
    def is_goal_state(self, state):
        return state.is_goal()
    
    def get_actions(self, state):
        if not state.is_valid():
            return []
        return state.get_actions()
    
    def get_next_state(self, state, action):
        next_state = state.apply_action(action)
        return next_state if next_state.is_valid() else None
    
    def get_action_cost(self, state, action, next_state):
        return 1  # Unit cost for each crossing

# Example usage
problem = MissionariesCannibalsProblem()
initial_state = problem.get_initial_state()

print("Initial State:")
print(initial_state)
print(f"Is goal: {problem.is_goal_state(initial_state)}")
print(f"Possible actions: {problem.get_actions(initial_state)}")
```

## State Space Analysis

### State Space Size

The size of the state space is crucial for algorithm selection:

```python
def analyze_state_space(problem):
    """Analyze the size and properties of a problem's state space"""
    visited = set()
    queue = [problem.get_initial_state()]
    visited.add(queue[0])
    
    while queue:
        state = queue.pop(0)
        actions = problem.get_actions(state)
        
        for action in actions:
            next_state = problem.get_next_state(state, action)
            if next_state and next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)
    
    return {
        'total_states': len(visited),
        'reachable_states': len(visited),
        'average_branching_factor': sum(len(problem.get_actions(s)) for s in visited) / len(visited)
    }

# Example analysis
eight_puzzle_analysis = analyze_state_space(problem)
print(f"8-Puzzle State Space Analysis: {eight_puzzle_analysis}")
```

### Branching Factor

The branching factor $b$ is the average number of successors per state:

$$b = \frac{\text{Total number of edges}}{\text{Total number of nodes}}$$

## Problem Characteristics

### 1. Adversarial vs. Cooperative

**Adversarial**: Multiple agents with conflicting goals (game playing)
**Cooperative**: Multiple agents working toward common goals

### 2. Real-time vs. Offline

**Real-time**: Must make decisions within time constraints
**Offline**: Can plan complete solution before execution

### 3. Sequential vs. Simultaneous

**Sequential**: Actions taken one at a time
**Simultaneous**: Multiple actions can occur simultaneously

## Best Practices for Problem Modeling

1. **Choose Appropriate State Representation**
   - Include all relevant information
   - Exclude irrelevant details
   - Use efficient data structures

2. **Define Clear Action Space**
   - Ensure completeness (all possible actions included)
   - Avoid redundant actions
   - Consider action costs

3. **Validate State Transitions**
   - Check for invalid states
   - Ensure consistency
   - Handle edge cases

4. **Optimize for Search Algorithms**
   - Use hashable state representations
   - Implement efficient equality checks
   - Consider symmetry and transpositions

## Summary

Effective problem modeling is the foundation of successful search algorithm application. Key considerations include:

- **State representation** that captures all necessary information
- **Action space** that is complete and non-redundant
- **Transition function** that is consistent and efficient
- **Goal conditions** that are clearly defined
- **Cost function** that reflects the problem's objectives

The choice of representation significantly impacts the efficiency and effectiveness of search algorithms, making careful modeling essential for solving complex problems. 