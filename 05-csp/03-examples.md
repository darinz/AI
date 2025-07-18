# Examples of Constraint Satisfaction Problems

## Introduction

This guide presents classic examples of Constraint Satisfaction Problems, demonstrating how real-world problems can be modeled and solved using the CSP framework. Each example includes problem description, mathematical formulation, and complete Python implementation.

## 1. Map Coloring Problem

### Problem Description

The map coloring problem is one of the most famous CSP examples. Given a map with regions, assign colors to each region so that no adjacent regions have the same color. The goal is to use the minimum number of colors.

### Mathematical Formulation

**Variables**: X = {X₁, X₂, ..., Xₙ} where Xᵢ represents region i
**Domains**: Dᵢ = {red, green, blue, ...} for all i
**Constraints**: Xᵢ ≠ Xⱼ for all adjacent regions (i, j)

### Python Implementation

```python
from typing import List, Dict, Set, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import networkx as nx

@dataclass
class MapRegion:
    """Represents a region in a map."""
    name: str
    neighbors: List[str]

class MapColoringCSP:
    """CSP for map coloring problems."""
    
    def __init__(self, regions: List[MapRegion], colors: List[str]):
        self.regions = {r.name: r for r in regions}
        self.colors = colors
        self.variables = list(self.regions.keys())
        self.domains = {var: colors.copy() for var in self.variables}
        self.constraints = self._create_constraints()
    
    def _create_constraints(self) -> List[Tuple[str, str]]:
        """Create adjacency constraints."""
        constraints = []
        for region in self.regions.values():
            for neighbor in region.neighbors:
                # Add constraint only once (avoid duplicates)
                if (neighbor, region.name) not in constraints:
                    constraints.append((region.name, neighbor))
        return constraints
    
    def is_valid_coloring(self, assignment: Dict[str, str]) -> bool:
        """Check if a coloring assignment is valid."""
        for var1, var2 in self.constraints:
            if var1 in assignment and var2 in assignment:
                if assignment[var1] == assignment[var2]:
                    return False
        return True
    
    def get_conflicts(self, assignment: Dict[str, str]) -> List[Tuple[str, str]]:
        """Get list of conflicting adjacent regions."""
        conflicts = []
        for var1, var2 in self.constraints:
            if var1 in assignment and var2 in assignment:
                if assignment[var1] == assignment[var2]:
                    conflicts.append((var1, var2))
        return conflicts
    
    def visualize_coloring(self, assignment: Dict[str, str] = None):
        """Visualize the map coloring using networkx."""
        G = nx.Graph()
        
        # Add nodes
        for region_name in self.regions.keys():
            G.add_node(region_name)
        
        # Add edges
        for var1, var2 in self.constraints:
            G.add_edge(var1, var2)
        
        # Set up the plot
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw the graph
        if assignment:
            # Color nodes based on assignment
            node_colors = [assignment.get(node, 'white') for node in G.nodes()]
            nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                   node_size=2000, font_size=10, font_weight='bold')
        else:
            nx.draw(G, pos, with_labels=True, node_size=2000, 
                   font_size=10, font_weight='bold')
        
        plt.title("Map Coloring Problem")
        plt.show()

# Australia Map Coloring Example
def create_australia_map():
    """Create the classic Australia map coloring problem."""
    regions = [
        MapRegion("WA", ["NT", "SA"]),
        MapRegion("NT", ["WA", "SA", "Q"]),
        MapRegion("SA", ["WA", "NT", "Q", "NSW", "V"]),
        MapRegion("Q", ["NT", "SA", "NSW"]),
        MapRegion("NSW", ["SA", "Q", "V"]),
        MapRegion("V", ["SA", "NSW"]),
        MapRegion("T", [])  # Tasmania has no neighbors
    ]
    
    colors = ["red", "green", "blue"]
    return MapColoringCSP(regions, colors)

def solve_map_coloring_backtracking(csp: MapColoringCSP) -> Dict[str, str]:
    """Solve map coloring using backtracking."""
    
    def backtrack(assignment: Dict[str, str]) -> Dict[str, str]:
        if len(assignment) == len(csp.variables):
            return assignment
        
        # Select unassigned variable
        unassigned = [var for var in csp.variables if var not in assignment]
        var = unassigned[0]
        
        # Try each color
        for color in csp.domains[var]:
            assignment[var] = color
            if csp.is_valid_coloring(assignment):
                result = backtrack(assignment)
                if result is not None:
                    return result
            del assignment[var]
        
        return None
    
    return backtrack({})

# Test the implementation
def test_map_coloring():
    """Test the map coloring implementation."""
    csp = create_australia_map()
    
    print("Australia Map Coloring Problem")
    print("Variables:", csp.variables)
    print("Colors:", csp.colors)
    print("Constraints:", csp.constraints)
    
    # Solve the problem
    solution = solve_map_coloring_backtracking(csp)
    
    if solution:
        print("\nSolution found:")
        for region, color in solution.items():
            print(f"{region}: {color}")
        
        print("\nConflicts:", csp.get_conflicts(solution))
        csp.visualize_coloring(solution)
    else:
        print("No solution found")

if __name__ == "__main__":
    test_map_coloring()
```

## 2. N-Queens Problem

### Problem Description

Place N queens on an N×N chessboard so that no two queens threaten each other. A queen can move horizontally, vertically, and diagonally.

### Mathematical Formulation

**Variables**: X = {X₁, X₂, ..., Xₙ} where Xᵢ represents the column position of queen in row i
**Domains**: Dᵢ = {1, 2, ..., N} for all i
**Constraints**: 
- Xᵢ ≠ Xⱼ (no two queens in same column)
- |Xᵢ - Xⱼ| ≠ |i - j| (no two queens on same diagonal)

### Python Implementation

```python
import numpy as np
from typing import List, Dict, Set, Optional

class NQueensCSP:
    """CSP for the N-Queens problem."""
    
    def __init__(self, n: int):
        self.n = n
        self.variables = [f"Q{i}" for i in range(n)]
        self.domains = {var: list(range(n)) for var in self.variables}
        self.constraints = self._create_constraints()
    
    def _create_constraints(self) -> List[Tuple[str, str]]:
        """Create constraints for the N-Queens problem."""
        constraints = []
        
        # Add constraints for all pairs of queens
        for i in range(self.n):
            for j in range(i + 1, self.n):
                constraints.append((f"Q{i}", f"Q{j}"))
        
        return constraints
    
    def is_valid_placement(self, assignment: Dict[str, int]) -> bool:
        """Check if queen placement is valid."""
        for var1, var2 in self.constraints:
            if var1 in assignment and var2 in assignment:
                col1, col2 = assignment[var1], assignment[var2]
                row1 = int(var1[1:])
                row2 = int(var2[1:])
                
                # Check same column
                if col1 == col2:
                    return False
                
                # Check same diagonal
                if abs(col1 - col2) == abs(row1 - row2):
                    return False
        
        return True
    
    def visualize_board(self, assignment: Dict[str, int]):
        """Visualize the chessboard with queens."""
        board = np.zeros((self.n, self.n), dtype=int)
        
        for var, col in assignment.items():
            row = int(var[1:])
            board[row, col] = 1
        
        print("Chessboard (1 = Queen, 0 = Empty):")
        print(board)
        
        # Visual representation
        for row in range(self.n):
            line = ""
            for col in range(self.n):
                if board[row, col] == 1:
                    line += "Q "
                else:
                    line += ". "
            print(line)

def solve_n_queens_backtracking(csp: NQueensCSP) -> Optional[Dict[str, int]]:
    """Solve N-Queens using backtracking with MRV heuristic."""
    
    def get_mrv_variable(assignment: Dict[str, int]) -> str:
        """Get variable with minimum remaining values."""
        unassigned = [var for var in csp.variables if var not in assignment]
        if not unassigned:
            return None
        
        # Count legal values for each unassigned variable
        var_counts = {}
        for var in unassigned:
            legal_values = 0
            for value in csp.domains[var]:
                test_assignment = assignment.copy()
                test_assignment[var] = value
                if csp.is_valid_placement(test_assignment):
                    legal_values += 1
            var_counts[var] = legal_values
        
        # Return variable with minimum legal values
        return min(var_counts.keys(), key=lambda v: var_counts[v])
    
    def backtrack(assignment: Dict[str, int]) -> Optional[Dict[str, int]]:
        if len(assignment) == len(csp.variables):
            return assignment
        
        var = get_mrv_variable(assignment)
        if var is None:
            return None
        
        # Try values in order (could add LCV heuristic here)
        for value in csp.domains[var]:
            assignment[var] = value
            if csp.is_valid_placement(assignment):
                result = backtrack(assignment)
                if result is not None:
                    return result
            del assignment[var]
        
        return None
    
    return backtrack({})

def solve_n_queens_all_solutions(csp: NQueensCSP) -> List[Dict[str, int]]:
    """Find all solutions to the N-Queens problem."""
    solutions = []
    
    def backtrack_all(assignment: Dict[str, int]):
        if len(assignment) == len(csp.variables):
            solutions.append(assignment.copy())
            return
        
        var = f"Q{len(assignment)}"  # Simple variable ordering
        
        for value in csp.domains[var]:
            assignment[var] = value
            if csp.is_valid_placement(assignment):
                backtrack_all(assignment)
            del assignment[var]
    
    backtrack_all({})
    return solutions

# Test the implementation
def test_n_queens():
    """Test the N-Queens implementation."""
    for n in [4, 8]:
        print(f"\n=== {n}-Queens Problem ===")
        csp = NQueensCSP(n)
        
        print(f"Variables: {csp.variables}")
        print(f"Domain size: {len(csp.domains[csp.variables[0]])}")
        print(f"Number of constraints: {len(csp.constraints)}")
        
        # Find one solution
        solution = solve_n_queens_backtracking(csp)
        
        if solution:
            print(f"\nSolution found:")
            for var, value in solution.items():
                print(f"{var}: column {value}")
            
            csp.visualize_board(solution)
        else:
            print("No solution found")
        
        # Find all solutions for small n
        if n <= 8:
            all_solutions = solve_n_queens_all_solutions(csp)
            print(f"\nTotal solutions: {len(all_solutions)}")

if __name__ == "__main__":
    test_n_queens()
```

## 3. Sudoku Problem

### Problem Description

Fill a 9×9 grid with digits 1-9 so that each row, column, and 3×3 box contains each digit exactly once.

### Mathematical Formulation

**Variables**: X = {Xᵢⱼ} where Xᵢⱼ represents cell at row i, column j
**Domains**: Dᵢⱼ = {1, 2, ..., 9} for empty cells, Dᵢⱼ = {v} for filled cells
**Constraints**: 
- AllDifferent(Xᵢ₁, Xᵢ₂, ..., Xᵢ₉) for each row i
- AllDifferent(X₁ⱼ, X₂ⱼ, ..., X₉ⱼ) for each column j
- AllDifferent(Xᵢⱼ, Xᵢⱼ₊₁, ..., Xᵢ₊₂ⱼ₊₂) for each 3×3 box

### Python Implementation

```python
from typing import List, Dict, Set, Optional, Tuple
import numpy as np

class SudokuCSP:
    """CSP for Sudoku puzzles."""
    
    def __init__(self, initial_board: List[List[int]]):
        self.size = 9
        self.box_size = 3
        self.initial_board = np.array(initial_board)
        self.variables = self._create_variables()
        self.domains = self._create_domains()
        self.constraints = self._create_constraints()
    
    def _create_variables(self) -> List[str]:
        """Create variable names for each cell."""
        variables = []
        for row in range(self.size):
            for col in range(self.size):
                variables.append(f"R{row}C{col}")
        return variables
    
    def _create_domains(self) -> Dict[str, List[int]]:
        """Create domains for each variable."""
        domains = {}
        for row in range(self.size):
            for col in range(self.size):
                var = f"R{row}C{col}"
                if self.initial_board[row, col] == 0:
                    domains[var] = list(range(1, 10))
                else:
                    domains[var] = [self.initial_board[row, col]]
        return domains
    
    def _create_constraints(self) -> List[List[str]]:
        """Create row, column, and box constraints."""
        constraints = []
        
        # Row constraints
        for row in range(self.size):
            row_vars = [f"R{row}C{col}" for col in range(self.size)]
            constraints.append(row_vars)
        
        # Column constraints
        for col in range(self.size):
            col_vars = [f"R{row}C{col}" for row in range(self.size)]
            constraints.append(col_vars)
        
        # Box constraints
        for box_row in range(0, self.size, self.box_size):
            for box_col in range(0, self.size, self.box_size):
                box_vars = []
                for row in range(box_row, box_row + self.box_size):
                    for col in range(box_col, box_col + self.box_size):
                        box_vars.append(f"R{row}C{col}")
                constraints.append(box_vars)
        
        return constraints
    
    def is_valid_assignment(self, assignment: Dict[str, int]) -> bool:
        """Check if assignment satisfies all constraints."""
        for constraint_vars in self.constraints:
            values = []
            for var in constraint_vars:
                if var in assignment:
                    values.append(assignment[var])
            
            # Check for duplicates
            if len(values) != len(set(values)):
                return False
        
        return True
    
    def get_board_from_assignment(self, assignment: Dict[str, int]) -> np.ndarray:
        """Convert assignment to board representation."""
        board = np.zeros((self.size, self.size), dtype=int)
        
        for var, value in assignment.items():
            row = int(var[1])
            col = int(var[3])
            board[row, col] = value
        
        return board
    
    def print_board(self, board: np.ndarray):
        """Print the Sudoku board."""
        for row in range(self.size):
            if row % 3 == 0 and row != 0:
                print("-" * 21)
            
            for col in range(self.size):
                if col % 3 == 0 and col != 0:
                    print("|", end=" ")
                
                if board[row, col] == 0:
                    print(".", end=" ")
                else:
                    print(board[row, col], end=" ")
            print()

def solve_sudoku_backtracking(csp: SudokuCSP) -> Optional[Dict[str, int]]:
    """Solve Sudoku using backtracking with forward checking."""
    
    def get_next_variable(assignment: Dict[str, int]) -> str:
        """Get next unassigned variable (simple ordering)."""
        for var in csp.variables:
            if var not in assignment:
                return var
        return None
    
    def forward_check(var: str, value: int, assignment: Dict[str, int]) -> bool:
        """Check if assigning value to var would leave any variable with empty domain."""
        test_assignment = assignment.copy()
        test_assignment[var] = value
        
        # Check each constraint involving var
        for constraint_vars in csp.constraints:
            if var in constraint_vars:
                # Count how many values are still possible for other variables in constraint
                for other_var in constraint_vars:
                    if other_var != var and other_var not in test_assignment:
                        legal_values = 0
                        for other_value in csp.domains[other_var]:
                            test_assignment[other_var] = other_value
                            if csp.is_valid_assignment(test_assignment):
                                legal_values += 1
                            del test_assignment[other_var]
                        
                        if legal_values == 0:
                            return False
        
        return True
    
    def backtrack(assignment: Dict[str, int]) -> Optional[Dict[str, int]]:
        if len(assignment) == len(csp.variables):
            return assignment
        
        var = get_next_variable(assignment)
        if var is None:
            return None
        
        # Try each value in domain
        for value in csp.domains[var]:
            assignment[var] = value
            if csp.is_valid_assignment(assignment) and forward_check(var, value, assignment):
                result = backtrack(assignment)
                if result is not None:
                    return result
            del assignment[var]
        
        return None
    
    return backtrack({})

# Example Sudoku puzzles
def create_easy_sudoku() -> List[List[int]]:
    """Create an easy Sudoku puzzle."""
    return [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

def test_sudoku():
    """Test the Sudoku implementation."""
    print("=== Sudoku Problem ===")
    
    # Create puzzle
    initial_board = create_easy_sudoku()
    csp = SudokuCSP(initial_board)
    
    print("Initial board:")
    csp.print_board(initial_board)
    
    print(f"\nVariables: {len(csp.variables)}")
    print(f"Constraints: {len(csp.constraints)}")
    
    # Solve the puzzle
    solution = solve_sudoku_backtracking(csp)
    
    if solution:
        print("\nSolution:")
        solution_board = csp.get_board_from_assignment(solution)
        csp.print_board(solution_board)
        
        # Verify solution
        print(f"\nSolution is valid: {csp.is_valid_assignment(solution)}")
    else:
        print("No solution found")

if __name__ == "__main__":
    test_sudoku()
```

## 4. Scheduling Problems

### Problem Description

Schedule a set of tasks with given durations, precedence constraints, and resource requirements to minimize completion time or satisfy other objectives.

### Mathematical Formulation

**Variables**: 
- Xᵢ = start time of task i
- Yᵢⱼ = 1 if task i uses resource j, 0 otherwise

**Domains**: 
- Xᵢ ∈ [0, T_max] where T_max is maximum completion time
- Yᵢⱼ ∈ {0, 1}

**Constraints**:
- Xᵢ + durationᵢ ≤ Xⱼ for precedence i → j
- Resource capacity constraints
- Time window constraints

### Python Implementation

```python
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
import heapq

@dataclass
class Task:
    """Represents a task in a scheduling problem."""
    id: str
    duration: int
    predecessors: List[str]
    resources: List[str]
    earliest_start: int = 0
    latest_start: int = None

class SchedulingCSP:
    """CSP for task scheduling problems."""
    
    def __init__(self, tasks: List[Task], resources: Dict[str, int]):
        self.tasks = {task.id: task for task in tasks}
        self.resources = resources
        self.variables = list(self.tasks.keys())
        self.domains = self._create_domains()
        self.constraints = self._create_constraints()
    
    def _create_domains(self) -> Dict[str, List[int]]:
        """Create domains for start times."""
        domains = {}
        
        # Calculate earliest and latest start times
        for task_id, task in self.tasks.items():
            earliest = task.earliest_start
            latest = task.latest_start or 1000  # Default large value
            
            domains[task_id] = list(range(earliest, latest + 1))
        
        return domains
    
    def _create_constraints(self) -> List[Tuple[str, str]]:
        """Create precedence constraints."""
        constraints = []
        
        for task_id, task in self.tasks.items():
            for pred_id in task.predecessors:
                constraints.append((pred_id, task_id))
        
        return constraints
    
    def is_valid_schedule(self, assignment: Dict[str, int]) -> bool:
        """Check if schedule satisfies all constraints."""
        # Check precedence constraints
        for pred_id, succ_id in self.constraints:
            if pred_id in assignment and succ_id in assignment:
                pred_task = self.tasks[pred_id]
                pred_end = assignment[pred_id] + pred_task.duration
                succ_start = assignment[succ_id]
                
                if pred_end > succ_start:
                    return False
        
        # Check resource constraints
        if not self._check_resource_constraints(assignment):
            return False
        
        return True
    
    def _check_resource_constraints(self, assignment: Dict[str, int]) -> bool:
        """Check if resource constraints are satisfied."""
        # For each time step, check resource usage
        max_time = max(assignment.values()) + max(task.duration for task in self.tasks.values())
        
        for time in range(max_time + 1):
            resource_usage = {res: 0 for res in self.resources.keys()}
            
            for task_id, start_time in assignment.items():
                task = self.tasks[task_id]
                if start_time <= time < start_time + task.duration:
                    for resource in task.resources:
                        resource_usage[resource] += 1
            
            # Check if any resource is over-allocated
            for resource, usage in resource_usage.items():
                if usage > self.resources[resource]:
                    return False
        
        return True
    
    def get_completion_time(self, assignment: Dict[str, int]) -> int:
        """Calculate the completion time of the schedule."""
        if not assignment:
            return 0
        
        completion_times = []
        for task_id, start_time in assignment.items():
            task = self.tasks[task_id]
            completion_times.append(start_time + task.duration)
        
        return max(completion_times)

def solve_scheduling_backtracking(csp: SchedulingCSP) -> Optional[Dict[str, int]]:
    """Solve scheduling problem using backtracking."""
    
    def get_critical_path_order() -> List[str]:
        """Get topological order based on precedence constraints."""
        # Simple topological sort
        in_degree = {task_id: 0 for task_id in csp.variables}
        
        for pred_id, succ_id in csp.constraints:
            in_degree[succ_id] += 1
        
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            task_id = queue.pop(0)
            order.append(task_id)
            
            for pred_id, succ_id in csp.constraints:
                if pred_id == task_id:
                    in_degree[succ_id] -= 1
                    if in_degree[succ_id] == 0:
                        queue.append(succ_id)
        
        return order
    
    def backtrack(assignment: Dict[str, int], task_order: List[str], index: int) -> Optional[Dict[str, int]]:
        if index >= len(task_order):
            return assignment
        
        task_id = task_order[index]
        
        # Try each possible start time
        for start_time in csp.domains[task_id]:
            assignment[task_id] = start_time
            if csp.is_valid_schedule(assignment):
                result = backtrack(assignment, task_order, index + 1)
                if result is not None:
                    return result
            del assignment[task_id]
        
        return None
    
    task_order = get_critical_path_order()
    return backtrack({}, task_order, 0)

# Example scheduling problem
def create_scheduling_example():
    """Create an example scheduling problem."""
    tasks = [
        Task("A", 3, [], ["R1"]),
        Task("B", 2, ["A"], ["R1"]),
        Task("C", 4, ["A"], ["R2"]),
        Task("D", 1, ["B", "C"], ["R1", "R2"]),
        Task("E", 2, ["D"], ["R1"])
    ]
    
    resources = {"R1": 1, "R2": 1}
    
    return SchedulingCSP(tasks, resources)

def test_scheduling():
    """Test the scheduling implementation."""
    print("=== Task Scheduling Problem ===")
    
    csp = create_scheduling_example()
    
    print("Tasks:")
    for task_id, task in csp.tasks.items():
        print(f"  {task_id}: duration={task.duration}, preds={task.predecessors}, resources={task.resources}")
    
    print(f"\nResources: {csp.resources}")
    print(f"Precedence constraints: {csp.constraints}")
    
    # Solve the problem
    solution = solve_scheduling_backtracking(csp)
    
    if solution:
        print("\nSolution:")
        for task_id, start_time in solution.items():
            task = csp.tasks[task_id]
            end_time = start_time + task.duration
            print(f"  {task_id}: {start_time} -> {end_time}")
        
        completion_time = csp.get_completion_time(solution)
        print(f"\nCompletion time: {completion_time}")
        print(f"Solution is valid: {csp.is_valid_schedule(solution)}")
    else:
        print("No solution found")

if __name__ == "__main__":
    test_scheduling()
```

## Summary

These examples demonstrate the versatility of the CSP framework:

1. **Map Coloring**: Shows how spatial relationships can be modeled as constraints
2. **N-Queens**: Demonstrates complex constraint patterns and multiple solution finding
3. **Sudoku**: Illustrates global constraints and forward checking
4. **Scheduling**: Shows how temporal and resource constraints can be combined

Key insights from these examples:

- **Problem Modeling**: The choice of variables and constraints significantly affects solution efficiency
- **Constraint Types**: Different problems require different constraint representations
- **Solution Methods**: The same CSP framework can be solved with various algorithms
- **Heuristics**: Variable and value ordering heuristics can dramatically improve performance

Understanding these classic examples provides a foundation for modeling and solving more complex real-world problems using CSPs. 