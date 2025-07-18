# Definitions in Constraint Satisfaction Problems

## Core Mathematical Concepts

### Variables and Domains

In a CSP, we work with a set of variables, each of which can take values from a specified domain.

**Definition**: Let X = {X₁, X₂, ..., Xₙ} be a finite set of variables, and let D = {D₁, D₂, ..., Dₙ} be a set of domains where Dᵢ is the domain of variable Xᵢ.

```python
from typing import List, Dict, Set, Any, Tuple
from dataclasses import dataclass

@dataclass
class Variable:
    """Represents a variable in a CSP."""
    name: str
    domain: List[Any]
    
    def __post_init__(self):
        if not self.domain:
            raise ValueError(f"Domain for variable {self.name} cannot be empty")
    
    def get_domain_size(self) -> int:
        """Return the size of the variable's domain."""
        return len(self.domain)
    
    def has_value(self, value: Any) -> bool:
        """Check if a value is in the variable's domain."""
        return value in self.domain

class VariableSet:
    """Represents a set of variables with their domains."""
    
    def __init__(self):
        self.variables: Dict[str, Variable] = {}
    
    def add_variable(self, name: str, domain: List[Any]):
        """Add a variable to the set."""
        self.variables[name] = Variable(name, domain)
    
    def get_variable(self, name: str) -> Variable:
        """Get a variable by name."""
        return self.variables[name]
    
    def get_all_variables(self) -> List[str]:
        """Get all variable names."""
        return list(self.variables.keys())
    
    def get_domain(self, name: str) -> List[Any]:
        """Get the domain of a variable."""
        return self.variables[name].domain
    
    def get_total_domain_size(self) -> int:
        """Get the total size of all domains combined."""
        return sum(var.get_domain_size() for var in self.variables.values())

# Example usage
def create_variable_set_example():
    """Create an example variable set."""
    var_set = VariableSet()
    
    # Add variables for a simple scheduling problem
    var_set.add_variable("Task1", ["morning", "afternoon", "evening"])
    var_set.add_variable("Task2", ["morning", "afternoon", "evening"])
    var_set.add_variable("Task3", ["morning", "afternoon"])
    
    print("Variables:", var_set.get_all_variables())
    print("Domain for Task1:", var_set.get_domain("Task1"))
    print("Total domain size:", var_set.get_total_domain_size())
    
    return var_set
```

### Constraints

Constraints are the heart of CSPs. They define which combinations of values are allowed.

**Definition**: A constraint C is a pair (scope, relation) where:
- **scope** is a subset of variables {Xᵢ₁, Xᵢ₂, ..., Xᵢₖ}
- **relation** is a subset of Dᵢ₁ × Dᵢ₂ × ... × Dᵢₖ

```python
@dataclass
class Constraint:
    """Represents a constraint in a CSP."""
    scope: List[str]
    relation: Set[Tuple[Any, ...]]
    
    def __post_init__(self):
        if not self.scope:
            raise ValueError("Constraint scope cannot be empty")
        if not self.relation:
            raise ValueError("Constraint relation cannot be empty")
        
        # Validate that relation tuples have correct arity
        arity = len(self.scope)
        for tuple_val in self.relation:
            if len(tuple_val) != arity:
                raise ValueError(f"Tuple {tuple_val} has wrong arity for constraint with {arity} variables")
    
    def get_arity(self) -> int:
        """Return the arity of the constraint (number of variables)."""
        return len(self.scope)
    
    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        """Check if the constraint is satisfied by the given assignment."""
        # If not all variables in scope are assigned, constraint is satisfied
        if not all(var in assignment for var in self.scope):
            return True
        
        # Extract values for variables in scope
        values = tuple(assignment[var] for var in self.scope)
        return values in self.relation
    
    def get_supporting_values(self, var: str, partial_assignment: Dict[str, Any]) -> Set[Any]:
        """Get values for variable var that satisfy this constraint given partial_assignment."""
        if var not in self.scope:
            return set()  # Variable not in this constraint's scope
        
        supporting_values = set()
        
        # Get the domain of the variable (this would come from the CSP)
        # For now, we'll assume we can get it from the partial assignment context
        for tuple_val in self.relation:
            # Check if this tuple is compatible with the partial assignment
            compatible = True
            for i, scope_var in enumerate(self.scope):
                if scope_var in partial_assignment:
                    if partial_assignment[scope_var] != tuple_val[i]:
                        compatible = False
                        break
            
            if compatible:
                # Find the value for our variable in this tuple
                var_index = self.scope.index(var)
                supporting_values.add(tuple_val[var_index])
        
        return supporting_values

class ConstraintSet:
    """Represents a set of constraints."""
    
    def __init__(self):
        self.constraints: List[Constraint] = []
        self.constraint_graph: Dict[str, Set[str]] = {}  # Adjacency list
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the set."""
        self.constraints.append(constraint)
        
        # Update constraint graph
        for var in constraint.scope:
            if var not in self.constraint_graph:
                self.constraint_graph[var] = set()
            for other_var in constraint.scope:
                if other_var != var:
                    self.constraint_graph[var].add(other_var)
    
    def get_constraints_involving(self, var: str) -> List[Constraint]:
        """Get all constraints that involve a specific variable."""
        return [c for c in self.constraints if var in c.scope]
    
    def get_neighbors(self, var: str) -> Set[str]:
        """Get variables that are constrained with the given variable."""
        return self.constraint_graph.get(var, set())
    
    def get_constraint_density(self) -> float:
        """Calculate the density of the constraint graph."""
        total_vars = len(self.constraint_graph)
        if total_vars <= 1:
            return 0.0
        
        total_edges = sum(len(neighbors) for neighbors in self.constraint_graph.values()) // 2
        max_edges = total_vars * (total_vars - 1) // 2
        return total_edges / max_edges if max_edges > 0 else 0.0

# Example: Creating different types of constraints
def create_constraint_examples():
    """Create examples of different types of constraints."""
    constraint_set = ConstraintSet()
    
    # Binary constraint: X ≠ Y
    binary_constraint = Constraint(
        scope=["X", "Y"],
        relation={(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)}
    )
    constraint_set.add_constraint(binary_constraint)
    
    # Unary constraint: X ≠ 1
    unary_constraint = Constraint(
        scope=["X"],
        relation={(2,), (3,)}
    )
    constraint_set.add_constraint(unary_constraint)
    
    # Global constraint: AllDifferent(X, Y, Z)
    global_constraint = Constraint(
        scope=["X", "Y", "Z"],
        relation={(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)}
    )
    constraint_set.add_constraint(global_constraint)
    
    print("Constraint density:", constraint_set.get_constraint_density())
    print("Neighbors of X:", constraint_set.get_neighbors("X"))
    
    return constraint_set
```

### Assignments and Solutions

**Definition**: An assignment A is a function A: X → ∪D such that A(Xᵢ) ∈ Dᵢ for all variables Xᵢ.

**Definition**: A solution is an assignment that satisfies all constraints.

```python
@dataclass
class Assignment:
    """Represents an assignment of values to variables."""
    values: Dict[str, Any]
    
    def __post_init__(self):
        if self.values is None:
            self.values = {}
    
    def assign(self, var: str, value: Any):
        """Assign a value to a variable."""
        self.values[var] = value
    
    def unassign(self, var: str):
        """Remove assignment for a variable."""
        if var in self.values:
            del self.values[var]
    
    def get_value(self, var: str) -> Any:
        """Get the value assigned to a variable."""
        return self.values.get(var)
    
    def is_assigned(self, var: str) -> bool:
        """Check if a variable is assigned a value."""
        return var in self.values
    
    def get_assigned_variables(self) -> List[str]:
        """Get list of assigned variables."""
        return list(self.values.keys())
    
    def is_complete(self, all_variables: List[str]) -> bool:
        """Check if all variables are assigned."""
        return all(var in self.values for var in all_variables)
    
    def copy(self) -> 'Assignment':
        """Create a copy of this assignment."""
        return Assignment(self.values.copy())

class CSPSolution:
    """Represents a complete CSP solution."""
    
    def __init__(self, csp, assignment: Assignment):
        self.csp = csp
        self.assignment = assignment
    
    def is_valid(self) -> bool:
        """Check if this is a valid solution."""
        if not self.assignment.is_complete(self.csp.variables):
            return False
        
        for constraint in self.csp.constraints:
            if not constraint.is_satisfied(self.assignment.values):
                return False
        
        return True
    
    def get_violated_constraints(self) -> List[Constraint]:
        """Get list of constraints that are violated by this solution."""
        violated = []
        for constraint in self.csp.constraints:
            if not constraint.is_satisfied(self.assignment.values):
                violated.append(constraint)
        return violated
    
    def get_violation_count(self) -> int:
        """Count the number of constraint violations."""
        return len(self.get_violated_constraints())
```

## Formal CSP Definition

### Complete Mathematical Definition

A Constraint Satisfaction Problem is formally defined as a 4-tuple (X, D, C, V) where:

- **X** = {X₁, X₂, ..., Xₙ} is a finite set of variables
- **D** = {D₁, D₂, ..., Dₙ} is a set of domains where Dᵢ ⊆ V and Dᵢ is the domain of Xᵢ
- **C** = {C₁, C₂, ..., Cₘ} is a set of constraints
- **V** is the universal set of values

Each constraint Cⱼ = (scopeⱼ, relationⱼ) where:
- scopeⱼ ⊆ X is the scope of the constraint
- relationⱼ ⊆ ∏_{Xᵢ ∈ scopeⱼ} Dᵢ is the relation of the constraint

```python
class FormalCSP:
    """Formal implementation of a CSP with mathematical rigor."""
    
    def __init__(self, variables: List[str], universal_values: Set[Any]):
        self.variables = variables
        self.universal_values = universal_values
        self.domains: Dict[str, Set[Any]] = {}
        self.constraints: List[Constraint] = []
        
        # Initialize all domains to universal values
        for var in variables:
            self.domains[var] = universal_values.copy()
    
    def set_domain(self, var: str, domain: Set[Any]):
        """Set the domain of a variable."""
        if var not in self.variables:
            raise ValueError(f"Variable {var} not in CSP")
        
        # Ensure domain is subset of universal values
        if not domain.issubset(self.universal_values):
            raise ValueError(f"Domain must be subset of universal values")
        
        self.domains[var] = domain.copy()
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the CSP."""
        # Validate constraint scope
        if not all(var in self.variables for var in constraint.scope):
            raise ValueError("Constraint scope contains variables not in CSP")
        
        # Validate constraint relation
        for tuple_val in constraint.relation:
            if len(tuple_val) != len(constraint.scope):
                raise ValueError("Constraint relation tuples have wrong arity")
            
            # Check that values are in appropriate domains
            for i, value in enumerate(tuple_val):
                var = constraint.scope[i]
                if value not in self.domains[var]:
                    raise ValueError(f"Value {value} not in domain of {var}")
        
        self.constraints.append(constraint)
    
    def get_domain_product(self, scope: List[str]) -> Set[Tuple[Any, ...]]:
        """Get the Cartesian product of domains for given scope."""
        domain_lists = [list(self.domains[var]) for var in scope]
        
        def cartesian_product(lists):
            if not lists:
                return {()}
            result = set()
            for item in lists[0]:
                for sub_product in cartesian_product(lists[1:]):
                    result.add((item,) + sub_product)
            return result
        
        return cartesian_product(domain_lists)
    
    def is_arc_consistent(self) -> bool:
        """Check if the CSP is arc consistent."""
        for constraint in self.constraints:
            if len(constraint.scope) == 2:  # Binary constraint
                var1, var2 = constraint.scope
                
                # Check if every value in domain of var1 has support in var2
                for val1 in self.domains[var1]:
                    has_support = False
                    for val2 in self.domains[var2]:
                        if (val1, val2) in constraint.relation:
                            has_support = True
                            break
                    if not has_support:
                        return False
                
                # Check if every value in domain of var2 has support in var1
                for val2 in self.domains[var2]:
                    has_support = False
                    for val1 in self.domains[var1]:
                        if (val1, val2) in constraint.relation:
                            has_support = True
                            break
                    if not has_support:
                        return False
        
        return True
    
    def get_solution_space_size(self) -> int:
        """Calculate the size of the solution space."""
        total = 1
        for var in self.variables:
            total *= len(self.domains[var])
        return total

# Example: Creating a formal CSP
def create_formal_csp_example():
    """Create an example of a formal CSP."""
    variables = ["A", "B", "C"]
    universal_values = {1, 2, 3, 4}
    
    csp = FormalCSP(variables, universal_values)
    
    # Set domains
    csp.set_domain("A", {1, 2, 3})
    csp.set_domain("B", {2, 3, 4})
    csp.set_domain("C", {1, 3, 4})
    
    # Add constraints
    # A ≠ B
    csp.add_constraint(Constraint(
        scope=["A", "B"],
        relation={(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 2), (3, 4)}
    ))
    
    # A + B = C
    csp.add_constraint(Constraint(
        scope=["A", "B", "C"],
        relation={(1, 2, 3), (1, 3, 4), (2, 1, 3), (2, 2, 4), (3, 1, 4)}
    ))
    
    print("Solution space size:", csp.get_solution_space_size())
    print("Is arc consistent:", csp.is_arc_consistent())
    
    return csp
```

## Constraint Types and Properties

### Constraint Arity

The **arity** of a constraint is the number of variables in its scope.

```python
def analyze_constraint_types(constraints: List[Constraint]) -> Dict[int, int]:
    """Analyze the distribution of constraint arities."""
    arity_counts = {}
    for constraint in constraints:
        arity = constraint.get_arity()
        arity_counts[arity] = arity_counts.get(arity, 0) + 1
    return arity_counts

def is_binary_csp(constraints: List[Constraint]) -> bool:
    """Check if all constraints are binary (arity ≤ 2)."""
    return all(c.get_arity() <= 2 for c in constraints)
```

### Constraint Tightness

The **tightness** of a constraint measures how restrictive it is.

```python
def calculate_constraint_tightness(constraint: Constraint, domains: Dict[str, List[Any]]) -> float:
    """Calculate the tightness of a constraint."""
    # Get the size of the constraint relation
    relation_size = len(constraint.relation)
    
    # Calculate the size of the Cartesian product of domains
    domain_product_size = 1
    for var in constraint.scope:
        domain_product_size *= len(domains[var])
    
    # Tightness = 1 - (relation_size / domain_product_size)
    return 1.0 - (relation_size / domain_product_size)

def analyze_csp_tightness(csp) -> Dict[Constraint, float]:
    """Analyze the tightness of all constraints in a CSP."""
    tightness = {}
    for constraint in csp.constraints:
        tightness[constraint] = calculate_constraint_tightness(constraint, csp.domains)
    return tightness
```

## Summary

The formal definitions and mathematical foundations of CSPs provide:

1. **Precise Language**: Clear mathematical notation for describing problems
2. **Algorithmic Basis**: Foundation for developing solution algorithms
3. **Complexity Analysis**: Framework for analyzing problem difficulty
4. **Implementation Guide**: Clear structure for building CSP solvers

Understanding these definitions is essential for:
- Modeling real-world problems as CSPs
- Implementing efficient solution algorithms
- Analyzing problem characteristics and difficulty
- Developing specialized solving techniques 