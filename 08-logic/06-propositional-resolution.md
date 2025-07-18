# Propositional Resolution

## Introduction

Resolution is a powerful inference rule that forms the foundation of automated theorem proving and logic programming. It is a complete and sound method for determining the satisfiability of propositional formulas and is the basis for many AI reasoning systems.

## What is Resolution?

Resolution is an inference rule that allows us to derive new clauses from existing ones by resolving complementary literals. It is particularly powerful because it is both sound and complete for propositional logic.

### Mathematical Definition

**Resolution Rule**: From two clauses containing complementary literals, derive a new clause.

**Formal Notation**:
```
A ∨ B
¬B ∨ C
-------
A ∨ C
```

**General Form**: If we have clauses C₁ and C₂, and literal L appears in C₁ while ¬L appears in C₂, then we can derive the resolvent by removing L and ¬L and combining the remaining literals.

## Understanding Resolution

### Basic Concept

Resolution works by finding pairs of complementary literals (a literal and its negation) in different clauses and eliminating them to create a new clause.

### Examples

1. **Simple Resolution**:
   - Clause 1: P ∨ Q
   - Clause 2: ¬Q ∨ R
   - Resolvent: P ∨ R

2. **Multiple Literals**:
   - Clause 1: P ∨ Q ∨ R
   - Clause 2: ¬Q ∨ S ∨ T
   - Resolvent: P ∨ R ∨ S ∨ T

3. **Unit Resolution**:
   - Clause 1: P
   - Clause 2: ¬P ∨ Q
   - Resolvent: Q

## Python Implementation

### Basic Resolution Implementation

```python
from typing import Set, List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

class Literal:
    """Represents a literal (atomic proposition or its negation)"""
    
    def __init__(self, variable: str, negated: bool = False):
        self.variable = variable
        self.negated = negated
    
    def __str__(self):
        if self.negated:
            return f"¬{self.variable}"
        else:
            return self.variable
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if isinstance(other, Literal):
            return self.variable == other.variable and self.negated == other.negated
        return False
    
    def __hash__(self):
        return hash((self.variable, self.negated))
    
    def is_complementary(self, other: 'Literal') -> bool:
        """Check if this literal is complementary to another"""
        return (self.variable == other.variable and 
                self.negated != other.negated)
    
    def negate(self) -> 'Literal':
        """Return the negation of this literal"""
        return Literal(self.variable, not self.negated)

class Clause:
    """Represents a clause (disjunction of literals)"""
    
    def __init__(self, literals: Set[Literal]):
        self.literals = literals.copy()
    
    def __str__(self):
        if not self.literals:
            return "⊥"  # Empty clause (contradiction)
        return " ∨ ".join(str(lit) for lit in sorted(self.literals, key=str))
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if isinstance(other, Clause):
            return self.literals == other.literals
        return False
    
    def __hash__(self):
        return hash(frozenset(self.literals))
    
    def is_empty(self) -> bool:
        """Check if this is the empty clause"""
        return len(self.literals) == 0
    
    def is_unit(self) -> bool:
        """Check if this is a unit clause (single literal)"""
        return len(self.literals) == 1
    
    def add_literal(self, literal: Literal):
        """Add a literal to this clause"""
        self.literals.add(literal)
    
    def remove_literal(self, literal: Literal):
        """Remove a literal from this clause"""
        self.literals.discard(literal)
    
    def get_variables(self) -> Set[str]:
        """Get all variables in this clause"""
        return {lit.variable for lit in self.literals}
    
    def evaluate(self, interpretation: Dict[str, bool]) -> bool:
        """Evaluate this clause under an interpretation"""
        for literal in self.literals:
            value = interpretation.get(literal.variable, False)
            if literal.negated:
                value = not value
            if value:  # If any literal is true, clause is true
                return True
        return False  # All literals are false

class Resolution:
    """Implementation of the resolution inference rule"""
    
    @staticmethod
    def resolve(clause1: Clause, clause2: Clause) -> Optional[Clause]:
        """
        Resolve two clauses if they contain complementary literals
        
        Args:
            clause1: First clause
            clause2: Second clause
            
        Returns:
            Resolvent clause, or None if no resolution is possible
        """
        # Find complementary literals
        for lit1 in clause1.literals:
            for lit2 in clause2.literals:
                if lit1.is_complementary(lit2):
                    # Create resolvent by removing complementary literals
                    resolvent_literals = set()
                    
                    # Add all literals from clause1 except lit1
                    for lit in clause1.literals:
                        if lit != lit1:
                            resolvent_literals.add(lit)
                    
                    # Add all literals from clause2 except lit2
                    for lit in clause2.literals:
                        if lit != lit2:
                            resolvent_literals.add(lit)
                    
                    return Clause(resolvent_literals)
        
        return None  # No complementary literals found
    
    @staticmethod
    def can_resolve(clause1: Clause, clause2: Clause) -> bool:
        """Check if two clauses can be resolved"""
        for lit1 in clause1.literals:
            for lit2 in clause2.literals:
                if lit1.is_complementary(lit2):
                    return True
        return False
    
    @staticmethod
    def find_resolvable_pairs(clauses: List[Clause]) -> List[Tuple[Clause, Clause]]:
        """Find all pairs of clauses that can be resolved"""
        pairs = []
        for i, clause1 in enumerate(clauses):
            for j, clause2 in enumerate(clauses):
                if i < j and Resolution.can_resolve(clause1, clause2):
                    pairs.append((clause1, clause2))
        return pairs

# Example usage
P = Literal('P')
not_P = Literal('P', negated=True)
Q = Literal('Q')
R = Literal('R')

# Create clauses
clause1 = Clause({P, Q})
clause2 = Clause({not_P, R})

print(f"Clause 1: {clause1}")
print(f"Clause 2: {clause2}")

# Resolve
resolvent = Resolution.resolve(clause1, clause2)
print(f"Resolvent: {resolvent}")

# Check if resolution is possible
can_resolve = Resolution.can_resolve(clause1, clause2)
print(f"Can resolve: {can_resolve}")
```

### Advanced Resolution System

```python
class ResolutionSystem:
    """Complete resolution system for theorem proving"""
    
    def __init__(self):
        self.clauses = []
        self.resolution_history = []
    
    def add_clause(self, clause: Clause):
        """Add a clause to the system"""
        self.clauses.append(clause)
    
    def add_formula(self, formula):
        """Convert a formula to CNF and add its clauses"""
        cnf_clauses = self.to_cnf(formula)
        for clause in cnf_clauses:
            self.add_clause(clause)
    
    def to_cnf(self, formula) -> List[Clause]:
        """Convert a formula to Conjunctive Normal Form"""
        # This is a simplified CNF conversion
        # In practice, you'd use more sophisticated algorithms
        
        if isinstance(formula, AtomicFormula):
            return [Clause({Literal(formula.variable)})]
        
        elif isinstance(formula, Negation):
            if isinstance(formula.formula, AtomicFormula):
                return [Clause({Literal(formula.formula.variable, negated=True)})]
            else:
                # Handle double negation and De Morgan's laws
                return self.to_cnf(formula.formula)
        
        elif isinstance(formula, Conjunction):
            left_clauses = self.to_cnf(formula.left)
            right_clauses = self.to_cnf(formula.right)
            return left_clauses + right_clauses
        
        elif isinstance(formula, Disjunction):
            left_clauses = self.to_cnf(formula.left)
            right_clauses = self.to_cnf(formula.right)
            
            # Distribute disjunction over conjunction
            result_clauses = []
            for left_clause in left_clauses:
                for right_clause in right_clauses:
                    combined_literals = left_clause.literals | right_clause.literals
                    result_clauses.append(Clause(combined_literals))
            
            return result_clauses
        
        elif isinstance(formula, Implication):
            # P → Q is equivalent to ¬P ∨ Q
            equivalent = Disjunction(Negation(formula.left), formula.right)
            return self.to_cnf(equivalent)
        
        elif isinstance(formula, Biconditional):
            # P ↔ Q is equivalent to (P → Q) ∧ (Q → P)
            left_imp = Implication(formula.left, formula.right)
            right_imp = Implication(formula.right, formula.left)
            equivalent = Conjunction(left_imp, right_imp)
            return self.to_cnf(equivalent)
        
        return []
    
    def resolution_refutation(self, goal_formula) -> Tuple[bool, List[str]]:
        """
        Use resolution refutation to prove a goal
        
        Args:
            goal_formula: The formula to prove
            
        Returns:
            Tuple of (is_provable, proof_steps)
        """
        # Negate the goal and add to clauses
        negated_goal = self.to_cnf(Negation(goal_formula))
        original_clauses = self.clauses.copy()
        
        for clause in negated_goal:
            self.add_clause(clause)
        
        proof_steps = []
        max_iterations = 1000
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            new_clauses = []
            
            # Try to resolve all pairs of clauses
            resolvable_pairs = Resolution.find_resolvable_pairs(self.clauses)
            
            for clause1, clause2 in resolvable_pairs:
                resolvent = Resolution.resolve(clause1, clause2)
                
                if resolvent is not None and resolvent not in self.clauses:
                    new_clauses.append(resolvent)
                    step = f"Resolve {clause1} and {clause2} → {resolvent}"
                    proof_steps.append(step)
                    
                    # Check if we've derived the empty clause
                    if resolvent.is_empty():
                        self.clauses = original_clauses  # Restore original clauses
                        return True, proof_steps
            
            # Add new clauses to the system
            self.clauses.extend(new_clauses)
            
            # If no new clauses were derived, the goal is not provable
            if not new_clauses:
                self.clauses = original_clauses  # Restore original clauses
                return False, proof_steps
        
        self.clauses = original_clauses  # Restore original clauses
        return False, proof_steps
    
    def is_satisfiable(self) -> bool:
        """Check if the set of clauses is satisfiable"""
        # Try to derive the empty clause
        empty_clause = Clause(set())
        is_provable, _ = self.resolution_refutation(empty_clause)
        return not is_provable  # If empty clause is provable, set is unsatisfiable

# Example usage
resolution_system = ResolutionSystem()

# Add some axioms
P = AtomicFormula('P')
Q = AtomicFormula('Q')
R = AtomicFormula('R')

# Add formulas: P → Q, Q → R, P
resolution_system.add_formula(Implication(P, Q))
resolution_system.add_formula(Implication(Q, R))
resolution_system.add_formula(P)

# Try to prove R
goal = R
is_provable, proof = resolution_system.resolution_refutation(goal)

print(f"Can prove {goal}: {is_provable}")
print("Proof steps:")
for i, step in enumerate(proof, 1):
    print(f"{i}. {step}")
```

## Resolution Refutation

Resolution refutation is a powerful method for proving theorems by contradiction.

### Method

1. **Negate the goal**: Convert the goal to be proved into its negation
2. **Add to premises**: Add the negated goal to the set of premises
3. **Apply resolution**: Repeatedly apply resolution to derive new clauses
4. **Check for contradiction**: If the empty clause (⊥) is derived, the original goal is valid

### Python Implementation

```python
class ResolutionRefutation:
    """Implementation of resolution refutation method"""
    
    def __init__(self):
        self.system = ResolutionSystem()
    
    def prove(self, premises: List, conclusion) -> Tuple[bool, List[str]]:
        """
        Prove a conclusion from premises using resolution refutation
        
        Args:
            premises: List of premise formulas
            conclusion: Formula to prove
            
        Returns:
            Tuple of (is_provable, proof_steps)
        """
        # Add premises to the system
        for premise in premises:
            self.system.add_formula(premise)
        
        # Use resolution refutation to prove the conclusion
        return self.system.resolution_refutation(conclusion)
    
    def prove_tautology(self, formula) -> Tuple[bool, List[str]]:
        """Prove that a formula is a tautology"""
        return self.system.resolution_refutation(formula)
    
    def check_entailment(self, premises: List, conclusion) -> bool:
        """Check if premises entail the conclusion"""
        is_provable, _ = self.prove(premises, conclusion)
        return is_provable

# Example: Prove modus ponens
refutation = ResolutionRefutation()

P = AtomicFormula('P')
Q = AtomicFormula('Q')

# Premises: P → Q, P
# Conclusion: Q
premises = [Implication(P, Q), P]
conclusion = Q

is_valid, proof = refutation.prove(premises, conclusion)
print(f"Modus Ponens is valid: {is_valid}")
print("Proof:")
for step in proof:
    print(f"  {step}")

# Example: Prove a tautology
tautology = Disjunction(P, Negation(P))  # P ∨ ¬P
is_tautology, proof = refutation.prove_tautology(tautology)
print(f"\nP ∨ ¬P is a tautology: {is_tautology}")
```

## Optimizations and Strategies

### 1. Unit Resolution

Unit resolution is a restricted form of resolution that only resolves unit clauses (single literals).

```python
class UnitResolution:
    """Unit resolution strategy"""
    
    @staticmethod
    def unit_resolve(clauses: List[Clause]) -> List[Clause]:
        """Apply unit resolution to a set of clauses"""
        new_clauses = []
        unit_clauses = [c for c in clauses if c.is_unit()]
        
        for unit_clause in unit_clauses:
            unit_literal = next(iter(unit_clause.literals))
            
            for clause in clauses:
                if clause != unit_clause:
                    # Check if clause contains the negation of the unit literal
                    for literal in clause.literals:
                        if literal.is_complementary(unit_literal):
                            # Create resolvent
                            resolvent_literals = clause.literals - {literal}
                            if resolvent_literals:  # Not empty
                                new_clause = Clause(resolvent_literals)
                                if new_clause not in new_clauses:
                                    new_clauses.append(new_clause)
        
        return new_clauses

# Example unit resolution
unit_system = UnitResolution()

clauses = [
    Clause({P}),  # Unit clause
    Clause({not_P, Q}),
    Clause({not_Q, R})
]

new_clauses = UnitResolution.unit_resolve(clauses)
print("Unit resolution results:")
for clause in new_clauses:
    print(f"  {clause}")
```

### 2. Pure Literal Elimination

Pure literals are literals that appear only positively or only negatively in the entire clause set.

```python
class PureLiteralElimination:
    """Pure literal elimination strategy"""
    
    @staticmethod
    def find_pure_literals(clauses: List[Clause]) -> Set[Literal]:
        """Find pure literals in a set of clauses"""
        all_literals = set()
        negated_literals = set()
        
        for clause in clauses:
            for literal in clause.literals:
                all_literals.add(literal)
                if literal.negated:
                    negated_literals.add(literal.negate())
                else:
                    negated_literals.add(literal.negate())
        
        # Pure literals are those that don't have their negation
        pure_literals = all_literals - negated_literals
        return pure_literals
    
    @staticmethod
    def eliminate_pure_literals(clauses: List[Clause]) -> List[Clause]:
        """Eliminate clauses containing pure literals"""
        pure_literals = PureLiteralElimination.find_pure_literals(clauses)
        
        # Remove clauses that contain pure literals
        filtered_clauses = []
        for clause in clauses:
            has_pure = any(lit in pure_literals for lit in clause.literals)
            if not has_pure:
                filtered_clauses.append(clause)
        
        return filtered_clauses

# Example pure literal elimination
pure_eliminator = PureLiteralElimination()

clauses = [
    Clause({P, Q}),
    Clause({not_Q, R}),
    Clause({not_R, S}),
    Clause({T})  # T is pure (only appears positively)
]

pure_literals = PureLiteralElimination.find_pure_literals(clauses)
print(f"Pure literals: {pure_literals}")

filtered_clauses = PureLiteralElimination.eliminate_pure_literals(clauses)
print("After pure literal elimination:")
for clause in filtered_clauses:
    print(f"  {clause}")
```

### 3. Subsumption

Subsumption eliminates clauses that are logically implied by other clauses.

```python
class Subsumption:
    """Subsumption strategy for clause elimination"""
    
    @staticmethod
    def subsumes(clause1: Clause, clause2: Clause) -> bool:
        """Check if clause1 subsumes clause2"""
        return clause1.literals.issubset(clause2.literals)
    
    @staticmethod
    def remove_subsumed_clauses(clauses: List[Clause]) -> List[Clause]:
        """Remove clauses that are subsumed by other clauses"""
        filtered_clauses = []
        
        for i, clause1 in enumerate(clauses):
            is_subsumed = False
            
            for j, clause2 in enumerate(clauses):
                if i != j and Subsumption.subsumes(clause2, clause1):
                    is_subsumed = True
                    break
            
            if not is_subsumed:
                filtered_clauses.append(clause1)
        
        return filtered_clauses

# Example subsumption
subsumption = Subsumption()

clauses = [
    Clause({P, Q}),      # P ∨ Q
    Clause({P, Q, R}),   # P ∨ Q ∨ R (subsumed by P ∨ Q)
    Clause({P}),         # P (subsumes P ∨ Q)
    Clause({Q, R})       # Q ∨ R
]

filtered_clauses = Subsumption.remove_subsumed_clauses(clauses)
print("After subsumption elimination:")
for clause in filtered_clauses:
    print(f"  {clause}")
```

## Applications

### 1. SAT Solving

Resolution is fundamental in SAT (Boolean satisfiability) solving:

```python
class SATSolver:
    """Simple SAT solver using resolution"""
    
    def __init__(self):
        self.resolution_system = ResolutionSystem()
    
    def solve(self, formula) -> Optional[Dict[str, bool]]:
        """
        Solve a SAT problem
        
        Args:
            formula: Formula to check for satisfiability
            
        Returns:
            Satisfying assignment if satisfiable, None otherwise
        """
        # Check if formula is unsatisfiable using resolution
        is_unsatisfiable, _ = self.resolution_system.resolution_refutation(formula)
        
        if is_unsatisfiable:
            return None  # Formula is unsatisfiable
        
        # If not unsatisfiable, try to find a satisfying assignment
        # This is a simplified approach - real SAT solvers use more sophisticated methods
        variables = formula.get_variables()
        
        # Try random assignments (in practice, use better heuristics)
        import random
        for _ in range(100):  # Try 100 random assignments
            assignment = {var: random.choice([True, False]) for var in variables}
            if formula.evaluate(assignment):
                return assignment
        
        return None

# Example SAT solving
sat_solver = SATSolver()

# Formula: (P ∨ Q) ∧ (¬P ∨ R) ∧ (¬Q ∨ R)
formula = Conjunction(
    Conjunction(
        Disjunction(P, Q),
        Disjunction(Negation(P), R)
    ),
    Disjunction(Negation(Q), R)
)

satisfying_assignment = sat_solver.solve(formula)
if satisfying_assignment:
    print(f"Satisfying assignment: {satisfying_assignment}")
else:
    print("Formula is unsatisfiable")
```

### 2. Logic Programming

Resolution is the foundation of logic programming languages like Prolog:

```python
class LogicProgram:
    """Simple logic programming system using resolution"""
    
    def __init__(self):
        self.rules = []  # List of (head, body) pairs
        self.facts = set()  # Set of atomic facts
    
    def add_rule(self, head: str, body: List[str]):
        """Add a rule: head :- body"""
        self.rules.append((head, body))
    
    def add_fact(self, fact: str):
        """Add a fact"""
        self.facts.add(fact)
    
    def query(self, goal: str) -> bool:
        """Answer a query using resolution"""
        # Convert to resolution refutation
        resolution_system = ResolutionSystem()
        
        # Add rules as implications
        for head, body in self.rules:
            if not body:  # Fact
                resolution_system.add_formula(AtomicFormula(head))
            else:  # Rule
                antecedent = AtomicFormula(body[0])
                for literal in body[1:]:
                    antecedent = Conjunction(antecedent, AtomicFormula(literal))
                implication = Implication(antecedent, AtomicFormula(head))
                resolution_system.add_formula(implication)
        
        # Add facts
        for fact in self.facts:
            resolution_system.add_formula(AtomicFormula(fact))
        
        # Try to prove the goal
        is_provable, _ = resolution_system.resolution_refutation(AtomicFormula(goal))
        return is_provable

# Example logic program
program = LogicProgram()

# Add rules
program.add_rule('parent', ['father'])
program.add_rule('parent', ['mother'])
program.add_rule('ancestor', ['parent'])
program.add_rule('ancestor', ['parent', 'ancestor'])

# Add facts
program.add_fact('father')
program.add_fact('mother')

# Query
result = program.query('ancestor')
print(f"ancestor: {result}")
```

## Key Concepts Summary

| Concept | Description | Mathematical Form | Python Implementation |
|---------|-------------|-------------------|----------------------|
| **Resolution** | Eliminate complementary literals | A∨B, ¬B∨C ⊢ A∨C | `Resolution.resolve()` |
| **Resolution Refutation** | Prove by contradiction | ¬φ, premises ⊢ ⊥ | `ResolutionRefutation.prove()` |
| **Unit Resolution** | Resolve with unit clauses | P, ¬P∨Q ⊢ Q | `UnitResolution.unit_resolve()` |
| **Pure Literal Elimination** | Remove pure literals | Simplify clause set | `PureLiteralElimination.eliminate_pure_literals()` |
| **Subsumption** | Remove redundant clauses | A subsumes A∨B | `Subsumption.remove_subsumed_clauses()` |
| **SAT Solving** | Boolean satisfiability | Find satisfying assignment | `SATSolver.solve()` |

## Best Practices

1. **Use resolution refutation** for theorem proving
2. **Apply optimizations** like unit resolution and pure literal elimination
3. **Handle large clause sets** efficiently with subsumption
4. **Combine with other methods** for better performance
5. **Use appropriate strategies** for different problem types
6. **Validate results** with multiple methods

Resolution remains one of the most powerful and widely used methods in automated reasoning and AI systems. 