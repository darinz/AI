# Propositional Logic Semantics

## Introduction

Semantics in propositional logic defines the meaning of logical expressions through truth assignments. While syntax tells us how to construct well-formed formulas, semantics tells us how to interpret and evaluate them.

## What are Semantics?

Semantics provide the interpretation function that maps logical formulas to truth values. They answer the question: "What does this formula mean, and when is it true?"

### Key Concepts
- **Interpretation**: Assignment of truth values to atomic propositions
- **Truth Function**: Mapping from formulas to truth values
- **Satisfaction**: When a formula is true under an interpretation
- **Validity**: When a formula is true under all interpretations

## Truth Tables

Truth tables are the most fundamental tool for understanding propositional logic semantics. They systematically show the truth value of compound propositions for all possible truth assignments to atomic propositions.

### Mathematical Definition

For a formula φ with variables P₁, P₂, ..., Pₙ, the truth table has:
- **Rows**: 2ⁿ possible truth assignments
- **Columns**: One for each variable plus one for the formula
- **Values**: True (T) or False (F)

### Python Implementation: Truth Table Generator

```python
from typing import Dict, List, Set
import itertools

class TruthTable:
    """Generate and analyze truth tables for propositional logic formulas"""
    
    def __init__(self, formula):
        self.formula = formula
        self.variables = sorted(list(formula.get_variables()))
        self.table = self._generate_table()
    
    def _generate_table(self) -> List[Dict]:
        """Generate all possible interpretations and their results"""
        table = []
        n = len(self.variables)
        
        # Generate all possible truth assignments
        for assignment in itertools.product([False, True], repeat=n):
            interpretation = dict(zip(self.variables, assignment))
            result = self.formula.evaluate(interpretation)
            interpretation['RESULT'] = result
            table.append(interpretation)
        
        return table
    
    def display(self, show_binary: bool = False):
        """Display the truth table"""
        if not self.variables:
            print(f"Formula: {self.formula}")
            print(f"Result: {self.formula.evaluate({})}")
            return
        
        # Header
        header = " | ".join(self.variables + ["RESULT"])
        print(f"Formula: {self.formula}")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        # Rows
        for row in self.table:
            values = []
            for var in self.variables:
                if show_binary:
                    values.append("1" if row[var] else "0")
                else:
                    values.append("T" if row[var] else "F")
            values.append("T" if row['RESULT'] else "F")
            print(" | ".join(values))
    
    def is_tautology(self) -> bool:
        """Check if the formula is a tautology (always true)"""
        return all(row['RESULT'] for row in self.table)
    
    def is_contradiction(self) -> bool:
        """Check if the formula is a contradiction (always false)"""
        return all(not row['RESULT'] for row in self.table)
    
    def is_satisfiable(self) -> bool:
        """Check if the formula is satisfiable (true under some interpretation)"""
        return any(row['RESULT'] for row in self.table)
    
    def is_unsatisfiable(self) -> bool:
        """Check if the formula is unsatisfiable (false under all interpretations)"""
        return not self.is_satisfiable()
    
    def get_satisfying_assignments(self) -> List[Dict]:
        """Get all interpretations that make the formula true"""
        return [row for row in self.table if row['RESULT']]
    
    def get_falsifying_assignments(self) -> List[Dict]:
        """Get all interpretations that make the formula false"""
        return [row for row in self.table if not row['RESULT']]
    
    def count_satisfying_assignments(self) -> int:
        """Count the number of satisfying assignments"""
        return len(self.get_satisfying_assignments())
    
    def get_truth_value_distribution(self) -> Dict[str, int]:
        """Get distribution of truth values"""
        true_count = self.count_satisfying_assignments()
        false_count = len(self.table) - true_count
        return {
            'true': true_count,
            'false': false_count,
            'total': len(self.table)
        }

# Example usage
P = AtomicFormula('P')
Q = AtomicFormula('Q')

# Test various formulas
formulas_to_test = [
    ("P", P),
    ("¬P", Negation(P)),
    ("P ∧ Q", Conjunction(P, Q)),
    ("P ∨ Q", Disjunction(P, Q)),
    ("P → Q", Implication(P, Q)),
    ("P ↔ Q", Biconditional(P, Q)),
    ("P ∨ ¬P", Disjunction(P, Negation(P))),  # Tautology
    ("P ∧ ¬P", Conjunction(P, Negation(P))),  # Contradiction
]

print("Truth Table Analysis:")
for name, formula in formulas_to_test:
    print(f"\n{name}:")
    tt = TruthTable(formula)
    tt.display()
    print(f"Tautology: {tt.is_tautology()}")
    print(f"Contradiction: {tt.is_contradiction()}")
    print(f"Satisfiable: {tt.is_satisfiable()}")
    print(f"Satisfying assignments: {tt.count_satisfying_assignments()}")
```

## Interpretation and Valuation

### Mathematical Definition

An **interpretation** (or **valuation**) is a function:
```
I: P → {True, False}
```
where P is the set of atomic propositions.

### Truth Function

The truth function V maps formulas to truth values under an interpretation:
```
V_I: F → {True, False}
```

Where F is the set of well-formed formulas, defined recursively:

1. **Atomic**: V_I(P) = I(P)
2. **Negation**: V_I(¬φ) = True iff V_I(φ) = False
3. **Conjunction**: V_I(φ ∧ ψ) = True iff V_I(φ) = True and V_I(ψ) = True
4. **Disjunction**: V_I(φ ∨ ψ) = True iff V_I(φ) = True or V_I(ψ) = True
5. **Implication**: V_I(φ → ψ) = True iff V_I(φ) = False or V_I(ψ) = True
6. **Biconditional**: V_I(φ ↔ ψ) = True iff V_I(φ) = V_I(ψ)

### Python Implementation: Interpretation System

```python
class Interpretation:
    """Represents an interpretation (truth assignment) for propositional logic"""
    
    def __init__(self, assignment: Dict[str, bool]):
        self.assignment = assignment.copy()
    
    def get_value(self, variable: str) -> bool:
        """Get the truth value of a variable"""
        return self.assignment.get(variable, False)
    
    def set_value(self, variable: str, value: bool):
        """Set the truth value of a variable"""
        self.assignment[variable] = value
    
    def evaluate_formula(self, formula) -> bool:
        """Evaluate a formula under this interpretation"""
        return formula.evaluate(self.assignment)
    
    def __str__(self):
        return str(self.assignment)
    
    def __repr__(self):
        return f"Interpretation({self.assignment})"

class SemanticAnalyzer:
    """Analyze semantic properties of propositional logic formulas"""
    
    @staticmethod
    def generate_all_interpretations(variables: Set[str]) -> List[Interpretation]:
        """Generate all possible interpretations for given variables"""
        interpretations = []
        var_list = sorted(list(variables))
        n = len(var_list)
        
        for assignment in itertools.product([False, True], repeat=n):
            interpretation_dict = dict(zip(var_list, assignment))
            interpretations.append(Interpretation(interpretation_dict))
        
        return interpretations
    
    @staticmethod
    def is_valid(formula, premises: List = None) -> bool:
        """Check if a formula is valid (true under all interpretations)"""
        if premises is None:
            premises = []
        
        all_vars = formula.get_variables()
        for premise in premises:
            all_vars.update(premise.get_variables())
        
        interpretations = SemanticAnalyzer.generate_all_interpretations(all_vars)
        
        for interp in interpretations:
            # Check if all premises are true
            premises_true = all(premise.evaluate(interp.assignment) for premise in premises)
            
            # If premises are true but conclusion is false, not valid
            if premises_true and not formula.evaluate(interp.assignment):
                return False
        
        return True
    
    @staticmethod
    def is_satisfiable(formula) -> bool:
        """Check if a formula is satisfiable"""
        variables = formula.get_variables()
        interpretations = SemanticAnalyzer.generate_all_interpretations(variables)
        
        return any(formula.evaluate(interp.assignment) for interp in interpretations)
    
    @staticmethod
    def find_satisfying_interpretation(formula) -> Interpretation:
        """Find an interpretation that satisfies the formula"""
        variables = formula.get_variables()
        interpretations = SemanticAnalyzer.generate_all_interpretations(variables)
        
        for interp in interpretations:
            if formula.evaluate(interp.assignment):
                return interp
        
        return None  # Formula is unsatisfiable
    
    @staticmethod
    def logical_equivalence(formula1, formula2) -> bool:
        """Check if two formulas are logically equivalent"""
        all_vars = formula1.get_variables() | formula2.get_variables()
        interpretations = SemanticAnalyzer.generate_all_interpretations(all_vars)
        
        for interp in interpretations:
            val1 = formula1.evaluate(interp.assignment)
            val2 = formula2.evaluate(interp.assignment)
            if val1 != val2:
                return False
        
        return True
    
    @staticmethod
    def logical_entailment(premises: List, conclusion) -> bool:
        """Check if premises logically entail the conclusion"""
        return SemanticAnalyzer.is_valid(conclusion, premises)

# Example usage
P = AtomicFormula('P')
Q = AtomicFormula('Q')
R = AtomicFormula('R')

analyzer = SemanticAnalyzer()

# Test logical equivalence
formula1 = Implication(P, Q)
formula2 = Disjunction(Negation(P), Q)
print(f"P → Q ≡ ¬P ∨ Q: {analyzer.logical_equivalence(formula1, formula2)}")

# Test logical entailment
premises = [Implication(P, Q), P]
conclusion = Q
print(f"P → Q, P ⊨ Q: {analyzer.logical_entailment(premises, conclusion)}")

# Find satisfying interpretation
complex_formula = Conjunction(
    Disjunction(P, Q),
    Implication(P, R)
)
satisfying_interp = analyzer.find_satisfying_interpretation(complex_formula)
print(f"Satisfying interpretation for (P ∨ Q) ∧ (P → R): {satisfying_interp}")
```

## Validity and Satisfiability

### Mathematical Definitions

**Valid (Tautology)**: A formula φ is valid if it is true under all interpretations.
```
⊨ φ iff for all I, V_I(φ) = True
```

**Satisfiable**: A formula φ is satisfiable if it is true under at least one interpretation.
```
φ is satisfiable iff there exists I such that V_I(φ) = True
```

**Contradiction (Unsatisfiable)**: A formula φ is a contradiction if it is false under all interpretations.
```
φ is a contradiction iff for all I, V_I(φ) = False
```

### Python Implementation: Validity Checker

```python
class ValidityChecker:
    """Check validity, satisfiability, and related semantic properties"""
    
    @staticmethod
    def check_validity(formula) -> Dict[str, bool]:
        """Comprehensive validity analysis"""
        tt = TruthTable(formula)
        
        return {
            'tautology': tt.is_tautology(),
            'contradiction': tt.is_contradiction(),
            'satisfiable': tt.is_satisfiable(),
            'unsatisfiable': tt.is_unsatisfiable(),
            'contingent': not tt.is_tautology() and not tt.is_contradiction()
        }
    
    @staticmethod
    def analyze_formula_class(formula) -> str:
        """Classify a formula based on its semantic properties"""
        properties = ValidityChecker.check_validity(formula)
        
        if properties['tautology']:
            return "Tautology"
        elif properties['contradiction']:
            return "Contradiction"
        else:
            return "Contingent"
    
    @staticmethod
    def find_counterexample(formula) -> Dict:
        """Find a counterexample if the formula is not valid"""
        if formula.is_tautology():
            return None
        
        tt = TruthTable(formula)
        falsifying = tt.get_falsifying_assignments()
        return falsifying[0] if falsifying else None

# Test various types of formulas
test_formulas = [
    ("P ∨ ¬P", Disjunction(P, Negation(P))),  # Tautology
    ("P ∧ ¬P", Conjunction(P, Negation(P))),  # Contradiction
    ("P ∧ Q", Conjunction(P, Q)),             # Contingent
    ("P → P", Implication(P, P)),             # Tautology
    ("P ↔ ¬P", Biconditional(P, Negation(P))), # Contradiction
]

print("Formula Classification:")
for name, formula in test_formulas:
    classification = ValidityChecker.analyze_formula_class(formula)
    properties = ValidityChecker.check_validity(formula)
    print(f"{name}: {classification}")
    print(f"  Properties: {properties}")
    
    if classification != "Tautology":
        counterexample = ValidityChecker.find_counterexample(formula)
        print(f"  Counterexample: {counterexample}")
```

## Semantic Properties

### Logical Equivalence

Two formulas φ and ψ are logically equivalent (φ ≡ ψ) if they have the same truth value under all interpretations.

```python
def test_logical_equivalences():
    """Test common logical equivalences"""
    P = AtomicFormula('P')
    Q = AtomicFormula('Q')
    R = AtomicFormula('R')
    
    equivalences = [
        # Double negation
        ("¬¬P", "P", Negation(Negation(P)), P),
        
        # De Morgan's laws
        ("¬(P ∧ Q)", "¬P ∨ ¬Q", 
         Negation(Conjunction(P, Q)), 
         Disjunction(Negation(P), Negation(Q))),
        
        ("¬(P ∨ Q)", "¬P ∧ ¬Q",
         Negation(Disjunction(P, Q)),
         Conjunction(Negation(P), Negation(Q))),
        
        # Implication
        ("P → Q", "¬P ∨ Q",
         Implication(P, Q),
         Disjunction(Negation(P), Q)),
        
        # Biconditional
        ("P ↔ Q", "(P → Q) ∧ (Q → P)",
         Biconditional(P, Q),
         Conjunction(Implication(P, Q), Implication(Q, P))),
        
        # Distributive laws
        ("P ∧ (Q ∨ R)", "(P ∧ Q) ∨ (P ∧ R)",
         Conjunction(P, Disjunction(Q, R)),
         Disjunction(Conjunction(P, Q), Conjunction(P, R))),
        
        ("P ∨ (Q ∧ R)", "(P ∨ Q) ∧ (P ∨ R)",
         Disjunction(P, Conjunction(Q, R)),
         Conjunction(Disjunction(P, Q), Disjunction(P, R))),
    ]
    
    print("Testing Logical Equivalences:")
    for name1, name2, formula1, formula2 in equivalences:
        equivalent = SemanticAnalyzer.logical_equivalence(formula1, formula2)
        print(f"{name1} ≡ {name2}: {equivalent}")

test_logical_equivalences()
```

### Logical Entailment

A set of formulas Γ entails a formula φ (Γ ⊨ φ) if φ is true under all interpretations where all formulas in Γ are true.

```python
def test_logical_entailments():
    """Test common logical entailments"""
    P = AtomicFormula('P')
    Q = AtomicFormula('Q')
    R = AtomicFormula('R')
    
    entailments = [
        # Modus ponens
        ([Implication(P, Q), P], Q, "Modus Ponens"),
        
        # Modus tollens
        ([Implication(P, Q), Negation(Q)], Negation(P), "Modus Tollens"),
        
        # Hypothetical syllogism
        ([Implication(P, Q), Implication(Q, R)], Implication(P, R), "Hypothetical Syllogism"),
        
        # Disjunctive syllogism
        ([Disjunction(P, Q), Negation(P)], Q, "Disjunctive Syllogism"),
        
        # Addition
        ([P], Disjunction(P, Q), "Addition"),
        
        # Simplification
        ([Conjunction(P, Q)], P, "Simplification"),
    ]
    
    print("\nTesting Logical Entailments:")
    for premises, conclusion, name in entailments:
        entails = SemanticAnalyzer.logical_entailment(premises, conclusion)
        print(f"{name}: {entails}")

test_logical_entailments()
```

## Semantic Tableaux

Semantic tableaux (truth trees) provide a systematic method for testing satisfiability and validity.

```python
class SemanticTableau:
    """Implement semantic tableau method for testing satisfiability"""
    
    def __init__(self, formula):
        self.formula = formula
        self.branches = [[Negation(formula)]]  # Start with negation for validity test
    
    def apply_rules(self):
        """Apply tableau rules to expand branches"""
        # This is a simplified implementation
        # Full tableau method requires more sophisticated rule application
        
        new_branches = []
        
        for branch in self.branches:
            # Check for contradictions
            if self._has_contradiction(branch):
                continue  # Close this branch
            
            # Apply rules to formulas in the branch
            for formula in branch:
                if isinstance(formula, Negation):
                    if isinstance(formula.formula, Negation):
                        # Double negation rule
                        new_branch = branch.copy()
                        new_branch.remove(formula)
                        new_branch.append(formula.formula.formula)
                        new_branches.append(new_branch)
                
                elif isinstance(formula, Conjunction):
                    # Conjunction rule: both conjuncts must be true
                    new_branch = branch.copy()
                    new_branch.remove(formula)
                    new_branch.extend([formula.left, formula.right])
                    new_branches.append(new_branch)
                
                elif isinstance(formula, Disjunction):
                    # Disjunction rule: split into two branches
                    branch1 = branch.copy()
                    branch1.remove(formula)
                    branch1.append(formula.left)
                    new_branches.append(branch1)
                    
                    branch2 = branch.copy()
                    branch2.remove(formula)
                    branch2.append(formula.right)
                    new_branches.append(branch2)
        
        self.branches = new_branches
    
    def _has_contradiction(self, branch) -> bool:
        """Check if a branch contains a contradiction"""
        formulas = set()
        negated_formulas = set()
        
        for formula in branch:
            if isinstance(formula, Negation):
                if isinstance(formula.formula, AtomicFormula):
                    negated_formulas.add(formula.formula.variable)
            elif isinstance(formula, AtomicFormula):
                formulas.add(formula.variable)
        
        return bool(formulas & negated_formulas)
    
    def is_valid(self) -> bool:
        """Check if the original formula is valid"""
        # Apply rules until no more rules can be applied
        while True:
            old_branches = self.branches.copy()
            self.apply_rules()
            if self.branches == old_branches:
                break
        
        # If all branches are closed, formula is valid
        return len(self.branches) == 0

# Test semantic tableau
P = AtomicFormula('P')
Q = AtomicFormula('Q')

# Test a tautology
tautology = Disjunction(P, Negation(P))
tableau = SemanticTableau(tautology)
print(f"P ∨ ¬P is valid: {tableau.is_valid()}")

# Test a contingent formula
contingent = Conjunction(P, Q)
tableau = SemanticTableau(contingent)
print(f"P ∧ Q is valid: {tableau.is_valid()}")
```

## Key Concepts Summary

| Concept | Description | Mathematical Notation | Python Implementation |
|---------|-------------|----------------------|----------------------|
| **Interpretation** | Truth assignment to variables | I: P → {T,F} | `Interpretation(assignment)` |
| **Truth Function** | Evaluation under interpretation | V_I(φ) | `formula.evaluate(interp)` |
| **Truth Table** | Systematic truth value display | 2ⁿ rows for n variables | `TruthTable(formula)` |
| **Valid/Tautology** | True under all interpretations | ⊨ φ | `formula.is_tautology()` |
| **Satisfiable** | True under some interpretation | ∃I: V_I(φ) = T | `formula.is_satisfiable()` |
| **Contradiction** | False under all interpretations | ⊨ ¬φ | `formula.is_contradiction()` |
| **Logical Equivalence** | Same truth value under all interpretations | φ ≡ ψ | `logical_equivalence(φ, ψ)` |
| **Logical Entailment** | Conclusion true when premises true | Γ ⊨ φ | `logical_entailment(premises, φ)` |

## Applications

### 1. Circuit Design
Truth tables are fundamental in digital logic design:
```python
def design_circuit():
    """Design a simple digital circuit using truth tables"""
    A = AtomicFormula('A')
    B = AtomicFormula('B')
    C = AtomicFormula('C')
    
    # Half-adder: S = A ⊕ B, C = A ∧ B
    S = Biconditional(A, B)  # XOR equivalent
    Cout = Conjunction(A, B)
    
    print("Half-Adder Truth Table:")
    print("A | B | S | C")
    print("--+---+---+---")
    
    for a in [False, True]:
        for b in [False, True]:
            interp = {'A': a, 'B': b}
            s_val = S.evaluate(interp)
            c_val = Cout.evaluate(interp)
            print(f"{'T' if a else 'F'} | {'T' if b else 'F'} | {'T' if s_val else 'F'} | {'T' if c_val else 'F'}")

design_circuit()
```

### 2. Boolean Function Minimization
Semantic analysis helps in minimizing boolean functions:
```python
def minimize_boolean_function():
    """Demonstrate boolean function minimization"""
    P = AtomicFormula('P')
    Q = AtomicFormula('Q')
    R = AtomicFormula('R')
    
    # Original function: (P ∧ Q ∧ R) ∨ (P ∧ Q ∧ ¬R) ∨ (P ∧ ¬Q ∧ R)
    original = Disjunction(
        Conjunction(Conjunction(P, Q), R),
        Disjunction(
            Conjunction(Conjunction(P, Q), Negation(R)),
            Conjunction(Conjunction(P, Negation(Q)), R)
        )
    )
    
    # Simplified: P ∧ (Q ∨ R)
    simplified = Conjunction(P, Disjunction(Q, R))
    
    # Verify they are equivalent
    equivalent = SemanticAnalyzer.logical_equivalence(original, simplified)
    print(f"Original and simplified functions are equivalent: {equivalent}")
    
    # Show truth tables
    print("\nOriginal function:")
    TruthTable(original).display()
    
    print("\nSimplified function:")
    TruthTable(simplified).display()

minimize_boolean_function()
```

Understanding propositional logic semantics is essential for:
- Building automated reasoning systems
- Designing digital circuits
- Verifying software specifications
- Analyzing logical arguments
- Implementing knowledge representation systems

The semantic analysis provides the foundation for all higher-level logical reasoning and AI applications. 