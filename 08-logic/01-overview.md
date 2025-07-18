# Overview: Logic in Artificial Intelligence

## Introduction

Logic serves as the mathematical foundation for artificial intelligence, providing formal methods for knowledge representation and automated reasoning. This guide explores how logical systems enable AI to make decisions, solve problems, and explain its reasoning process.

## What is Logic in AI?

Logic in AI refers to the use of formal mathematical systems to:
- Represent knowledge in a precise, unambiguous manner
- Perform automated inference and deduction
- Ensure consistency and validity in AI decision-making
- Provide explainable and interpretable AI systems

## Historical Context

The relationship between logic and AI dates back to the early days of computer science:

- **1950s**: Alan Turing's work on computability and the Turing Test
- **1960s**: Development of automated theorem proving systems
- **1970s**: Expert systems using rule-based reasoning
- **1980s**: Logic programming languages like Prolog
- **1990s**: Description logics for knowledge representation
- **2000s**: Semantic web and ontologies
- **2010s-present**: Integration with machine learning and explainable AI

## Types of Logic in AI

### 1. Propositional Logic
- **Scope**: Simple true/false statements
- **Use case**: Basic reasoning systems, digital circuits
- **Example**: "If it rains, then the ground is wet"

### 2. First-Order Logic (Predicate Logic)
- **Scope**: Quantified statements with variables
- **Use case**: Knowledge representation, expert systems
- **Example**: "All humans are mortal"

### 3. Modal Logic
- **Scope**: Necessity and possibility
- **Use case**: Belief systems, temporal reasoning
- **Example**: "It is necessary that all bachelors are unmarried"

### 4. Temporal Logic
- **Scope**: Time-dependent statements
- **Use case**: Planning, scheduling, verification
- **Example**: "The light will be green eventually"

### 5. Description Logic
- **Scope**: Knowledge representation for ontologies
- **Use case**: Semantic web, biomedical ontologies
- **Example**: "A student is a person who studies"

## Mathematical Foundations

### Formal Language Theory

A logical system consists of:
1. **Alphabet**: Set of symbols
2. **Syntax**: Rules for forming well-formed formulas
3. **Semantics**: Meaning of formulas
4. **Inference Rules**: Methods for deriving conclusions

### Mathematical Notation

Common symbols used in logic:
- `¬` (negation): NOT
- `∧` (conjunction): AND
- `∨` (disjunction): OR
- `→` (implication): IMPLIES
- `↔` (biconditional): IFF
- `∀` (universal quantifier): FOR ALL
- `∃` (existential quantifier): EXISTS

## Python Implementation: Basic Logic Framework

Let's create a basic framework for working with logic in Python:

```python
from abc import ABC, abstractmethod
from typing import Set, List, Dict, Any
from enum import Enum

class TruthValue(Enum):
    TRUE = True
    FALSE = False
    UNKNOWN = None

class Formula(ABC):
    """Abstract base class for logical formulas"""
    
    @abstractmethod
    def evaluate(self, interpretation: Dict[str, bool]) -> bool:
        """Evaluate the formula under a given interpretation"""
        pass
    
    @abstractmethod
    def get_variables(self) -> Set[str]:
        """Get all variables in the formula"""
        pass
    
    def is_tautology(self) -> bool:
        """Check if the formula is a tautology (always true)"""
        variables = self.get_variables()
        if not variables:
            return self.evaluate({})
        
        # Generate all possible interpretations
        interpretations = self._generate_interpretations(variables)
        return all(self.evaluate(interp) for interp in interpretations)
    
    def is_contradiction(self) -> bool:
        """Check if the formula is a contradiction (always false)"""
        return not self.is_satisfiable()
    
    def is_satisfiable(self) -> bool:
        """Check if the formula is satisfiable (true under some interpretation)"""
        variables = self.get_variables()
        if not variables:
            return self.evaluate({})
        
        interpretations = self._generate_interpretations(variables)
        return any(self.evaluate(interp) for interp in interpretations)
    
    def _generate_interpretations(self, variables: Set[str]) -> List[Dict[str, bool]]:
        """Generate all possible truth assignments for given variables"""
        var_list = list(variables)
        n = len(var_list)
        interpretations = []
        
        for i in range(2**n):
            interpretation = {}
            for j in range(n):
                interpretation[var_list[j]] = bool((i >> j) & 1)
            interpretations.append(interpretation)
        
        return interpretations

class AtomicFormula(Formula):
    """Represents an atomic proposition (single variable)"""
    
    def __init__(self, variable: str):
        self.variable = variable
    
    def evaluate(self, interpretation: Dict[str, bool]) -> bool:
        return interpretation.get(self.variable, False)
    
    def get_variables(self) -> Set[str]:
        return {self.variable}
    
    def __str__(self):
        return self.variable
    
    def __repr__(self):
        return f"AtomicFormula('{self.variable}')"

class Negation(Formula):
    """Represents the negation of a formula"""
    
    def __init__(self, formula: Formula):
        self.formula = formula
    
    def evaluate(self, interpretation: Dict[str, bool]) -> bool:
        return not self.formula.evaluate(interpretation)
    
    def get_variables(self) -> Set[str]:
        return self.formula.get_variables()
    
    def __str__(self):
        return f"¬({self.formula})"
    
    def __repr__(self):
        return f"Negation({self.formula})"

class BinaryOperator(Formula):
    """Base class for binary logical operators"""
    
    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right
    
    def get_variables(self) -> Set[str]:
        return self.left.get_variables() | self.right.get_variables()

class Conjunction(BinaryOperator):
    """Represents logical AND"""
    
    def evaluate(self, interpretation: Dict[str, bool]) -> bool:
        return self.left.evaluate(interpretation) and self.right.evaluate(interpretation)
    
    def __str__(self):
        return f"({self.left} ∧ {self.right})"
    
    def __repr__(self):
        return f"Conjunction({self.left}, {self.right})"

class Disjunction(BinaryOperator):
    """Represents logical OR"""
    
    def evaluate(self, interpretation: Dict[str, bool]) -> bool:
        return self.left.evaluate(interpretation) or self.right.evaluate(interpretation)
    
    def __str__(self):
        return f"({self.left} ∨ {self.right})"
    
    def __repr__(self):
        return f"Disjunction({self.left}, {self.right})"

class Implication(BinaryOperator):
    """Represents logical implication (→)"""
    
    def evaluate(self, interpretation: Dict[str, bool]) -> bool:
        # P → Q is equivalent to ¬P ∨ Q
        return (not self.left.evaluate(interpretation)) or self.right.evaluate(interpretation)
    
    def __str__(self):
        return f"({self.left} → {self.right})"
    
    def __repr__(self):
        return f"Implication({self.left}, {self.right})"

class Biconditional(BinaryOperator):
    """Represents logical biconditional (↔)"""
    
    def evaluate(self, interpretation: Dict[str, bool]) -> bool:
        # P ↔ Q is equivalent to (P → Q) ∧ (Q → P)
        left_val = self.left.evaluate(interpretation)
        right_val = self.right.evaluate(interpretation)
        return left_val == right_val
    
    def __str__(self):
        return f"({self.left} ↔ {self.right})"
    
    def __repr__(self):
        return f"Biconditional({self.left}, {self.right})"
```

## Example Usage

```python
# Create atomic propositions
P = AtomicFormula('P')
Q = AtomicFormula('Q')
R = AtomicFormula('R')

# Create compound formulas
not_P = Negation(P)
P_and_Q = Conjunction(P, Q)
P_or_Q = Disjunction(P, Q)
P_implies_Q = Implication(P, Q)
P_iff_Q = Biconditional(P, Q)

# Test tautologies
print("Testing tautologies:")
print(f"P ∨ ¬P (Law of Excluded Middle): {P_or_Q.is_tautology()}")  # False, should be P ∨ ¬P
print(f"¬(P ∧ ¬P) (Law of Non-Contradiction): {Negation(Conjunction(P, Negation(P))).is_tautology()}")  # True

# Test satisfiability
print("\nTesting satisfiability:")
print(f"P ∧ Q is satisfiable: {P_and_Q.is_satisfiable()}")  # True
print(f"P ∧ ¬P is satisfiable: {Conjunction(P, Negation(P)).is_satisfiable()}")  # False

# Evaluate under specific interpretation
interpretation = {'P': True, 'Q': False}
print(f"\nEvaluation under {interpretation}:")
print(f"P: {P.evaluate(interpretation)}")
print(f"¬P: {not_P.evaluate(interpretation)}")
print(f"P ∧ Q: {P_and_Q.evaluate(interpretation)}")
print(f"P ∨ Q: {P_or_Q.evaluate(interpretation)}")
print(f"P → Q: {P_implies_Q.evaluate(interpretation)}")
```

## Applications in AI

### 1. Expert Systems
Logic-based expert systems use rules to make decisions:
```python
class ExpertSystem:
    def __init__(self):
        self.rules = []
        self.facts = set()
    
    def add_rule(self, antecedent: Formula, consequent: str):
        """Add a rule: if antecedent is true, then consequent holds"""
        self.rules.append((antecedent, consequent))
    
    def add_fact(self, fact: str):
        """Add a known fact"""
        self.facts.add(fact)
    
    def infer(self) -> Set[str]:
        """Perform forward chaining inference"""
        new_facts = set()
        changed = True
        
        while changed:
            changed = False
            for antecedent, consequent in self.rules:
                if consequent not in self.facts:
                    # Create interpretation from current facts
                    interpretation = {fact: True for fact in self.facts}
                    if antecedent.evaluate(interpretation):
                        self.facts.add(consequent)
                        new_facts.add(consequent)
                        changed = True
        
        return new_facts

# Example: Medical diagnosis system
medical_system = ExpertSystem()

# Define symptoms and conditions
fever = AtomicFormula('fever')
cough = AtomicFormula('cough')
fatigue = AtomicFormula('fatigue')
flu = AtomicFormula('flu')
cold = AtomicFormula('cold')

# Add rules
medical_system.add_rule(
    Conjunction(fever, Conjunction(cough, fatigue)), 
    'flu'
)
medical_system.add_rule(
    Conjunction(cough, Negation(fever)), 
    'cold'
)

# Add observed symptoms
medical_system.add_fact('fever')
medical_system.add_fact('cough')
medical_system.add_fact('fatigue')

# Infer diagnosis
diagnosis = medical_system.infer()
print(f"Diagnosis: {diagnosis}")
```

### 2. Automated Theorem Proving
Logic enables automated proof systems:
```python
class TheoremProver:
    def __init__(self):
        self.axioms = []
        self.theorems = []
    
    def add_axiom(self, formula: Formula):
        """Add an axiom to the system"""
        self.axioms.append(formula)
    
    def prove(self, goal: Formula) -> bool:
        """Attempt to prove a goal formula"""
        # Simple implementation using truth table method
        all_vars = set()
        for axiom in self.axioms:
            all_vars.update(axiom.get_variables())
        all_vars.update(goal.get_variables())
        
        # Check if axioms logically entail the goal
        interpretations = self._generate_interpretations(all_vars)
        
        for interp in interpretations:
            # If all axioms are true but goal is false, goal is not entailed
            axioms_true = all(axiom.evaluate(interp) for axiom in self.axioms)
            goal_false = not goal.evaluate(interp)
            
            if axioms_true and goal_false:
                return False
        
        return True
    
    def _generate_interpretations(self, variables: Set[str]) -> List[Dict[str, bool]]:
        """Generate all possible truth assignments"""
        var_list = list(variables)
        n = len(var_list)
        interpretations = []
        
        for i in range(2**n):
            interpretation = {}
            for j in range(n):
                interpretation[var_list[j]] = bool((i >> j) & 1)
            interpretations.append(interpretation)
        
        return interpretations

# Example: Prove De Morgan's Law
prover = TheoremProver()

P = AtomicFormula('P')
Q = AtomicFormula('Q')

# De Morgan's Law: ¬(P ∧ Q) ↔ (¬P ∨ ¬Q)
demorgan_law = Biconditional(
    Negation(Conjunction(P, Q)),
    Disjunction(Negation(P), Negation(Q))
)

# No axioms needed for this tautology
is_provable = prover.prove(demorgan_law)
print(f"De Morgan's Law is provable: {is_provable}")
```

## Key Concepts Summary

| Concept | Description | Mathematical Form | Python Implementation |
|---------|-------------|-------------------|----------------------|
| **Atomic Formula** | Single proposition | P, Q, R | `AtomicFormula('P')` |
| **Negation** | Logical NOT | ¬P | `Negation(P)` |
| **Conjunction** | Logical AND | P ∧ Q | `Conjunction(P, Q)` |
| **Disjunction** | Logical OR | P ∨ Q | `Disjunction(P, Q)` |
| **Implication** | Logical IF-THEN | P → Q | `Implication(P, Q)` |
| **Biconditional** | Logical IFF | P ↔ Q | `Biconditional(P, Q)` |
| **Tautology** | Always true | ⊨ φ | `formula.is_tautology()` |
| **Contradiction** | Always false | ⊨ ¬φ | `formula.is_contradiction()` |
| **Satisfiable** | True under some interpretation | ∃I: I ⊨ φ | `formula.is_satisfiable()` |

## Challenges and Limitations

### 1. Computational Complexity
- **Problem**: Truth table method is exponential in number of variables
- **Solution**: Use more efficient algorithms like resolution or SAT solvers

### 2. Expressiveness
- **Problem**: Propositional logic is limited to simple statements
- **Solution**: Use first-order logic for more complex reasoning

### 3. Uncertainty
- **Problem**: Logic assumes perfect knowledge
- **Solution**: Integrate with probabilistic methods

### 4. Common Sense
- **Problem**: Capturing everyday knowledge is difficult
- **Solution**: Use large knowledge bases and ontologies

## Future Directions

1. **Integration with Machine Learning**: Combining logical reasoning with statistical learning
2. **Neural-Symbolic AI**: Hybrid systems that combine neural networks with symbolic reasoning
3. **Explainable AI**: Using logic to provide interpretable explanations for AI decisions
4. **Knowledge Graphs**: Large-scale logical knowledge representation
5. **Automated Theorem Proving**: Advanced proof systems for mathematics and verification

Logic remains fundamental to AI, providing the theoretical foundation for knowledge representation, automated reasoning, and explainable AI systems. Understanding logic is essential for building intelligent systems that can reason, explain their decisions, and maintain consistency in their knowledge. 