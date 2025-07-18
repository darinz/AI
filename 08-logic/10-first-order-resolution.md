# First Order Resolution

## Introduction

First-order resolution extends propositional resolution to handle quantified statements, variables, and complex predicates. It is a powerful and complete method for automated theorem proving in first-order logic, forming the foundation for logic programming languages like Prolog.

## What is First-Order Resolution?

First-order resolution is an inference rule that combines unification with resolution to handle quantified formulas. It can derive new clauses from existing ones by finding complementary literals and applying appropriate substitutions.

### Mathematical Definition

**Resolution Rule**: From two clauses containing complementary literals, derive a new clause after applying the most general unifier.

**Formal Notation**:
```
C₁ ∨ L₁
C₂ ∨ ¬L₂
--------
σ(C₁ ∨ C₂)
```

Where σ is the most general unifier of L₁ and L₂.

## Key Components

### 1. Unification

Unification is the process of finding a substitution that makes two terms or literals identical.

### 2. Most General Unifier (MGU)

The MGU is the most general substitution that unifies two terms, allowing for the most flexible resolution.

### 3. Skolemization

Skolemization converts existential quantifiers to Skolem functions, eliminating existential quantifiers.

## Python Implementation: First-Order Resolution System

```python
from typing import Set, List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import copy

class FirstOrderLiteral:
    """Represents a literal in first-order logic"""
    
    def __init__(self, predicate: str, arguments: List['Term'], negated: bool = False):
        self.predicate = predicate
        self.arguments = arguments
        self.negated = negated
    
    def __str__(self):
        literal_str = f"{self.predicate}({', '.join(str(arg) for arg in self.arguments)})"
        if self.negated:
            return f"¬{literal_str}"
        return literal_str
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if isinstance(other, FirstOrderLiteral):
            return (self.predicate == other.predicate and 
                    self.arguments == other.arguments and 
                    self.negated == other.negated)
        return False
    
    def __hash__(self):
        return hash((self.predicate, tuple(self.arguments), self.negated))
    
    def is_complementary(self, other: 'FirstOrderLiteral') -> bool:
        """Check if this literal is complementary to another"""
        return (self.predicate == other.predicate and 
                self.arguments == other.arguments and 
                self.negated != other.negated)
    
    def negate(self) -> 'FirstOrderLiteral':
        """Return the negation of this literal"""
        return FirstOrderLiteral(self.predicate, self.arguments, not self.negated)
    
    def get_variables(self) -> Set[str]:
        """Get all variables in this literal"""
        variables = set()
        for arg in self.arguments:
            if isinstance(arg, Variable):
                variables.add(arg.name)
            elif isinstance(arg, Function):
                variables.update(arg.get_variables())
        return variables
    
    def substitute(self, substitution: Dict[str, 'Term']) -> 'FirstOrderLiteral':
        """Apply a substitution to this literal"""
        substituted_args = []
        for arg in self.arguments:
            if isinstance(arg, Variable) and arg.name in substitution:
                substituted_args.append(substitution[arg.name])
            else:
                substituted_args.append(arg)
        
        return FirstOrderLiteral(self.predicate, substituted_args, self.negated)

class FirstOrderClause:
    """Represents a clause in first-order logic"""
    
    def __init__(self, literals: Set[FirstOrderLiteral]):
        self.literals = literals.copy()
    
    def __str__(self):
        if not self.literals:
            return "⊥"  # Empty clause
        return " ∨ ".join(str(lit) for lit in sorted(self.literals, key=str))
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if isinstance(other, FirstOrderClause):
            return self.literals == other.literals
        return False
    
    def __hash__(self):
        return hash(frozenset(self.literals))
    
    def is_empty(self) -> bool:
        """Check if this is the empty clause"""
        return len(self.literals) == 0
    
    def add_literal(self, literal: FirstOrderLiteral):
        """Add a literal to this clause"""
        self.literals.add(literal)
    
    def remove_literal(self, literal: FirstOrderLiteral):
        """Remove a literal from this clause"""
        self.literals.discard(literal)
    
    def get_variables(self) -> Set[str]:
        """Get all variables in this clause"""
        variables = set()
        for literal in self.literals:
            variables.update(literal.get_variables())
        return variables
    
    def substitute(self, substitution: Dict[str, 'Term']) -> 'FirstOrderClause':
        """Apply a substitution to this clause"""
        substituted_literals = set()
        for literal in self.literals:
            substituted_literals.add(literal.substitute(substitution))
        return FirstOrderClause(substituted_literals)

class FirstOrderUnification:
    """Unification algorithm for first-order logic"""
    
    @staticmethod
    def unify(term1: 'Term', term2: 'Term') -> Optional[Dict[str, 'Term']]:
        """
        Unify two terms
        
        Args:
            term1: First term
            term2: Second term
            
        Returns:
            Most general unifier, or None if not unifiable
        """
        return FirstOrderUnification._unify_terms(term1, term2, {})
    
    @staticmethod
    def _unify_terms(term1: 'Term', term2: 'Term', substitution: Dict[str, 'Term']) -> Optional[Dict[str, 'Term']]:
        """Recursive unification with current substitution"""
        
        # Apply current substitution to both terms
        term1_subst = FirstOrderUnification._apply_substitution_to_term(term1, substitution)
        term2_subst = FirstOrderUnification._apply_substitution_to_term(term2, substitution)
        
        # If terms are identical, return current substitution
        if term1_subst == term2_subst:
            return substitution
        
        # If one term is a variable
        if isinstance(term1_subst, Variable):
            return FirstOrderUnification._unify_variable(term1_subst, term2_subst, substitution)
        elif isinstance(term2_subst, Variable):
            return FirstOrderUnification._unify_variable(term2_subst, term1_subst, substitution)
        
        # If both are constants, they must be equal
        if isinstance(term1_subst, Constant) and isinstance(term2_subst, Constant):
            if term1_subst == term2_subst:
                return substitution
            else:
                return None
        
        # If both are functions, unify function names and arguments
        if isinstance(term1_subst, Function) and isinstance(term2_subst, Function):
            if term1_subst.name != term2_subst.name:
                return None
            
            if len(term1_subst.arguments) != len(term2_subst.arguments):
                return None
            
            current_substitution = substitution.copy()
            for arg1, arg2 in zip(term1_subst.arguments, term2_subst.arguments):
                result = FirstOrderUnification._unify_terms(arg1, arg2, current_substitution)
                if result is None:
                    return None
                current_substitution = result
            
            return current_substitution
        
        return None
    
    @staticmethod
    def _unify_variable(var: 'Variable', term: 'Term', substitution: Dict[str, 'Term']) -> Optional[Dict[str, 'Term']]:
        """Unify a variable with a term"""
        
        # Check if variable is already bound
        if var.name in substitution:
            return FirstOrderUnification._unify_terms(substitution[var.name], term, substitution)
        
        # Check if term contains the variable (occurs check)
        if FirstOrderUnification._occurs_in(var, term, substitution):
            return None
        
        # Create new substitution
        new_substitution = substitution.copy()
        new_substitution[var.name] = term
        return new_substitution
    
    @staticmethod
    def _occurs_in(var: 'Variable', term: 'Term', substitution: Dict[str, 'Term']) -> bool:
        """Check if variable occurs in term (occurs check)"""
        
        if isinstance(term, Variable):
            if term.name == var.name:
                return True
            if term.name in substitution:
                return FirstOrderUnification._occurs_in(var, substitution[term.name], substitution)
            return False
        
        elif isinstance(term, Constant):
            return False
        
        elif isinstance(term, Function):
            for arg in term.arguments:
                if FirstOrderUnification._occurs_in(var, arg, substitution):
                    return True
            return False
        
        return False
    
    @staticmethod
    def _apply_substitution_to_term(term: 'Term', substitution: Dict[str, 'Term']) -> 'Term':
        """Apply substitution to a term"""
        
        if isinstance(term, Variable):
            if term.name in substitution:
                return substitution[term.name]
            return term
        
        elif isinstance(term, Constant):
            return term
        
        elif isinstance(term, Function):
            substituted_args = [FirstOrderUnification._apply_substitution_to_term(arg, substitution) for arg in term.arguments]
            return Function(term.name, substituted_args)
        
        return term

class FirstOrderResolution:
    """Implementation of first-order resolution"""
    
    @staticmethod
    def resolve(clause1: FirstOrderClause, clause2: FirstOrderClause) -> List[FirstOrderClause]:
        """
        Resolve two clauses using first-order resolution
        
        Args:
            clause1: First clause
            clause2: Second clause
            
        Returns:
            List of possible resolvents
        """
        resolvents = []
        
        # Try to resolve each pair of literals
        for lit1 in clause1.literals:
            for lit2 in clause2.literals:
                if lit1.is_complementary(lit2):
                    # Try to unify the literals
                    unifier = FirstOrderUnification.unify(lit1, lit2)
                    
                    if unifier is not None:
                        # Create resolvent
                        resolvent_literals = set()
                        
                        # Add all literals from clause1 except lit1
                        for lit in clause1.literals:
                            if lit != lit1:
                                substituted_lit = lit.substitute(unifier)
                                resolvent_literals.add(substituted_lit)
                        
                        # Add all literals from clause2 except lit2
                        for lit in clause2.literals:
                            if lit != lit2:
                                substituted_lit = lit.substitute(unifier)
                                resolvent_literals.add(substituted_lit)
                        
                        resolvent = FirstOrderClause(resolvent_literals)
                        resolvents.append(resolvent)
        
        return resolvents
    
    @staticmethod
    def can_resolve(clause1: FirstOrderClause, clause2: FirstOrderClause) -> bool:
        """Check if two clauses can be resolved"""
        for lit1 in clause1.literals:
            for lit2 in clause2.literals:
                if lit1.is_complementary(lit2):
                    unifier = FirstOrderUnification.unify(lit1, lit2)
                    if unifier is not None:
                        return True
        return False
    
    @staticmethod
    def find_resolvable_pairs(clauses: List[FirstOrderClause]) -> List[Tuple[FirstOrderClause, FirstOrderClause]]:
        """Find all pairs of clauses that can be resolved"""
        pairs = []
        for i, clause1 in enumerate(clauses):
            for j, clause2 in enumerate(clauses):
                if i < j and FirstOrderResolution.can_resolve(clause1, clause2):
                    pairs.append((clause1, clause2))
        return pairs

# Example usage
x = Variable('x')
y = Variable('y')
socrates = Constant('Socrates')
plato = Constant('Plato')

# Create literals
human_x = FirstOrderLiteral('Human', [x])
mortal_x = FirstOrderLiteral('Mortal', [x])
human_socrates = FirstOrderLiteral('Human', [socrates])
not_mortal_socrates = FirstOrderLiteral('Mortal', [socrates], negated=True)

# Create clauses
clause1 = FirstOrderClause({human_x, mortal_x})  # Human(x) ∨ Mortal(x)
clause2 = FirstOrderClause({not_mortal_socrates})  # ¬Mortal(Socrates)

print("First-Order Resolution Example:")
print(f"Clause 1: {clause1}")
print(f"Clause 2: {clause2}")

# Resolve
resolvents = FirstOrderResolution.resolve(clause1, clause2)
print(f"Resolvents: {resolvents}")
```

## Preprocessing Steps

### 1. Conversion to Prenex Normal Form

```python
class PrenexNormalForm:
    """Convert formulas to prenex normal form"""
    
    @staticmethod
    def to_prenex_normal_form(formula: 'Formula') -> 'Formula':
        """Convert a formula to prenex normal form"""
        # This is a simplified implementation
        # Full PNF conversion requires more sophisticated algorithms
        
        if isinstance(formula, (UniversalQuantifier, ExistentialQuantifier)):
            return formula
        
        elif isinstance(formula, Negation):
            if isinstance(formula.formula, UniversalQuantifier):
                # ¬∀x φ(x) ≡ ∃x ¬φ(x)
                return ExistentialQuantifier(
                    formula.formula.variable,
                    PrenexNormalForm.to_prenex_normal_form(Negation(formula.formula.formula))
                )
            elif isinstance(formula.formula, ExistentialQuantifier):
                # ¬∃x φ(x) ≡ ∀x ¬φ(x)
                return UniversalQuantifier(
                    formula.formula.variable,
                    PrenexNormalForm.to_prenex_normal_form(Negation(formula.formula.formula))
                )
            else:
                return Negation(PrenexNormalForm.to_prenex_normal_form(formula.formula))
        
        elif isinstance(formula, (Conjunction, Disjunction, Implication, Biconditional)):
            # Convert subformulas to PNF first
            left_pnf = PrenexNormalForm.to_prenex_normal_form(formula.left)
            right_pnf = PrenexNormalForm.to_prenex_normal_form(formula.right)
            
            # Then combine (simplified)
            return formula.__class__(left_pnf, right_pnf)
        
        return formula

# Example PNF conversion
x = Variable('x')
y = Variable('y')
P = lambda x: AtomicFormula('P', [x])
Q = lambda x: AtomicFormula('Q', [x])

# Formula: (∀x P(x)) ∧ (∃y Q(y))
formula = Conjunction(
    UniversalQuantifier('x', P(x)),
    ExistentialQuantifier('y', Q(y))
)

pnf_formula = PrenexNormalForm.to_prenex_normal_form(formula)
print(f"Original: {formula}")
print(f"PNF: {pnf_formula}")
```

### 2. Skolemization

```python
class Skolemization:
    """Convert existential quantifiers to Skolem functions"""
    
    def __init__(self):
        self.skolem_counter = 0
        self.universal_variables = []
    
    def skolemize(self, formula: 'Formula') -> 'Formula':
        """Convert existential quantifiers to Skolem functions"""
        return self._skolemize_formula(formula)
    
    def _skolemize_formula(self, formula: 'Formula') -> 'Formula':
        """Recursively skolemize a formula"""
        
        if isinstance(formula, UniversalQuantifier):
            # Add variable to universal list
            self.universal_variables.append(formula.variable)
            result = self._skolemize_formula(formula.formula)
            self.universal_variables.pop()  # Remove variable
            return UniversalQuantifier(formula.variable, result)
        
        elif isinstance(formula, ExistentialQuantifier):
            # Replace existential quantifier with Skolem function
            skolem_term = self._create_skolem_term(formula.variable)
            
            # Substitute the variable with the Skolem term
            substituted = self._substitute_variable(formula.formula, formula.variable, skolem_term)
            
            return self._skolemize_formula(substituted)
        
        elif isinstance(formula, Negation):
            return Negation(self._skolemize_formula(formula.formula))
        
        elif isinstance(formula, (Conjunction, Disjunction, Implication, Biconditional)):
            left_skolemized = self._skolemize_formula(formula.left)
            right_skolemized = self._skolemize_formula(formula.right)
            return formula.__class__(left_skolemized, right_skolemized)
        
        return formula
    
    def _create_skolem_term(self, variable: str) -> 'Term':
        """Create a Skolem term for a variable"""
        self.skolem_counter += 1
        skolem_name = f"skolem_{self.skolem_counter}"
        
        if not self.universal_variables:
            # No universal quantifiers, create constant
            return Constant(skolem_name)
        else:
            # Create function with universal variables as arguments
            args = [Variable(var) for var in self.universal_variables]
            return Function(skolem_name, args)
    
    def _substitute_variable(self, formula: 'Formula', variable: str, term: 'Term') -> 'Formula':
        """Substitute a variable with a term in a formula"""
        
        if isinstance(formula, AtomicFormula):
            substituted_args = []
            for arg in formula.arguments:
                if isinstance(arg, Variable) and arg.name == variable:
                    substituted_args.append(term)
                else:
                    substituted_args.append(arg)
            return AtomicFormula(formula.predicate, substituted_args)
        
        elif isinstance(formula, Negation):
            return Negation(self._substitute_variable(formula.formula, variable, term))
        
        elif isinstance(formula, (Conjunction, Disjunction, Implication, Biconditional)):
            left_substituted = self._substitute_variable(formula.left, variable, term)
            right_substituted = self._substitute_variable(formula.right, variable, term)
            return formula.__class__(left_substituted, right_substituted)
        
        elif isinstance(formula, (UniversalQuantifier, ExistentialQuantifier)):
            if formula.variable == variable:
                return formula  # Don't substitute bound variables
            else:
                return formula.__class__(formula.variable, 
                                       self._substitute_variable(formula.formula, variable, term))
        
        return formula

# Example skolemization
skolemizer = Skolemization()

# Formula: ∀x∃y P(x,y)
x = Variable('x')
y = Variable('y')
P_xy = AtomicFormula('P', [x, y])

formula = UniversalQuantifier('x', ExistentialQuantifier('y', P_xy))

skolemized = skolemizer.skolemize(formula)
print(f"Original: {formula}")
print(f"Skolemized: {skolemized}")
```

### 3. Conversion to Conjunctive Normal Form

```python
class CNFConverter:
    """Convert formulas to Conjunctive Normal Form"""
    
    @staticmethod
    def to_cnf(formula: 'Formula') -> List[FirstOrderClause]:
        """Convert a formula to CNF"""
        # This is a simplified implementation
        # Full CNF conversion requires more sophisticated algorithms
        
        if isinstance(formula, AtomicFormula):
            literal = FirstOrderLiteral(formula.predicate, formula.arguments)
            return [FirstOrderClause({literal})]
        
        elif isinstance(formula, Negation):
            if isinstance(formula.formula, AtomicFormula):
                literal = FirstOrderLiteral(formula.formula.predicate, formula.formula.arguments, negated=True)
                return [FirstOrderClause({literal})]
            else:
                # Handle double negation and De Morgan's laws
                return CNFConverter.to_cnf(formula.formula)
        
        elif isinstance(formula, Conjunction):
            left_clauses = CNFConverter.to_cnf(formula.left)
            right_clauses = CNFConverter.to_cnf(formula.right)
            return left_clauses + right_clauses
        
        elif isinstance(formula, Disjunction):
            left_clauses = CNFConverter.to_cnf(formula.left)
            right_clauses = CNFConverter.to_cnf(formula.right)
            
            # Distribute disjunction over conjunction
            result_clauses = []
            for left_clause in left_clauses:
                for right_clause in right_clauses:
                    combined_literals = left_clause.literals | right_clause.literals
                    result_clauses.append(FirstOrderClause(combined_literals))
            
            return result_clauses
        
        elif isinstance(formula, Implication):
            # P → Q is equivalent to ¬P ∨ Q
            equivalent = Disjunction(Negation(formula.left), formula.right)
            return CNFConverter.to_cnf(equivalent)
        
        elif isinstance(formula, Biconditional):
            # P ↔ Q is equivalent to (P → Q) ∧ (Q → P)
            left_imp = Implication(formula.left, formula.right)
            right_imp = Implication(formula.right, formula.left)
            equivalent = Conjunction(left_imp, right_imp)
            return CNFConverter.to_cnf(equivalent)
        
        return []

# Example CNF conversion
P = AtomicFormula('P', [x])
Q = AtomicFormula('Q', [x])
R = AtomicFormula('R', [x])

# Formula: (P ∨ Q) ∧ (¬P ∨ R)
formula = Conjunction(
    Disjunction(P, Q),
    Disjunction(Negation(P), R)
)

cnf_clauses = CNFConverter.to_cnf(formula)
print(f"Original: {formula}")
print("CNF Clauses:")
for clause in cnf_clauses:
    print(f"  {clause}")
```

## Complete First-Order Resolution System

```python
class FirstOrderResolutionSystem:
    """Complete first-order resolution system"""
    
    def __init__(self):
        self.clauses = []
        self.resolution_history = []
    
    def add_clause(self, clause: FirstOrderClause):
        """Add a clause to the system"""
        self.clauses.append(clause)
    
    def add_formula(self, formula: 'Formula'):
        """Convert a formula to CNF and add its clauses"""
        # Convert to PNF
        pnf_formula = PrenexNormalForm.to_prenex_normal_form(formula)
        
        # Skolemize
        skolemizer = Skolemization()
        skolemized_formula = skolemizer.skolemize(pnf_formula)
        
        # Convert to CNF
        cnf_clauses = CNFConverter.to_cnf(skolemized_formula)
        
        for clause in cnf_clauses:
            self.add_clause(clause)
    
    def resolution_refutation(self, goal_formula: 'Formula') -> Tuple[bool, List[str]]:
        """
        Use resolution refutation to prove a goal
        
        Args:
            goal_formula: The formula to prove
            
        Returns:
            Tuple of (is_provable, proof_steps)
        """
        # Negate the goal and add to clauses
        negated_goal = CNFConverter.to_cnf(Negation(goal_formula))
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
            resolvable_pairs = FirstOrderResolution.find_resolvable_pairs(self.clauses)
            
            for clause1, clause2 in resolvable_pairs:
                resolvents = FirstOrderResolution.resolve(clause1, clause2)
                
                for resolvent in resolvents:
                    if resolvent not in self.clauses and resolvent not in new_clauses:
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
        empty_clause = FirstOrderClause(set())
        is_provable, _ = self.resolution_refutation(empty_clause)
        return not is_provable  # If empty clause is provable, set is unsatisfiable

# Example usage
resolution_system = FirstOrderResolutionSystem()

# Add formulas: ∀x (Human(x) → Mortal(x)), Human(Socrates)
x = Variable('x')
socrates = Constant('Socrates')

human_x = AtomicFormula('Human', [x])
mortal_x = AtomicFormula('Mortal', [x])

# Add universal implication: ∀x (Human(x) → Mortal(x))
universal_implication = UniversalQuantifier('x', Implication(human_x, mortal_x))
resolution_system.add_formula(universal_implication)

# Add fact: Human(Socrates)
human_socrates = AtomicFormula('Human', [socrates])
resolution_system.add_formula(human_socrates)

# Try to prove Mortal(Socrates)
goal = AtomicFormula('Mortal', [socrates])
is_provable, proof = resolution_system.resolution_refutation(goal)

print(f"Can prove {goal}: {is_provable}")
print("Proof steps:")
for i, step in enumerate(proof, 1):
    print(f"{i}. {step}")
```

## Applications

### 1. Logic Programming

```python
class LogicProgram:
    """Simple logic programming system using first-order resolution"""
    
    def __init__(self):
        self.resolution_system = FirstOrderResolutionSystem()
        self.rules = []
        self.facts = []
    
    def add_rule(self, head: str, body: List[str]):
        """Add a rule: head :- body"""
        # Convert to implication: body → head
        if not body:  # Fact
            self.facts.append(head)
        else:
            # Create implication
            antecedent = AtomicFormula(body[0], [])
            for literal in body[1:]:
                antecedent = Conjunction(antecedent, AtomicFormula(literal, []))
            implication = Implication(antecedent, AtomicFormula(head, []))
            self.rules.append(implication)
            self.resolution_system.add_formula(implication)
    
    def add_fact(self, fact: str):
        """Add a fact"""
        self.facts.append(fact)
        self.resolution_system.add_formula(AtomicFormula(fact, []))
    
    def query(self, goal: str) -> bool:
        """Answer a query using first-order resolution"""
        return self.resolution_system.resolution_refutation(AtomicFormula(goal, []))[0]

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

### 2. Automated Theorem Proving

```python
class FirstOrderTheoremProver:
    """Automated theorem prover using first-order resolution"""
    
    def __init__(self):
        self.resolution_system = FirstOrderResolutionSystem()
        self.axioms = []
        self.theorems = []
    
    def add_axiom(self, formula: 'Formula'):
        """Add an axiom to the system"""
        self.axioms.append(formula)
        self.resolution_system.add_formula(formula)
    
    def prove(self, goal: 'Formula') -> bool:
        """Attempt to prove a goal"""
        return self.resolution_system.resolution_refutation(goal)[0]
    
    def prove_with_proof(self, goal: 'Formula') -> Tuple[bool, List[str]]:
        """Attempt to prove a goal and return the proof"""
        return self.resolution_system.resolution_refutation(goal)

# Example theorem proving
prover = FirstOrderTheoremProver()

# Add axioms
x = Variable('x')
y = Variable('y')
socrates = Constant('Socrates')

# Axiom 1: ∀x (Human(x) → Mortal(x))
human_x = AtomicFormula('Human', [x])
mortal_x = AtomicFormula('Mortal', [x])
axiom1 = UniversalQuantifier('x', Implication(human_x, mortal_x))
prover.add_axiom(axiom1)

# Axiom 2: Human(Socrates)
human_socrates = AtomicFormula('Human', [socrates])
prover.add_axiom(human_socrates)

# Try to prove Mortal(Socrates)
goal = AtomicFormula('Mortal', [socrates])
can_prove, proof = prover.prove_with_proof(goal)

print(f"Can prove {goal}: {can_prove}")
if can_prove:
    print("Proof:")
    for step in proof:
        print(f"  {step}")
```

## Key Concepts Summary

| Concept | Description | Mathematical Form | Python Implementation |
|---------|-------------|-------------------|----------------------|
| **First-Order Resolution** | Resolve with unification | C₁∨L₁, C₂∨¬L₂ ⊢ σ(C₁∨C₂) | `FirstOrderResolution.resolve()` |
| **Unification** | Find MGU of terms | σ(t₁) = σ(t₂) | `FirstOrderUnification.unify()` |
| **Skolemization** | Eliminate existential quantifiers | ∃x φ(x) → φ(skolem()) | `Skolemization.skolemize()` |
| **Prenex Normal Form** | Move quantifiers to front | ∀x∃y φ(x,y) | `PrenexNormalForm.to_prenex_normal_form()` |
| **CNF Conversion** | Convert to clause form | (L₁∨L₂) ∧ (L₃∨L₄) | `CNFConverter.to_cnf()` |
| **Resolution Refutation** | Prove by contradiction | ¬φ, axioms ⊢ ⊥ | `resolution_refutation()` |

## Best Practices

1. **Use proper unification** with occurs check to avoid infinite loops
2. **Apply preprocessing steps** in the correct order (PNF → Skolemization → CNF)
3. **Handle variable scoping** carefully during substitutions
4. **Use resolution strategies** to improve efficiency
5. **Validate results** with multiple methods
6. **Consider computational complexity** for large problems

First-order resolution provides a powerful and complete method for automated reasoning in first-order logic, though it requires careful handling of its complexity and preprocessing requirements. 