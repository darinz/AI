# First Order Modus Ponens

## Introduction

First-order modus ponens extends the propositional version to handle quantified statements and complex predicates. It is a fundamental inference rule in first-order logic that enables reasoning about objects, properties, and relationships in a domain.

## What is First-Order Modus Ponens?

First-order modus ponens allows us to derive specific instances from universal statements. It combines universal quantification with instantiation to perform deductive reasoning.

### Mathematical Definition

**Rule**: From ∀x (P(x) → Q(x)) and P(a), infer Q(a)

**Formal Notation**:
```
∀x (P(x) → Q(x))
P(a)
--------
Q(a)
```

**General Form**: If we have a universal implication ∀x (φ(x) → ψ(x)) and we know that φ(a) holds for a specific object a, then we can conclude that ψ(a) holds.

## Understanding the Rule

### Key Components

1. **Universal Implication**: ∀x (P(x) → Q(x)) - "For all x, if P(x) then Q(x)"
2. **Instance**: P(a) - "P holds for the specific object a"
3. **Conclusion**: Q(a) - "Q holds for the specific object a"

### Examples in Natural Language

1. **Classical Example**:
   - Rule: "All humans are mortal"
   - Instance: "Socrates is human"
   - Conclusion: "Socrates is mortal"

2. **Mathematical Example**:
   - Rule: "All positive numbers are greater than zero"
   - Instance: "5 is a positive number"
   - Conclusion: "5 is greater than zero"

3. **Programming Example**:
   - Rule: "All valid inputs produce valid outputs"
   - Instance: "This input is valid"
   - Conclusion: "This input will produce a valid output"

## Python Implementation

### Basic First-Order Modus Ponens

```python
from typing import Dict, List, Set, Optional, Tuple
from abc import ABC, abstractmethod

class FirstOrderModusPonens:
    """Implementation of first-order modus ponens"""
    
    @staticmethod
    def apply(universal_implication: 'UniversalQuantifier', instance: 'AtomicFormula') -> Optional['AtomicFormula']:
        """
        Apply first-order modus ponens
        
        Args:
            universal_implication: Universal implication ∀x (P(x) → Q(x))
            instance: Instance P(a) of the antecedent
            
        Returns:
            Conclusion Q(a), or None if not applicable
        """
        if not isinstance(universal_implication, UniversalQuantifier):
            raise ValueError("First premise must be a universal quantifier")
        
        # Extract the quantified formula
        quantified_formula = universal_implication.formula
        
        if not isinstance(quantified_formula, Implication):
            raise ValueError("Quantified formula must be an implication")
        
        # Check if the instance matches the antecedent pattern
        antecedent = quantified_formula.left
        consequent = quantified_formula.right
        
        # Try to find a substitution that makes the antecedent match the instance
        substitution = FirstOrderModusPonens._find_substitution(antecedent, instance)
        
        if substitution is not None:
            # Apply the substitution to the consequent
            conclusion = FirstOrderModusPonens._apply_substitution(consequent, substitution)
            return conclusion
        
        return None
    
    @staticmethod
    def _find_substitution(pattern: 'Formula', instance: 'AtomicFormula') -> Optional[Dict[str, 'Term']]:
        """
        Find a substitution that makes the pattern match the instance
        
        Args:
            pattern: The pattern to match (usually the antecedent)
            instance: The instance to match against
            
        Returns:
            Substitution dictionary, or None if no match
        """
        if isinstance(pattern, AtomicFormula) and isinstance(instance, AtomicFormula):
            if pattern.predicate != instance.predicate:
                return None
            
            if len(pattern.arguments) != len(instance.arguments):
                return None
            
            substitution = {}
            
            for pattern_arg, instance_arg in zip(pattern.arguments, instance.arguments):
                if isinstance(pattern_arg, Variable):
                    # Variable can be substituted
                    if pattern_arg.name in substitution:
                        # Variable already has a substitution
                        if substitution[pattern_arg.name] != instance_arg:
                            return None  # Conflicting substitution
                    else:
                        substitution[pattern_arg.name] = instance_arg
                else:
                    # Constant or function - must match exactly
                    if pattern_arg != instance_arg:
                        return None
            
            return substitution
        
        return None
    
    @staticmethod
    def _apply_substitution(formula: 'Formula', substitution: Dict[str, 'Term']) -> 'Formula':
        """
        Apply a substitution to a formula
        
        Args:
            formula: The formula to substitute into
            substitution: The substitution to apply
            
        Returns:
            The formula with substitution applied
        """
        if isinstance(formula, AtomicFormula):
            substituted_args = []
            for arg in formula.arguments:
                if isinstance(arg, Variable) and arg.name in substitution:
                    substituted_args.append(substitution[arg.name])
                else:
                    substituted_args.append(arg)
            return AtomicFormula(formula.predicate, substituted_args)
        
        elif isinstance(formula, Negation):
            return Negation(FirstOrderModusPonens._apply_substitution(formula.formula, substitution))
        
        elif isinstance(formula, (Conjunction, Disjunction, Implication, Biconditional)):
            left_substituted = FirstOrderModusPonens._apply_substitution(formula.left, substitution)
            right_substituted = FirstOrderModusPonens._apply_substitution(formula.right, substitution)
            return formula.__class__(left_substituted, right_substituted)
        
        elif isinstance(formula, (UniversalQuantifier, ExistentialQuantifier)):
            # Don't substitute for bound variables
            new_substitution = {k: v for k, v in substitution.items() if k != formula.variable}
            return formula.__class__(formula.variable, 
                                   FirstOrderModusPonens._apply_substitution(formula.formula, new_substitution))
        
        return formula
    
    @staticmethod
    def is_applicable(universal_implication: 'UniversalQuantifier', instance: 'AtomicFormula') -> bool:
        """
        Check if first-order modus ponens can be applied
        
        Args:
            universal_implication: Universal implication
            instance: Instance to check
            
        Returns:
            True if modus ponens can be applied
        """
        try:
            return FirstOrderModusPonens.apply(universal_implication, instance) is not None
        except:
            return False
    
    @staticmethod
    def get_conclusion(universal_implication: 'UniversalQuantifier', instance: 'AtomicFormula') -> 'AtomicFormula':
        """
        Get the conclusion of first-order modus ponens
        
        Args:
            universal_implication: Universal implication
            instance: Instance
            
        Returns:
            The conclusion
        """
        conclusion = FirstOrderModusPonens.apply(universal_implication, instance)
        if conclusion is None:
            raise ValueError("First-order modus ponens not applicable")
        return conclusion

# Example usage
x = Variable('x')
socrates = Constant('Socrates')
plato = Constant('Plato')

# Create universal implication: ∀x (Human(x) → Mortal(x))
human_x = AtomicFormula('Human', [x])
mortal_x = AtomicFormula('Mortal', [x])
implication = Implication(human_x, mortal_x)
universal_implication = UniversalQuantifier('x', implication)

# Create instance: Human(Socrates)
human_socrates = AtomicFormula('Human', [socrates])

# Apply first-order modus ponens
if FirstOrderModusPonens.is_applicable(universal_implication, human_socrates):
    conclusion = FirstOrderModusPonens.get_conclusion(universal_implication, human_socrates)
    print(f"First-order Modus Ponens:")
    print(f"  {universal_implication}")
    print(f"  {human_socrates}")
    print(f"  ⊢ {conclusion}")
```

### Advanced Implementation with Unification

```python
class Unification:
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
        return Unification._unify_terms(term1, term2, {})
    
    @staticmethod
    def _unify_terms(term1: 'Term', term2: 'Term', substitution: Dict[str, 'Term']) -> Optional[Dict[str, 'Term']]:
        """Recursive unification with current substitution"""
        
        # Apply current substitution to both terms
        term1_subst = Unification._apply_substitution_to_term(term1, substitution)
        term2_subst = Unification._apply_substitution_to_term(term2, substitution)
        
        # If terms are identical, return current substitution
        if term1_subst == term2_subst:
            return substitution
        
        # If one term is a variable
        if isinstance(term1_subst, Variable):
            return Unification._unify_variable(term1_subst, term2_subst, substitution)
        elif isinstance(term2_subst, Variable):
            return Unification._unify_variable(term2_subst, term1_subst, substitution)
        
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
                result = Unification._unify_terms(arg1, arg2, current_substitution)
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
            return Unification._unify_terms(substitution[var.name], term, substitution)
        
        # Check if term contains the variable (occurs check)
        if Unification._occurs_in(var, term, substitution):
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
                return Unification._occurs_in(var, substitution[term.name], substitution)
            return False
        
        elif isinstance(term, Constant):
            return False
        
        elif isinstance(term, Function):
            for arg in term.arguments:
                if Unification._occurs_in(var, arg, substitution):
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
            substituted_args = [Unification._apply_substitution_to_term(arg, substitution) for arg in term.arguments]
            return Function(term.name, substituted_args)
        
        return term

class AdvancedFirstOrderModusPonens:
    """Advanced first-order modus ponens using unification"""
    
    @staticmethod
    def apply_with_unification(universal_implication: 'UniversalQuantifier', instance: 'AtomicFormula') -> List['AtomicFormula']:
        """
        Apply first-order modus ponens using unification
        
        Args:
            universal_implication: Universal implication
            instance: Instance
            
        Returns:
            List of possible conclusions
        """
        if not isinstance(universal_implication, UniversalQuantifier):
            raise ValueError("First premise must be a universal quantifier")
        
        quantified_formula = universal_implication.formula
        
        if not isinstance(quantified_formula, Implication):
            raise ValueError("Quantified formula must be an implication")
        
        antecedent = quantified_formula.left
        consequent = quantified_formula.right
        
        # Find unifier between antecedent and instance
        unifier = Unification.unify(antecedent, instance)
        
        if unifier is not None:
            # Apply unifier to consequent
            conclusion = AdvancedFirstOrderModusPonens._apply_substitution_to_formula(consequent, unifier)
            return [conclusion]
        
        return []
    
    @staticmethod
    def _apply_substitution_to_formula(formula: 'Formula', substitution: Dict[str, 'Term']) -> 'Formula':
        """Apply substitution to a formula"""
        
        if isinstance(formula, AtomicFormula):
            substituted_args = [Unification._apply_substitution_to_term(arg, substitution) for arg in formula.arguments]
            return AtomicFormula(formula.predicate, substituted_args)
        
        elif isinstance(formula, Negation):
            return Negation(AdvancedFirstOrderModusPonens._apply_substitution_to_formula(formula.formula, substitution))
        
        elif isinstance(formula, (Conjunction, Disjunction, Implication, Biconditional)):
            left_substituted = AdvancedFirstOrderModusPonens._apply_substitution_to_formula(formula.left, substitution)
            right_substituted = AdvancedFirstOrderModusPonens._apply_substitution_to_formula(formula.right, substitution)
            return formula.__class__(left_substituted, right_substituted)
        
        elif isinstance(formula, (UniversalQuantifier, ExistentialQuantifier)):
            # Don't substitute for bound variables
            new_substitution = {k: v for k, v in substitution.items() if k != formula.variable}
            return formula.__class__(formula.variable, 
                                   AdvancedFirstOrderModusPonens._apply_substitution_to_formula(formula.formula, new_substitution))
        
        return formula

# Example with unification
x = Variable('x')
y = Variable('y')
socrates = Constant('Socrates')
plato = Constant('Plato')

# Universal implication: ∀x (Human(x) → Mortal(x))
human_x = AtomicFormula('Human', [x])
mortal_x = AtomicFormula('Mortal', [x])
implication = Implication(human_x, mortal_x)
universal_implication = UniversalQuantifier('x', implication)

# Instance: Human(Socrates)
human_socrates = AtomicFormula('Human', [socrates])

# Apply with unification
conclusions = AdvancedFirstOrderModusPonens.apply_with_unification(universal_implication, human_socrates)
print(f"Conclusions using unification: {conclusions}")
```

## Complex Examples

### 1. Multiple Variables

```python
def demonstrate_multiple_variables():
    """Demonstrate first-order modus ponens with multiple variables"""
    x = Variable('x')
    y = Variable('y')
    socrates = Constant('Socrates')
    plato = Constant('Plato')
    
    # Universal implication: ∀x∀y (Parent(x,y) → Loves(x,y))
    parent_xy = AtomicFormula('Parent', [x, y])
    loves_xy = AtomicFormula('Loves', [x, y])
    implication = Implication(parent_xy, loves_xy)
    universal_implication = UniversalQuantifier('x', UniversalQuantifier('y', implication))
    
    # Instance: Parent(Socrates, Plato)
    parent_instance = AtomicFormula('Parent', [socrates, plato])
    
    # Apply first-order modus ponens
    if FirstOrderModusPonens.is_applicable(universal_implication, parent_instance):
        conclusion = FirstOrderModusPonens.get_conclusion(universal_implication, parent_instance)
        print(f"Multiple variables example:")
        print(f"  {universal_implication}")
        print(f"  {parent_instance}")
        print(f"  ⊢ {conclusion}")

demonstrate_multiple_variables()
```

### 2. Complex Predicates

```python
def demonstrate_complex_predicates():
    """Demonstrate first-order modus ponens with complex predicates"""
    x = Variable('x')
    y = Variable('y')
    socrates = Constant('Socrates')
    plato = Constant('Plato')
    
    # Universal implication: ∀x (Student(x) ∧ Studies(x) → Passes(x))
    student_x = AtomicFormula('Student', [x])
    studies_x = AtomicFormula('Studies', [x])
    passes_x = AtomicFormula('Passes', [x])
    
    antecedent = Conjunction(student_x, studies_x)
    implication = Implication(antecedent, passes_x)
    universal_implication = UniversalQuantifier('x', implication)
    
    # Instance: Student(Socrates) ∧ Studies(Socrates)
    student_socrates = AtomicFormula('Student', [socrates])
    studies_socrates = AtomicFormula('Studies', [socrates])
    instance = Conjunction(student_socrates, studies_socrates)
    
    print(f"Complex predicates example:")
    print(f"  Rule: {universal_implication}")
    print(f"  Instance: {instance}")
    print("  Note: This requires handling complex antecedents")

demonstrate_complex_predicates()
```

### 3. Function Terms

```python
def demonstrate_function_terms():
    """Demonstrate first-order modus ponens with function terms"""
    x = Variable('x')
    socrates = Constant('Socrates')
    
    # Universal implication: ∀x (Human(father(x)) → Mortal(father(x)))
    father_x = Function('father', [x])
    human_father_x = AtomicFormula('Human', [father_x])
    mortal_father_x = AtomicFormula('Mortal', [father_x])
    
    implication = Implication(human_father_x, mortal_father_x)
    universal_implication = UniversalQuantifier('x', implication)
    
    # Instance: Human(father(Socrates))
    father_socrates = Function('father', [socrates])
    human_father_socrates = AtomicFormula('Human', [father_socrates])
    
    # Apply first-order modus ponens
    if FirstOrderModusPonens.is_applicable(universal_implication, human_father_socrates):
        conclusion = FirstOrderModusPonens.get_conclusion(universal_implication, human_father_socrates)
        print(f"Function terms example:")
        print(f"  {universal_implication}")
        print(f"  {human_father_socrates}")
        print(f"  ⊢ {conclusion}")

demonstrate_function_terms()
```

## Applications in AI Systems

### 1. Expert Systems

```python
class FirstOrderExpertSystem:
    """Expert system using first-order modus ponens"""
    
    def __init__(self):
        self.rules = []  # List of universal implications
        self.facts = []  # List of ground facts
        self.inference_history = []
    
    def add_rule(self, universal_implication: 'UniversalQuantifier'):
        """Add a universal rule"""
        self.rules.append(universal_implication)
    
    def add_fact(self, fact: 'AtomicFormula'):
        """Add a ground fact"""
        self.facts.append(fact)
    
    def forward_chain(self, max_iterations: int = 100) -> List['AtomicFormula']:
        """Perform forward chaining using first-order modus ponens"""
        new_facts = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            cycle_new_facts = []
            
            for rule in self.rules:
                for fact in self.facts:
                    if FirstOrderModusPonens.is_applicable(rule, fact):
                        try:
                            conclusion = FirstOrderModusPonens.get_conclusion(rule, fact)
                            
                            if conclusion not in self.facts and conclusion not in cycle_new_facts:
                                cycle_new_facts.append(conclusion)
                                new_facts.append(conclusion)
                                
                                # Record inference
                                self.inference_history.append({
                                    'rule': rule,
                                    'fact': fact,
                                    'conclusion': conclusion
                                })
                        except:
                            continue
            
            # Add new facts
            self.facts.extend(cycle_new_facts)
            
            # If no new facts, stop
            if not cycle_new_facts:
                break
        
        return new_facts
    
    def explain_inference(self, fact: 'AtomicFormula') -> List[str]:
        """Explain how a fact was inferred"""
        explanation = []
        
        for inference in self.inference_history:
            if inference['conclusion'] == fact:
                explanation.append(f"Applied rule: {inference['rule']}")
                explanation.append(f"To fact: {inference['fact']}")
                explanation.append(f"Derived: {inference['conclusion']}")
                explanation.append("")
        
        return explanation

# Example expert system
expert = FirstOrderExpertSystem()

# Add rules
x = Variable('x')
y = Variable('y')
socrates = Constant('Socrates')
plato = Constant('Plato')

# Rule 1: All humans are mortal
human_x = AtomicFormula('Human', [x])
mortal_x = AtomicFormula('Mortal', [x])
rule1 = UniversalQuantifier('x', Implication(human_x, mortal_x))
expert.add_rule(rule1)

# Rule 2: All mortals die
die_x = AtomicFormula('Die', [x])
rule2 = UniversalQuantifier('x', Implication(mortal_x, die_x))
expert.add_rule(rule2)

# Add facts
expert.add_fact(AtomicFormula('Human', [socrates]))
expert.add_fact(AtomicFormula('Human', [plato]))

# Perform inference
new_facts = expert.forward_chain()
print("Expert system inference:")
for fact in new_facts:
    print(f"  Derived: {fact}")

# Explain inference
for fact in new_facts:
    explanation = expert.explain_inference(fact)
    print(f"\nExplanation for {fact}:")
    for line in explanation:
        print(f"  {line}")
```

### 2. Knowledge Representation

```python
class KnowledgeRepresentation:
    """Knowledge representation using first-order modus ponens"""
    
    def __init__(self):
        self.domain = set()
        self.predicates = {}
        self.functions = {}
        self.rules = []
        self.facts = []
    
    def add_domain_element(self, element: str):
        """Add an element to the domain"""
        self.domain.add(element)
    
    def add_predicate(self, name: str, arity: int, extension: Set[tuple]):
        """Add a predicate with its extension"""
        self.predicates[name] = {'arity': arity, 'extension': extension}
    
    def add_function(self, name: str, arity: int, mapping: Dict[tuple, str]):
        """Add a function with its mapping"""
        self.functions[name] = {'arity': arity, 'mapping': mapping}
    
    def add_rule(self, rule: 'UniversalQuantifier'):
        """Add a universal rule"""
        self.rules.append(rule)
    
    def add_fact(self, fact: 'AtomicFormula'):
        """Add a ground fact"""
        self.facts.append(fact)
    
    def query(self, query: 'AtomicFormula') -> bool:
        """Query the knowledge base"""
        # This is a simplified implementation
        # In practice, you'd use theorem proving
        
        # For now, just check if the query is directly in facts
        return query in self.facts
    
    def infer(self, max_steps: int = 10) -> List['AtomicFormula']:
        """Perform inference using first-order modus ponens"""
        expert = FirstOrderExpertSystem()
        
        for rule in self.rules:
            expert.add_rule(rule)
        
        for fact in self.facts:
            expert.add_fact(fact)
        
        return expert.forward_chain(max_steps)

# Example knowledge representation
kr = KnowledgeRepresentation()

# Add domain elements
kr.add_domain_element('Socrates')
kr.add_domain_element('Plato')
kr.add_domain_element('Aristotle')

# Add predicates
kr.add_predicate('Human', 1, {('Socrates',), ('Plato',), ('Aristotle',)})
kr.add_predicate('Mortal', 1, {('Socrates',), ('Plato',), ('Aristotle',)})
kr.add_predicate('Philosopher', 1, {('Socrates',), ('Plato',), ('Aristotle',)})

# Add rules
x = Variable('x')
human_x = AtomicFormula('Human', [x])
mortal_x = AtomicFormula('Mortal', [x])
philosopher_x = AtomicFormula('Philosopher', [x])

rule1 = UniversalQuantifier('x', Implication(human_x, mortal_x))
rule2 = UniversalQuantifier('x', Implication(philosopher_x, human_x))

kr.add_rule(rule1)
kr.add_rule(rule2)

# Add facts
socrates = Constant('Socrates')
kr.add_fact(AtomicFormula('Philosopher', [socrates]))

# Perform inference
inferences = kr.infer()
print("Knowledge representation inferences:")
for inference in inferences:
    print(f"  {inference}")
```

## Key Concepts Summary

| Concept | Description | Mathematical Form | Python Implementation |
|---------|-------------|-------------------|----------------------|
| **First-Order Modus Ponens** | From ∀x(P(x)→Q(x)) and P(a), infer Q(a) | Universal + Instance → Conclusion | `FirstOrderModusPonens.apply()` |
| **Unification** | Find substitution making terms equal | σ such that σ(t₁) = σ(t₂) | `Unification.unify()` |
| **Substitution** | Replace variables with terms | [x/t] | `_apply_substitution()` |
| **Variable Binding** | Quantifiers bind variables | ∀x, ∃x | `UniversalQuantifier`, `ExistentialQuantifier` |
| **Ground Terms** | Terms without variables | Constants, ground functions | `Constant`, ground `Function` |
| **Complex Antecedents** | Multi-literal antecedents | P(x) ∧ Q(x) → R(x) | Handle with extended rules |

## Best Practices

1. **Use proper variable scoping** to avoid conflicts
2. **Apply occurs check** during unification to prevent infinite loops
3. **Handle complex antecedents** with appropriate inference rules
4. **Validate premises** before applying modus ponens
5. **Use unification** for flexible matching
6. **Document the domain** and intended interpretations

First-order modus ponens is essential for building sophisticated reasoning systems that can handle quantified knowledge and complex relationships between objects. 