# Inference Rules

## Introduction

Inference rules are formal procedures for deriving new logical statements from existing ones. They form the foundation of logical reasoning and automated theorem proving systems. Understanding inference rules is crucial for building AI systems that can perform deductive reasoning.

## What are Inference Rules?

Inference rules are formal patterns that allow us to derive valid conclusions from given premises. They ensure that if the premises are true, the conclusion is guaranteed to be true.

### Key Properties

1. **Soundness**: An inference rule is sound if it only derives true conclusions from true premises
2. **Completeness**: An inference system is complete if it can derive all valid conclusions
3. **Validity**: The conclusion follows logically from the premises

## Basic Inference Rules

### 1. Modus Ponens

**Rule**: From P → Q and P, infer Q

**Mathematical Form**:
```
P → Q
P
-----
Q
```

**Explanation**: If we know that P implies Q, and we know that P is true, then we can conclude that Q is true.

```python
class ModusPonens:
    """Implement Modus Ponens inference rule"""
    
    @staticmethod
    def apply(premise1: Implication, premise2: Formula) -> Formula:
        """Apply Modus Ponens: from P → Q and P, infer Q"""
        if isinstance(premise1, Implication):
            # Check if premise2 matches the antecedent
            if premise1.left == premise2:
                return premise1.right
            else:
                raise ValueError("Second premise does not match the antecedent")
        else:
            raise ValueError("First premise must be an implication")
    
    @staticmethod
    def is_applicable(premise1: Formula, premise2: Formula) -> bool:
        """Check if Modus Ponens can be applied to the premises"""
        return (isinstance(premise1, Implication) and 
                premise1.left == premise2)
    
    @staticmethod
    def get_conclusion(premise1: Implication, premise2: Formula) -> Formula:
        """Get the conclusion of Modus Ponens"""
        if ModusPonens.is_applicable(premise1, premise2):
            return premise1.right
        else:
            raise ValueError("Modus Ponens not applicable")

# Example usage
P = AtomicFormula('P')
Q = AtomicFormula('Q')

implication = Implication(P, Q)
premise = P

if ModusPonens.is_applicable(implication, premise):
    conclusion = ModusPonens.get_conclusion(implication, premise)
    print(f"Modus Ponens: {implication}, {premise} ⊢ {conclusion}")
```

### 2. Modus Tollens

**Rule**: From P → Q and ¬Q, infer ¬P

**Mathematical Form**:
```
P → Q
¬Q
-----
¬P
```

**Explanation**: If P implies Q, and Q is false, then P must be false.

```python
class ModusTollens:
    """Implement Modus Tollens inference rule"""
    
    @staticmethod
    def apply(premise1: Implication, premise2: Negation) -> Negation:
        """Apply Modus Tollens: from P → Q and ¬Q, infer ¬P"""
        if isinstance(premise1, Implication) and isinstance(premise2, Negation):
            # Check if premise2 negates the consequent
            if premise1.right == premise2.formula:
                return Negation(premise1.left)
            else:
                raise ValueError("Second premise does not negate the consequent")
        else:
            raise ValueError("Premises must be implication and negation")
    
    @staticmethod
    def is_applicable(premise1: Formula, premise2: Formula) -> bool:
        """Check if Modus Tollens can be applied"""
        return (isinstance(premise1, Implication) and 
                isinstance(premise2, Negation) and
                premise1.right == premise2.formula)
    
    @staticmethod
    def get_conclusion(premise1: Implication, premise2: Negation) -> Negation:
        """Get the conclusion of Modus Tollens"""
        if ModusTollens.is_applicable(premise1, premise2):
            return Negation(premise1.left)
        else:
            raise ValueError("Modus Tollens not applicable")

# Example usage
not_Q = Negation(Q)

if ModusTollens.is_applicable(implication, not_Q):
    conclusion = ModusTollens.get_conclusion(implication, not_Q)
    print(f"Modus Tollens: {implication}, {not_Q} ⊢ {conclusion}")
```

### 3. Hypothetical Syllogism

**Rule**: From P → Q and Q → R, infer P → R

**Mathematical Form**:
```
P → Q
Q → R
-----
P → R
```

**Explanation**: If P implies Q, and Q implies R, then P implies R.

```python
class HypotheticalSyllogism:
    """Implement Hypothetical Syllogism inference rule"""
    
    @staticmethod
    def apply(premise1: Implication, premise2: Implication) -> Implication:
        """Apply Hypothetical Syllogism: from P → Q and Q → R, infer P → R"""
        if isinstance(premise1, Implication) and isinstance(premise2, Implication):
            # Check if the consequent of first matches antecedent of second
            if premise1.right == premise2.left:
                return Implication(premise1.left, premise2.right)
            else:
                raise ValueError("Consequent of first premise does not match antecedent of second")
        else:
            raise ValueError("Both premises must be implications")
    
    @staticmethod
    def is_applicable(premise1: Formula, premise2: Formula) -> bool:
        """Check if Hypothetical Syllogism can be applied"""
        return (isinstance(premise1, Implication) and 
                isinstance(premise2, Implication) and
                premise1.right == premise2.left)
    
    @staticmethod
    def get_conclusion(premise1: Implication, premise2: Implication) -> Implication:
        """Get the conclusion of Hypothetical Syllogism"""
        if HypotheticalSyllogism.is_applicable(premise1, premise2):
            return Implication(premise1.left, premise2.right)
        else:
            raise ValueError("Hypothetical Syllogism not applicable")

# Example usage
R = AtomicFormula('R')
implication2 = Implication(Q, R)

if HypotheticalSyllogism.is_applicable(implication, implication2):
    conclusion = HypotheticalSyllogism.get_conclusion(implication, implication2)
    print(f"Hypothetical Syllogism: {implication}, {implication2} ⊢ {conclusion}")
```

### 4. Disjunctive Syllogism

**Rule**: From P ∨ Q and ¬P, infer Q

**Mathematical Form**:
```
P ∨ Q
¬P
-----
Q
```

**Explanation**: If either P or Q is true, and P is false, then Q must be true.

```python
class DisjunctiveSyllogism:
    """Implement Disjunctive Syllogism inference rule"""
    
    @staticmethod
    def apply(premise1: Disjunction, premise2: Negation) -> Formula:
        """Apply Disjunctive Syllogism: from P ∨ Q and ¬P, infer Q"""
        if isinstance(premise1, Disjunction) and isinstance(premise2, Negation):
            # Check if premise2 negates one of the disjuncts
            if premise1.left == premise2.formula:
                return premise1.right
            elif premise1.right == premise2.formula:
                return premise1.left
            else:
                raise ValueError("Second premise does not negate either disjunct")
        else:
            raise ValueError("Premises must be disjunction and negation")
    
    @staticmethod
    def is_applicable(premise1: Formula, premise2: Formula) -> bool:
        """Check if Disjunctive Syllogism can be applied"""
        return (isinstance(premise1, Disjunction) and 
                isinstance(premise2, Negation) and
                (premise1.left == premise2.formula or 
                 premise1.right == premise2.formula))
    
    @staticmethod
    def get_conclusion(premise1: Disjunction, premise2: Negation) -> Formula:
        """Get the conclusion of Disjunctive Syllogism"""
        if DisjunctiveSyllogism.is_applicable(premise1, premise2):
            if premise1.left == premise2.formula:
                return premise1.right
            else:
                return premise1.left
        else:
            raise ValueError("Disjunctive Syllogism not applicable")

# Example usage
disjunction = Disjunction(P, Q)
not_P = Negation(P)

if DisjunctiveSyllogism.is_applicable(disjunction, not_P):
    conclusion = DisjunctiveSyllogism.get_conclusion(disjunction, not_P)
    print(f"Disjunctive Syllogism: {disjunction}, {not_P} ⊢ {conclusion}")
```

### 5. Addition

**Rule**: From P, infer P ∨ Q

**Mathematical Form**:
```
P
-----
P ∨ Q
```

**Explanation**: If P is true, then "P or Q" is also true, regardless of Q.

```python
class Addition:
    """Implement Addition inference rule"""
    
    @staticmethod
    def apply(premise: Formula, additional_formula: Formula) -> Disjunction:
        """Apply Addition: from P, infer P ∨ Q"""
        return Disjunction(premise, additional_formula)
    
    @staticmethod
    def get_conclusion(premise: Formula, additional_formula: Formula) -> Disjunction:
        """Get the conclusion of Addition"""
        return Disjunction(premise, additional_formula)

# Example usage
conclusion = Addition.get_conclusion(P, Q)
print(f"Addition: {P} ⊢ {conclusion}")
```

### 6. Simplification

**Rule**: From P ∧ Q, infer P

**Mathematical Form**:
```
P ∧ Q
-----
P
```

**Explanation**: If both P and Q are true, then P is true.

```python
class Simplification:
    """Implement Simplification inference rule"""
    
    @staticmethod
    def apply(premise: Conjunction, which_part: str = 'left') -> Formula:
        """Apply Simplification: from P ∧ Q, infer P or Q"""
        if isinstance(premise, Conjunction):
            if which_part == 'left':
                return premise.left
            elif which_part == 'right':
                return premise.right
            else:
                raise ValueError("which_part must be 'left' or 'right'")
        else:
            raise ValueError("Premise must be a conjunction")
    
    @staticmethod
    def get_left_conclusion(premise: Conjunction) -> Formula:
        """Get the left part of a conjunction"""
        if isinstance(premise, Conjunction):
            return premise.left
        else:
            raise ValueError("Premise must be a conjunction")
    
    @staticmethod
    def get_right_conclusion(premise: Conjunction) -> Formula:
        """Get the right part of a conjunction"""
        if isinstance(premise, Conjunction):
            return premise.right
        else:
            raise ValueError("Premise must be a conjunction")

# Example usage
conjunction = Conjunction(P, Q)
left_conclusion = Simplification.get_left_conclusion(conjunction)
right_conclusion = Simplification.get_right_conclusion(conjunction)
print(f"Simplification: {conjunction} ⊢ {left_conclusion}")
print(f"Simplification: {conjunction} ⊢ {right_conclusion}")
```

## Advanced Inference Rules

### 7. Conjunction

**Rule**: From P and Q, infer P ∧ Q

**Mathematical Form**:
```
P
Q
-----
P ∧ Q
```

```python
class ConjunctionRule:
    """Implement Conjunction inference rule"""
    
    @staticmethod
    def apply(premise1: Formula, premise2: Formula) -> Conjunction:
        """Apply Conjunction: from P and Q, infer P ∧ Q"""
        return Conjunction(premise1, premise2)
    
    @staticmethod
    def get_conclusion(premise1: Formula, premise2: Formula) -> Conjunction:
        """Get the conclusion of Conjunction"""
        return Conjunction(premise1, premise2)

# Example usage
conclusion = ConjunctionRule.get_conclusion(P, Q)
print(f"Conjunction: {P}, {Q} ⊢ {conclusion}")
```

### 8. Resolution

**Rule**: From P ∨ Q and ¬P ∨ R, infer Q ∨ R

**Mathematical Form**:
```
P ∨ Q
¬P ∨ R
-----
Q ∨ R
```

```python
class Resolution:
    """Implement Resolution inference rule"""
    
    @staticmethod
    def apply(premise1: Disjunction, premise2: Disjunction) -> Disjunction:
        """Apply Resolution: from P ∨ Q and ¬P ∨ R, infer Q ∨ R"""
        if isinstance(premise1, Disjunction) and isinstance(premise2, Disjunction):
            # Find complementary literals
            if isinstance(premise1.left, Negation) and premise1.left.formula == premise2.left:
                return Disjunction(premise1.right, premise2.right)
            elif isinstance(premise1.left, Negation) and premise1.left.formula == premise2.right:
                return Disjunction(premise1.right, premise2.left)
            elif isinstance(premise1.right, Negation) and premise1.right.formula == premise2.left:
                return Disjunction(premise1.left, premise2.right)
            elif isinstance(premise1.right, Negation) and premise1.right.formula == premise2.right:
                return Disjunction(premise1.left, premise2.left)
            else:
                raise ValueError("No complementary literals found")
        else:
            raise ValueError("Both premises must be disjunctions")
    
    @staticmethod
    def is_applicable(premise1: Disjunction, premise2: Disjunction) -> bool:
        """Check if Resolution can be applied"""
        if not (isinstance(premise1, Disjunction) and isinstance(premise2, Disjunction)):
            return False
        
        # Check for complementary literals
        literals1 = [premise1.left, premise1.right]
        literals2 = [premise2.left, premise2.right]
        
        for lit1 in literals1:
            for lit2 in literals2:
                if isinstance(lit1, Negation) and lit1.formula == lit2:
                    return True
                elif isinstance(lit2, Negation) and lit2.formula == lit1:
                    return True
        
        return False

# Example usage
not_P = Negation(P)
disjunction1 = Disjunction(not_P, Q)
disjunction2 = Disjunction(P, R)

if Resolution.is_applicable(disjunction1, disjunction2):
    conclusion = Resolution.apply(disjunction1, disjunction2)
    print(f"Resolution: {disjunction1}, {disjunction2} ⊢ {conclusion}")
```

## Inference System

Let's create a comprehensive inference system that can apply multiple rules:

```python
class InferenceSystem:
    """A system for applying multiple inference rules"""
    
    def __init__(self):
        self.rules = {
            'modus_ponens': ModusPonens,
            'modus_tollens': ModusTollens,
            'hypothetical_syllogism': HypotheticalSyllogism,
            'disjunctive_syllogism': DisjunctiveSyllogism,
            'addition': Addition,
            'simplification': Simplification,
            'conjunction': ConjunctionRule,
            'resolution': Resolution
        }
    
    def apply_rule(self, rule_name: str, premises: List[Formula], **kwargs) -> Formula:
        """Apply a specific inference rule"""
        if rule_name not in self.rules:
            raise ValueError(f"Unknown rule: {rule_name}")
        
        rule_class = self.rules[rule_name]
        
        if rule_name == 'modus_ponens':
            return rule_class.apply(premises[0], premises[1])
        elif rule_name == 'modus_tollens':
            return rule_class.apply(premises[0], premises[1])
        elif rule_name == 'hypothetical_syllogism':
            return rule_class.apply(premises[0], premises[1])
        elif rule_name == 'disjunctive_syllogism':
            return rule_class.apply(premises[0], premises[1])
        elif rule_name == 'addition':
            return rule_class.apply(premises[0], premises[1])
        elif rule_name == 'simplification':
            return rule_class.apply(premises[0], kwargs.get('which_part', 'left'))
        elif rule_name == 'conjunction':
            return rule_class.apply(premises[0], premises[1])
        elif rule_name == 'resolution':
            return rule_class.apply(premises[0], premises[1])
    
    def find_applicable_rules(self, premises: List[Formula]) -> List[str]:
        """Find all rules that can be applied to the given premises"""
        applicable_rules = []
        
        for rule_name, rule_class in self.rules.items():
            try:
                if rule_name == 'modus_ponens' and len(premises) >= 2:
                    if ModusPonens.is_applicable(premises[0], premises[1]):
                        applicable_rules.append(rule_name)
                elif rule_name == 'modus_tollens' and len(premises) >= 2:
                    if ModusTollens.is_applicable(premises[0], premises[1]):
                        applicable_rules.append(rule_name)
                elif rule_name == 'hypothetical_syllogism' and len(premises) >= 2:
                    if HypotheticalSyllogism.is_applicable(premises[0], premises[1]):
                        applicable_rules.append(rule_name)
                elif rule_name == 'disjunctive_syllogism' and len(premises) >= 2:
                    if DisjunctiveSyllogism.is_applicable(premises[0], premises[1]):
                        applicable_rules.append(rule_name)
                elif rule_name == 'addition' and len(premises) >= 1:
                    applicable_rules.append(rule_name)
                elif rule_name == 'simplification' and len(premises) >= 1:
                    if isinstance(premises[0], Conjunction):
                        applicable_rules.append(rule_name)
                elif rule_name == 'conjunction' and len(premises) >= 2:
                    applicable_rules.append(rule_name)
                elif rule_name == 'resolution' and len(premises) >= 2:
                    if Resolution.is_applicable(premises[0], premises[1]):
                        applicable_rules.append(rule_name)
            except:
                continue
        
        return applicable_rules
    
    def forward_chain(self, premises: List[Formula], max_steps: int = 10) -> List[Formula]:
        """Perform forward chaining inference"""
        knowledge_base = premises.copy()
        new_conclusions = []
        
        for step in range(max_steps):
            step_conclusions = []
            
            # Try to apply rules to current knowledge base
            for i in range(len(knowledge_base)):
                for j in range(i + 1, len(knowledge_base)):
                    applicable_rules = self.find_applicable_rules([knowledge_base[i], knowledge_base[j]])
                    
                    for rule_name in applicable_rules:
                        try:
                            conclusion = self.apply_rule(rule_name, [knowledge_base[i], knowledge_base[j]])
                            if conclusion not in knowledge_base and conclusion not in step_conclusions:
                                step_conclusions.append(conclusion)
                                new_conclusions.append(conclusion)
                        except:
                            continue
            
            # Add new conclusions to knowledge base
            knowledge_base.extend(step_conclusions)
            
            # If no new conclusions, stop
            if not step_conclusions:
                break
        
        return new_conclusions

# Example usage
inference_system = InferenceSystem()

# Example proof using multiple rules
premises = [
    Implication(P, Q),           # P → Q
    Implication(Q, R),           # Q → R
    P                           # P
]

print("Starting premises:")
for i, premise in enumerate(premises, 1):
    print(f"{i}. {premise}")

# Apply Hypothetical Syllogism
conclusion1 = inference_system.apply_rule('hypothetical_syllogism', [premises[0], premises[1]])
print(f"3. {conclusion1} (Hypothetical Syllogism from 1, 2)")

# Apply Modus Ponens
conclusion2 = inference_system.apply_rule('modus_ponens', [conclusion1, premises[2]])
print(f"4. {conclusion2} (Modus Ponens from 3, 1)")

# Forward chaining example
print("\nForward chaining results:")
new_conclusions = inference_system.forward_chain(premises)
for i, conclusion in enumerate(new_conclusions, 1):
    print(f"New conclusion {i}: {conclusion}")
```

## Soundness and Completeness

### Soundness

An inference rule is **sound** if it only derives true conclusions from true premises.

```python
def test_soundness():
    """Test the soundness of inference rules"""
    P = AtomicFormula('P')
    Q = AtomicFormula('Q')
    R = AtomicFormula('R')
    
    # Test Modus Ponens soundness
    implication = Implication(P, Q)
    premise = P
    
    # Create interpretation where premises are true
    interpretation = {'P': True, 'Q': True}
    
    premise1_true = implication.evaluate(interpretation)
    premise2_true = premise.evaluate(interpretation)
    
    if premise1_true and premise2_true:
        conclusion = ModusPonens.get_conclusion(implication, premise)
        conclusion_true = conclusion.evaluate(interpretation)
        print(f"Modus Ponens soundness: {conclusion_true}")
    
    # Test with false premises
    interpretation_false = {'P': True, 'Q': False}
    premise1_false = implication.evaluate(interpretation_false)
    premise2_false = premise.evaluate(interpretation_false)
    
    if premise1_false and premise2_false:
        conclusion = ModusPonens.get_conclusion(implication, premise)
        conclusion_false = conclusion.evaluate(interpretation_false)
        print(f"Modus Ponens with false premises: {conclusion_false}")

test_soundness()
```

### Completeness

An inference system is **complete** if it can derive all valid conclusions.

```python
def test_completeness():
    """Test completeness by checking if we can derive known valid conclusions"""
    P = AtomicFormula('P')
    Q = AtomicFormula('Q')
    
    # Test if we can derive P ∨ ¬P (law of excluded middle)
    # This requires more sophisticated proof techniques
    print("Completeness testing requires advanced proof methods")
    print("Basic inference rules may not be complete for all valid conclusions")

test_completeness()
```

## Key Concepts Summary

| Rule | Premises | Conclusion | Python Implementation |
|------|----------|------------|----------------------|
| **Modus Ponens** | P → Q, P | Q | `ModusPonens.apply(implication, premise)` |
| **Modus Tollens** | P → Q, ¬Q | ¬P | `ModusTollens.apply(implication, negation)` |
| **Hypothetical Syllogism** | P → Q, Q → R | P → R | `HypotheticalSyllogism.apply(imp1, imp2)` |
| **Disjunctive Syllogism** | P ∨ Q, ¬P | Q | `DisjunctiveSyllogism.apply(disjunction, negation)` |
| **Addition** | P | P ∨ Q | `Addition.apply(premise, additional)` |
| **Simplification** | P ∧ Q | P | `Simplification.apply(conjunction, 'left')` |
| **Conjunction** | P, Q | P ∧ Q | `ConjunctionRule.apply(premise1, premise2)` |
| **Resolution** | P ∨ Q, ¬P ∨ R | Q ∨ R | `Resolution.apply(disjunction1, disjunction2)` |

## Applications

### 1. Automated Theorem Proving

Inference rules are fundamental in automated theorem proving systems:

```python
class TheoremProver:
    """Simple theorem prover using inference rules"""
    
    def __init__(self):
        self.inference_system = InferenceSystem()
        self.axioms = []
        self.theorems = []
    
    def add_axiom(self, formula: Formula):
        """Add an axiom to the system"""
        self.axioms.append(formula)
    
    def prove(self, goal: Formula) -> bool:
        """Attempt to prove a goal using available axioms and inference rules"""
        premises = self.axioms.copy()
        
        # Try forward chaining
        new_conclusions = self.inference_system.forward_chain(premises)
        
        # Check if goal is in conclusions
        all_formulas = premises + new_conclusions
        return any(self.logical_equivalence(formula, goal) for formula in all_formulas)
    
    def logical_equivalence(self, formula1: Formula, formula2: Formula) -> bool:
        """Check if two formulas are logically equivalent"""
        # Simple implementation - in practice, this would be more sophisticated
        return str(formula1) == str(formula2)

# Example theorem proving
prover = TheoremProver()

# Add axioms
P = AtomicFormula('P')
Q = AtomicFormula('Q')
R = AtomicFormula('R')

prover.add_axiom(Implication(P, Q))
prover.add_axiom(Implication(Q, R))
prover.add_axiom(P)

# Try to prove R
goal = R
can_prove = prover.prove(goal)
print(f"Can prove {goal}: {can_prove}")
```

### 2. Expert Systems

Inference rules are used in expert systems for decision making:

```python
class ExpertSystem:
    """Simple expert system using inference rules"""
    
    def __init__(self):
        self.inference_system = InferenceSystem()
        self.rules = []
        self.facts = []
    
    def add_rule(self, antecedent: Formula, consequent: str):
        """Add a rule to the system"""
        self.rules.append((antecedent, consequent))
    
    def add_fact(self, fact: str):
        """Add a fact to the system"""
        self.facts.append(fact)
    
    def infer(self) -> List[str]:
        """Perform inference to derive new facts"""
        # Convert facts to formulas
        fact_formulas = [AtomicFormula(fact) for fact in self.facts]
        
        # Apply inference rules
        new_conclusions = self.inference_system.forward_chain(fact_formulas)
        
        # Convert back to strings
        new_facts = []
        for conclusion in new_conclusions:
            if isinstance(conclusion, AtomicFormula):
                new_facts.append(conclusion.variable)
        
        return new_facts

# Example expert system
expert = ExpertSystem()

# Add rules
fever = AtomicFormula('fever')
cough = AtomicFormula('cough')
flu = AtomicFormula('flu')

expert.add_rule(Conjunction(fever, cough), 'flu')

# Add facts
expert.add_fact('fever')
expert.add_fact('cough')

# Infer
conclusions = expert.infer()
print(f"Inferred facts: {conclusions}")
```

Understanding inference rules is essential for:
- Building automated reasoning systems
- Implementing expert systems
- Developing theorem provers
- Creating knowledge-based AI systems
- Understanding logical arguments

Inference rules provide the formal foundation for all deductive reasoning in AI and logic. 