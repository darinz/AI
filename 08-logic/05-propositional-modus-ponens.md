# Propositional Modus Ponens

## Introduction

Modus Ponens is one of the most fundamental and widely used inference rules in logic. It forms the basis for deductive reasoning and is essential for automated theorem proving, expert systems, and logical programming.

## What is Modus Ponens?

Modus Ponens (Latin for "method of affirming") is an inference rule that allows us to derive a conclusion from a conditional statement and its antecedent.

### Mathematical Definition

**Rule**: From P → Q and P, infer Q

**Formal Notation**:
```
P → Q
P
-----
Q
```

**Logical Form**: If we have an implication "If P, then Q" and we know that P is true, then we can conclude that Q is true.

## Understanding the Rule

### Intuitive Explanation

Modus Ponens captures the most basic form of logical reasoning:
1. We have a rule: "If condition P is met, then result Q follows"
2. We observe that condition P is indeed met
3. Therefore, we can conclude that result Q follows

### Examples in Natural Language

1. **Medical Diagnosis**:
   - Rule: "If a patient has a fever and cough, then they have the flu"
   - Observation: "The patient has a fever and cough"
   - Conclusion: "The patient has the flu"

2. **Weather Prediction**:
   - Rule: "If it rains, then the ground will be wet"
   - Observation: "It is raining"
   - Conclusion: "The ground is wet"

3. **Programming Logic**:
   - Rule: "If x > 0, then x is positive"
   - Observation: "x = 5"
   - Conclusion: "x is positive"

## Mathematical Foundation

### Truth Table Analysis

Let's analyze Modus Ponens using truth tables:

```python
from typing import Dict, List, Set
import itertools

class ModusPonensAnalyzer:
    """Analyze Modus Ponens using truth tables"""
    
    @staticmethod
    def create_truth_table():
        """Create truth table for Modus Ponens"""
        P = AtomicFormula('P')
        Q = AtomicFormula('Q')
        implication = Implication(P, Q)
        
        print("Modus Ponens Truth Table Analysis:")
        print("P | Q | P→Q | P | Q (conclusion)")
        print("--+---+-----+---+---------------")
        
        for p_val in [False, True]:
            for q_val in [False, True]:
                interpretation = {'P': p_val, 'Q': q_val}
                imp_val = implication.evaluate(interpretation)
                p_val_str = "T" if p_val else "F"
                q_val_str = "T" if q_val else "F"
                imp_val_str = "T" if imp_val else "F"
                
                # Check if Modus Ponens is applicable (P→Q is true and P is true)
                if imp_val and p_val:
                    conclusion = "T"
                    valid = "✓"
                elif imp_val and not p_val:
                    conclusion = "?"
                    valid = "N/A"
                else:
                    conclusion = "?"
                    valid = "N/A"
                
                print(f"{p_val_str} | {q_val_str} |  {imp_val_str}  | {p_val_str} | {conclusion} {valid}")

# Run the analysis
ModusPonensAnalyzer.create_truth_table()
```

### Soundness Proof

Modus Ponens is **sound**, meaning it only derives true conclusions from true premises.

**Proof by Truth Table**:
- When P → Q is true and P is true, Q must be true
- This is the only case where Modus Ponens is applicable
- In all other cases, either the premises are not both true, or the rule is not applicable

## Python Implementation

### Basic Modus Ponens Implementation

```python
class ModusPonens:
    """Implementation of Modus Ponens inference rule"""
    
    @staticmethod
    def apply(premise1: Implication, premise2: Formula) -> Formula:
        """
        Apply Modus Ponens: from P → Q and P, infer Q
        
        Args:
            premise1: The implication P → Q
            premise2: The antecedent P
            
        Returns:
            The consequent Q
            
        Raises:
            ValueError: If premises don't match Modus Ponens pattern
        """
        if not isinstance(premise1, Implication):
            raise ValueError("First premise must be an implication")
        
        if premise1.left != premise2:
            raise ValueError("Second premise must match the antecedent of the implication")
        
        return premise1.right
    
    @staticmethod
    def is_applicable(premise1: Formula, premise2: Formula) -> bool:
        """
        Check if Modus Ponens can be applied to the given premises
        
        Args:
            premise1: First premise
            premise2: Second premise
            
        Returns:
            True if Modus Ponens can be applied, False otherwise
        """
        try:
            return (isinstance(premise1, Implication) and 
                    premise1.left == premise2)
        except:
            return False
    
    @staticmethod
    def get_conclusion(premise1: Implication, premise2: Formula) -> Formula:
        """
        Get the conclusion of Modus Ponens without applying it
        
        Args:
            premise1: The implication
            premise2: The antecedent
            
        Returns:
            The consequent
        """
        if ModusPonens.is_applicable(premise1, premise2):
            return premise1.right
        else:
            raise ValueError("Modus Ponens not applicable to these premises")
    
    @staticmethod
    def validate_premises(premise1: Formula, premise2: Formula) -> bool:
        """
        Validate that premises are in the correct form for Modus Ponens
        
        Args:
            premise1: First premise
            premise2: Second premise
            
        Returns:
            True if premises are valid, False otherwise
        """
        if not isinstance(premise1, Implication):
            return False
        
        if premise1.left != premise2:
            return False
        
        return True

# Example usage
P = AtomicFormula('P')
Q = AtomicFormula('Q')

# Create premises
implication = Implication(P, Q)
antecedent = P

# Check if Modus Ponens is applicable
if ModusPonens.is_applicable(implication, antecedent):
    conclusion = ModusPonens.get_conclusion(implication, antecedent)
    print(f"Modus Ponens: {implication}, {antecedent} ⊢ {conclusion}")
else:
    print("Modus Ponens not applicable")
```

### Advanced Modus Ponens with Error Handling

```python
class AdvancedModusPonens:
    """Advanced implementation with comprehensive error handling and validation"""
    
    @staticmethod
    def apply_safe(premise1: Formula, premise2: Formula) -> tuple[Formula, bool, str]:
        """
        Safely apply Modus Ponens with detailed error reporting
        
        Args:
            premise1: First premise
            premise2: Second premise
            
        Returns:
            Tuple of (conclusion, success, error_message)
        """
        try:
            # Validate premises
            if not isinstance(premise1, Implication):
                return None, False, "First premise must be an implication"
            
            if premise1.left != premise2:
                return None, False, f"Second premise '{premise2}' does not match antecedent '{premise1.left}'"
            
            # Apply the rule
            conclusion = premise1.right
            return conclusion, True, "Success"
            
        except Exception as e:
            return None, False, f"Error applying Modus Ponens: {str(e)}"
    
    @staticmethod
    def apply_with_verification(premise1: Implication, premise2: Formula, 
                               interpretation: Dict[str, bool]) -> tuple[Formula, bool]:
        """
        Apply Modus Ponens and verify the result under a given interpretation
        
        Args:
            premise1: The implication
            premise2: The antecedent
            interpretation: Truth assignment for variables
            
        Returns:
            Tuple of (conclusion, verification_passed)
        """
        try:
            # Apply Modus Ponens
            conclusion = ModusPonens.apply(premise1, premise2)
            
            # Verify premises are true under interpretation
            premise1_true = premise1.evaluate(interpretation)
            premise2_true = premise2.evaluate(interpretation)
            
            if premise1_true and premise2_true:
                # Verify conclusion is true
                conclusion_true = conclusion.evaluate(interpretation)
                return conclusion, conclusion_true
            else:
                return conclusion, False  # Premises not both true
                
        except Exception:
            return None, False
    
    @staticmethod
    def find_modus_ponens_pairs(formulas: List[Formula]) -> List[tuple]:
        """
        Find all possible Modus Ponens applications in a set of formulas
        
        Args:
            formulas: List of formulas to search
            
        Returns:
            List of (implication, antecedent, consequent) tuples
        """
        pairs = []
        
        for i, formula1 in enumerate(formulas):
            for j, formula2 in enumerate(formulas):
                if i != j and ModusPonens.is_applicable(formula1, formula2):
                    conclusion = ModusPonens.get_conclusion(formula1, formula2)
                    pairs.append((formula1, formula2, conclusion))
        
        return pairs

# Example usage with error handling
P = AtomicFormula('P')
Q = AtomicFormula('Q')
R = AtomicFormula('R')

# Valid application
implication = Implication(P, Q)
antecedent = P

conclusion, success, message = AdvancedModusPonens.apply_safe(implication, antecedent)
print(f"Valid case: {message}")
if success:
    print(f"Conclusion: {conclusion}")

# Invalid application
conclusion, success, message = AdvancedModusPonens.apply_safe(implication, Q)
print(f"Invalid case: {message}")

# Verification
interpretation = {'P': True, 'Q': True}
conclusion, verified = AdvancedModusPonens.apply_with_verification(implication, antecedent, interpretation)
print(f"Verification: {verified}")

# Find pairs
formulas = [implication, antecedent, Implication(Q, R), Q]
pairs = AdvancedModusPonens.find_modus_ponens_pairs(formulas)
print(f"Found {len(pairs)} Modus Ponens pairs")
```

## Applications in AI Systems

### 1. Forward Chaining

Modus Ponens is the core of forward chaining in rule-based systems:

```python
class ForwardChainingSystem:
    """Forward chaining system using Modus Ponens"""
    
    def __init__(self):
        self.rules = []  # List of (antecedent, consequent) pairs
        self.facts = set()  # Set of known facts
        self.inferred_facts = []  # History of inferences
    
    def add_rule(self, antecedent: Formula, consequent: str):
        """Add a rule to the system"""
        self.rules.append((antecedent, consequent))
    
    def add_fact(self, fact: str):
        """Add a fact to the knowledge base"""
        self.facts.add(fact)
    
    def forward_chain(self, max_iterations: int = 100) -> List[str]:
        """
        Perform forward chaining using Modus Ponens
        
        Args:
            max_iterations: Maximum number of inference cycles
            
        Returns:
            List of newly inferred facts
        """
        new_facts = []
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            cycle_new_facts = []
            
            for antecedent, consequent in self.rules:
                if consequent not in self.facts:
                    # Create interpretation from current facts
                    interpretation = {fact: True for fact in self.facts}
                    
                    # Check if antecedent is true
                    if antecedent.evaluate(interpretation):
                        self.facts.add(consequent)
                        cycle_new_facts.append(consequent)
                        new_facts.append(consequent)
            
            # If no new facts in this cycle, stop
            if not cycle_new_facts:
                break
        
        return new_facts
    
    def explain_inference(self, fact: str) -> List[str]:
        """Explain how a fact was inferred"""
        explanation = []
        
        for antecedent, consequent in self.rules:
            if consequent == fact:
                # Find the facts that made the antecedent true
                interpretation = {f: True for f in self.facts}
                if antecedent.evaluate(interpretation):
                    explanation.append(f"Rule: {antecedent} → {consequent}")
                    explanation.append(f"Antecedent '{antecedent}' is true because:")
                    
                    # Find which facts made the antecedent true
                    if isinstance(antecedent, AtomicFormula):
                        if antecedent.variable in self.facts:
                            explanation.append(f"  - {antecedent.variable} is a known fact")
                    elif isinstance(antecedent, Conjunction):
                        if antecedent.left.evaluate(interpretation):
                            explanation.append(f"  - {antecedent.left} is true")
                        if antecedent.right.evaluate(interpretation):
                            explanation.append(f"  - {antecedent.right} is true")
        
        return explanation

# Example forward chaining system
system = ForwardChainingSystem()

# Add rules
fever = AtomicFormula('fever')
cough = AtomicFormula('cough')
fatigue = AtomicFormula('fatigue')
flu = AtomicFormula('flu')
cold = AtomicFormula('cold')

system.add_rule(Conjunction(fever, Conjunction(cough, fatigue)), 'flu')
system.add_rule(Conjunction(cough, Negation(fever)), 'cold')

# Add facts
system.add_fact('fever')
system.add_fact('cough')
system.add_fact('fatigue')

# Perform inference
new_facts = system.forward_chain()
print(f"Inferred facts: {new_facts}")

# Explain inference
for fact in new_facts:
    explanation = system.explain_inference(fact)
    print(f"\nExplanation for '{fact}':")
    for line in explanation:
        print(f"  {line}")
```

### 2. Expert Systems

Modus Ponens is fundamental in expert systems:

```python
class ExpertSystem:
    """Expert system using Modus Ponens for medical diagnosis"""
    
    def __init__(self):
        self.forward_chaining = ForwardChainingSystem()
        self.confidence_scores = {}  # Track confidence in conclusions
    
    def add_rule_with_confidence(self, antecedent: Formula, consequent: str, confidence: float):
        """Add a rule with confidence score"""
        self.forward_chaining.add_rule(antecedent, consequent)
        self.confidence_scores[consequent] = confidence
    
    def add_fact_with_confidence(self, fact: str, confidence: float):
        """Add a fact with confidence score"""
        self.forward_chaining.add_fact(fact)
        self.confidence_scores[fact] = confidence
    
    def diagnose(self) -> Dict[str, float]:
        """Perform diagnosis using Modus Ponens"""
        # Perform forward chaining
        new_facts = self.forward_chaining.forward_chain()
        
        # Calculate confidence scores for conclusions
        diagnoses = {}
        for fact in new_facts:
            if fact in self.confidence_scores:
                diagnoses[fact] = self.confidence_scores[fact]
        
        return diagnoses

# Example medical expert system
medical_system = ExpertSystem()

# Add rules with confidence scores
medical_system.add_rule_with_confidence(
    Conjunction(AtomicFormula('fever'), Conjunction(AtomicFormula('cough'), AtomicFormula('fatigue'))),
    'flu', 0.8
)
medical_system.add_rule_with_confidence(
    Conjunction(AtomicFormula('cough'), Negation(AtomicFormula('fever'))),
    'cold', 0.6
)

# Add symptoms with confidence
medical_system.add_fact_with_confidence('fever', 0.9)
medical_system.add_fact_with_confidence('cough', 0.8)
medical_system.add_fact_with_confidence('fatigue', 0.7)

# Perform diagnosis
diagnoses = medical_system.diagnose()
print("Medical Diagnosis:")
for condition, confidence in diagnoses.items():
    print(f"  {condition}: {confidence:.2f} confidence")
```

### 3. Automated Theorem Proving

Modus Ponens is essential in automated theorem proving:

```python
class TheoremProver:
    """Simple theorem prover using Modus Ponens"""
    
    def __init__(self):
        self.axioms = []
        self.theorems = []
        self.proof_history = []
    
    def add_axiom(self, formula: Formula):
        """Add an axiom to the system"""
        self.axioms.append(formula)
    
    def prove_using_modus_ponens(self, goal: Formula) -> List[str]:
        """
        Attempt to prove a goal using Modus Ponens
        
        Args:
            goal: The formula to prove
            
        Returns:
            List of proof steps
        """
        proof_steps = []
        current_knowledge = self.axioms.copy()
        
        # Try to find Modus Ponens applications
        for i, formula1 in enumerate(current_knowledge):
            for j, formula2 in enumerate(current_knowledge):
                if i != j and ModusPonens.is_applicable(formula1, formula2):
                    conclusion = ModusPonens.get_conclusion(formula1, formula2)
                    
                    if conclusion not in current_knowledge:
                        current_knowledge.append(conclusion)
                        step = f"Modus Ponens: {formula1}, {formula2} ⊢ {conclusion}"
                        proof_steps.append(step)
                        
                        # Check if we've reached our goal
                        if conclusion == goal:
                            return proof_steps
        
        return proof_steps

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
proof = prover.prove_using_modus_ponens(goal)

print("Theorem Proving with Modus Ponens:")
for i, step in enumerate(proof, 1):
    print(f"{i}. {step}")
```

## Common Mistakes and Pitfalls

### 1. Confusing Modus Ponens with Modus Tollens

```python
def demonstrate_common_mistakes():
    """Demonstrate common mistakes with Modus Ponens"""
    P = AtomicFormula('P')
    Q = AtomicFormula('Q')
    
    # Correct Modus Ponens
    implication = Implication(P, Q)
    antecedent = P
    
    print("Correct Modus Ponens:")
    if ModusPonens.is_applicable(implication, antecedent):
        conclusion = ModusPonens.get_conclusion(implication, antecedent)
        print(f"  {implication}, {antecedent} ⊢ {conclusion}")
    
    # Common mistake: Modus Tollens (incorrectly called Modus Ponens)
    not_Q = Negation(Q)
    print("\nCommon mistake (this is actually Modus Tollens):")
    print(f"  {implication}, {not_Q} ⊢ ¬P")
    print("  This is NOT Modus Ponens!")
    
    # Another common mistake: affirming the consequent
    print("\nAnother common mistake (affirming the consequent):")
    print(f"  {implication}, {Q} ⊢ P")
    print("  This is a logical fallacy!")

demonstrate_common_mistakes()
```

### 2. Handling Complex Antecedents

```python
def handle_complex_antecedents():
    """Show how to handle complex antecedents in Modus Ponens"""
    P = AtomicFormula('P')
    Q = AtomicFormula('Q')
    R = AtomicFormula('R')
    
    # Complex antecedent: P ∧ Q
    complex_antecedent = Conjunction(P, Q)
    implication = Implication(complex_antecedent, R)
    
    print("Complex Antecedent Example:")
    print(f"Rule: {implication}")
    print(f"Antecedent: {complex_antecedent}")
    
    # This requires both P and Q to be true
    print("For Modus Ponens to apply, both P and Q must be true")
    
    # Check if we have both parts
    if ModusPonens.is_applicable(implication, complex_antecedent):
        conclusion = ModusPonens.get_conclusion(implication, complex_antecedent)
        print(f"Conclusion: {conclusion}")

handle_complex_antecedents()
```

## Key Concepts Summary

| Concept | Description | Mathematical Form | Python Implementation |
|---------|-------------|-------------------|----------------------|
| **Modus Ponens** | From P→Q and P, infer Q | P→Q, P ⊢ Q | `ModusPonens.apply(imp, ant)` |
| **Soundness** | Only derives true conclusions | Valid inference rule | `apply_with_verification()` |
| **Forward Chaining** | Repeated application of rules | Systematic inference | `ForwardChainingSystem` |
| **Expert Systems** | Rule-based reasoning | Knowledge representation | `ExpertSystem` |
| **Theorem Proving** | Automated proof generation | Formal verification | `TheoremProver` |

## Best Practices

1. **Always validate premises** before applying Modus Ponens
2. **Check for soundness** by verifying premises are true
3. **Use error handling** to catch invalid applications
4. **Document inference chains** for explainability
5. **Consider confidence scores** in uncertain domains
6. **Avoid logical fallacies** like affirming the consequent

Modus Ponens remains one of the most fundamental tools in logical reasoning and AI systems, providing the foundation for deductive inference and automated reasoning. 