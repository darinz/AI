# Explainability and Interpretability

## Introduction

Explainability and interpretability are crucial aspects of AI systems, especially in domains where transparency, accountability, and trust are essential. Logic-based AI systems provide natural explainability through their formal structure, making them particularly valuable for applications requiring clear reasoning processes.

## What are Explainability and Interpretability?

### Definitions

- **Explainability**: The ability to provide human-understandable explanations for AI decisions
- **Interpretability**: The degree to which a human can understand the cause of a decision
- **Transparency**: The ability to see through the decision-making process
- **Accountability**: The ability to trace decisions back to their sources

### Key Characteristics

1. **Transparent Reasoning**: Each inference step can be traced and understood
2. **Verifiable Conclusions**: Proofs can be checked for correctness
3. **Human-Readable**: Logical statements are interpretable by humans
4. **Causal Relationships**: Logic captures cause-and-effect relationships

## Why Explainability Matters

### 1. Trust and Adoption

Users are more likely to trust and adopt AI systems when they can understand how decisions are made.

### 2. Regulatory Compliance

Many industries require explainable AI for regulatory compliance (e.g., GDPR, financial regulations).

### 3. Error Detection

Explainable systems make it easier to detect and correct errors in reasoning.

### 4. Knowledge Discovery

Explanation can reveal new insights and knowledge patterns.

## Mathematical Foundations

### Logical Explanation Framework

An explanation in logic consists of:
1. **Premises**: The facts and rules used
2. **Inference Steps**: The logical reasoning process
3. **Conclusion**: The final decision or answer
4. **Justification**: Why each step is valid

### Formal Definition

Given a logical system with:
- Set of formulas Γ (premises)
- Formula φ (conclusion)
- Proof π: Γ ⊢ φ

An explanation is a tuple (Γ, π, φ) where π is a sequence of inference steps that derive φ from Γ.

## Python Implementation: Explanation Framework

```python
from typing import List, Dict, Set, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class InferenceStep:
    """Represents a single step in logical inference"""
    
    def __init__(self, rule_name: str, premises: List['Formula'], conclusion: 'Formula', justification: str):
        self.rule_name = rule_name
        self.premises = premises
        self.conclusion = conclusion
        self.justification = justification
    
    def __str__(self):
        premises_str = ", ".join(str(p) for p in self.premises)
        return f"{self.rule_name}: {premises_str} ⊢ {self.conclusion}"
    
    def __repr__(self):
        return self.__str__()

class Proof:
    """Represents a complete logical proof"""
    
    def __init__(self):
        self.steps = []
        self.assumptions = set()
        self.conclusions = set()
    
    def add_step(self, step: InferenceStep):
        """Add an inference step to the proof"""
        self.steps.append(step)
        self.conclusions.add(step.conclusion)
    
    def add_assumption(self, assumption: 'Formula'):
        """Add an assumption to the proof"""
        self.assumptions.add(assumption)
    
    def get_explanation(self) -> str:
        """Generate a human-readable explanation"""
        explanation = []
        explanation.append("Proof Explanation:")
        explanation.append("=" * 50)
        
        # List assumptions
        if self.assumptions:
            explanation.append("Assumptions:")
            for i, assumption in enumerate(self.assumptions, 1):
                explanation.append(f"  {i}. {assumption}")
            explanation.append("")
        
        # List inference steps
        explanation.append("Inference Steps:")
        for i, step in enumerate(self.steps, 1):
            explanation.append(f"  {i}. {step}")
            explanation.append(f"     Justification: {step.justification}")
            explanation.append("")
        
        # Final conclusion
        if self.steps:
            final_conclusion = self.steps[-1].conclusion
            explanation.append(f"Final Conclusion: {final_conclusion}")
        
        return "\n".join(explanation)
    
    def is_valid(self) -> bool:
        """Check if the proof is valid"""
        # This is a simplified validation
        # In practice, you'd implement more sophisticated validation
        return len(self.steps) > 0
    
    def get_dependencies(self, conclusion: 'Formula') -> Set['Formula']:
        """Get the dependencies for a specific conclusion"""
        dependencies = set()
        
        for step in self.steps:
            if step.conclusion == conclusion:
                dependencies.update(step.premises)
                # Recursively get dependencies of premises
                for premise in step.premises:
                    dependencies.update(self.get_dependencies(premise))
        
        return dependencies

class ExplainableInferenceSystem:
    """Inference system that provides explanations"""
    
    def __init__(self):
        self.rules = {}
        self.facts = set()
        self.proofs = {}
    
    def add_rule(self, name: str, rule_function, description: str):
        """Add an inference rule with description"""
        self.rules[name] = {
            'function': rule_function,
            'description': description
        }
    
    def add_fact(self, fact: 'Formula'):
        """Add a fact to the knowledge base"""
        self.facts.add(fact)
    
    def infer_with_explanation(self, goal: 'Formula') -> Tuple[bool, Proof]:
        """Perform inference and return explanation"""
        proof = Proof()
        
        # Add facts as assumptions
        for fact in self.facts:
            proof.add_assumption(fact)
        
        # Try to derive the goal
        success = self._derive_goal(goal, proof)
        
        return success, proof
    
    def _derive_goal(self, goal: 'Formula', proof: Proof) -> bool:
        """Recursively derive a goal"""
        # Check if goal is already proven
        if goal in proof.conclusions:
            return True
        
        # Check if goal is a fact
        if goal in self.facts:
            step = InferenceStep(
                "Fact", 
                [], 
                goal, 
                f"{goal} is a known fact"
            )
            proof.add_step(step)
            return True
        
        # Try to apply inference rules
        for rule_name, rule_info in self.rules.items():
            rule_func = rule_info['function']
            description = rule_info['description']
            
            # Try to apply the rule
            try:
                result = rule_func(goal, proof)
                if result:
                    premises, conclusion = result
                    step = InferenceStep(
                        rule_name,
                        premises,
                        conclusion,
                        description
                    )
                    proof.add_step(step)
                    return True
            except:
                continue
        
        return False
    
    def explain_decision(self, decision: 'Formula') -> str:
        """Explain a specific decision"""
        success, proof = self.infer_with_explanation(decision)
        
        if success:
            return proof.get_explanation()
        else:
            return f"Could not derive {decision} from available knowledge"

# Example usage
class ModusPonensRule:
    """Modus Ponens inference rule with explanation"""
    
    @staticmethod
    def apply(goal: 'Formula', proof: Proof) -> Optional[Tuple[List['Formula'], 'Formula']]:
        """Apply Modus Ponens to derive the goal"""
        # Look for implications in the proof
        for step in proof.steps:
            if isinstance(step.conclusion, Implication):
                # Check if we have the antecedent
                antecedent = step.conclusion.left
                consequent = step.conclusion.right
                
                if antecedent in proof.conclusions and consequent == goal:
                    return [step.conclusion, antecedent], goal
        
        return None

# Create explainable inference system
explainable_system = ExplainableInferenceSystem()

# Add modus ponens rule
explainable_system.add_rule(
    "Modus Ponens",
    ModusPonensRule.apply,
    "If P → Q and P, then Q"
)

# Add facts
P = AtomicFormula('P')
Q = AtomicFormula('Q')
implication = Implication(P, Q)

explainable_system.add_fact(implication)
explainable_system.add_fact(P)

# Infer with explanation
success, proof = explainable_system.infer_with_explanation(Q)
print("Explainable Inference Example:")
print(proof.get_explanation())
```

## Types of Explanations

### 1. Step-by-Step Explanations

```python
class StepByStepExplainer:
    """Provides step-by-step explanations for logical reasoning"""
    
    def __init__(self):
        self.step_counter = 0
    
    def explain_step(self, step: InferenceStep) -> str:
        """Explain a single inference step"""
        self.step_counter += 1
        
        explanation = f"Step {self.step_counter}:\n"
        explanation += f"  Rule: {step.rule_name}\n"
        explanation += f"  Premises: {', '.join(str(p) for p in step.premises)}\n"
        explanation += f"  Conclusion: {step.conclusion}\n"
        explanation += f"  Reasoning: {step.justification}\n"
        
        return explanation
    
    def explain_proof(self, proof: Proof) -> str:
        """Explain an entire proof step by step"""
        explanation = "Step-by-Step Proof Explanation:\n"
        explanation += "=" * 50 + "\n\n"
        
        for step in proof.steps:
            explanation += self.explain_step(step) + "\n"
        
        return explanation

# Example step-by-step explanation
step_explainer = StepByStepExplainer()
print("Step-by-Step Explanation:")
print(step_explainer.explain_proof(proof))
```

### 2. Natural Language Explanations

```python
class NaturalLanguageExplainer:
    """Converts logical proofs to natural language explanations"""
    
    def __init__(self):
        self.predicate_descriptions = {}
        self.rule_descriptions = {}
    
    def add_predicate_description(self, predicate: str, description: str):
        """Add a natural language description for a predicate"""
        self.predicate_descriptions[predicate] = description
    
    def add_rule_description(self, rule: str, description: str):
        """Add a natural language description for a rule"""
        self.rule_descriptions[rule] = description
    
    def formula_to_natural_language(self, formula: 'Formula') -> str:
        """Convert a formula to natural language"""
        if isinstance(formula, AtomicFormula):
            predicate = formula.predicate
            args = [str(arg) for arg in formula.arguments]
            
            if predicate in self.predicate_descriptions:
                description = self.predicate_descriptions[predicate]
                # Replace placeholders with actual arguments
                for i, arg in enumerate(args):
                    description = description.replace(f"{{{i}}}", arg)
                return description
            else:
                return f"{predicate}({', '.join(args)})"
        
        elif isinstance(formula, Negation):
            inner = self.formula_to_natural_language(formula.formula)
            return f"not {inner}"
        
        elif isinstance(formula, Conjunction):
            left = self.formula_to_natural_language(formula.left)
            right = self.formula_to_natural_language(formula.right)
            return f"{left} and {right}"
        
        elif isinstance(formula, Disjunction):
            left = self.formula_to_natural_language(formula.left)
            right = self.formula_to_natural_language(formula.right)
            return f"{left} or {right}"
        
        elif isinstance(formula, Implication):
            antecedent = self.formula_to_natural_language(formula.left)
            consequent = self.formula_to_natural_language(formula.right)
            return f"if {antecedent}, then {consequent}"
        
        elif isinstance(formula, UniversalQuantifier):
            inner = self.formula_to_natural_language(formula.formula)
            return f"for all {formula.variable}, {inner}"
        
        elif isinstance(formula, ExistentialQuantifier):
            inner = self.formula_to_natural_language(formula.formula)
            return f"there exists {formula.variable} such that {inner}"
        
        return str(formula)
    
    def explain_proof_naturally(self, proof: Proof) -> str:
        """Explain a proof in natural language"""
        explanation = "Natural Language Explanation:\n"
        explanation += "=" * 50 + "\n\n"
        
        # Explain assumptions
        if proof.assumptions:
            explanation += "We start with the following knowledge:\n"
            for assumption in proof.assumptions:
                explanation += f"  • {self.formula_to_natural_language(assumption)}\n"
            explanation += "\n"
        
        # Explain each step
        for i, step in enumerate(proof.steps, 1):
            explanation += f"Step {i}: "
            
            if step.rule_name in self.rule_descriptions:
                rule_desc = self.rule_descriptions[step.rule_name]
            else:
                rule_desc = step.rule_name
            
            explanation += f"Using {rule_desc}, "
            
            if step.premises:
                premises_nl = [self.formula_to_natural_language(p) for p in step.premises]
                explanation += f"from {', '.join(premises_nl)}, "
            
            conclusion_nl = self.formula_to_natural_language(step.conclusion)
            explanation += f"we conclude that {conclusion_nl}.\n\n"
        
        return explanation

# Example natural language explanation
nl_explainer = NaturalLanguageExplainer()

# Add descriptions
nl_explainer.add_predicate_description('Human', '{0} is human')
nl_explainer.add_predicate_description('Mortal', '{0} is mortal')
nl_explainer.add_rule_description('Modus Ponens', 'the rule that if P implies Q and P is true, then Q is true')

print("Natural Language Explanation:")
print(nl_explainer.explain_proof_naturally(proof))
```

### 3. Visual Explanations

```python
class VisualExplainer:
    """Creates visual representations of logical proofs"""
    
    def __init__(self):
        self.node_counter = 0
    
    def proof_to_graph(self, proof: Proof) -> str:
        """Convert a proof to a graph representation"""
        graph = "digraph Proof {\n"
        graph += "  rankdir=TB;\n"
        graph += "  node [shape=box, style=filled, fillcolor=lightblue];\n\n"
        
        # Create nodes for assumptions
        for assumption in proof.assumptions:
            node_id = f"assumption_{self.node_counter}"
            self.node_counter += 1
            graph += f'  {node_id} [label="{assumption}"];\n'
        
        # Create nodes for inference steps
        for i, step in enumerate(proof.steps):
            step_id = f"step_{i}"
            conclusion_id = f"conclusion_{i}"
            
            # Create step node
            graph += f'  {step_id} [label="{step.rule_name}", fillcolor=lightgreen];\n'
            
            # Create conclusion node
            graph += f'  {conclusion_id} [label="{step.conclusion}"];\n'
            
            # Create edges from premises to step
            for premise in step.premises:
                # Find the node that contains this premise
                for j, prev_step in enumerate(proof.steps[:i]):
                    if prev_step.conclusion == premise:
                        graph += f'  conclusion_{j} -> {step_id};\n'
                        break
                else:
                    # Premise is an assumption
                    for assumption in proof.assumptions:
                        if assumption == premise:
                            graph += f'  assumption_{list(proof.assumptions).index(assumption)} -> {step_id};\n'
                            break
            
            # Create edge from step to conclusion
            graph += f'  {step_id} -> {conclusion_id};\n'
        
        graph += "}\n"
        return graph
    
    def proof_to_tree(self, proof: Proof) -> str:
        """Convert a proof to a tree representation"""
        tree = "Proof Tree:\n"
        tree += "=" * 30 + "\n\n"
        
        # Start with assumptions
        if proof.assumptions:
            tree += "Assumptions:\n"
            for assumption in proof.assumptions:
                tree += f"  ├─ {assumption}\n"
            tree += "\n"
        
        # Build tree for each step
        for i, step in enumerate(proof.steps):
            tree += f"Step {i+1} ({step.rule_name}):\n"
            
            # Show premises
            for j, premise in enumerate(step.premises):
                if j == len(step.premises) - 1:
                    tree += f"  ├─ {premise}\n"
                else:
                    tree += f"  ├─ {premise}\n"
            
            # Show conclusion
            tree += f"  └─ {step.conclusion}\n\n"
        
        return tree

# Example visual explanation
visual_explainer = VisualExplainer()
print("Visual Explanation (DOT format):")
print(visual_explainer.proof_to_graph(proof))
print("\nTree Representation:")
print(visual_explainer.proof_to_tree(proof))
```

## Applications

### 1. Medical Diagnosis Systems

```python
class MedicalDiagnosisExplainer:
    """Explainable medical diagnosis system"""
    
    def __init__(self):
        self.symptoms = {}
        self.conditions = {}
        self.rules = []
        self.explanations = {}
    
    def add_symptom(self, symptom: str, description: str):
        """Add a symptom with description"""
        self.symptoms[symptom] = description
    
    def add_condition(self, condition: str, description: str):
        """Add a condition with description"""
        self.conditions[condition] = description
    
    def add_rule(self, symptoms: List[str], condition: str, confidence: float, explanation: str):
        """Add a diagnostic rule"""
        self.rules.append({
            'symptoms': symptoms,
            'condition': condition,
            'confidence': confidence,
            'explanation': explanation
        })
    
    def diagnose(self, observed_symptoms: List[str]) -> List[Dict]:
        """Perform diagnosis with explanations"""
        diagnoses = []
        
        for rule in self.rules:
            # Check if all symptoms in the rule are observed
            if all(symptom in observed_symptoms for symptom in rule['symptoms']):
                diagnosis = {
                    'condition': rule['condition'],
                    'confidence': rule['confidence'],
                    'explanation': rule['explanation'],
                    'supporting_symptoms': rule['symptoms']
                }
                diagnoses.append(diagnosis)
        
        return diagnoses
    
    def explain_diagnosis(self, diagnosis: Dict) -> str:
        """Explain a specific diagnosis"""
        explanation = f"Diagnosis: {diagnosis['condition']}\n"
        explanation += f"Confidence: {diagnosis['confidence']:.2f}\n\n"
        explanation += f"Reasoning:\n"
        explanation += f"  {diagnosis['explanation']}\n\n"
        explanation += f"Supporting symptoms:\n"
        
        for symptom in diagnosis['supporting_symptoms']:
            if symptom in self.symptoms:
                explanation += f"  • {self.symptoms[symptom]}\n"
            else:
                explanation += f"  • {symptom}\n"
        
        return explanation

# Example medical diagnosis system
medical_system = MedicalDiagnosisExplainer()

# Add symptoms
medical_system.add_symptom('fever', 'Elevated body temperature')
medical_system.add_symptom('cough', 'Persistent coughing')
medical_system.add_symptom('fatigue', 'Extreme tiredness')
medical_system.add_symptom('headache', 'Pain in the head')

# Add conditions
medical_system.add_condition('flu', 'Influenza virus infection')
medical_system.add_condition('cold', 'Common cold')
medical_system.add_condition('migraine', 'Severe headache condition')

# Add diagnostic rules
medical_system.add_rule(
    ['fever', 'cough', 'fatigue'],
    'flu',
    0.8,
    'The combination of fever, cough, and fatigue strongly suggests influenza'
)

medical_system.add_rule(
    ['cough', 'headache'],
    'cold',
    0.6,
    'Cough with headache suggests a common cold'
)

medical_system.add_rule(
    ['headache'],
    'migraine',
    0.4,
    'Severe headache could indicate migraine'
)

# Perform diagnosis
observed_symptoms = ['fever', 'cough', 'fatigue']
diagnoses = medical_system.diagnose(observed_symptoms)

print("Medical Diagnosis with Explanations:")
for diagnosis in diagnoses:
    print(medical_system.explain_diagnosis(diagnosis))
    print("-" * 50)
```

### 2. Legal Reasoning Systems

```python
class LegalReasoningExplainer:
    """Explainable legal reasoning system"""
    
    def __init__(self):
        self.laws = {}
        self.cases = {}
        self.rules = []
    
    def add_law(self, law_id: str, description: str, text: str):
        """Add a law with description"""
        self.laws[law_id] = {
            'description': description,
            'text': text
        }
    
    def add_case(self, case_id: str, facts: List[str], outcome: str, reasoning: str):
        """Add a legal case"""
        self.cases[case_id] = {
            'facts': facts,
            'outcome': outcome,
            'reasoning': reasoning
        }
    
    def add_rule(self, law_id: str, conditions: List[str], conclusion: str):
        """Add a legal rule"""
        self.rules.append({
            'law_id': law_id,
            'conditions': conditions,
            'conclusion': conclusion
        })
    
    def analyze_case(self, facts: List[str]) -> Dict:
        """Analyze a legal case with explanations"""
        analysis = {
            'applicable_laws': [],
            'relevant_cases': [],
            'reasoning': [],
            'conclusion': None
        }
        
        # Find applicable laws
        for rule in self.rules:
            if all(condition in facts for condition in rule['conditions']):
                law_info = self.laws[rule['law_id']]
                analysis['applicable_laws'].append({
                    'law_id': rule['law_id'],
                    'description': law_info['description'],
                    'text': law_info['text'],
                    'conclusion': rule['conclusion']
                })
        
        # Find relevant cases
        for case_id, case in self.cases.items():
            # Simple similarity check (in practice, use more sophisticated methods)
            common_facts = set(facts) & set(case['facts'])
            if len(common_facts) > 0:
                analysis['relevant_cases'].append({
                    'case_id': case_id,
                    'facts': case['facts'],
                    'outcome': case['outcome'],
                    'reasoning': case['reasoning'],
                    'similarity': len(common_facts) / len(set(facts) | set(case['facts']))
                })
        
        # Generate reasoning
        if analysis['applicable_laws']:
            analysis['reasoning'].append("Based on applicable laws:")
            for law in analysis['applicable_laws']:
                analysis['reasoning'].append(f"  • {law['description']}: {law['conclusion']}")
        
        if analysis['relevant_cases']:
            analysis['reasoning'].append("\nBased on similar cases:")
            for case in analysis['relevant_cases']:
                analysis['reasoning'].append(f"  • {case['case_id']}: {case['outcome']}")
        
        return analysis
    
    def explain_analysis(self, analysis: Dict) -> str:
        """Explain the legal analysis"""
        explanation = "Legal Analysis Explanation:\n"
        explanation += "=" * 50 + "\n\n"
        
        # Applicable laws
        if analysis['applicable_laws']:
            explanation += "Applicable Laws:\n"
            for law in analysis['applicable_laws']:
                explanation += f"  Law: {law['description']}\n"
                explanation += f"  Text: {law['text']}\n"
                explanation += f"  Conclusion: {law['conclusion']}\n\n"
        
        # Relevant cases
        if analysis['relevant_cases']:
            explanation += "Relevant Cases:\n"
            for case in analysis['relevant_cases']:
                explanation += f"  Case: {case['case_id']}\n"
                explanation += f"  Facts: {', '.join(case['facts'])}\n"
                explanation += f"  Outcome: {case['outcome']}\n"
                explanation += f"  Reasoning: {case['reasoning']}\n"
                explanation += f"  Similarity: {case['similarity']:.2f}\n\n"
        
        # Reasoning
        if analysis['reasoning']:
            explanation += "Legal Reasoning:\n"
            for step in analysis['reasoning']:
                explanation += f"  {step}\n"
        
        return explanation

# Example legal reasoning system
legal_system = LegalReasoningExplainer()

# Add laws
legal_system.add_law(
    'LAW001',
    'Contract Formation',
    'A contract requires offer, acceptance, and consideration'
)

legal_system.add_law(
    'LAW002',
    'Breach of Contract',
    'A party breaches a contract by failing to perform as promised'
)

# Add cases
legal_system.add_case(
    'CASE001',
    ['offer_made', 'acceptance_given', 'consideration_provided'],
    'Contract is valid',
    'All elements of contract formation were present'
)

# Add rules
legal_system.add_rule(
    'LAW001',
    ['offer_made', 'acceptance_given', 'consideration_provided'],
    'Contract is valid'
)

legal_system.add_rule(
    'LAW002',
    ['contract_valid', 'performance_failed'],
    'Breach of contract occurred'
)

# Analyze a case
case_facts = ['offer_made', 'acceptance_given', 'consideration_provided', 'performance_failed']
analysis = legal_system.analyze_case(case_facts)

print("Legal Analysis with Explanations:")
print(legal_system.explain_analysis(analysis))
```

## Challenges and Solutions

### 1. Complexity Management

```python
class ComplexityManager:
    """Manages complexity in explanations"""
    
    def __init__(self):
        self.max_steps = 10
        self.max_premises = 5
        self.abbreviation_rules = {}
    
    def simplify_proof(self, proof: Proof) -> Proof:
        """Simplify a proof for better explanation"""
        simplified_proof = Proof()
        
        # Add only essential assumptions
        essential_assumptions = self._find_essential_assumptions(proof)
        for assumption in essential_assumptions:
            simplified_proof.add_assumption(assumption)
        
        # Add only key inference steps
        key_steps = self._find_key_steps(proof)
        for step in key_steps:
            simplified_proof.add_step(step)
        
        return simplified_proof
    
    def _find_essential_assumptions(self, proof: Proof) -> Set['Formula']:
        """Find assumptions that are actually used"""
        used_assumptions = set()
        
        for step in proof.steps:
            for premise in step.premises:
                if premise in proof.assumptions:
                    used_assumptions.add(premise)
        
        return used_assumptions
    
    def _find_key_steps(self, proof: Proof) -> List[InferenceStep]:
        """Find the most important inference steps"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated heuristics
        return proof.steps[:self.max_steps]
    
    def add_abbreviation(self, pattern: str, abbreviation: str):
        """Add an abbreviation rule"""
        self.abbreviation_rules[pattern] = abbreviation
    
    def abbreviate_explanation(self, explanation: str) -> str:
        """Apply abbreviations to explanation"""
        abbreviated = explanation
        
        for pattern, abbreviation in self.abbreviation_rules.items():
            abbreviated = abbreviated.replace(pattern, abbreviation)
        
        return abbreviated

# Example complexity management
complexity_manager = ComplexityManager()
complexity_manager.add_abbreviation('Modus Ponens', 'MP')
complexity_manager.add_abbreviation('Universal Quantifier', '∀')

simplified_proof = complexity_manager.simplify_proof(proof)
abbreviated_explanation = complexity_manager.abbreviate_explanation(proof.get_explanation())

print("Simplified Explanation:")
print(abbreviated_explanation)
```

### 2. User Customization

```python
class UserCustomizableExplainer:
    """Explainable system that adapts to user preferences"""
    
    def __init__(self):
        self.user_preferences = {
            'detail_level': 'medium',  # low, medium, high
            'explanation_style': 'natural',  # formal, natural, visual
            'max_steps': 10,
            'include_justifications': True
        }
    
    def set_preference(self, key: str, value: str):
        """Set a user preference"""
        if key in self.user_preferences:
            self.user_preferences[key] = value
    
    def explain_for_user(self, proof: Proof) -> str:
        """Generate explanation tailored to user preferences"""
        if self.user_preferences['explanation_style'] == 'natural':
            explainer = NaturalLanguageExplainer()
            return explainer.explain_proof_naturally(proof)
        
        elif self.user_preferences['explanation_style'] == 'visual':
            explainer = VisualExplainer()
            return explainer.proof_to_tree(proof)
        
        else:  # formal
            return proof.get_explanation()
    
    def get_explanation_summary(self, proof: Proof) -> str:
        """Get a summary explanation based on detail level"""
        if self.user_preferences['detail_level'] == 'low':
            return f"Derived {proof.steps[-1].conclusion} in {len(proof.steps)} steps"
        
        elif self.user_preferences['detail_level'] == 'medium':
            return self.explain_for_user(proof)
        
        else:  # high
            return self.explain_for_user(proof) + "\n\n" + f"Total steps: {len(proof.steps)}"

# Example user customization
user_explainer = UserCustomizableExplainer()

# Set user preferences
user_explainer.set_preference('detail_level', 'high')
user_explainer.set_preference('explanation_style', 'natural')

# Generate customized explanation
customized_explanation = user_explainer.explain_for_user(proof)
print("Customized Explanation:")
print(customized_explanation)
```

## Key Concepts Summary

| Concept | Description | Implementation | Benefits |
|---------|-------------|----------------|----------|
| **Step-by-Step Explanation** | Detailed inference process | `StepByStepExplainer` | Complete transparency |
| **Natural Language** | Human-readable explanations | `NaturalLanguageExplainer` | Accessibility |
| **Visual Representation** | Graphical proof structure | `VisualExplainer` | Intuitive understanding |
| **Complexity Management** | Simplified explanations | `ComplexityManager` | Usability |
| **User Customization** | Adaptive explanations | `UserCustomizableExplainer` | Personalization |
| **Domain-Specific** | Tailored to application | `MedicalDiagnosisExplainer` | Relevance |

## Best Practices

1. **Provide multiple explanation types** for different users
2. **Manage complexity** to avoid overwhelming users
3. **Use natural language** when possible
4. **Include justifications** for each step
5. **Allow user customization** of explanation style
6. **Validate explanations** for correctness
7. **Test with real users** to ensure understandability

Explainability and interpretability are essential for building trustworthy AI systems, and logic-based approaches provide a natural foundation for creating transparent and understandable reasoning systems. 