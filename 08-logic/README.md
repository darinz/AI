# Logic in Artificial Intelligence

This section covers the fundamental concepts of logic as applied to artificial intelligence, from basic propositional logic to advanced first-order logic and reasoning systems.

## Overview

Logic provides the formal foundation for knowledge representation and automated reasoning in AI systems. It enables us to:
- Represent knowledge in a precise, unambiguous way
- Perform automated inference and deduction
- Build systems that can reason about complex domains
- Ensure consistency and validity in AI decision-making

Logic serves as the backbone for expert systems, automated theorem proving, and many other AI applications.

## Propositional Logic Syntax

Propositional logic deals with simple declarative statements (propositions) and their logical relationships.

### Basic Components
- **Atomic propositions**: Simple statements that are either true or false (e.g., P, Q, R)
- **Logical connectives**:
  - NOT (¬): Negation
  - AND (∧): Conjunction
  - OR (∨): Disjunction
  - IMPLIES (→): Implication
  - IFF (↔): Biconditional

### Well-Formed Formulas
- Atomic propositions are well-formed formulas
- If φ and ψ are well-formed formulas, then:
  - ¬φ is a well-formed formula
  - φ ∧ ψ is a well-formed formula
  - φ ∨ ψ is a well-formed formula
  - φ → ψ is a well-formed formula
  - φ ↔ ψ is a well-formed formula

## Propositional Logic Semantics

Semantics define the meaning of logical expressions through truth assignments.

### Truth Tables
Truth tables systematically show the truth value of compound propositions for all possible truth assignments to atomic propositions.

### Interpretation
An interpretation assigns truth values (true/false) to all atomic propositions in a formula.

### Validity and Satisfiability
- **Valid**: A formula that is true under all interpretations
- **Satisfiable**: A formula that is true under at least one interpretation
- **Unsatisfiable**: A formula that is false under all interpretations

## Inference Rules

Inference rules are formal procedures for deriving new logical statements from existing ones.

### Basic Inference Rules
1. **Modus Ponens**: From P → Q and P, infer Q
2. **Modus Tollens**: From P → Q and ¬Q, infer ¬P
3. **Hypothetical Syllogism**: From P → Q and Q → R, infer P → R
4. **Disjunctive Syllogism**: From P ∨ Q and ¬P, infer Q
5. **Addition**: From P, infer P ∨ Q
6. **Simplification**: From P ∧ Q, infer P

### Soundness and Completeness
- **Sound**: An inference rule is sound if it only derives true conclusions from true premises
- **Complete**: An inference system is complete if it can derive all valid conclusions

## Propositional Modus Ponens

Modus Ponens is one of the most fundamental inference rules in logic.

### Rule
If we have:
- P → Q (implication)
- P (antecedent)

Then we can conclude:
- Q (consequent)

### Example
- If it rains, then the ground will be wet. (R → W)
- It is raining. (R)
- Therefore, the ground is wet. (W)

### Implementation
Modus Ponens is the basis for forward chaining in rule-based systems and expert systems.

## Propositional Resolution

Resolution is a powerful inference rule that forms the basis for automated theorem proving.

### Resolution Rule
From two clauses containing complementary literals, derive a new clause:
- (A ∨ B) and (¬B ∨ C) resolve to (A ∨ C)

### Resolution Refutation
To prove that a set of premises entails a conclusion:
1. Negate the conclusion
2. Add it to the premises
3. Convert all formulas to conjunctive normal form (CNF)
4. Apply resolution repeatedly
5. If we derive the empty clause (contradiction), the original conclusion is valid

### Example
To prove: {P → Q, Q → R} ⊨ P → R
1. Negate: ¬(P → R) ≡ P ∧ ¬R
2. Convert to CNF: {¬P ∨ Q, ¬Q ∨ R, P, ¬R}
3. Resolve: (¬P ∨ Q) and P → Q, then Q and (¬Q ∨ R) → R, then R and ¬R → ⊥

## First Order Logic

First-order logic extends propositional logic by introducing:
- **Variables**: x, y, z
- **Predicates**: P(x), Q(x,y)
- **Functions**: f(x), g(x,y)
- **Quantifiers**: ∀ (universal), ∃ (existential)

### Syntax
- **Terms**: Variables, constants, and function applications
- **Atomic formulas**: Predicate applications
- **Formulas**: Atomic formulas combined with logical connectives and quantifiers

### Examples
- ∀x (Student(x) → Studies(x))
- ∃x (Student(x) ∧ Smart(x))
- ∀x∀y (Parent(x,y) → Loves(x,y))

## First Order Modus Ponens

First-order modus ponens extends the propositional version to handle quantified statements.

### Rule
If we have:
- ∀x (P(x) → Q(x)) (universal implication)
- P(a) (instance of antecedent)

Then we can conclude:
- Q(a) (instance of consequent)

### Example
- All humans are mortal. ∀x (Human(x) → Mortal(x))
- Socrates is human. Human(Socrates)
- Therefore, Socrates is mortal. Mortal(Socrates)

### Unification
First-order modus ponens uses unification to match predicates with variables:
- P(x) unifies with P(Socrates) when x = Socrates

## Explainability and Interpretability

Logic-based AI systems provide natural explainability through their formal structure.

### Advantages
- **Transparent reasoning**: Each inference step can be traced
- **Verifiable conclusions**: Proofs can be checked for correctness
- **Human-readable**: Logical statements are interpretable by humans
- **Causal relationships**: Logic captures cause-and-effect relationships

### Applications
- **Expert systems**: Medical diagnosis, legal reasoning
- **Automated theorem proving**: Mathematical proofs
- **Knowledge representation**: Ontologies, semantic web
- **Decision support systems**: Business rules, regulations

### Challenges
- **Scalability**: Complex domains require large knowledge bases
- **Incompleteness**: Real-world knowledge is often incomplete
- **Uncertainty**: Logic struggles with probabilistic information
- **Common sense**: Capturing everyday knowledge is difficult

## First Order Resolution

First-order resolution extends propositional resolution to handle quantified statements.

### Preprocessing Steps
1. **Convert to prenex normal form**: Move all quantifiers to the front
2. **Skolemization**: Replace existential quantifiers with Skolem functions
3. **Convert to conjunctive normal form**: Distribute disjunctions over conjunctions

### Resolution with Unification
- Use unification to match complementary literals
- Apply substitution to make literals identical
- Resolve to produce new clauses

### Example
Premises:
- ∀x (Student(x) → Studies(x))
- ∀x (Studies(x) → Passes(x))
- Student(John)

To prove: Passes(John)

Resolution steps:
1. Skolemize and convert to CNF
2. Unify and resolve clauses
3. Derive Passes(John)

### Applications
- **Automated theorem proving**: Mathematical proofs
- **Logic programming**: Prolog
- **Knowledge-based systems**: Expert systems
- **Natural language processing**: Semantic analysis

## Key Concepts Summary

| Concept | Description | Application |
|---------|-------------|-------------|
| **Propositional Logic** | Simple true/false statements | Basic reasoning systems |
| **First-Order Logic** | Quantified statements with variables | Complex knowledge representation |
| **Inference Rules** | Formal procedures for deduction | Automated reasoning |
| **Resolution** | Powerful proof method | Theorem proving |
| **Explainability** | Transparent reasoning process | Trustworthy AI systems |

## Further Reading

- **Classical Logic**: Study of formal deductive systems
- **Modal Logic**: Logic of necessity and possibility
- **Temporal Logic**: Reasoning about time and change
- **Description Logic**: Family of knowledge representation languages
- **Logic Programming**: Programming paradigm based on logic

Logic continues to be fundamental in AI, providing the theoretical foundation for knowledge representation, automated reasoning, and explainable AI systems. 