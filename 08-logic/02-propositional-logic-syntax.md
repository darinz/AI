# Propositional Logic Syntax

## Introduction

Propositional logic syntax defines the formal rules for constructing well-formed logical expressions. Understanding syntax is crucial for building logical systems, as it ensures that our formulas are grammatically correct and can be properly interpreted.

## What is Propositional Logic?

Propositional logic (also called sentential logic) is the branch of logic that deals with simple declarative statements (propositions) and their logical relationships. It is the foundation upon which more complex logical systems are built.

### Key Characteristics
- **Atomic**: Deals with indivisible statements
- **Boolean**: Each statement is either true or false
- **Compositional**: Complex statements are built from simpler ones
- **Formal**: Has precise rules for construction and interpretation

## Basic Components

### 1. Atomic Propositions

Atomic propositions are the building blocks of propositional logic. They represent simple, indivisible statements that can be either true or false.

**Mathematical Definition**: An atomic proposition is a symbol from a set of propositional variables.

**Examples**:
- P: "It is raining"
- Q: "The ground is wet"
- R: "The sun is shining"
- S: "I am happy"

### 2. Logical Connectives

Logical connectives are operators that combine atomic propositions to form compound statements.

#### Primary Connectives

| Symbol | Name | Meaning | Arity |
|--------|------|---------|-------|
| ¬ | Negation | NOT | Unary |
| ∧ | Conjunction | AND | Binary |
| ∨ | Disjunction | OR | Binary |
| → | Implication | IF-THEN | Binary |
| ↔ | Biconditional | IF AND ONLY IF | Binary |

#### Mathematical Definitions

**Negation (¬)**:
- If φ is a formula, then ¬φ is a formula
- ¬φ is true when φ is false, and false when φ is true

**Conjunction (∧)**:
- If φ and ψ are formulas, then φ ∧ ψ is a formula
- φ ∧ ψ is true when both φ and ψ are true

**Disjunction (∨)**:
- If φ and ψ are formulas, then φ ∨ ψ is a formula
- φ ∨ ψ is true when at least one of φ or ψ is true

**Implication (→)**:
- If φ and ψ are formulas, then φ → ψ is a formula
- φ → ψ is false only when φ is true and ψ is false

**Biconditional (↔)**:
- If φ and ψ are formulas, then φ ↔ ψ is a formula
- φ ↔ ψ is true when φ and ψ have the same truth value

## Well-Formed Formulas (WFFs)

### Recursive Definition

A well-formed formula is defined recursively as follows:

1. **Base Case**: Every atomic proposition is a well-formed formula
2. **Inductive Cases**:
   - If φ is a well-formed formula, then ¬φ is a well-formed formula
   - If φ and ψ are well-formed formulas, then:
     - φ ∧ ψ is a well-formed formula
     - φ ∨ ψ is a well-formed formula
     - φ → ψ is a well-formed formula
     - φ ↔ ψ is a well-formed formula
3. **Closure**: Nothing else is a well-formed formula

### Examples of Well-Formed Formulas

**Valid WFFs**:
- P (atomic)
- ¬P (negation)
- P ∧ Q (conjunction)
- (P ∨ Q) → R (implication with disjunction)
- ¬(P ∧ Q) ↔ (¬P ∨ ¬Q) (biconditional with negation)

**Invalid WFFs**:
- P ∧ (missing right operand)
- ∧ P (missing left operand)
- P ∧ ∧ Q (double connective)
- (P (unmatched parenthesis)

## Python Implementation: Syntax Parser

Let's implement a parser for propositional logic syntax:

```python
from abc import ABC, abstractmethod
from typing import Set, List, Dict, Optional, Tuple
from enum import Enum
import re

class TokenType(Enum):
    ATOM = "ATOM"
    NEGATION = "NEGATION"
    CONJUNCTION = "CONJUNCTION"
    DISJUNCTION = "DISJUNCTION"
    IMPLICATION = "IMPLICATION"
    BICONDITIONAL = "BICONDITIONAL"
    LEFT_PAREN = "LEFT_PAREN"
    RIGHT_PAREN = "RIGHT_PAREN"
    EOF = "EOF"

class Token:
    def __init__(self, type: TokenType, value: str, position: int):
        self.type = type
        self.value = value
        self.position = position
    
    def __str__(self):
        return f"Token({self.type}, '{self.value}', pos={self.position})"
    
    def __repr__(self):
        return self.__str__()

class Lexer:
    """Lexical analyzer for propositional logic"""
    
    def __init__(self, text: str):
        self.text = text
        self.position = 0
        self.current_char = self.text[0] if text else None
    
    def advance(self):
        """Move to the next character"""
        self.position += 1
        if self.position >= len(self.text):
            self.current_char = None
        else:
            self.current_char = self.text[self.position]
    
    def skip_whitespace(self):
        """Skip whitespace characters"""
        while self.current_char and self.current_char.isspace():
            self.advance()
    
    def get_atom(self) -> Token:
        """Extract an atomic proposition"""
        start_pos = self.position
        atom = ""
        
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            atom += self.current_char
            self.advance()
        
        return Token(TokenType.ATOM, atom, start_pos)
    
    def get_next_token(self) -> Token:
        """Get the next token from the input"""
        while self.current_char:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            if self.current_char.isalpha():
                return self.get_atom()
            
            if self.current_char == '¬':
                token = Token(TokenType.NEGATION, '¬', self.position)
                self.advance()
                return token
            
            if self.current_char == '∧':
                token = Token(TokenType.CONJUNCTION, '∧', self.position)
                self.advance()
                return token
            
            if self.current_char == '∨':
                token = Token(TokenType.DISJUNCTION, '∨', self.position)
                self.advance()
                return token
            
            if self.current_char == '→':
                token = Token(TokenType.IMPLICATION, '→', self.position)
                self.advance()
                return token
            
            if self.current_char == '↔':
                token = Token(TokenType.BICONDITIONAL, '↔', self.position)
                self.advance()
                return token
            
            if self.current_char == '(':
                token = Token(TokenType.LEFT_PAREN, '(', self.position)
                self.advance()
                return token
            
            if self.current_char == ')':
                token = Token(TokenType.RIGHT_PAREN, ')', self.position)
                self.advance()
                return token
            
            raise SyntaxError(f"Invalid character '{self.current_char}' at position {self.position}")
        
        return Token(TokenType.EOF, '', self.position)

class Parser:
    """Parser for propositional logic formulas"""
    
    def __init__(self, lexer: Lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()
    
    def eat(self, token_type: TokenType):
        """Consume a token of the expected type"""
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token.type}")
    
    def factor(self):
        """Parse atomic propositions and negations"""
        token = self.current_token
        
        if token.type == TokenType.ATOM:
            self.eat(TokenType.ATOM)
            return AtomicFormula(token.value)
        
        elif token.type == TokenType.NEGATION:
            self.eat(TokenType.NEGATION)
            return Negation(self.factor())
        
        elif token.type == TokenType.LEFT_PAREN:
            self.eat(TokenType.LEFT_PAREN)
            node = self.expr()
            self.eat(TokenType.RIGHT_PAREN)
            return node
        
        else:
            raise SyntaxError(f"Unexpected token: {token}")
    
    def term(self):
        """Parse conjunctions (highest precedence)"""
        node = self.factor()
        
        while self.current_token.type == TokenType.CONJUNCTION:
            token = self.current_token
            self.eat(TokenType.CONJUNCTION)
            node = Conjunction(node, self.factor())
        
        return node
    
    def expr(self):
        """Parse disjunctions, implications, and biconditionals"""
        node = self.term()
        
        while self.current_token.type in [TokenType.DISJUNCTION, TokenType.IMPLICATION, TokenType.BICONDITIONAL]:
            token = self.current_token
            
            if token.type == TokenType.DISJUNCTION:
                self.eat(TokenType.DISJUNCTION)
                node = Disjunction(node, self.term())
            
            elif token.type == TokenType.IMPLICATION:
                self.eat(TokenType.IMPLICATION)
                node = Implication(node, self.term())
            
            elif token.type == TokenType.BICONDITIONAL:
                self.eat(TokenType.BICONDITIONAL)
                node = Biconditional(node, self.term())
        
        return node
    
    def parse(self):
        """Parse the entire formula"""
        return self.expr()

# Import the Formula classes from the previous guide
from typing import Set, Dict

class Formula(ABC):
    @abstractmethod
    def evaluate(self, interpretation: Dict[str, bool]) -> bool:
        pass
    
    @abstractmethod
    def get_variables(self) -> Set[str]:
        pass

class AtomicFormula(Formula):
    def __init__(self, variable: str):
        self.variable = variable
    
    def evaluate(self, interpretation: Dict[str, bool]) -> bool:
        return interpretation.get(self.variable, False)
    
    def get_variables(self) -> Set[str]:
        return {self.variable}
    
    def __str__(self):
        return self.variable

class Negation(Formula):
    def __init__(self, formula: Formula):
        self.formula = formula
    
    def evaluate(self, interpretation: Dict[str, bool]) -> bool:
        return not self.formula.evaluate(interpretation)
    
    def get_variables(self) -> Set[str]:
        return self.formula.get_variables()
    
    def __str__(self):
        return f"¬({self.formula})"

class BinaryOperator(Formula):
    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right
    
    def get_variables(self) -> Set[str]:
        return self.left.get_variables() | self.right.get_variables()

class Conjunction(BinaryOperator):
    def evaluate(self, interpretation: Dict[str, bool]) -> bool:
        return self.left.evaluate(interpretation) and self.right.evaluate(interpretation)
    
    def __str__(self):
        return f"({self.left} ∧ {self.right})"

class Disjunction(BinaryOperator):
    def evaluate(self, interpretation: Dict[str, bool]) -> bool:
        return self.left.evaluate(interpretation) or self.right.evaluate(interpretation)
    
    def __str__(self):
        return f"({self.left} ∨ {self.right})"

class Implication(BinaryOperator):
    def evaluate(self, interpretation: Dict[str, bool]) -> bool:
        return (not self.left.evaluate(interpretation)) or self.right.evaluate(interpretation)
    
    def __str__(self):
        return f"({self.left} → {self.right})"

class Biconditional(BinaryOperator):
    def evaluate(self, interpretation: Dict[str, bool]) -> bool:
        left_val = self.left.evaluate(interpretation)
        right_val = self.right.evaluate(interpretation)
        return left_val == right_val
    
    def __str__(self):
        return f"({self.left} ↔ {self.right})"
```

## Syntax Validation

Let's implement a syntax validator:

```python
class SyntaxValidator:
    """Validates propositional logic syntax"""
    
    @staticmethod
    def is_well_formed(formula_str: str) -> bool:
        """Check if a string represents a well-formed formula"""
        try:
            lexer = Lexer(formula_str)
            parser = Parser(lexer)
            parser.parse()
            return True
        except (SyntaxError, IndexError):
            return False
    
    @staticmethod
    def get_syntax_errors(formula_str: str) -> List[str]:
        """Get detailed syntax error messages"""
        errors = []
        
        # Check for basic syntax issues
        if not formula_str.strip():
            errors.append("Empty formula")
            return errors
        
        # Check parentheses balance
        if formula_str.count('(') != formula_str.count(')'):
            errors.append("Unmatched parentheses")
        
        # Check for invalid characters
        valid_chars = set('¬∧∨→↔() \t\n\r')
        for i, char in enumerate(formula_str):
            if not (char.isalnum() or char == '_' or char in valid_chars):
                errors.append(f"Invalid character '{char}' at position {i}")
        
        # Try to parse
        try:
            lexer = Lexer(formula_str)
            parser = Parser(lexer)
            parser.parse()
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
        except IndexError:
            errors.append("Unexpected end of formula")
        
        return errors

# Example usage
validator = SyntaxValidator()

test_formulas = [
    "P",                    # Valid
    "¬P",                   # Valid
    "P ∧ Q",                # Valid
    "(P ∨ Q) → R",          # Valid
    "P ∧",                  # Invalid
    "∧ P",                  # Invalid
    "P ∧ ∧ Q",              # Invalid
    "(P",                   # Invalid
    "P @ Q",                # Invalid character
]

print("Syntax validation results:")
for formula in test_formulas:
    is_valid = validator.is_well_formed(formula)
    errors = validator.get_syntax_errors(formula)
    print(f"'{formula}': {'✓' if is_valid else '✗'}")
    if errors:
        print(f"  Errors: {errors}")
```

## Operator Precedence

Understanding operator precedence is crucial for parsing and interpreting logical formulas correctly.

### Precedence Rules (from highest to lowest)

1. **Parentheses**: `()` - Highest precedence
2. **Negation**: `¬` - Unary operator
3. **Conjunction**: `∧` - Binary operator
4. **Disjunction**: `∨` - Binary operator
5. **Implication**: `→` - Binary operator
6. **Biconditional**: `↔` - Binary operator - Lowest precedence

### Examples

| Formula | Parsed as | Meaning |
|---------|-----------|---------|
| `P ∧ Q ∨ R` | `(P ∧ Q) ∨ R` | (P AND Q) OR R |
| `P → Q ∧ R` | `P → (Q ∧ R)` | P IMPLIES (Q AND R) |
| `¬P ∧ Q` | `(¬P) ∧ Q` | (NOT P) AND Q |
| `P ↔ Q → R` | `P ↔ (Q → R)` | P IFF (Q IMPLIES R) |

## Formula Complexity

### Measures of Complexity

1. **Length**: Number of symbols in the formula
2. **Depth**: Maximum nesting level of operators
3. **Number of variables**: Count of distinct atomic propositions
4. **Number of operators**: Count of logical connectives

```python
class FormulaAnalyzer:
    """Analyze properties of logical formulas"""
    
    @staticmethod
    def get_length(formula: Formula) -> int:
        """Get the length of the formula (number of symbols)"""
        if isinstance(formula, AtomicFormula):
            return 1
        elif isinstance(formula, Negation):
            return 1 + FormulaAnalyzer.get_length(formula.formula)
        elif isinstance(formula, BinaryOperator):
            return 1 + FormulaAnalyzer.get_length(formula.left) + FormulaAnalyzer.get_length(formula.right)
        return 0
    
    @staticmethod
    def get_depth(formula: Formula) -> int:
        """Get the maximum nesting depth of operators"""
        if isinstance(formula, AtomicFormula):
            return 0
        elif isinstance(formula, Negation):
            return 1 + FormulaAnalyzer.get_depth(formula.formula)
        elif isinstance(formula, BinaryOperator):
            return 1 + max(
                FormulaAnalyzer.get_depth(formula.left),
                FormulaAnalyzer.get_depth(formula.right)
            )
        return 0
    
    @staticmethod
    def get_variable_count(formula: Formula) -> int:
        """Get the number of distinct variables"""
        return len(formula.get_variables())
    
    @staticmethod
    def get_operator_count(formula: Formula) -> int:
        """Get the number of logical operators"""
        if isinstance(formula, AtomicFormula):
            return 0
        elif isinstance(formula, Negation):
            return 1 + FormulaAnalyzer.get_operator_count(formula.formula)
        elif isinstance(formula, BinaryOperator):
            return 1 + FormulaAnalyzer.get_operator_count(formula.left) + FormulaAnalyzer.get_operator_count(formula.right)
        return 0

# Example analysis
P = AtomicFormula('P')
Q = AtomicFormula('Q')
R = AtomicFormula('R')

complex_formula = Biconditional(
    Implication(P, Conjunction(Q, R)),
    Disjunction(Negation(P), R)
)

analyzer = FormulaAnalyzer()
print(f"Formula: {complex_formula}")
print(f"Length: {analyzer.get_length(complex_formula)}")
print(f"Depth: {analyzer.get_depth(complex_formula)}")
print(f"Variables: {analyzer.get_variable_count(complex_formula)}")
print(f"Operators: {analyzer.get_operator_count(complex_formula)}")
```

## Normal Forms

### Conjunctive Normal Form (CNF)

A formula is in CNF if it is a conjunction of disjunctions of literals (atomic propositions or their negations).

**Form**: (L₁ ∨ L₂ ∨ ... ∨ Lₙ) ∧ (M₁ ∨ M₂ ∨ ... ∨ Mₘ) ∧ ...

```python
def to_cnf(formula: Formula) -> Formula:
    """Convert a formula to Conjunctive Normal Form"""
    # This is a simplified implementation
    # Full CNF conversion requires more sophisticated algorithms
    
    if isinstance(formula, AtomicFormula):
        return formula
    
    elif isinstance(formula, Negation):
        if isinstance(formula.formula, AtomicFormula):
            return formula
        # Handle double negation
        elif isinstance(formula.formula, Negation):
            return to_cnf(formula.formula.formula)
        # De Morgan's laws
        elif isinstance(formula.formula, Conjunction):
            return to_cnf(Disjunction(
                Negation(formula.formula.left),
                Negation(formula.formula.right)
            ))
        elif isinstance(formula.formula, Disjunction):
            return to_cnf(Conjunction(
                Negation(formula.formula.left),
                Negation(formula.formula.right)
            ))
    
    elif isinstance(formula, Conjunction):
        return Conjunction(to_cnf(formula.left), to_cnf(formula.right))
    
    elif isinstance(formula, Disjunction):
        left_cnf = to_cnf(formula.left)
        right_cnf = to_cnf(formula.right)
        
        # Distribute disjunction over conjunction
        if isinstance(left_cnf, Conjunction):
            return to_cnf(Conjunction(
                Disjunction(left_cnf.left, right_cnf),
                Disjunction(left_cnf.right, right_cnf)
            ))
        elif isinstance(right_cnf, Conjunction):
            return to_cnf(Conjunction(
                Disjunction(left_cnf, right_cnf.left),
                Disjunction(left_cnf, right_cnf.right)
            ))
        else:
            return Disjunction(left_cnf, right_cnf)
    
    return formula
```

## Key Concepts Summary

| Concept | Description | Mathematical Notation | Python Implementation |
|---------|-------------|----------------------|----------------------|
| **Atomic Proposition** | Basic true/false statement | P, Q, R | `AtomicFormula('P')` |
| **Negation** | Logical NOT | ¬P | `Negation(P)` |
| **Conjunction** | Logical AND | P ∧ Q | `Conjunction(P, Q)` |
| **Disjunction** | Logical OR | P ∨ Q | `Disjunction(P, Q)` |
| **Implication** | Logical IF-THEN | P → Q | `Implication(P, Q)` |
| **Biconditional** | Logical IFF | P ↔ Q | `Biconditional(P, Q)` |
| **Well-Formed Formula** | Grammatically correct expression | Recursive definition | `Parser.parse()` |
| **Operator Precedence** | Order of operations | ¬ > ∧ > ∨ > → > ↔ | Built into parser |
| **Normal Form** | Standard representation | CNF, DNF | `to_cnf()` |

## Common Syntax Errors and Solutions

### 1. Missing Operands
**Error**: `P ∧`
**Solution**: Always provide both operands for binary operators

### 2. Unmatched Parentheses
**Error**: `(P ∧ Q`
**Solution**: Ensure every opening parenthesis has a matching closing one

### 3. Invalid Characters
**Error**: `P @ Q`
**Solution**: Use only valid logical symbols and alphanumeric characters

### 4. Operator Precedence Confusion
**Error**: Misinterpreting `P ∧ Q ∨ R`
**Solution**: Use parentheses to clarify intended meaning: `(P ∧ Q) ∨ R`

## Best Practices

1. **Use parentheses liberally** to avoid ambiguity
2. **Choose meaningful variable names** for atomic propositions
3. **Validate syntax** before processing formulas
4. **Consider operator precedence** when writing formulas
5. **Use consistent notation** throughout your work
6. **Document complex formulas** with comments or explanations

Understanding propositional logic syntax is essential for building logical systems, writing correct formulas, and implementing automated reasoning systems. The formal rules ensure that our logical expressions are well-defined and can be properly interpreted by both humans and machines. 