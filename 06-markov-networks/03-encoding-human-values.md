# Encoding Human Values in Markov Networks

## Introduction

Encoding human values in AI systems is crucial for ensuring that these systems behave in ways that align with human preferences, ethical principles, and societal norms. Markov networks provide a powerful framework for representing and reasoning about human values in a probabilistic setting.

## What are Human Values?

### Definition

Human values are the principles, standards, and qualities that people consider important in life. In the context of AI systems, human values include:

- **Moral Principles**: Fundamental ethical rules (e.g., "do no harm")
- **Preferences**: Individual or collective choices and desires
- **Social Norms**: Cultural and societal expectations
- **Risk Attitudes**: How people evaluate uncertainty and potential losses

### Value Types

1. **Intrinsic Values**: Values that are good in themselves (e.g., happiness, knowledge)
2. **Instrumental Values**: Values that help achieve other values (e.g., money, education)
3. **Moral Values**: Values related to right and wrong behavior (e.g., fairness, honesty)
4. **Aesthetic Values**: Values related to beauty and artistic appreciation

## Mathematical Framework

### Value Representation

Human values can be represented in Markov networks through:

1. **Utility Functions**: Potential functions that encode preferences
2. **Constraint Potentials**: Functions that penalize violations of moral principles
3. **Social Potentials**: Functions that model group dynamics and norms
4. **Risk Potentials**: Functions that capture attitudes toward uncertainty

### Mathematical Formulation

A value-augmented Markov network has the form:

P(X) = (1/Z) ∏ᵢ φᵢ(Cᵢ) ∏ⱼ ψⱼ(Vⱼ)

Where:
- φᵢ are standard potential functions
- ψⱼ are value-based potential functions
- Vⱼ are value-related variables or constraints

## Python Implementation

### Basic Value Encoding Framework

```python
import numpy as np
from typing import List, Dict, Set, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import random

class ValueType(Enum):
    """Types of human values."""
    UTILITY = "utility"
    MORAL = "moral"
    SOCIAL = "social"
    RISK = "risk"

@dataclass
class ValueFunction:
    """Represents a value function in a Markov network."""
    name: str
    value_type: ValueType
    variables: List[str]
    function: Callable[[Dict[str, Any]], float]
    weight: float = 1.0
    
    def evaluate(self, assignment: Dict[str, Any]) -> float:
        """Evaluate the value function for a given assignment."""
        return self.weight * self.function(assignment)

class ValueAugmentedMarkovNetwork:
    """Markov network augmented with human values."""
    
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]]):
        self.variables = variables
        self.domains = domains
        self.standard_potentials: List[PotentialFunction] = []
        self.value_functions: List[ValueFunction] = []
        self.graph = nx.Graph()
        
        # Initialize graph
        for var in variables:
            self.graph.add_node(var)
    
    def add_standard_potential(self, potential: PotentialFunction):
        """Add a standard potential function."""
        self.standard_potentials.append(potential)
        
        # Update graph
        for i, var1 in enumerate(potential.variables):
            for var2 in potential.variables[i+1:]:
                self.graph.add_edge(var1, var2)
    
    def add_value_function(self, value_func: ValueFunction):
        """Add a value function."""
        self.value_functions.append(value_func)
        
        # Update graph for value-related variables
        for i, var1 in enumerate(value_func.variables):
            for var2 in value_func.variables[i+1:]:
                self.graph.add_edge(var1, var2)
    
    def compute_joint_probability(self, assignment: Dict[str, Any]) -> float:
        """Compute joint probability including value functions."""
        # Standard potentials
        standard_prob = 1.0
        for potential in self.standard_potentials:
            standard_prob *= potential.get_value(assignment)
        
        # Value functions (converted to potentials)
        value_prob = 1.0
        for value_func in self.value_functions:
            value_prob *= np.exp(value_func.evaluate(assignment))
        
        return standard_prob * value_prob
    
    def compute_value_score(self, assignment: Dict[str, Any]) -> float:
        """Compute the total value score for an assignment."""
        total_value = 0.0
        for value_func in self.value_functions:
            total_value += value_func.evaluate(assignment)
        return total_value

# Example: Medical Decision Making
def create_medical_decision_network():
    """Create a Markov network for medical decision making with human values."""
    
    # Variables: treatment decision, patient outcome, cost, side effects
    variables = ['treatment', 'outcome', 'cost', 'side_effects']
    domains = {
        'treatment': ['surgery', 'medication', 'watchful_waiting'],
        'outcome': ['cured', 'improved', 'no_change', 'worse'],
        'cost': ['low', 'medium', 'high'],
        'side_effects': ['none', 'mild', 'severe']
    }
    
    # Create network
    network = ValueAugmentedMarkovNetwork(variables, domains)
    
    # Standard potentials (medical relationships)
    # Treatment affects outcome
    treatment_outcome = PotentialFunction(
        variables=['treatment', 'outcome'],
        values={
            ('surgery', 'cured'): 3.0,
            ('surgery', 'improved'): 2.0,
            ('surgery', 'no_change'): 1.0,
            ('surgery', 'worse'): 0.5,
            ('medication', 'cured'): 1.5,
            ('medication', 'improved'): 2.5,
            ('medication', 'no_change'): 1.5,
            ('medication', 'worse'): 0.8,
            ('watchful_waiting', 'cured'): 0.5,
            ('watchful_waiting', 'improved'): 1.0,
            ('watchful_waiting', 'no_change'): 2.0,
            ('watchful_waiting', 'worse'): 1.5
        }
    )
    
    # Treatment affects cost
    treatment_cost = PotentialFunction(
        variables=['treatment', 'cost'],
        values={
            ('surgery', 'high'): 2.0,
            ('surgery', 'medium'): 1.0,
            ('surgery', 'low'): 0.5,
            ('medication', 'high'): 1.0,
            ('medication', 'medium'): 2.0,
            ('medication', 'low'): 1.5,
            ('watchful_waiting', 'high'): 0.5,
            ('watchful_waiting', 'medium'): 1.0,
            ('watchful_waiting', 'low'): 2.0
        }
    )
    
    # Treatment affects side effects
    treatment_side_effects = PotentialFunction(
        variables=['treatment', 'side_effects'],
        values={
            ('surgery', 'severe'): 2.0,
            ('surgery', 'mild'): 1.5,
            ('surgery', 'none'): 0.5,
            ('medication', 'severe'): 1.0,
            ('medication', 'mild'): 2.0,
            ('medication', 'none'): 1.5,
            ('watchful_waiting', 'severe'): 0.5,
            ('watchful_waiting', 'mild'): 1.0,
            ('watchful_waiting', 'none'): 2.0
        }
    )
    
    network.add_standard_potential(treatment_outcome)
    network.add_standard_potential(treatment_cost)
    network.add_standard_potential(treatment_side_effects)
    
    # Value functions
    # Utility: prefer better outcomes
    outcome_utility = ValueFunction(
        name="outcome_utility",
        value_type=ValueType.UTILITY,
        variables=['outcome'],
        function=lambda assignment: {
            'cured': 10.0,
            'improved': 7.0,
            'no_change': 5.0,
            'worse': 0.0
        }[assignment['outcome']],
        weight=1.0
    )
    
    # Moral: avoid severe side effects
    side_effect_moral = ValueFunction(
        name="side_effect_moral",
        value_type=ValueType.MORAL,
        variables=['side_effects'],
        function=lambda assignment: {
            'none': 0.0,
            'mild': -2.0,
            'severe': -10.0
        }[assignment['side_effects']],
        weight=2.0
    )
    
    # Social: consider cost to society
    cost_social = ValueFunction(
        name="cost_social",
        value_type=ValueType.SOCIAL,
        variables=['cost'],
        function=lambda assignment: {
            'low': 0.0,
            'medium': -1.0,
            'high': -3.0
        }[assignment['cost']],
        weight=0.5
    )
    
    network.add_value_function(outcome_utility)
    network.add_value_function(side_effect_moral)
    network.add_value_function(cost_social)
    
    return network

def test_medical_decision_network():
    """Test the medical decision network with human values."""
    network = create_medical_decision_network()
    
    # Test different treatment decisions
    treatments = ['surgery', 'medication', 'watchful_waiting']
    
    print("Medical Decision Analysis with Human Values:")
    print("=" * 50)
    
    for treatment in treatments:
        print(f"\nTreatment: {treatment}")
        
        # Generate all possible outcomes for this treatment
        for outcome in network.domains['outcome']:
            for cost in network.domains['cost']:
                for side_effects in network.domains['side_effects']:
                    assignment = {
                        'treatment': treatment,
                        'outcome': outcome,
                        'cost': cost,
                        'side_effects': side_effects
                    }
                    
                    prob = network.compute_joint_probability(assignment)
                    value_score = network.compute_value_score(assignment)
                    
                    print(f"  {outcome}, {cost} cost, {side_effects} side effects:")
                    print(f"    Probability: {prob:.4f}")
                    print(f"    Value Score: {value_score:.2f}")
    
    return network

if __name__ == "__main__":
    test_medical_decision_network()
```

## Value Learning

### Preference Learning

Learning human values from observed behavior or preferences.

```python
class PreferenceLearner:
    """Learn human values from preference data."""
    
    def __init__(self, network: ValueAugmentedMarkovNetwork):
        self.network = network
    
    def learn_from_preferences(self, preferences: List[Tuple[Dict[str, Any], Dict[str, Any]]]):
        """Learn value function weights from preference data."""
        # Each preference is (preferred_assignment, less_preferred_assignment)
        
        # Initialize weights
        weights = {func.name: func.weight for func in self.network.value_functions}
        
        # Simple gradient-based learning
        learning_rate = 0.01
        num_iterations = 1000
        
        for iteration in range(num_iterations):
            total_loss = 0.0
            
            for preferred, less_preferred in preferences:
                # Compute value differences
                preferred_value = self.network.compute_value_score(preferred)
                less_preferred_value = self.network.compute_value_score(less_preferred)
                
                # Loss: preferred should have higher value
                loss = max(0, less_preferred_value - preferred_value + 1.0)
                total_loss += loss
                
                # Update weights (simplified gradient)
                if loss > 0:
                    for func in self.network.value_functions:
                        preferred_contribution = func.function(preferred)
                        less_preferred_contribution = func.function(less_preferred)
                        
                        gradient = less_preferred_contribution - preferred_contribution
                        weights[func.name] -= learning_rate * gradient
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Total Loss: {total_loss:.4f}")
        
        # Update network weights
        for func in self.network.value_functions:
            func.weight = weights[func.name]
        
        return weights

# Example: Learning from medical preferences
def create_preference_data():
    """Create example preference data for medical decisions."""
    preferences = [
        # Prefer surgery with good outcome over medication with poor outcome
        (
            {'treatment': 'surgery', 'outcome': 'cured', 'cost': 'high', 'side_effects': 'mild'},
            {'treatment': 'medication', 'outcome': 'worse', 'cost': 'medium', 'side_effects': 'none'}
        ),
        # Prefer medication over surgery when side effects are severe
        (
            {'treatment': 'medication', 'outcome': 'improved', 'cost': 'medium', 'side_effects': 'mild'},
            {'treatment': 'surgery', 'outcome': 'cured', 'cost': 'high', 'side_effects': 'severe'}
        ),
        # Prefer watchful waiting when cost is a major concern
        (
            {'treatment': 'watchful_waiting', 'outcome': 'improved', 'cost': 'low', 'side_effects': 'none'},
            {'treatment': 'surgery', 'outcome': 'cured', 'cost': 'high', 'side_effects': 'mild'}
        )
    ]
    
    return preferences

def test_preference_learning():
    """Test learning human values from preferences."""
    network = create_medical_decision_network()
    learner = PreferenceLearner(network)
    
    # Create preference data
    preferences = create_preference_data()
    
    print("Learning Human Values from Preferences:")
    print("=" * 40)
    
    # Show initial weights
    print("\nInitial weights:")
    for func in network.value_functions:
        print(f"  {func.name}: {func.weight}")
    
    # Learn weights
    learned_weights = learner.learn_from_preferences(preferences)
    
    # Show learned weights
    print("\nLearned weights:")
    for func_name, weight in learned_weights.items():
        print(f"  {func_name}: {weight:.4f}")
    
    return network, learner, learned_weights
```

## Value Aggregation

### Combining Multiple Value Systems

When multiple stakeholders have different values, we need to aggregate them.

```python
class ValueAggregator:
    """Aggregate values from multiple stakeholders."""
    
    def __init__(self, networks: List[ValueAugmentedMarkovNetwork]):
        self.networks = networks
        self.weights = [1.0 / len(networks)] * len(networks)  # Equal weights initially
    
    def set_stakeholder_weights(self, weights: List[float]):
        """Set weights for different stakeholders."""
        if len(weights) != len(self.networks):
            raise ValueError("Number of weights must match number of networks")
        
        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]
    
    def compute_aggregated_value(self, assignment: Dict[str, Any]) -> float:
        """Compute aggregated value across all stakeholders."""
        total_value = 0.0
        
        for i, network in enumerate(self.networks):
            value = network.compute_value_score(assignment)
            total_value += self.weights[i] * value
        
        return total_value
    
    def find_optimal_assignment(self) -> Dict[str, Any]:
        """Find assignment that maximizes aggregated value."""
        # Generate all possible assignments
        all_assignments = self._generate_all_assignments()
        
        best_assignment = None
        best_value = float('-inf')
        
        for assignment in all_assignments:
            value = self.compute_aggregated_value(assignment)
            if value > best_value:
                best_value = value
                best_assignment = assignment
        
        return best_assignment
    
    def _generate_all_assignments(self) -> List[Dict[str, Any]]:
        """Generate all possible assignments."""
        # Use the first network's variables and domains
        network = self.networks[0]
        return network._generate_all_assignments()

# Example: Medical decision with multiple stakeholders
def create_stakeholder_networks():
    """Create networks representing different stakeholders."""
    networks = []
    
    # Patient network (values personal health and comfort)
    patient_network = create_medical_decision_network()
    # Modify weights to emphasize personal outcomes
    for func in patient_network.value_functions:
        if func.name == "outcome_utility":
            func.weight = 2.0
        elif func.name == "side_effect_moral":
            func.weight = 3.0
        elif func.name == "cost_social":
            func.weight = 0.1
    networks.append(patient_network)
    
    # Doctor network (values medical effectiveness and safety)
    doctor_network = create_medical_decision_network()
    # Modify weights to emphasize medical outcomes
    for func in doctor_network.value_functions:
        if func.name == "outcome_utility":
            func.weight = 3.0
        elif func.name == "side_effect_moral":
            func.weight = 2.0
        elif func.name == "cost_social":
            func.weight = 0.5
    networks.append(doctor_network)
    
    # Insurance network (values cost-effectiveness)
    insurance_network = create_medical_decision_network()
    # Modify weights to emphasize cost
    for func in insurance_network.value_functions:
        if func.name == "outcome_utility":
            func.weight = 1.0
        elif func.name == "side_effect_moral":
            func.weight = 1.0
        elif func.name == "cost_social":
            func.weight = 3.0
    networks.append(insurance_network)
    
    return networks

def test_value_aggregation():
    """Test value aggregation across multiple stakeholders."""
    networks = create_stakeholder_networks()
    aggregator = ValueAggregator(networks)
    
    print("Value Aggregation Across Stakeholders:")
    print("=" * 40)
    
    # Test different weight combinations
    weight_combinations = [
        [0.5, 0.3, 0.2],  # Patient-focused
        [0.3, 0.5, 0.2],  # Doctor-focused
        [0.2, 0.3, 0.5],  # Cost-focused
        [0.33, 0.33, 0.34]  # Balanced
    ]
    
    for i, weights in enumerate(weight_combinations):
        print(f"\nWeight combination {i+1}: {weights}")
        aggregator.set_stakeholder_weights(weights)
        
        optimal = aggregator.find_optimal_assignment()
        optimal_value = aggregator.compute_aggregated_value(optimal)
        
        print(f"Optimal decision: {optimal}")
        print(f"Aggregated value: {optimal_value:.2f}")
    
    return aggregator
```

## Ethical Considerations

### Bias Detection and Mitigation

```python
class BiasDetector:
    """Detect and mitigate biases in value representations."""
    
    def __init__(self, network: ValueAugmentedMarkovNetwork):
        self.network = network
    
    def detect_demographic_bias(self, demographic_var: str, 
                               demographic_values: List[Any]) -> Dict[Any, float]:
        """Detect bias across demographic groups."""
        bias_scores = {}
        
        for value in demographic_values:
            # Compute average value scores for this demographic
            total_value = 0.0
            count = 0
            
            for assignment in self.network._generate_all_assignments():
                if assignment.get(demographic_var) == value:
                    value_score = self.network.compute_value_score(assignment)
                    total_value += value_score
                    count += 1
            
            if count > 0:
                bias_scores[value] = total_value / count
        
        return bias_scores
    
    def mitigate_bias(self, demographic_var: str, target_fairness: float = 0.1):
        """Mitigate bias by adjusting value function weights."""
        bias_scores = self.detect_demographic_bias(demographic_var, 
                                                  self.network.domains[demographic_var])
        
        # Compute fairness metric
        values = list(bias_scores.values())
        fairness = max(values) - min(values)
        
        if fairness > target_fairness:
            # Adjust weights to reduce bias
            # This is a simplified approach - in practice, more sophisticated methods would be used
            print(f"Detected bias: {fairness:.3f}, target: {target_fairness}")
            print("Applying bias mitigation...")
            
            # Simple mitigation: reduce weights of value functions that contribute to bias
            for func in self.network.value_functions:
                func.weight *= 0.9  # Reduce weight slightly
```

## Summary

Encoding human values in Markov networks provides a powerful framework for:

1. **Value Representation**: Capturing diverse human values in a mathematical framework
2. **Preference Learning**: Inferring values from observed behavior
3. **Value Aggregation**: Combining multiple stakeholder perspectives
4. **Bias Detection**: Identifying and mitigating unfair value representations

Key considerations:
- **Value Diversity**: Different people and cultures have different values
- **Value Conflicts**: Values may conflict and require trade-offs
- **Value Evolution**: Values may change over time
- **Ethical Responsibility**: Careful consideration of whose values to encode

Understanding value encoding is essential for:
- Building AI systems that align with human values
- Making decisions that respect diverse perspectives
- Ensuring fairness and avoiding bias
- Creating AI systems that are trustworthy and beneficial 