# Markov Decision Processes (MDPs)

This section covers the fundamental concepts and algorithms of Markov Decision Processes, a mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision maker.

## Table of Contents

1. [Overview](#overview)
2. [Modeling](#modeling)
3. [Policy Evaluation](#policy-evaluation)
4. [Value Iteration](#value-iteration)
5. [Reinforcement Learning](#reinforcement-learning)
6. [Model-based Monte Carlo](#model-based-monte-carlo)
7. [Model-free Monte Carlo](#model-free-monte-carlo)
8. [SARSA](#sarsa)
9. [Q-learning](#q-learning)
10. [Epsilon-greedy](#epsilon-greedy)
11. [Function Approximation](#function-approximation)

## Overview

Markov Decision Processes (MDPs) provide a mathematical framework for modeling sequential decision-making problems under uncertainty. MDPs are characterized by:

- **States**: The possible situations the agent can be in
- **Actions**: The choices available to the agent in each state
- **Transitions**: How the environment changes based on actions
- **Rewards**: The immediate feedback for taking actions
- **Policy**: A strategy that maps states to actions

MDPs are fundamental to reinforcement learning and are used in robotics, game playing, autonomous systems, and many other AI applications.

## Modeling

MDP modeling involves defining the core components:

- **State Space (S)**: Set of all possible states
- **Action Space (A)**: Set of all possible actions
- **Transition Function P(s'|s,a)**: Probability of reaching state s' from state s after taking action a
- **Reward Function R(s,a,s')**: Expected reward for transitioning from s to s' via action a
- **Discount Factor γ**: Determines the importance of future rewards

The Markov property ensures that the future depends only on the current state and action, not the history.

## Policy Evaluation

Policy evaluation computes the value function for a given policy. The value function V^π(s) represents the expected cumulative reward starting from state s and following policy π.

**Bellman Equation for Policy Evaluation:**
```
V^π(s) = Σ P(s'|s,π(s)) [R(s,π(s),s') + γV^π(s')]
```

This equation is solved iteratively using dynamic programming techniques.

## Value Iteration

Value iteration is an algorithm for finding the optimal policy by iteratively improving the value function. It combines policy evaluation and policy improvement in a single algorithm.

**Bellman Optimality Equation:**
```
V*(s) = max_a Σ P(s'|s,a) [R(s,a,s') + γV*(s')]
```

The algorithm converges to the optimal value function, from which the optimal policy can be extracted.

## Reinforcement Learning

Reinforcement learning is learning through interaction with an environment. Key concepts include:

- **Exploration vs Exploitation**: Balancing trying new actions vs using known good actions
- **Temporal Difference Learning**: Learning from the difference between predicted and actual rewards
- **On-policy vs Off-policy**: Whether the learning policy matches the behavior policy

## Model-based Monte Carlo

Model-based Monte Carlo methods learn a model of the environment and use it for planning:

- **Model Learning**: Estimate transition probabilities and rewards from experience
- **Planning**: Use the learned model to compute optimal policies
- **Certainty Equivalence**: Treat the learned model as if it were the true model

## Model-free Monte Carlo

Model-free Monte Carlo methods learn directly from experience without building an explicit model:

- **Episode-based Learning**: Learn from complete sequences of experience
- **First-visit vs Every-visit**: How to handle multiple visits to the same state
- **Monte Carlo Control**: Combine policy evaluation with policy improvement

## SARSA

SARSA (State-Action-Reward-State-Action) is an on-policy temporal difference learning algorithm:

- **Q-learning Update**: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
- **On-policy**: Uses the same policy for behavior and learning
- **Conservative**: Tends to be more conservative than off-policy methods

## Q-learning

Q-learning is an off-policy temporal difference learning algorithm:

- **Q-learning Update**: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
- **Off-policy**: Can learn about the optimal policy while following a different behavior policy
- **Convergence**: Guaranteed to converge to the optimal Q-function under certain conditions

## Epsilon-greedy

Epsilon-greedy is a simple exploration strategy:

- **Exploitation**: Choose the best action with probability 1-ε
- **Exploration**: Choose a random action with probability ε
- **Trade-off**: Balances exploration and exploitation
- **Decay**: Often ε is decreased over time to focus more on exploitation

## Function Approximation

Function approximation allows MDPs to scale to large or continuous state spaces:

- **Linear Approximation**: Q(s,a) = Σ w_i φ_i(s,a)
- **Neural Networks**: Deep Q-networks and other neural network approaches
- **Feature Engineering**: Designing good features for approximation
- **Convergence**: Theoretical guarantees may not hold with function approximation

## Applications

MDPs and reinforcement learning have numerous applications:

- **Game Playing**: AlphaGo, chess engines, video game AI
- **Robotics**: Navigation, manipulation, autonomous vehicles
- **Operations Research**: Inventory management, resource allocation
- **Finance**: Portfolio optimization, trading strategies
- **Healthcare**: Treatment planning, medical diagnosis

## Key Algorithms Summary

| Algorithm | Type | Policy | Model Required |
|-----------|------|--------|----------------|
| Value Iteration | Dynamic Programming | Optimal | Yes |
| Policy Iteration | Dynamic Programming | Optimal | Yes |
| Monte Carlo | Model-free | Any | No |
| SARSA | Temporal Difference | On-policy | No |
| Q-learning | Temporal Difference | Off-policy | No |

## Further Reading

- Sutton & Barto: "Reinforcement Learning: An Introduction"
- Puterman: "Markov Decision Processes"
- Bertsekas: "Dynamic Programming and Optimal Control"

## Implementation Notes

When implementing MDP algorithms, consider:

- **State Representation**: How to efficiently represent and store states
- **Action Selection**: Balancing exploration and exploitation
- **Convergence Criteria**: When to stop iterative algorithms
- **Computational Complexity**: Scaling to large state spaces
- **Parameter Tuning**: Learning rates, discount factors, exploration rates 