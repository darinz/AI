# EM Algorithm Overview

## Introduction

The Expectation-Maximization (EM) algorithm is a powerful iterative method for parameter estimation in probabilistic models with latent (hidden) variables, such as Bayesian networks with missing data or hidden states. EM is widely used in machine learning, statistics, and artificial intelligence.

## When is EM Used?

- When some variables in the model are unobserved (latent)
- When data is missing or incomplete
- In mixture models (e.g., Gaussian Mixture Models)
- In Hidden Markov Models (HMMs) and Bayesian networks with hidden nodes

## EM Algorithm Steps

The EM algorithm alternates between two steps:

1. **Expectation Step (E-step):**
   - Compute the expected value of the log-likelihood with respect to the current estimate of the distribution over the latent variables.
   - Intuitively: "Fill in" the missing data using the current parameters.

2. **Maximization Step (M-step):**
   - Maximize the expected log-likelihood found in the E-step with respect to the model parameters.
   - Intuitively: Update the parameters as if the filled-in data were real.

This process is repeated until convergence.

## Mathematical Formulation

Given:
- Observed data: $X$
- Latent variables: $Z$
- Model parameters: $\theta$
- Complete-data likelihood: $P(X, Z | \theta)$

The EM algorithm maximizes the marginal likelihood:
$$P(X | \theta) = \sum_Z P(X, Z | \theta)$$

### E-step:
Compute the expected complete-data log-likelihood:
$$Q(\theta | \theta^{(t)}) = \mathbb{E}_{Z | X, \theta^{(t)}} [\log P(X, Z | \theta)]$$

### M-step:
Update parameters to maximize $Q$:
$$\theta^{(t+1)} = \arg\max_{\theta} Q(\theta | \theta^{(t)})$$

## Intuition

- The E-step "guesses" the missing data using the current model.
- The M-step updates the model as if the guesses were correct.
- Each iteration increases (or leaves unchanged) the data likelihood.
- EM is guaranteed to converge to a local maximum of the likelihood.

## Applications in Bayesian Networks

- Learning parameters with missing data
- Learning parameters in models with hidden variables (e.g., mixture models, HMMs)
- Clustering and density estimation

## Limitations

- May converge to local, not global, maxima
- Can be slow if the E-step or M-step is computationally expensive
- Sensitive to initialization

## Key Takeaways

- EM is a general approach for maximum likelihood estimation with latent variables
- Alternates between estimating missing data and updating parameters
- Widely used in unsupervised learning, clustering, and probabilistic modeling 