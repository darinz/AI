# EM Algorithm Examples

## Example 1: Gaussian Mixture Model (GMM)

The EM algorithm is commonly used to fit a Gaussian Mixture Model to data when the component assignments are unknown.

### Problem Setup
- Data: $X = \{x_1, x_2, ..., x_N\}$
- Model: $K$ Gaussian components with means $\mu_k$, variances $\sigma_k^2$, and mixing weights $\pi_k$
- Latent variable: $z_i$ (component assignment for $x_i$)

### EM Steps for GMM

#### E-step:
Compute the responsibility $\gamma_{ik}$ (probability that $x_i$ belongs to component $k$):
$$
gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \sigma_k^2)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i | \mu_j, \sigma_j^2)}
$$

#### M-step:
Update parameters using the responsibilities:
- $N_k = \sum_{i=1}^N \gamma_{ik}$
- $\mu_k = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} x_i$
- $\sigma_k^2 = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (x_i - \mu_k)^2$
- $\pi_k = \frac{N_k}{N}$

### Python Example: 1D GMM

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate synthetic data
np.random.seed(42)
N = 500
X = np.concatenate([
    np.random.normal(-2, 0.5, N//2),
    np.random.normal(3, 1.0, N//2)
])

# Number of components
K = 2

# Initialize parameters
mu = np.random.choice(X, K)
sigma = np.ones(K)
pi = np.ones(K) / K

# EM algorithm
n_iter = 20
for iteration in range(n_iter):
    # E-step: compute responsibilities
    gamma = np.zeros((N, K))
    for k in range(K):
        gamma[:, k] = pi[k] * norm.pdf(X, mu[k], sigma[k])
    gamma /= gamma.sum(axis=1, keepdims=True)

    # M-step: update parameters
    N_k = gamma.sum(axis=0)
    mu = (gamma * X[:, np.newaxis]).sum(axis=0) / N_k
    sigma = np.sqrt((gamma * (X[:, np.newaxis] - mu) ** 2).sum(axis=0) / N_k)
    pi = N_k / N

# Plot results
plt.hist(X, bins=30, density=True, alpha=0.5, label='Data')
x_grid = np.linspace(X.min(), X.max(), 1000)
for k in range(K):
    plt.plot(x_grid, pi[k] * norm.pdf(x_grid, mu[k], sigma[k]), label=f'Component {k+1}')
plt.title('Gaussian Mixture Model (EM)')
plt.legend()
plt.show()
```

## Example 2: EM for Hidden Markov Model (HMM) - Baum-Welch Algorithm

The EM algorithm for HMMs is known as the Baum-Welch algorithm. It learns the transition, emission, and initial probabilities from observed sequences.

### Python Example: Simple HMM

```python
import numpy as np

# States: 0 = Rainy, 1 = Sunny
# Observations: 0 = Walk, 1 = Shop, 2 = Clean
N_states = 2
N_obs = 3
T = 100

# Generate synthetic observation sequence (for demonstration)
np.random.seed(42)
obs_seq = np.random.choice(N_obs, size=T)

# Initialize parameters randomly
A = np.random.dirichlet(np.ones(N_states), size=N_states)  # Transition
B = np.random.dirichlet(np.ones(N_obs), size=N_states)     # Emission
pi = np.random.dirichlet(np.ones(N_states))                # Initial

# EM (Baum-Welch) algorithm
n_iter = 10
for iteration in range(n_iter):
    # Forward pass
    alpha = np.zeros((T, N_states))
    alpha[0] = pi * B[:, obs_seq[0]]
    for t in range(1, T):
        for j in range(N_states):
            alpha[t, j] = (alpha[t-1] @ A[:, j]) * B[j, obs_seq[t]]
    # Backward pass
    beta = np.zeros((T, N_states))
    beta[-1] = 1
    for t in range(T-2, -1, -1):
        for i in range(N_states):
            beta[t, i] = (A[i] * B[:, obs_seq[t+1]] * beta[t+1]).sum()
    # Compute gamma and xi
    gamma = (alpha * beta)
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.zeros((T-1, N_states, N_states))
    for t in range(T-1):
        denom = (alpha[t][:, None] * A * B[:, obs_seq[t+1]] * beta[t+1]).sum()
        for i in range(N_states):
            numer = alpha[t, i] * A[i] * B[:, obs_seq[t+1]] * beta[t+1]
            xi[t, i, :] = numer / denom
    # M-step: update parameters
    pi = gamma[0]
    A = xi.sum(axis=0) / xi.sum(axis=(0, 2), keepdims=True)
    for j in range(N_states):
        for k in range(N_obs):
            mask = (obs_seq == k)
            B[j, k] = gamma[mask, j].sum() / gamma[:, j].sum()
    # Normalize B
    B /= B.sum(axis=1, keepdims=True)

print('Learned initial probabilities:', pi)
print('Learned transition matrix:', A)
print('Learned emission matrix:', B)
```

## Key Takeaways
- EM is practical for learning in models with hidden variables
- GMM and HMMs are classic applications
- Python code can be adapted for more complex Bayesian networks with latent variables 