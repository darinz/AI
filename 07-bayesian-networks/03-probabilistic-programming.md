# Probabilistic Programming with Bayesian Networks

## Overview

Probabilistic programming provides a powerful framework for building and reasoning with Bayesian networks. It allows us to specify complex probabilistic models declaratively and perform inference automatically.

## What is Probabilistic Programming?

Probabilistic programming combines the expressiveness of programming languages with the power of probabilistic reasoning. It enables us to:
- Define complex probabilistic models using familiar programming constructs
- Automatically perform inference using sophisticated algorithms
- Handle uncertainty in a principled way

## Language Integration

### Using PyMC for Bayesian Networks

PyMC is a popular probabilistic programming library for Python that provides excellent support for Bayesian networks.

```python
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

class PyMCBayesianNetwork:
    def __init__(self):
        self.model = None
        self.trace = None
    
    def create_weather_model(self):
        """Create a weather Bayesian network using PyMC"""
        
        with pm.Model() as model:
            # Prior for Rain
            rain_prob = pm.Beta('rain_prob', alpha=1, beta=4)  # Prior belief: 20% chance of rain
            rain = pm.Bernoulli('rain', p=rain_prob)
            
            # Sprinkler depends on Rain
            sprinkler_prob_no_rain = pm.Beta('sprinkler_prob_no_rain', alpha=2, beta=3)
            sprinkler_prob_rain = pm.Beta('sprinkler_prob_rain', alpha=1, beta=99)
            
            # Use pm.math.switch to conditionally select probability
            sprinkler_p = pm.math.switch(rain, sprinkler_prob_rain, sprinkler_prob_no_rain)
            sprinkler = pm.Bernoulli('sprinkler', p=sprinkler_p)
            
            # WetGrass depends on both Rain and Sprinkler
            wetgrass_prob_rain_sprinkler = pm.Beta('wetgrass_prob_rain_sprinkler', alpha=99, beta=1)
            wetgrass_prob_rain_no_sprinkler = pm.Beta('wetgrass_prob_rain_no_sprinkler', alpha=9, beta=1)
            wetgrass_prob_no_rain_sprinkler = pm.Beta('wetgrass_prob_no_rain_sprinkler', alpha=9, beta=1)
            wetgrass_prob_no_rain_no_sprinkler = pm.Beta('wetgrass_prob_no_rain_no_sprinkler', alpha=1, beta=99)
            
            # Select probability based on Rain and Sprinkler
            wetgrass_p = pm.math.switch(
                rain,
                pm.math.switch(sprinkler, wetgrass_prob_rain_sprinkler, wetgrass_prob_rain_no_sprinkler),
                pm.math.switch(sprinkler, wetgrass_prob_no_rain_sprinkler, wetgrass_prob_no_rain_no_sprinkler)
            )
            
            wetgrass = pm.Bernoulli('wetgrass', p=wetgrass_p)
            
            self.model = model
        
        return model
    
    def sample_prior(self, n_samples=1000):
        """Sample from the prior distribution"""
        with self.model:
            prior_samples = pm.sample_prior_predictive(samples=n_samples)
        return prior_samples
    
    def sample_posterior(self, data=None, n_samples=1000):
        """Sample from the posterior distribution"""
        with self.model:
            if data is not None:
                # Condition on observed data
                with pm.Data():
                    pm.Bernoulli('rain_obs', p=self.model.rain_prob, observed=data['rain'])
                    pm.Bernoulli('sprinkler_obs', p=self.model.sprinkler_p, observed=data['sprinkler'])
                    pm.Bernoulli('wetgrass_obs', p=self.model.wetgrass_p, observed=data['wetgrass'])
            
            # Sample from posterior
            self.trace = pm.sample(n_samples, return_inferencedata=True)
        
        return self.trace
    
    def plot_posterior(self):
        """Plot posterior distributions"""
        if self.trace is not None:
            az.plot_posterior(self.trace)
            plt.show()
    
    def get_posterior_summary(self):
        """Get summary statistics of posterior distributions"""
        if self.trace is not None:
            return az.summary(self.trace)
        return None

# Example: Using PyMC for Weather Network
def weather_network_example():
    """Example of using PyMC for the weather Bayesian network"""
    
    # Create the model
    pymc_bn = PyMCBayesianNetwork()
    model = pymc_bn.create_weather_model()
    
    print("PyMC Weather Model Created")
    print("Model variables:", [var.name for var in model.vars])
    
    # Sample from prior
    prior_samples = pymc_bn.sample_prior(n_samples=1000)
    print(f"Prior samples shape: {prior_samples.prior_predictive['rain'].shape}")
    
    # Generate some synthetic data
    np.random.seed(42)
    n_obs = 100
    data = {
        'rain': np.random.choice([0, 1], size=n_obs, p=[0.8, 0.2]),
        'sprinkler': np.random.choice([0, 1], size=n_obs, p=[0.6, 0.4]),
        'wetgrass': np.random.choice([0, 1], size=n_obs, p=[0.7, 0.3])
    }
    
    # Sample from posterior
    trace = pymc_bn.sample_posterior(data=data, n_samples=1000)
    
    # Get posterior summary
    summary = pymc_bn.get_posterior_summary()
    print("\nPosterior Summary:")
    print(summary)
    
    return pymc_bn

# Run the example
if __name__ == "__main__":
    weather_model = weather_network_example()
```

## Model Specification

### Declarative Model Definition

Probabilistic programming allows us to specify models declaratively, focusing on the relationships between variables rather than the computational details.

```python
import pymc as pm
import numpy as np

class DeclarativeBayesianNetwork:
    def __init__(self):
        self.models = {}
    
    def create_medical_diagnosis_model(self):
        """Create a medical diagnosis model declaratively"""
        
        with pm.Model() as model:
            # Patient characteristics
            age = pm.Normal('age', mu=50, sigma=15)
            gender = pm.Bernoulli('gender', p=0.5)
            
            # Disease probability depends on age and gender
            disease_logit = pm.Normal('disease_logit', mu=0, sigma=1)
            disease_prob = pm.math.sigmoid(disease_logit + 0.1 * (age - 50) / 15 + 0.2 * (gender - 0.5))
            disease = pm.Bernoulli('disease', p=disease_prob)
            
            # Symptoms depend on disease
            symptom1_prob = pm.math.switch(disease, 0.8, 0.1)
            symptom1 = pm.Bernoulli('symptom1', p=symptom1_prob)
            
            symptom2_prob = pm.math.switch(disease, 0.7, 0.15)
            symptom2 = pm.Bernoulli('symptom2', p=symptom2_prob)
            
            # Tests depend on symptoms
            test1_prob = pm.math.switch(symptom1, 0.9, 0.05)
            test1 = pm.Bernoulli('test1', p=test1_prob)
            
            test2_prob = pm.math.switch(symptom2, 0.85, 0.1)
            test2 = pm.Bernoulli('test2', p=test2_prob)
            
            self.models['medical'] = model
        
        return model
    
    def create_financial_risk_model(self):
        """Create a financial risk assessment model"""
        
        with pm.Model() as model:
            # Economic factors
            gdp_growth = pm.Normal('gdp_growth', mu=0.02, sigma=0.01)
            interest_rate = pm.Normal('interest_rate', mu=0.03, sigma=0.005)
            
            # Company-specific factors
            company_size = pm.Categorical('company_size', p=[0.3, 0.4, 0.3])  # Small, Medium, Large
            industry = pm.Categorical('industry', p=[0.2, 0.3, 0.3, 0.2])  # Tech, Finance, Manufacturing, Retail
            
            # Risk factors
            market_risk = pm.Normal('market_risk', mu=0.1, sigma=0.05)
            credit_risk = pm.Normal('credit_risk', mu=0.05, sigma=0.02)
            
            # Risk depends on all factors
            risk_logit = (pm.Normal('risk_logit', mu=0, sigma=1) + 
                         0.5 * (gdp_growth - 0.02) / 0.01 +
                         0.3 * (interest_rate - 0.03) / 0.005 +
                         0.2 * (market_risk - 0.1) / 0.05 +
                         0.1 * (credit_risk - 0.05) / 0.02)
            
            risk_prob = pm.math.sigmoid(risk_logit)
            high_risk = pm.Bernoulli('high_risk', p=risk_prob)
            
            # Default probability depends on risk
            default_prob = pm.math.switch(high_risk, 0.15, 0.02)
            default = pm.Bernoulli('default', p=default_prob)
            
            self.models['financial'] = model
        
        return model

# Example: Declarative Models
def declarative_models_example():
    """Example of creating declarative Bayesian network models"""
    
    declarative_bn = DeclarativeBayesianNetwork()
    
    # Create medical diagnosis model
    medical_model = declarative_bn.create_medical_diagnosis_model()
    print("Medical diagnosis model created with variables:")
    print([var.name for var in medical_model.vars])
    
    # Create financial risk model
    financial_model = declarative_bn.create_financial_risk_model()
    print("\nFinancial risk model created with variables:")
    print([var.name for var in financial_model.vars])
    
    return declarative_bn

# Run the example
if __name__ == "__main__":
    models = declarative_models_example()
```

## Inference Engines

### Automatic Inference with PyMC

PyMC provides multiple inference algorithms that can be applied automatically to any model.

```python
class InferenceEngine:
    def __init__(self, model):
        self.model = model
        self.trace = None
    
    def run_nuts_sampling(self, n_samples=2000, tune=1000):
        """Run No-U-Turn Sampling (NUTS) - default PyMC sampler"""
        with self.model:
            self.trace = pm.sample(n_samples, tune=tune, return_inferencedata=True)
        return self.trace
    
    def run_metropolis_sampling(self, n_samples=2000, tune=1000):
        """Run Metropolis-Hastings sampling"""
        with self.model:
            self.trace = pm.sample(n_samples, tune=tune, step=pm.Metropolis(), 
                                 return_inferencedata=True)
        return self.trace
    
    def run_variational_inference(self, n_samples=1000):
        """Run Automatic Differentiation Variational Inference (ADVI)"""
        with self.model:
            # Find MAP estimate first
            map_estimate = pm.find_MAP()
            
            # Run variational inference
            self.trace = pm.sample(n_samples, return_inferencedata=True)
        
        return self.trace
    
    def run_prior_predictive(self, n_samples=1000):
        """Generate prior predictive samples"""
        with self.model:
            prior_samples = pm.sample_prior_predictive(samples=n_samples)
        return prior_samples
    
    def run_posterior_predictive(self, n_samples=1000):
        """Generate posterior predictive samples"""
        if self.trace is None:
            raise ValueError("Must run inference first")
        
        with self.model:
            posterior_samples = pm.sample_posterior_predictive(
                self.trace, samples=n_samples
            )
        return posterior_samples
    
    def diagnose_convergence(self):
        """Diagnose MCMC convergence"""
        if self.trace is None:
            raise ValueError("Must run inference first")
        
        # Plot trace plots
        az.plot_trace(self.trace)
        plt.show()
        
        # Plot rank plots
        az.plot_rank(self.trace)
        plt.show()
        
        # Get convergence diagnostics
        summary = az.summary(self.trace)
        print("Convergence Diagnostics:")
        print(f"Effective sample sizes: {summary['ess_bulk'].min():.0f} - {summary['ess_bulk'].max():.0f}")
        print(f"R-hat values: {summary['r_hat'].min():.3f} - {summary['r_hat'].max():.3f}")

# Example: Inference Engine
def inference_engine_example():
    """Example of using different inference engines"""
    
    # Create a simple model
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0, sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=1)
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=np.random.normal(0, 1, 100))
    
    # Create inference engine
    engine = InferenceEngine(model)
    
    # Run different inference methods
    print("Running NUTS sampling...")
    nuts_trace = engine.run_nuts_sampling(n_samples=1000)
    
    print("Running Metropolis sampling...")
    engine.run_metropolis_sampling(n_samples=1000)
    
    print("Running variational inference...")
    engine.run_variational_inference(n_samples=1000)
    
    # Diagnose convergence
    engine.diagnose_convergence()
    
    return engine

# Run the example
if __name__ == "__main__":
    inference_engine = inference_engine_example()
```

## Software Tools

### Overview of Popular Probabilistic Programming Languages

```python
class ProbabilisticProgrammingTools:
    def __init__(self):
        self.tools = {
            'PyMC': {
                'language': 'Python',
                'strengths': ['Excellent documentation', 'Rich ecosystem', 'NUTS sampler'],
                'weaknesses': ['Limited to Python', 'Steep learning curve'],
                'best_for': ['General Bayesian modeling', 'Research', 'Production']
            },
            'Stan': {
                'language': 'Stan (C++)',
                'strengths': ['Very fast', 'Excellent diagnostics', 'Hamiltonian Monte Carlo'],
                'weaknesses': ['Separate language to learn', 'Less flexible'],
                'best_for': ['High-performance inference', 'Complex models']
            },
            'JAGS': {
                'language': 'R',
                'strengths': ['Simple syntax', 'Good for beginners', 'Gibbs sampling'],
                'weaknesses': ['Slower than Stan', 'Limited to conjugate models'],
                'best_for': ['Learning', 'Simple models', 'Teaching']
            },
            'Edward2': {
                'language': 'Python (TensorFlow)',
                'strengths': ['Deep learning integration', 'Variational inference', 'GPU support'],
                'weaknesses': ['Less mature', 'Steeper learning curve'],
                'best_for': ['Deep probabilistic models', 'Variational inference']
            }
        }
    
    def compare_tools(self):
        """Compare different probabilistic programming tools"""
        print("Probabilistic Programming Tools Comparison:")
        print("=" * 60)
        
        for tool, info in self.tools.items():
            print(f"\n{tool}:")
            print(f"  Language: {info['language']}")
            print(f"  Strengths: {', '.join(info['strengths'])}")
            print(f"  Weaknesses: {', '.join(info['weaknesses'])}")
            print(f"  Best for: {', '.join(info['best_for'])}")
    
    def get_recommendation(self, use_case):
        """Get tool recommendation based on use case"""
        recommendations = {
            'beginner': 'JAGS',
            'research': 'PyMC',
            'production': 'Stan',
            'deep_learning': 'Edward2',
            'teaching': 'JAGS'
        }
        
        return recommendations.get(use_case, 'PyMC')

# Example: Tool Comparison
def tool_comparison_example():
    """Example of comparing probabilistic programming tools"""
    
    tools = ProbabilisticProgrammingTools()
    tools.compare_tools()
    
    print(f"\nRecommendation for beginners: {tools.get_recommendation('beginner')}")
    print(f"Recommendation for research: {tools.get_recommendation('research')}")
    print(f"Recommendation for production: {tools.get_recommendation('production')}")
    
    return tools

# Run the example
if __name__ == "__main__":
    tool_comparison = tool_comparison_example()
```

## Advanced Features

### Custom Distributions and Likelihoods

```python
class CustomBayesianNetwork:
    def __init__(self):
        self.models = {}
    
    def create_custom_model(self):
        """Create a model with custom distributions"""
        
        with pm.Model() as model:
            # Custom prior for a parameter
            def custom_prior(name, mu, sigma):
                return pm.TruncatedNormal(name, mu=mu, sigma=sigma, lower=0, upper=1)
            
            # Custom likelihood function
            def custom_likelihood(value, mu, sigma):
                return pm.Normal.dist(mu=mu, sigma=sigma).logp(value)
            
            # Model parameters
            mu = custom_prior('mu', mu=0, sigma=1)
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Observations
            y = pm.Custom('y', logp=custom_likelihood, mu=mu, sigma=sigma, 
                         observed=np.random.normal(0, 1, 50))
            
            self.models['custom'] = model
        
        return model
    
    def create_hierarchical_model(self):
        """Create a hierarchical Bayesian network"""
        
        with pm.Model() as model:
            # Global parameters
            global_mu = pm.Normal('global_mu', mu=0, sigma=1)
            global_sigma = pm.HalfNormal('global_sigma', sigma=1)
            
            # Group-specific parameters
            n_groups = 5
            group_mu = pm.Normal('group_mu', mu=global_mu, sigma=global_sigma, shape=n_groups)
            group_sigma = pm.HalfNormal('group_sigma', sigma=1, shape=n_groups)
            
            # Observations for each group
            for i in range(n_groups):
                n_obs = 20
                y = pm.Normal(f'y_group_{i}', mu=group_mu[i], sigma=group_sigma[i],
                             observed=np.random.normal(i, 1, n_obs))
            
            self.models['hierarchical'] = model
        
        return model

# Example: Advanced Features
def advanced_features_example():
    """Example of advanced probabilistic programming features"""
    
    custom_bn = CustomBayesianNetwork()
    
    # Create custom model
    custom_model = custom_bn.create_custom_model()
    print("Custom model created")
    
    # Create hierarchical model
    hierarchical_model = custom_bn.create_hierarchical_model()
    print("Hierarchical model created")
    
    return custom_bn

# Run the example
if __name__ == "__main__":
    advanced_bn = advanced_features_example()
```

## Key Takeaways

1. **Probabilistic programming** provides a powerful framework for Bayesian networks
2. **PyMC** is an excellent choice for Python-based probabilistic programming
3. **Declarative model specification** focuses on relationships rather than computation
4. **Automatic inference** handles the complex computational details
5. **Multiple inference algorithms** are available for different scenarios
6. **Custom distributions and likelihoods** enable flexible modeling
7. **Hierarchical models** can capture complex dependency structures

## Exercises

1. Create a Bayesian network for a recommendation system using PyMC
2. Implement a custom likelihood function for a specific domain
3. Build a hierarchical model for multi-group data analysis
4. Compare different inference algorithms on the same model
5. Create a probabilistic programming model for time series data 