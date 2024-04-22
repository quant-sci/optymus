# Stochastic functions and optimization

## Stochastic functions

A **stochastic function** is a mathematical function that incorporates randomness in its output. In other words, for a given input, the function does not produce a single deterministic value but rather a probability distribution over possible values. This randomness can be due to various factors, such as measurement errors, inherent variability in the system being modeled, or deliberate introduction of noise.

Consider the movement of a stock price over time. We could model this as a stochastic function:

$$S(t) = S_0 + \mu t + \sigma W(t)$$

- $S(t)$ represents the stock price at time t
- $S_0$​ is the initial stock price
- $\mu$ is the average drift of the stock price
- $\sigma$ is the volatility (standard deviation) of price changes
- $W(t)$ is a Wiener process (a standard model for random fluctuations)

The Wiener process introduces randomness, making the stock price evolution unpredictable at any given point in time.

## Stochastic optimization

Stochastic optimization methods are algorithms designed to solve optimization problems involving stochastic functions. These methods aim to find the optimal solution (e.g., minimum or maximum) of a function whose output is subject to random variations. 

Here's how stochastic optimization methods generally work: 

1. Sample the stochastic function: Since the function's output is random, we need to evaluate it multiple times for different realizations of the random variable ω. This provides us with a set of noisy observations of the function. 
2. Use the observations to estimate the function's properties: Based on the sampled values, we can estimate the function's expected value, gradient, or other relevant characteristics. 
3. Update the solution based on the estimated properties: Using the estimated information, we update the current solution towards the optimum. This update step often involves algorithms similar to those used in deterministic optimization, but with modifications to account for the noise. 
4. Repeat steps 1-3 iteratively: The process of sampling, estimation, and update is repeated until convergence to a satisfactory solution is achieved.