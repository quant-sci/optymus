# Batch and minibatch

In machine learning algorithms the objective function usually decomposes as a sum over the training sample. Optimization algorithms for machine learning typically compute each update to the parameters based on the expected value of the cost function estimated using only a subset of the terms of the full cost function.

Most of the properties of $J$ used by optimization algorithms are also expectations over the training set. The most commonly used property is the gradient 

$$\nabla_{\theta}J(\theta)=\mathbb{E}_{x,y\sim\hat{p}_{data}}\nabla_{\theta}log~p_{model}(x,y;\theta)$$

Optimization algorithms that use the entire training set are called **batch** or **deterministic** gradient methods. Optimization algorithms that use only a single example at a time are sometimes called **stochastic** or **online** gradient methods. Optimization algorithms that use some number between 2 and $n-1$ training examples are called **minibatch** methods. Is also common to call these methods as **stochastic**.

A canonical example of minibatch is the stochastic gradient descent method. The minibatch sizes are generally driven by the following factors: (i) larger batches provide a more accurate estimate of the gradient; (ii) multicore architectures are usually underutilized by extremely small batches. 

There are some hardware considerations about the batch size in GPU context. Small batches can offer a regularizing effect, perhaps due to the noise they add to the learning process.

First-order methods are usually relatively robust and can handle smaller batch sizes like 100. Second-order methods typically require much larger batch sizes like 10.000. The minibatches must be selected randomly. A motivation for minibatch SGD is that it follows the gradient of the true _generalization error_ so long as no examples are repeated;