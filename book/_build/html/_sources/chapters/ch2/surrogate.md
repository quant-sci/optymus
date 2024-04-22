# Surrogate loss function and early stopping

Not rare, the loss function can't be optimized efficiently. We can use a **surrogate loss function** in this case.

Some advantages of surrogate:

- negative log-likelihood of the correct class is typically used as a surrogate for the 0-1 loss;
- negative log-likelihood allows the model to estimate the conditional probability of the classes, given the input, and if the model can do it well, then it picks the classes that yield the least classification error in expectation;

An important difference between optimization in general and optimization as we use for training algorithms: (i) training algorithms do not usually halt at the local minimum; (ii) a machine learning algorithm usually minimizes a **surrogate loss function** but halts when a convergence criterion based on **early stopping** is satisfied; (iii) training often halts while the surrogate loss function still has large derivatives. This is very different from the pure optimization setting, where an optimization algorithm is considered to have converged when the gradient becomes very small.