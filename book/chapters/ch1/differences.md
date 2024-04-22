# The difference between learning and pure optimization

In most machine learning scenarios, we define some performance measure $P$, defined for the test set and may be intractable. But in this case, we optimize $P$ only indirectly because we reduce a cost function $J(\theta)$ in the \textbf{hope} that doing so will optimize $P$. 

For example, $P$ can be the binary accuracy measure as 

$$\frac{TP+TN}{TP+TN+FP+FN}$$

Typically, the cost function can be written as an average over the training set, 

$$J(\theta)=\mathbb{E}_{(x,y)\sim \hat{p}_{data}}\mathcal{L}(f(x, \theta), y)$$

where $\mathcal{L}$ is the loss function, $f(x,\theta)$ is the predicted output when the input is $x$, and $\hat{p}_{data}$ is the empirical distribution. Considering a supervised learning scenario, $y$ is the target output. 

The equation defines an objective function for the training set. Considering the minimization scenario with the objective function have the expectation taking across the data generating distribution $p_{data}$ rather than just over the finite training set, we write the cost function as 

$$J(\theta)=\mathbb{E}_{(x,y)\sim p_{data}}\mathcal{L}(f(x, \theta), y)$$