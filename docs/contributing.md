Topics
========

## Non-Linear Regression

1. Add capabilities for regularization:Use scipy.optimize.minimize function to optimize and build the loss function as shown in ParametricModelInference.
2. Enable non-explicit constrained optimization: Use scipy.optimize.minimize to optimize. 
3. Accurate degree of freedom calculations for non-linear models: Currently, the model degrees of freedom are computed by considering the number of model parameters. But for true non-linear functions, these should be computed as shown in Hastie et al, [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) Pg 233. 
4. For L2 regularization for non-linear models, see Eq. 7.34 on Pg 233 of Hastie et al, [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/).  

For 3, 4: Develop code using simulations, see documentation for ensemble model 
inference degrees of freedom calculation.


## Parametric model inference

1. For linear models and ridge, there is a closed form
    solution. Var[b]=σ2(X′X)−1.
    https://stats.stackexchange.com/questions/68151/how-to-derive-variance-covariance-matrix-of-coefficients-in-linear-regression
    Ridge: https://online.stat.psu.edu/stat857/node/155/ under Properties of Ridge estimator.
2. Allow users to compute Hessian through a first-order approximation for
    functions that might not be 2-times differenciable. Refer to Niclas Borgin's lectures.
    (J'J)

## General
1. Enable model signficance tests: 
    Analogous to F-tests for ordinary least squares regression. Implement F-tests for parametric models (with appropriate assumptions) and various distributions.
    
    For non-parametric models (like random forests), enable these tests.

## Documentation
1. Write more documentation for internal methods.
    
