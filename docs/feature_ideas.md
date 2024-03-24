Topics
========

## Non-Linear Regression

1. Add capabilities for regularization: Use scipy.optimize.minimize function to optimize and build the loss function as shown in ParametricModelInference.
2. Enable non-explicit constrained optimization: Use scipy.optimize.minimize to optimize - Create loss function as in ParametricModelInference class.
3. Accurate degree of freedom calculations for non-linear models: Currently, the model degrees of freedom are computed by considering the number of model parameters. But for true non-linear functions, these should be computed as shown in Hastie et al, [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) Pg 233. 
4. For L2 regularization for non-linear models, see Eq. 7.34 on Pg 233 of Hastie et al, [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/).  

    For 3, 4: Develop code using simulations, see documentation for ensemble model inference degrees of freedom calculation.

5. Implement cross-validation with sklearn grid search. 
6. See if it can be used with sklearn pipelines.

## Parametric model inference

1. For linear models and ridge, there is a closed form
    solution. Var[b]=σ2(X′X)−1. See [here](https://stats.stackexchange.com/questions/68151/how-to-derive-variance-covariance-matrix-of-coefficients-in-linear-regression).
    [For ridge: ](https://online.stat.psu.edu/stat857/node/155/)under Properties of Ridge estimator.
2. Allow users to compute Hessian through a first-order approximation for
    functions that might not be 2-times differenciable. Refer to Niclas Borgin's lectures.
    (J'J)
3. Extend to classification models such as LogisticRegression, etc.
4. Simplify and show examples for computing prediction intervals for non-normal distributions. Allow users to specify a distribution, use the stat generator function idea from ensemble model inference, and get prediction intervals.


## Ensemble models
1. Create tests for other kinds of models such as gradient boosting and classification models.
2. Create tests for model significance. Basically to answer the question: is the model better than the null model? 
        1. Refer to the book [Nonparametric Statistical Methods](https://www.wiley.com/en-us/Nonparametric+Statistical+Methods,+3rd+Edition-p-9780470387375) and see which tests can be used as non-parametric version of the F-test. 
        2. To achieve this, we will require to compute degrees of freedom of non-parametric models. The MC algorithm for 
        this has been discussed in DOI: 10.1080/01621459.1998.10474094. 
        Algorithm to do so is mentioned in Pg 122, algorithm 1
            1. Basic idea of the method is to create t perturbations in Y and measure the efects on $\hat{y}$. 
        Results can be compared with URL: https://arxiv.org/pdf/1911.00190.pdf Pg 11 and 12. 

4. Create wrappers for more non-normal distributions as done in the [example](../examples/random_forest_non-normal_distribution.py).
5. Incorporate sample weights to compute prediction and confidence intervals.
6. Investigate: Attempt to reproduce results from `fit$predicted` and `fit$mse` from `pred$individual` in the `randomForest` library in R. Hasn't been possible thus far.


## General
1. Enable model signficance tests: 
    Analogous to F-tests for ordinary least squares regression. Implement F-tests for parametric models (with appropriate assumptions) and various distributions.

## Tests
1. Write test for cases in error propagation.py where there are errors in X (i.e., X_err is not None)

## Documentation
1. Write more documentation for internal methods.
2. More documentation for class attributes.
