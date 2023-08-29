# Parametric Model Inference

## Estimating model degrees of freedom

### 1. No regularization

#### Linear models
Typically for a linear model with intercept such as:

$$ y = \beta_0 + \textbf{X}_{n \times p}\beta_{p \times 1} $$ 

The degrees of freedom are p.

For linear models without intercept such as:

$$ y = \textbf{X}_{n \times p}\beta_{p \times 1} $$ 

The degrees of freedom are also p. 

#### Non-linear models

Technically, the degrees of freedom for any general non-linear model is given as: 

$$ df(\hat{y}) = \frac{\sum_{i=1}^N Cov(\hat{y_i}, y_i)}{\sigma_\epsilon^2}  $$

As mentioned in ESL Pg 233 [Ref 1]. 

However, in the literature, some people have approximated the model degrees of freedom in the same way as that for linear models.[Ref 2.] In this version (0.1.0), the degrees of freedom computed by the function `set_model_dof` which returns the class attribute `model_dof`, the degrees of freedom are computed by considering the non-linear model as a linear model as done in Ref 2. Thus, the foregoing discussion is about linear models. 

I.e., for a non-linear model given by:

$$ y = f(\textbf{X}_{n \times p}, \beta_{p \times 1}) $$

Degrees of freedom = $p$. 

In version 0.1.0 for this code, we assume that any non-linear model is of this form. 

However, we also allow users to specify a model intercept as the first parameter. This can be done using the argument `fit_intercept=True` in the `NonLinearRegression` model class. In this case, we assume that the model is of the form:

$$ y = \beta_0 + f(\textbf{X}_{n \times p}, \beta_{p \times 1}) $$


### 2. L1 Regularization

For L1 regularization, such as LASSO models, etc., the number of model degrees of freedom are given by the number of non-zero coefficients predicted as any given regularization penalty ($\lambda$). [ESL Pg. 79, Ref 1]

### 3. L2 Regularization

For L2 regularization such as ridge, the degrees of freedom are computed as given in ESL Pg 68 [Ref 1]. 

For some linear model as shown above, with loss function

$$ \mathcal{L(\hat{y}, y)} = r(\hat{y}, y)^Tr(\hat{y}, y) + \lambda ||\beta_p||_2^2$$

The degrees of freedom are:

$$ df(\lambda) = \sum_{j=1}^p \frac{d_j^2}{d_j^2+\lambda} $$

Where $d_j$ is the $j^{th}$ diagonal entry of the $ \textbf{D}$ matrix containing the singular values of the design matrix $\textbf{X} $ in the equation

$$ \textbf{X} = \textbf{U}  \textbf{D}  \textbf{V}^T$$

### 4. L1+L2 regularization

In such cases, for instance elastic net, the degrees of freedom are computed by considering the set of non-zero coefficients for the given hyperparameters. 

Let's call this set the active set $\mathcal{A}$. 

Then, the degrees of freedom are calculated as

$$ df(\lambda) = \sum_{j=1, j\in\mathcal{A}}^p \frac{d_j^2}{d_j^2+\lambda} $$

This is inferred from the degrees of freedom computation in this case as discussed in a talk by Hui Zou [Ref 3] where 

$$ df(\lambda_1, \lambda_2) = Tr(\textbf{H}_{\lambda_2}(\mathcal{A})) $$


## Residual (error) degrees of freedom

For training data with $n$ samples,

Total dof = $n-1$

Residual degrees of freedom (dof) = Error dof = Total dof - model dof

## Estimating mean sum of squares ($\hat{\sigma}^2$)

Given as residual sum of squares divided by error degrees of freedom.

$$ \hat{\sigma}^2 =  \frac{r(\hat{y}, y)^Tr(\hat{y}, y)} {\mathrm{Error\ dof}}$$

## Calculation of the variance-covariance matrix

This is discussed in a course handout by Niclas Börlin at CMU. [Ref 4] 

Basically, it is related to the Hessian matrix of the loss with respect to the parameters.

For some loss function which can be written in terms of model parameters $ \mathcal{L}(\beta) $, the variance-covariance matrix of the fitted parameters can be given as

$$ Var(\hat{\beta}) = \hat{\sigma}^2(\nabla^2\mathcal{L}(\hat{\beta}))^{-1}$$

## Estimating confidence and prediction intervals

### Confidence intervals
Confidence intervals are estimated using the propagation of uncertainties principle that the variances are additive. [Refs 5, 6]

For some response dependent on inputs and parameters, 

$$ y = f(\bf{X}, \bf{\beta})  $$

The uncertainties are given by

$$ \delta y = \delta f(\bf{X}, \bf{\beta}) $$
$$ \delta y= \sqrt{(\nabla_{\textbf{X}}f) \delta\textbf{X} (\nabla_{\textbf{X}}f)^T
        + (\nabla_{\bm{\beta}}f) \delta\bm{\beta} (\nabla_{\bm{\beta}}f)^T} $$

### Prediction intervals

For parametric model, these outputs are an expression of the expected value of the responses. Thus, their variance is

$$  Var( E(\hat{y} | \mathbf{X}, \bm{\beta})) $$

which was calculated from the previous equation. 

However, to get prediction intervals, we wish to estimate

$$ Var(\hat{y} | \mathbf{X}, \bm{\beta}) $$

This is given by adding the MSE to the confidence interval variance.

$$ Var(\hat{y} | \mathbf{X}, \bm{\beta}) = Var(E(\hat{y} | \mathbf{X}, \bm{\beta})) + 
\hat{\sigma}^2 $$




# References
1. Hastie et al 2009, Elements of Statistical Learning, URL: https://hastie.su.domains/ElemStatLearn/.
2. Ruckstuhl, A., Introduction to Nonlinear Regression, https://stat.ethz.ch/~stahel/courses/cheming/nlreg10E.pdf.
3. Zou, H., Hastie, T., Regularization and Variable
Selection via the Elastic Net, https://hastie.su.domains/TALKS/enet_talk.pdf, Pg 22.
4. Nonlinear Optimization, https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts.pdf. 
5. Propagation of Uncertainties, https://web.physics.utah.edu/~belz/phys3719/lecture10.pdf; 
6. Lecture 11: Standard Error, Propagation of
Error, Central Limit Theorem in the Real World
, https://www.stat.cmu.edu/~cshalizi/36-220/lecture-11.pdf