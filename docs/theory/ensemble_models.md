Ensemble Models
================

## Confidence intervals
Confidence intervals (CIs) are computed using the answer given by Greg Snow in Ref 1. 

Basically, CI measures the standard deviation of the mean of the response $ Var(E(\hat{y_i} | X_i)) $. 

So, we simply consider the distribution of the predictions made by each tree and take its variance. 

Equivalently, we compute the appropriate quantile for the desired confidence interval and return the corresponding value from the distribution.

This hasn't been benchmarked against anything but it seems reasonable because it is mentioned by Hastie et al. in ESL (Ref 4).

Basically, the idea is that for any quantity S predicted from bootstrap samples,

$$ \widehat{Var}[S(Z)] = \frac{1}{B-1} \sum_{b=1}^{B} (S(Z^{*b}) - \bar{S}^*)^2 $$

Where $\bar{S}^* = \sum_b S(Z^{*b}) /B$.



## Prediction intervals



There are various prediction intervals reported in literature as discussed in Ref 2. 

However, to decide an appropriate benchmark, we use the test used by Ref 2. That is, asymptotically (limit of large data), a prediction interval $\mathcal{I}_\alpha$ with confidence level ($1-\alpha$) should be such that

$$ \mathbb{P}(Y \in  \mathcal{I}_\alpha ) \approx 1 - \alpha$$

This type of coverage (spread of the prediction interval) is said to be the Type I coverage. 

We compared two kinds of prediction intervals.


### 1. RMSE from overall error distribution (`marginal` method in code)


This approach was suggested in Zhang et al in Ref 2 and it is called *OOB Prediction Intervals* in the work.

Basically, the idea is that the prediction interval for any $\hat{y_i}$ can be computed from the distribution of training residuals computed with *Out-of-bag (OOB)* training samples. It is given as: 

$$ \{ D_i \equiv y_i  - \hat{y}_{(i)} \}_{i=1}^N $$

The two-sided $1-\alpha$ prediction inteval is then obtained as:
$$ \hat{Y} + D_{[n, \alpha/2]} \leq Y \leq \hat{Y} + D_{[n, 1-\alpha/2]} $$

Where $D_{[n, \gamma]}$ denotes the $\gamma$ quantile of the distribution with $n$ training samples. 

#### Test
In benchmarking applications, we noticed that this approach leads to a coverage of $1-\alpha$ as expected and is thus **recommended by default**.


### 2. RMSE from each tree (`individual` method in code)

This approach was suggested by Greg Snow in Ref 1.

The basic idea was to compute RMSE for each tree as 

$$ \sigma^2_k = \frac{1}{N}\sum_{i=1}^N(y_{ik} - y_i)^2 $$

Then for each sample as predicted by each tree, draw a sample from the assumed distribution ($\mathcal{D}$) with mean 0 and variance $\sigma^2_k$. Finally, the interval was estimated by the required quantiles of this distribution.

Effectively, this is equivalent to pooling the variance estimates from trees and adding it to the SD of the confidence interval.

$$ \sigma^2_{pooled} = \frac{1}{K} \sum_{k=1}^K \sigma^2_k $$

$$ Var(\hat{y}_i | X_i) = Var(E(\hat{y_i} | X_i)) + \sigma^2_{pooled} $$

#### Test

However, in applications (both in [Python](../../tests/benchmarking/ensemble_model_prediction_interval.py) and [R](../../tests/benchmarking/ensemble_model_validation.R)), we notice that this approach leads to larger coverage than the confidence level. In other words, this apprach leads to a larger estimate of uncertainty. 

So, this approach is *NOT* recommended by default.


#### Issues
Before reproducing this approach, we tried to validate this approach in R (see benchmarking in [R](../../tests/benchmarking/ensemble_model_validation.R)). However, we were unable to reproduce the predictions and `fit$mse` values reported in the `randomForest` library by Breiman [Ref 3]. 

This can be investigated in future.



## References
1. Answer by Greg Snow on [Stackoverflow](https://stats.stackexchange.com/questions/56895/do-the-predictions-of-a-random-forest-model-have-a-prediction-interval).
2. Zhang et al. (2020). DOI: 10.1080/00031305.2019.1585288.
3. Breiman, 2001. DOI: 10.1023/A:1010933404324.
4. Hastie et al 2009, Elements of Statistical Learning, URL: https://hastie.su.domains/ElemStatLearn/