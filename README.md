
![Version badge](https://img.shields.io/badge/version-0.1.1-blue)
![Python badge](https://img.shields.io/badge/python-3.9|3.10|3.11-blue?logo=python)
![License badge](https://img.shields.io/badge/License-MIT-blue)
![Format badge](https://img.shields.io/badge/code_format-black-black)
![Linting badge](https://img.shields.io/badge/code_linting-flake8-black)
![Test badge](https://img.shields.io/badge/tests-pytest-black?logo=pytest)
[![tests](https://github.com/architdatar/ml_uncertainty/actions/workflows/run_tests.yml/badge.svg)](https://github.com/architdatar/ml_uncertainty/actions/workflows/run_tests.yml)
[![PyPI Downloads](https://static.pepy.tech/badge/ml-uncertainty)](https://pepy.tech/projects/ml-uncertainty)

![ML Uncertainty](./docs/images/ML_uncertainty_logo.jpg)
=============================

ML Uncertainty is a Python package which provides a scikit-learn-like interface to obtain prediction intervals and model parameter error estimation for machine learning models.

All in less than 4 lines of code.

Getting started
----
Install from PyPI with
```
pip install ml-uncertainty
```

## Examples
**View:** View all [examples](./examples). 

**Run:** To run examples, some additional packages are required since they require plots for visualization. Install these using:
```
pip install matplotlib seaborn jupyter scikit-fda
```

### First example: Linear regression
Consider a linear regression model fit with scikit-learn. The uncertainty estimation can be done as follows:

```Python
# Fit model with sklearn.
regr = LinearRegression(fit_intercept=True)
regr.fit(X_expt, y_expt)

# Set up error estimation with ML uncertainty. 
inf = ParametricModelInference()
inf.set_up_model_inference(X_expt, y_expt, regr)

# Obtain parameter error estimate with ML Uncertainty
df_feature_imp = inf.get_parameter_errors()

# Obtain prediction intervals with ML Uncertainty.
df_int = inf.get_intervals(X_expt, confidence_level=95.0, distribution="t")
```

The result looks like:
![img](./docs/images/linear_regression.jpg)

Find the full example [here](./examples/linear_regression.ipynb).

### Other examples

* [Parameter error estimation](examples/parametric_model.ipynb)
* [Non-linear regression](examples/non_linear_regression_arrhenius.py)
* [Weighted non-linear least squares regression](examples/weighted_non_linear_regression_arrhenius.ipynb)
* [Error Propagation](examples/error_propagation.py)
* [Regression splines](examples/spline_wage_data.ipynb)
* [Regression with periodic data](examples/fourier_basis.ipynb)
* [Random forest regression ](examples/ensemble_model.py)


Intended audience
----
This package is intended to benefit data scientists and ML enthusiasts. 

Motivation
----
* Too often in machine learning, we fit complex models, but cannot quantity their precision via prediction intervals or feature significance.

* This is especially true of the scikit-learn environment which is extremely easy to use but does not offer these functionalities.

* However, in many use cases, especially where we have small and fat datasets, these are insights are critical to produce reliable models and insights. 

* Enter ML Uncertainty! This provides an easy API to get all these insights from models.

* It takes scikit-learn fitted models as inputs and uses appropriate statistics to quantify the uncertainties in ML models.

Computing stats as easy as:

```Python
# Set up the model inference.
inf = ParametricModelInference()
inf.set_up_model_inference(X_train=X, y_train=y, estimator=regr)

# Get parameter importance estimates.
df_imp = inf.get_parameter_errors()

# Get prediction intervals.
df_int = inf.get_intervals(X)
```

Features
--------

1. **Model parameter significance testing:** Tests whether the given model parameters are truly significant or not.

     For ensemble models, it can inform if given features are truly important or if they just seem so due to the instability of the model.

2. **Prediction intervals:** Can produce prediction and confidence intervals for parametric and non-parametric ML models.

3. **Error propagation:** Propagates error from input / model parameters to the outputs.

4. **Non-Linear regression:** Scikit-learn-style API to fit non-linear models. 

Installation
------------
### Dependencies
Python versions: See badges above.\
Packages: See [requirements.txt](./requirements.txt).

### User installation
See [./docs/installation.md](/docs/installation.md).

## Theoretical foundations

Discussion about the theory used can be found here:

* [Parametric models](docs/theory/parametric_models.md)
* [Ensemble models](docs/theory/ensemble_models.md)


## Benchmarking
`NonLinearRegression`, `ParametricModelInference`, and `ErrorPropagation` classes have been benchmarked against the Python [statsmodels](https://www.statsmodels.org/stable/index.html) package. The codes for this can be found [here](tests/benchmarking/). 

To run these benchmarking codes, please install statsmodels using:
```
pip install statsmodels==0.14.0
```

The `EnsembleModelInference` does not have a code to benchmark it against to the best of my knowledge. However, the code follows the ideas developed in the work by [Zhang et al. (2020)](https://www.tandfonline.com/doi/abs/10.1080/00031305.2019.1585288?journalCode=utas20). The test is that a $(1-\alpha)\times100$ % prediction interval must contain $(1-\alpha)$ proportion of the training data. See benchmarking codes [here](tests/benchmarking/). 

Author
-------
Archit Datar (architdatar@gmail.com)


Credits
-------

1. This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

    [Cookiecutter](https://github.com/audreyr/cookiecutter)

    [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage)

2. Some functions in `ParametricModelInference` are adopted from a [Github repo](https://github.com/sriki18/adnls/) by sriki18.
