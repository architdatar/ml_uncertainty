#!/usr/bin/env python'

"""Tests for `ensemble_model_inference` package"""

import pytest
import numpy as np
from ml_uncertainty.model_inference.parametric_model_inference import (
    ParametricModelInference,
)


@pytest.fixture
def model_fit():
    """Sample model fit: Elastic net"""

    from sklearn.datasets import make_regression
    from sklearn.linear_model import ElasticNet

    np.random.seed(1)

    # Create a test case for elastic net regression and test the inference with
    # the created class.
    X, y = make_regression(n_samples=20, n_features=2, n_informative=2, noise=1)

    regr = ElasticNet(alpha=0.010, l1_ratio=0.5)

    regr.fit(X, y)

    return X, y, regr


def test_parameter_errors(model_fit):
    """Tests that parameter errors can be computed properly."""

    X, y, regr = model_fit

    inf = ParametricModelInference()

    inf.set_up_model_inference(X_train=X, y_train=y, estimator=regr)

    # Should return a dataframe of parameters and their CIs.
    # Each element corresponding to one output.
    df_imp = inf.get_parameter_errors()

    # Make sure that it is the right shape.
    assert (
        df_imp.shape[0] == regr.coef_.shape[0] + 1  # +1 due to the intercept term.
    ), "First dimension of the feature importance dataframe \
            differs from the number of coefficients in the model."


def test_confidence_intervals(model_fit):
    """Check if model can compute confidence intervals."""

    X, y, regr = model_fit

    inf = ParametricModelInference()

    inf.set_up_model_inference(X_train=X, y_train=y, estimator=regr)

    # Compute confidence intervals.
    df_int = inf.get_intervals(X)

    assert (
        df_int.shape[0] == X.shape[0]
    ), "Shape of the returned interval dataframe does not match the \
            number of samples."
