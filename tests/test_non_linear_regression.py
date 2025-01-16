#!/usr/bin/env python'

"""Tests for `error_propagation` package"""

import autograd.numpy as np
import numpy.testing
from .arrhenius_model import arrhenius_model, T_expt, k_expt, fitted_params
from ml_uncertainty.non_linear_regression import NonLinearRegression
from ml_uncertainty.model_inference import ParametricModelInference
from scipy.stats import spearmanr as srcc
import pytest

np.random.seed(1)


@pytest.fixture()
def fit_quadratic_model():
    """Quadratic model fitting."""

    def model(X, params):
        return params[0] + X[:, 0] * params[1] + X[:, 0] ** 2 * params[2]

    # Generate X and y data.
    X_expt = np.linspace(0, 10, 100).reshape((-1, 1))
    true_params = np.array([1.0, 1.0, 1.0])
    y_expt = model(X_expt, true_params) + np.random.normal(
        loc=0, scale=1, size=X_expt.shape[0]
    )

    # Non-linear regression.
    nlr = NonLinearRegression(model, p0_length=3, fit_intercept=True)

    nlr.fit(X_expt, y_expt)

    # Predict y from the model.
    y_pred = nlr.predict(X_expt)

    return X_expt, true_params, y_expt, nlr, y_pred


def test_quadratic_model(fit_quadratic_model):
    """Creates a simple non-linear model and tests it."""

    _, true_params, y_expt, nlr, y_pred = fit_quadratic_model

    assert y_pred.shape == y_expt.shape, "Shape of y pred and y expt is different."
    assert (
        nlr.coef_.shape == true_params.shape
    ), "Shape of fitted coefficients \
                    does not match the shape of true coeffients."

    # Check tha that the predicted and true parameters are in the same order
    # of magnitude.
    assert (
        np.linalg.norm(nlr.coef_ - true_params) < 3
    ), "Fitted parameters are different from true parameters."


def test_quadratic_model_inference(fit_quadratic_model):
    """Tests that the model inference of the quadratic model is
    proceeding accurately.
    """

    X_expt, true_params, y_expt, nlr, y_pred = fit_quadratic_model

    inf = ParametricModelInference()

    inf.set_up_model_inference(X_train=X_expt, y_train=y_expt, estimator=nlr)

    # Get prediction intervals on the features.
    df_feature_imp = inf.get_parameter_errors()

    # Getting prediction intervals for X_expt values.
    df_int = inf.get_intervals(X_expt)

    # Check shape of the predicted outputs.
    assert (
        df_feature_imp.shape[0] == true_params.shape[0]
    ), "Feature importance dataframe shape does not match the shape \
            of the true parameters."

    assert (
        df_int.shape[0] == X_expt.shape[0]
    ), "Prediction interval dataframe \
        shape does not match the shape of the input array."

    # Check the predictions made in error_propagation are the same as those made
    # by NLR.
    numpy.testing.assert_almost_equal(
        y_pred,
        df_int["mean"].values,
        decimal=3,
        err_msg="Predicted values from NLR prediction and those from \
                    error_propagation get_intervals method differ from each other.",
    )


def test_arrhenius_model():
    """Tests the non-linear regression for an Arrhenius model.
    We deliberately do not linearize this model so as to try it
    out for a non-linear case.
    """

    # Instantiate the class
    nlr = NonLinearRegression(model=arrhenius_model, p0_length=2)

    # Ensure that the shape of T_expt is (m,1) rather than (m,).
    T_expt_mat = T_expt.reshape((-1, 1))

    # Fit the model and get the parameters.
    nlr.fit(T_expt_mat, k_expt, p0=fitted_params * 0.5)

    k_pred = nlr.predict(T_expt_mat)

    assert k_pred.shape == k_expt.shape, "Shape of k pred and k expt is different."
    assert (
        nlr.coef_.shape == fitted_params.shape
    ), "Shape of fitted coefficients \
                    does not match the shape of true parameters."

    # Check that the ranks of the predicted values and the and the experimental
    # values are the same; i.e., Spearman rank correlation coefficients = 1.
    assert (
        srcc(k_expt, k_pred).correlation == 1
    ), "Predicted value \
        from nlr does not follow the same order as the experimental values. \
        This is unexpected behavior."


def test_arrhenius_model_inference():
    """Test that the non-linear regression function can work with
    parametric model inference to do model inference.
    """

    # Instantiate the class
    nlr = NonLinearRegression(model=arrhenius_model, p0_length=2)

    # Ensure that the shape of T_expt is (m,1) rather than (m,).
    T_expt_mat = T_expt.reshape((-1, 1))

    # Fit the model and get the parameters.
    nlr.fit(T_expt_mat, k_expt, p0=fitted_params * 0.5)

    # Model inference
    inf = ParametricModelInference()

    inf.set_up_model_inference(X_train=T_expt_mat, y_train=k_expt, estimator=nlr)

    df_feature_imp = inf.get_parameter_errors()

    # Compute confidence intervals for new T values.
    T_new = np.linspace(T_expt.min(), T_expt.max(), 50).reshape((-1, 1))

    k_new = nlr.predict(T_new)

    df_int = inf.get_intervals(T_new)

    # Check shape of the predicted outputs.
    assert (
        df_feature_imp.shape[0] == fitted_params.shape[0]
    ), "Feature importance dataframe shape does not match the shape \
            of the fitted parameters."

    assert (
        df_int.shape[0] == T_new.shape[0]
    ), "Prediction interval dataframe \
        shape does not match the shape of the input."

    # Check the predictions made in error_propagation are the same as those made
    # by NLR.
    numpy.testing.assert_almost_equal(
        k_new,
        df_int["mean"].values,
        decimal=3,
        err_msg="Predicted values from NLR prediction and those from \
                    error_propagation get_intervals method differ from each other.",
    )
