"""Tests that the WLS function works well.

Compares with the example from statsmodels.
Source: https://www.statsmodels.org/dev/examples/notebooks/generated/wls.html

Also see the benchmarking test in
tests/benchmarking/sm_wls.py
"""

import os
import pytest
import autograd.numpy as np
from ml_uncertainty.non_linear_regression import NonLinearRegression
from ml_uncertainty.model_inference import ParametricModelInference
from sklearn.linear_model import LinearRegression


np.random.seed(1024)
file_path = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def set_up_test():

    # Load the expected arrays from the sm outputs.
    path_to_dir = f"{file_path}/benchmarking/sm_wls_outputs"
    x = np.loadtxt(f"{path_to_dir}/x.csv")
    y = np.loadtxt(f"{path_to_dir}/y.csv")
    w = np.loadtxt(f"{path_to_dir}/w.csv")
    sm_params = np.loadtxt(f"{path_to_dir}/sm_params.csv")
    sm_params_se = np.loadtxt(f"{path_to_dir}/sm_params_se.csv")
    sm_pred_df_bounds = np.loadtxt(
        f"{path_to_dir}/sm_pred_df_bounds.csv", delimiter=","
    )

    # In this example, they have used an abridged version on this
    # model for fitting, but they have used the true model for generating y.
    # The model from which y is truly generated is shown in
    # tests/benchmarking/sm_wls.py
    def model(X_arr, coef_):
        """
        $$y = \beta_0 x + \beta_1 x $$
        """
        x = X_arr[:, 0]
        beta0, beta1 = coef_
        y = beta0 + beta1 * x
        return y

    return x, y, w, model, sm_params, sm_params_se, sm_pred_df_bounds


def test_WLS(set_up_test):

    # Set up test
    x, y, w, model, sm_params, sm_params_se, sm_pred_df_bounds = set_up_test

    X_arr = x.reshape((-1, 1))

    # Quantify weights
    weights = 1.0 / w ** 2

    nlr = NonLinearRegression(model=model, p0_length=2, fit_intercept=True)

    nlr.fit(X_arr, y, sample_weight=1.0 / w ** 2)

    inf = ParametricModelInference()

    inf.set_up_model_inference(X_arr, y, nlr, y_train_weights=weights)

    # Get prediction intervals on the features.
    df_feature_imp = inf.get_parameter_errors()

    # Compute prediction intervals.
    df_int = inf.get_intervals(
        X_arr, confidence_level=95.0, distribution="t", y_weights=weights
    )

    # Set up tests to compare with statsmodels
    # Compare sm_best_fit_params with nlr.coef_.
    assert (
        np.linalg.norm(sm_params - inf.best_fit_params) < 1e-1
    ), "Best fit parameters not equal"

    # Compare pred_sm_df with df_feature_imp
    assert (
        np.linalg.norm(sm_params_se - df_feature_imp["std"].values) < 1e-3
    ), "Standard error of parameter values are different"

    # Compare pred_sm_df with df_int
    assert (
        np.linalg.norm(
            (sm_pred_df_bounds - df_int[["lower_bound", "upper_bound"]].values)
            / df_int.shape[0]
        )
        < 1e-1
    ), "Prediction intervals predicted are different"


def test_WLS_with_sklearn(set_up_test):
    """Repeat this example with sklearn OLS fitting."""

    # Set up test
    x, y, w, model, sm_params, sm_params_se, sm_pred_df_bounds = set_up_test

    X_arr = x.reshape((-1, 1))

    # Quantify weights
    weights = 1.0 / w ** 2

    # Fit with sklearn
    regr = LinearRegression(fit_intercept=True)

    regr.fit(X_arr, y, sample_weight=weights)

    inf = ParametricModelInference()

    inf.set_up_model_inference(X_arr, y, regr, y_train_weights=weights)

    df_feature_imp = inf.get_parameter_errors()

    df_int = inf.get_intervals(
        X_arr, confidence_level=95.0, distribution="t", y_weights=weights
    )

    # Set up tests to compare with statsmodels
    # Compare sm_best_fit_params with nlr.coef_.
    assert (
        np.linalg.norm(sm_params - inf.best_fit_params) < 1e-1
    ), "Best fit parameters not equal"

    # Compare pred_sm_df with df_feature_imp
    assert (
        np.linalg.norm(sm_params_se - df_feature_imp["std"].values) < 1e-3
    ), "Standard error of parameter values are different"

    # Compare pred_sm_df with df_int
    assert (
        np.linalg.norm(
            (sm_pred_df_bounds - df_int[["lower_bound", "upper_bound"]].values)
            / df_int.shape[0]
        )
        < 1e-1
    ), "Prediction intervals predicted are different"
