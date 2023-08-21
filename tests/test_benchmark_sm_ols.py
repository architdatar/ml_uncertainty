"""Benchmark a variety of models against results from statsmodels.

However, we don't use statsmodels here as we do not need the users
to otherwise use statsmodels library.

We have run a script 'benchmarking/sm_linear_models.py' to generate
arrays containing the benchmark values for the set of
data to be with with a linear model.
"""

import pytest
import autograd.numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from ml_uncertainty.non_linear_regression import NonLinearRegression
from ml_uncertainty.model_inference import ParametricModelInference
import os

np.random.seed(1)
file_path = os.path.dirname(os.path.abspath(__file__))


def linear_model(X, beta):
    """ """
    return X @ beta


@pytest.fixture
def get_sm_data_for_1D_test():
    """Gets sm data for the 1D test."""

    X_expt = np.linspace(0, 10, 1000).reshape((-1, 1))
    true_params = np.array([1.0])

    # Shift the mean by 500 to test that there is no centering effect.
    X_expt += 500

    y_expt = linear_model(X_expt, true_params) + np.random.normal(
        loc=0, scale=1, size=X_expt.shape[0]
    )

    # Load the expected arrays from the sm outputs.
    path_to_dir = f"{file_path}/benchmarking/sm_outputs"
    sm_best_fit_params = np.loadtxt(f"{path_to_dir}/1D_best_fit_params.csv")
    sm_param_errors = np.loadtxt(f"{path_to_dir}/1D_param_errors.csv")
    sm_pred_bounds = np.loadtxt(f"{path_to_dir}/1D_pred_bounds.csv", delimiter=",")

    return (
        X_expt,
        true_params,
        y_expt,
        sm_best_fit_params,
        sm_param_errors,
        sm_pred_bounds,
    )


def test_1D(get_sm_data_for_1D_test):
    """When X is 1D, test that our code matches with statsmodels results."""

    # Get test inputs and outputs.
    (
        X_expt,
        true_params,
        y_expt,
        sm_best_fit_params,
        sm_param_errors,
        sm_pred_bounds,
    ) = get_sm_data_for_1D_test

    # Fit with NLR.
    nlr = NonLinearRegression(
        model=linear_model, p0_length=true_params.shape[0], fit_intercept=False
    )

    nlr.fit(X_expt, y_expt)

    inf = ParametricModelInference()

    inf.set_up_model_inference(X_expt, y_expt, nlr)

    # Get prediction intervals on the features.
    df_feature_imp = inf.get_parameter_errors()

    # Getting prediction intervals for X_expt values.
    df_int = inf.get_intervals(X_expt, confidence_level=95.0, distribution="t")

    # Compare sm_best_fit_params with nlr.coef_
    assert (
        np.linalg.norm(sm_best_fit_params - inf.best_fit_params) < 1e-1
    ), "Best fit parameters not equal"

    # Compare pred_sm_df with df_feature_imp
    assert (
        np.linalg.norm(sm_param_errors - df_feature_imp["std"].values) < 1e-1
    ), "Standard error of parameter values are different"

    # Compare pred_sm_df with df_int
    assert (
        np.linalg.norm(
            (sm_pred_bounds - df_int[["lower_bound", "upper_bound"]].values)
            / df_int.shape[0]
        )
        < 1e-1
    ), "Prediction intervals predicted are different"


@pytest.fixture
def get_sm_data_for_2D_test():
    """Gets inputs and expected outputs for 2D test"""

    X_expt, _ = make_regression(
        n_samples=500, n_features=2, n_informative=2, random_state=1
    )
    true_params = np.array([1.0, 1.0])

    # Means of X shifted by 500 to make sure that the analysis stays
    # valid irrespective of the mean of X.
    X_expt = X_expt + 500

    y_expt = linear_model(X_expt, true_params) + np.random.normal(
        loc=0, scale=1, size=X_expt.shape[0]
    )

    # Load the expected arrays from the sm outputs.
    path_to_dir = f"{file_path}/benchmarking/sm_outputs"
    sm_best_fit_params = np.loadtxt(f"{path_to_dir}/2D_best_fit_params.csv")
    sm_param_errors = np.loadtxt(f"{path_to_dir}/2D_param_errors.csv")
    sm_pred_bounds = np.loadtxt(f"{path_to_dir}/2D_pred_bounds.csv", delimiter=",")

    return (
        X_expt,
        true_params,
        y_expt,
        sm_best_fit_params,
        sm_param_errors,
        sm_pred_bounds,
    )


def test_2D(get_sm_data_for_2D_test):
    """Tests a 2D linear model."""

    (
        X_expt,
        true_params,
        y_expt,
        sm_best_fit_params,
        sm_param_errors,
        sm_pred_bounds,
    ) = get_sm_data_for_2D_test

    # Fit with NLR.
    nlr = NonLinearRegression(model=linear_model, p0_length=true_params.shape[0])

    nlr.fit(X_expt, y_expt)

    inf = ParametricModelInference()

    inf.set_up_model_inference(X_expt, y_expt, nlr)

    # Get prediction intervals on the features.
    df_feature_imp = inf.get_parameter_errors()

    # Getting prediction intervals for X_expt values.
    df_int = inf.get_intervals(X_expt, confidence_level=95.0, distribution="t")

    # Compare sm_best_fit_params with nlr.coef_.
    # There are differences due to the random draw in y.
    assert (
        np.linalg.norm(sm_best_fit_params - nlr.coef_) < 1e-1
    ), "Best fit parameters not equal"

    # Compare pred_sm_df with df_feature_imp
    assert (
        np.linalg.norm(sm_param_errors - df_feature_imp["std"].values) < 1e-1
    ), "Standard error of parameter values are different"

    # Compare pred_sm_df with df_int
    assert (
        np.linalg.norm(
            (sm_pred_bounds - df_int[["lower_bound", "upper_bound"]].values)
            / df_int.shape[0]
        )
        < 1e-1
    ), "Prediction intervals predicted are different"


def test_2D_for_sklearn(get_sm_data_for_2D_test):
    """Repeats the 1D example but benchmarks it with sklearn."""

    # Get test inputs and outputs.
    (
        X_expt,
        true_params,
        y_expt,
        sm_best_fit_params,
        sm_param_errors,
        sm_pred_bounds,
    ) = get_sm_data_for_2D_test

    # Fit model with sklearn.
    regr = LinearRegression(fit_intercept=False)

    regr.fit(X_expt, y_expt)

    inf = ParametricModelInference()

    inf.set_up_model_inference(X_expt, y_expt, regr)

    df_feature_imp = inf.get_parameter_errors()

    df_int = inf.get_intervals(X_expt, confidence_level=95.0, distribution="t")

    # Compare with benchmark results.
    assert (
        np.linalg.norm(sm_best_fit_params - regr.coef_) < 1e-1
    ), "Best fit parameters not equal"

    # Compare pred_sm_df with df_feature_imp
    assert (
        np.linalg.norm(sm_param_errors - df_feature_imp["std"].values) < 1e-1
    ), "Standard error of parameter values are different"

    # Compare pred_sm_df with df_int
    assert (
        np.linalg.norm(
            (sm_pred_bounds - df_int[["lower_bound", "upper_bound"]].values)
            / df_int.shape[0]
        )
        < 1e-1
    ), "Prediction intervals predicted are different"
