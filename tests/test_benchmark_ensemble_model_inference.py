"""Tests that the approaches for ensemble model inference are correct.
"""


import pytest
import numpy as np
from ml_uncertainty.model_inference import EnsembleModelInference
from sklearn.ensemble import RandomForestRegressor

np.random.seed(1)


@pytest.fixture()
def set_up_random_forest():
    """Sets up RF for benchmarking."""

    X = (np.random.random(size=10000) * 10 + 5).reshape((-1, 1))

    int_for_split = int(0.8 * X.shape[0])
    X_train = X[:int_for_split, :]
    X_test = X[int_for_split:, :]

    def model(X, coef_):
        """ """
        return np.dot(X, coef_)

    # Training y
    true_params = np.array([1.0])
    noise = np.random.normal(loc=0, scale=1, size=X.shape[0])
    y = model(X, true_params) + noise

    y_train = y[:int_for_split]
    y_test = y[int_for_split:]

    # Be sure to have a large number of trees (n_estimators)
    # and max_samples < 1, if we want to analyze the train OOB features.
    regr = RandomForestRegressor(
        max_depth=10, random_state=0, max_samples=0.7, n_estimators=500
    )

    regr.fit(X_train, y_train)

    y_pred_train = regr.predict(X_train)
    y_pred_test = regr.predict(X_test)

    return (
        X,
        y,
        X_train,
        y_train,
        X_test,
        y_test,
        true_params,
        noise,
        y,
        regr,
        y_pred_train,
        y_pred_test,
    )


def compute_coverage(df_int, y):
    """Gets coverage from the dataframe."""

    lower_bound = df_int["lower_bound"].values
    upper_bound = df_int["upper_bound"].values

    coverage = ((lower_bound < y) & (upper_bound > y)).sum() / y.shape[0]

    return coverage


def test_prediction_interval_marginal(set_up_random_forest):
    """Computes prediction interval on the model and checks
    that it has the right coverage.
    Uses the 'marginal' method to obtain the prediction interval.
    """

    (
        X,
        y,
        X_train,
        y_train,
        X_test,
        y_test,
        true_params,
        noise,
        y,
        regr,
        y_pred_train,
        y_pred_test,
    ) = set_up_random_forest

    inf = EnsembleModelInference()

    inf.set_up_model_inference(X_train, y_train, regr, variance_type_to_use="marginal")

    # Gets prediction intervals NOT using SD.
    df_int_list_1 = inf.get_intervals(
        X_train,
        is_train_data=True,
        type_="prediction",
        estimate_from_SD=False,
        confidence_level=90.0,
        return_full_distribution=False,
    )

    df_int_1 = df_int_list_1[0]
    coverage_1 = compute_coverage(df_int_1, y_train)

    # Test when prediction  intervals are from SD.
    df_int_list_2 = inf.get_intervals(
        X_train,
        is_train_data=True,
        type_="prediction",
        estimate_from_SD=True,
        confidence_level=90.0,
        return_full_distribution=False,
    )
    df_int_2 = df_int_list_2[0]
    coverage_2 = compute_coverage(df_int_2, y_train)

    # Tests coverage
    assert (
        np.abs(coverage_1 - 0.90) < 0.02
    ), "Coverage for marginal with non-SD\
            deviates from confidence_level."

    assert (
        np.abs(coverage_2 - 0.90) < 0.02
    ), "Coverage for marginal with SD\
            deviates from confidence_level."


def test_prediction_interval_individual(set_up_random_forest):
    """Computes prediction interval on the model and checks
    that it has the right coverage.
    Uses the individual trees to obtain an estimate of the prediction
    intervals.
    """

    (
        X,
        y,
        X_train,
        y_train,
        X_test,
        y_test,
        true_params,
        noise,
        y,
        regr,
        y_pred_train,
        y_pred_test,
    ) = set_up_random_forest

    inf = EnsembleModelInference()

    inf.set_up_model_inference(
        X_train, y_train, regr, variance_type_to_use="individual"
    )

    # Gets prediction intervals NOT using SD.
    df_int_list_1 = inf.get_intervals(
        X_train,
        is_train_data=True,
        type_="prediction",
        estimate_from_SD=False,
        confidence_level=90.0,
        return_full_distribution=False,
    )
    df_int_1 = df_int_list_1[0]
    coverage_1 = compute_coverage(df_int_1, y_train)

    # Test when prediction  intervals are from SD.
    df_int_list_2 = inf.get_intervals(
        X_train,
        is_train_data=True,
        type_="prediction",
        estimate_from_SD=True,
        confidence_level=90.0,
        return_full_distribution=False,
    )
    df_int_2 = df_int_list_2[0]
    coverage_2 = compute_coverage(df_int_2, y_train)

    # Expected coverages

    # Tests coverage
    assert (
        np.abs(coverage_1 - 0.95) < 0.02
    ), "Coverage for individual with non-SD\
            deviates from expected value."

    assert (
        np.abs(coverage_2 - 0.95) < 0.02
    ), "Coverage for individual with SD\
            deviates from expected value."


def test_confidence_intervals(set_up_random_forest):
    """Computes confidence intervals."""

    (
        X,
        y,
        X_train,
        y_train,
        X_test,
        y_test,
        true_params,
        noise,
        y,
        regr,
        y_pred_train,
        y_pred_test,
    ) = set_up_random_forest

    inf = EnsembleModelInference()

    inf.set_up_model_inference(
        X_train, y_train, regr, variance_type_to_use="individual"
    )

    # Gets prediction intervals NOT using SD.
    df_int_list_1 = inf.get_intervals(
        X_train,
        is_train_data=True,
        type_="confidence",
        estimate_from_SD=False,
        confidence_level=90.0,
        return_full_distribution=False,
    )
    df_int_1 = df_int_list_1[0]
    coverage_1 = compute_coverage(df_int_1, y_train)

    # Test when prediction  intervals are from SD.
    df_int_list_2 = inf.get_intervals(
        X_train,
        is_train_data=True,
        type_="confidence",
        estimate_from_SD=True,
        confidence_level=90.0,
        return_full_distribution=False,
    )
    df_int_2 = df_int_list_2[0]
    coverage_2 = compute_coverage(df_int_2, y_train)

    # Expected coverages

    # Tests coverage against expected coverage
    assert (
        np.abs(coverage_1 - 0.47) < 0.02
    ), "Coverage for individual with non-SD\
            deviates from expected value."

    assert (
        np.abs(coverage_2 - 0.50) < 0.02
    ), "Coverage for individual with SD\
            deviates from expected value."
