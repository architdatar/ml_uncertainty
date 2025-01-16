"""Tests ensemble model for more complex cases:
specifically 3 cases:
1. Multioutput linear
2. Single output non-linear
3. Errors from non-linear distribution
"""

import pytest
import numpy as np
from ml_uncertainty.model_inference import EnsembleModelInference
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

np.random.seed(1)


@pytest.fixture
def set_up_multioutput_linear():
    """ """

    X, y = make_regression(
        n_features=4,
        n_informative=2,
        random_state=0,
        shuffle=False,
        n_samples=10000,
        noise=1,
        n_targets=3,
    )
    regr = RandomForestRegressor(
        max_depth=2, random_state=0, max_samples=0.7, n_estimators=500
    )
    regr.fit(X, y)

    return [X, y, regr]


def compute_coverage(df_int, y):
    """Gets coverage from the dataframe."""

    lower_bound = df_int["lower_bound"].values
    upper_bound = df_int["upper_bound"].values

    coverage = ((lower_bound < y) & (upper_bound > y)).sum() / y.shape[0]

    return coverage


def test_multioutput_linear(set_up_multioutput_linear):
    """Tests that the model works well on
    multioutput cases.
    """

    X, y, regr = set_up_multioutput_linear

    inf = EnsembleModelInference()

    inf.set_up_model_inference(X, y, regr)

    # Feature importances
    df_feature_imp_list = inf.get_feature_importance_intervals(
        confidence_level=90.0, side="two-sided"
    )

    # Intervals
    df_int_list = inf.get_intervals(
        X,
        is_train_data=True,
        type_="prediction",
        confidence_level=90.0,
    )
    df_int_0 = df_int_list[0]
    coverage_0 = compute_coverage(df_int_0, y[:, 0])

    df_int_1 = df_int_list[1]
    coverage_1 = compute_coverage(df_int_1, y[:, 1])

    df_int_2 = df_int_list[2]
    coverage_2 = compute_coverage(df_int_2, y[:, 2])

    # Test that the computed coverages are close to the confidence interval.
    assert (
        len(df_feature_imp_list) == y.shape[1]
    ), "Feature importance list length \
        differs from the number of targets."

    assert (
        np.abs(coverage_0 - 0.90) < 0.02
    ), "Coverage for first variable\
        deviates from expected value."

    assert (
        np.abs(coverage_1 - 0.90) < 0.02
    ), "Coverage for second variable\
        deviates from expected value."

    assert (
        np.abs(coverage_2 - 0.90) < 0.02
    ), "Coverage for third variable\
        deviates from expected value."


@pytest.fixture()
def set_up_non_linear_regression():
    """Sets up non-linear regression."""

    X = np.linspace(-10, 10, 10000).reshape((-1, 1))

    def sigmoid(X):
        x = X[:, 0]
        return 1 / (1 + np.exp(-x))

    y_true = sigmoid(X)
    noise = np.random.normal(scale=0.1, size=X.shape[0])

    y = y_true + noise

    # Split into train and test.
    train_idx = np.random.choice(
        np.arange(X.shape[0]), size=int(0.8 * X.shape[0]), replace=False
    )

    X_train = X[train_idx, :]
    y_train = y[train_idx]
    mask = np.ones(X.shape[0], dtype=bool)
    mask[train_idx] = False
    X_test = X[mask, :]
    y_test = y[mask]

    regr = RandomForestRegressor(
        max_depth=10, random_state=0, max_samples=0.7, n_estimators=500
    )

    regr.fit(X, y)

    return X_train, y_train, X_test, y_test, regr


def test_non_linear_function(set_up_non_linear_regression):
    """Tests non-linear function."""

    X_train, y_train, X_test, y_test, regr = set_up_non_linear_regression

    inf = EnsembleModelInference()

    inf.set_up_model_inference(X_train, y_train, regr)

    # Get intervals
    df_int_train = inf.get_intervals(
        X_train, is_train_data=True, type_="prediction", confidence_level=90.0
    )[0]

    df_int_test = inf.get_intervals(
        X_test, is_train_data=False, type_="prediction", confidence_level=90.0
    )[0]

    coverage_train = compute_coverage(df_int_train, y_train)
    coverage_test = compute_coverage(df_int_test, y_test)

    # Tests coverage
    assert (
        np.abs(coverage_train - 0.90) < 0.02
    ), "Coverage for train\
            deviates from confidence_level."

    # The coverage for test set is different than that of the train set.
    assert (
        np.abs(coverage_test - 0.90) < 0.02
    ), "Coverage for test\
            deviates from confidence_level."


@pytest.fixture()
def set_up_non_normal_dist():
    """Sets up regression with non-normal distribution of errors."""

    X = np.linspace(-10, 10, 10000).reshape((-1, 1))

    def linear_model(X, coef_):
        return coef_[0] + np.dot(X, coef_[1:])

    true_params = np.array([1.0, 1.0])
    y_true = linear_model(X, true_params)

    # Poisson distribution.
    noise = np.random.poisson(lam=1, size=X.shape[0]) - 1

    y = y_true + noise

    # Split into train and test.
    train_idx = np.random.choice(
        np.arange(X.shape[0]), size=int(0.8 * X.shape[0]), replace=False
    )

    X_train = X[train_idx, :]
    y_train = y[train_idx]
    mask = np.ones(X.shape[0], dtype=bool)
    mask[train_idx] = False
    X_test = X[mask, :]
    y_test = y[mask]

    regr = RandomForestRegressor(
        max_depth=10, random_state=0, max_samples=0.7, n_estimators=500
    )

    regr.fit(X, y)

    return X_train, y_train, X_test, y_test, regr, noise, true_params


def test_non_normal_error_distribution_marginal(set_up_non_normal_dist):
    """Tests inference for regression with non-normal error distributions."""

    X_train, y_train, X_test, y_test, regr, noise, true_params = set_up_non_normal_dist

    inf = EnsembleModelInference()

    def poisson(scale=1, size=None):
        r"""Wrapper around numpy's Poisson function to specify scale.
        Note: For poisson, the mean, and variance are both=$\lambda$.
        So, we adjust for the mean, and "scale", i.e., standard deviation.
        """

        lam = scale ** 2
        return np.random.poisson(lam=lam, size=size) - lam

    inf.set_up_model_inference(X_train, y_train, regr, distribution=poisson)

    df_int_train = inf.get_intervals(
        X_train, is_train_data=True, confidence_level=90.0
    )[0]
    coverage_train = compute_coverage(df_int_train, y_train)

    df_int_test = inf.get_intervals(X_test, is_train_data=False, confidence_level=90.0)[
        0
    ]
    coverage_test = compute_coverage(df_int_test, y_test)

    # Tests coverage
    assert (
        np.abs(coverage_train - 0.90) < 0.05
    ), "Coverage for train\
            deviates from confidence_level."

    # The coverage for test set is different than that of the train set.
    assert (
        np.abs(coverage_test - 0.90) < 0.05
    ), "Coverage for test\
            deviates from confidence_level."


def test_non_normal_error_distribution_ind(set_up_non_normal_dist):
    """Tests inference for regression with non-normal error distributions.
    Tests this with 'individual' error predictions.
    """

    X_train, y_train, X_test, y_test, regr, noise, true_params = set_up_non_normal_dist

    inf = EnsembleModelInference()

    def poisson(scale=1, size=None):
        r"""Wrapper around numpy's Poisson function to specify scale.
        Note: For poisson, the mean, and variance are both=$\lambda$.
        So, we adjust for the mean, and "scale", i.e., standard deviation.
        """

        lam = scale ** 2
        return np.random.poisson(lam=lam, size=size) - lam

    inf.set_up_model_inference(
        X_train, y_train, regr, distribution=poisson, variance_type_to_use="individual"
    )

    df_int_train = inf.get_intervals(
        X_train, is_train_data=True, confidence_level=90.0
    )[0]
    coverage_train = compute_coverage(df_int_train, y_train)

    df_int_test = inf.get_intervals(X_test, is_train_data=False, confidence_level=90.0)[
        0
    ]
    coverage_test = compute_coverage(df_int_test, y_test)

    # Tests coverage: In this case, it is different
    # from the 90% since this prediction interval has more coverage
    # than confidence level.
    assert (
        np.abs(coverage_train - 0.99) < 0.02
    ), "Coverage for train\
            deviates from confidence_level."

    # The coverage for test set is different than that of the train set.
    assert (
        np.abs(coverage_test - 0.99) < 0.02
    ), "Coverage for test\
            deviates from confidence_level."
