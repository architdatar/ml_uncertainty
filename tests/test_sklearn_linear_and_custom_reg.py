"""Benchmark LASSO and ridge models and custom regularizatio.
"""

import pytest
import autograd.numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from ml_uncertainty.model_inference import ParametricModelInference
from ml_uncertainty.model_inference.common_model_functions import (
    ordinary_residual,
    least_squares_loss,
)

# linear_model,
# linear_model_with_intercept,

np.random.seed(1)


def linear_model(X, beta):
    """ """
    return beta[0] + X @ beta[1:]


@pytest.fixture
def fit_sklearn_lasso_model():
    """ """
    X_expt, _ = make_regression(
        n_samples=500, n_features=2, n_informative=2, random_state=1
    )

    # Normalize
    X_expt = (X_expt - X_expt.mean(axis=0)) / X_expt.std(axis=0)
    true_params = np.array([1.0, 1.0, 1.0])

    y_expt = linear_model(X_expt, true_params) + np.random.normal(
        loc=0, scale=1, size=X_expt.shape[0]
    )

    regr = Lasso(alpha=1e-1, fit_intercept=True)

    regr.fit(X_expt, y_expt)

    return X_expt, true_params, y_expt, regr


def test_Lasso_model(fit_sklearn_lasso_model):
    """
    Fits LASSO model and does model analysis.
    """

    # Sklearn Lasso.
    X_expt, true_params, y_expt, regr = fit_sklearn_lasso_model

    # Model inference.
    inf = ParametricModelInference()

    inf.set_up_model_inference(X_expt, y_expt, regr)

    df_feature_imp = inf.get_parameter_errors()

    df_int = inf.get_intervals(X_expt)

    assert len(regr.coef_) == 2, "Shape of coef_ not equal to features."

    assert (
        df_feature_imp.shape[0] == inf.best_fit_params.shape[0]
    ), "Standard parameter shape is wrong."

    assert df_int.shape[0] == X_expt.shape[0], "Prediction interval shape is wrong."


def test_custom_regularization(fit_sklearn_lasso_model):
    """
    Do parametric model inference using custom parameters.
    """

    # Sklearn Lasso.
    X_expt, true_params, y_expt, regr = fit_sklearn_lasso_model

    def l1_reg(coef_, alpha=1.0):
        return alpha * np.linalg.norm(coef_, ord=2)

    # If model if fitted with intercept, add intercept to best_fit_params
    # as the first argument.

    inf = ParametricModelInference()

    inf.set_up_model_inference(
        X_train=X_expt,
        y_train=y_expt,
        model=linear_model,
        model_kwargs={},
        intercept=regr.intercept_,
        best_fit_params=np.concatenate((np.array([regr.intercept_]), regr.coef_)),
        residual=ordinary_residual,
        residual_kwargs={},
        loss=least_squares_loss,
        loss_kwargs={},
        regularization="custom",
        custom_reg=l1_reg,
        custom_reg_kwargs=dict(alpha=1e-1),
        model_dof=2,
    )

    df_feature_imp = inf.get_parameter_errors()

    df_int = inf.get_intervals(X_expt)

    # LASSO model inference.
    inf_Lasso = ParametricModelInference()
    inf_Lasso.set_up_model_inference(X_expt, y_expt, regr)
    df_feature_imp_Lasso = inf_Lasso.get_parameter_errors()
    df_int_Lasso = inf_Lasso.get_intervals(X_expt)

    # Check that this matches with the Lasso implementation
    # from the parametric model inference.
    assert (
        np.linalg.norm(df_feature_imp.values - df_feature_imp_Lasso.values)
        / df_feature_imp.shape[0]
        < 1e-3
    ), "Standard errors \
                between custom regularization and Lasso fit don't match."

    assert (
        np.linalg.norm(df_int.values - df_int_Lasso.values) / df_int.shape[0] < 1e-3
    ), "Confidence intervals \
                between custom regularization and Lasso fit don't match."
