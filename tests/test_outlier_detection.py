"""
Here, we test the utilities to detect outliers and get associated information.
The goal is to test each of the key functions one by one.

NOTE: This approach, especially, the computation of standardized residuals
and, therefore, test of normality, and leverage, is ONLY applicable to
linear models. 
To extend to non-linear models, we will have to first linearize those around 
$\beta^*$. And then, instead of using X directly, like we do for linear models,
we will have to use $d/\beta f(X, \beta)$.
The idea is that around $\beta^*$, we can make the argument that 
the model may behave like a linear model.
If substitite a linear model here, we should get f'(X,\beta) = X.
For non-linear model, the full formula is f(X, \beta) = f(X, \beta^*) + d/d\beta f(X, \beta) | \beta = \beta^*

This would be valid if the remaining terms are much smaller than this. 
We should also set up a test to ensure that this is true. 
"""

from ml_uncertainty.model_inference.data_testing import residualAnalysis
from tests.test_benchmark_sm_ols import linear_model
from ml_uncertainty.non_linear_regression import NonLinearRegression
from ml_uncertainty.model_inference import ParametricModelInference
import pytest
import numpy as np
import os
import statsmodels.api as sm
from sklearn.linear_model import Ridge


np.random.seed(1)
file_path = os.path.dirname(os.path.abspath(__file__))


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

    return (
        X_expt,
        true_params,
        y_expt,
    )


def test_outlier_detection_OLS(get_sm_data_for_1D_test):
    (X_expt, true_params, y_expt) = get_sm_data_for_1D_test

    # Fit with NLR.
    nlr = NonLinearRegression(
        model=linear_model, p0_length=true_params.shape[0], fit_intercept=False
    )

    nlr.fit(X_expt, y_expt)
    y_pred = nlr.predict(X_expt)

    inf = ParametricModelInference()

    inf.set_up_model_inference(X_expt, y_expt, nlr)

    ra = residualAnalysis(
        y_expt,
        y_pred,
        "test_var",
        "test_var_pred",
        X_expt,
        inf_model=inf,
        l2_regularization=None,
        l1_regularization=None,
        coef_=inf.best_fit_params,
    )

    diag_vals = ra.get_hat_matrix().diagonal()

    # Build the SM model.
    model = sm.OLS(y_expt, X_expt)

    results = model.fit()

    influence = results.get_influence()
    hat_diag = influence.hat_matrix_diag

    np.testing.assert_array_almost_equal(hat_diag, diag_vals)


def test_outlier_detection_ridge(get_sm_data_for_1D_test):
    (X_expt, true_params, y_expt) = get_sm_data_for_1D_test

    ridge_model = Ridge(alpha=1.0, fit_intercept=False)
    ridge_model.fit(X_expt, y_expt)
    ridge_model.coef_ = true_params

    y_pred = ridge_model.predict(X_expt)

    inf = ParametricModelInference()

    inf.set_up_model_inference(X_expt, y_expt, ridge_model)

    ra = residualAnalysis(
        y_expt,
        y_pred,
        "test_var",
        "test_var_pred",
        X_expt,
        inf_model=inf,
        l2_regularization=1.0,
        l1_regularization=None,
        coef_=inf.best_fit_params,
    )

    diag_vals = ra.get_hat_matrix().diagonal()

    # Build the SM model.
    model = sm.OLS(y_expt, X_expt)

    # For statsmodels, the hat matrix cannot be computed for the regularized
    # fitting, even for an ordinary linear model. 
    # This is one of the limitations that ML uncertainty can solve. 
    # This can be seen in the docs for statsmodels.
    # https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit_regularized.html#statsmodels.regression.linear_model.OLS.fit_regularized
    # https://www.statsmodels.org/stable/generated/statsmodels.base.elastic_net.RegularizedResults.html#statsmodels.base.elastic_net.RegularizedResults
    
    # Ideally, we would be using this command.
    # results = model.fit_regularized(method="elastic_net", alpha=1.0, L1_wt=0)
    # influence = results.get_influence()
    # hat_diag = influence.hat_matrix_diag

    # np.testing.assert_array_almost_equal(hat_diag, diag_vals)
    