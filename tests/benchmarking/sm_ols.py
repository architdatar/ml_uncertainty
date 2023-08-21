"""We benchmark SM examples from website.
https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html

NOTE: This example is not added to tests. as this example and sm_linear_models.py
tackle almost the same problem.
"""

#%%
# Statsmodels example

import numpy as np
import statsmodels.api as sm
from ml_uncertainty.non_linear_regression import NonLinearRegression
from ml_uncertainty.model_inference import ParametricModelInference

np.random.seed(9876789)

# Statsmodels example
nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x ** 2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)

X = sm.add_constant(X)
y = np.dot(X, beta) + e

model = sm.OLS(y, X)
results = model.fit()
# print(results.summary())
sm_params = results.params
sm_param_se = np.sqrt(np.diag(results.cov_params()))  # parameter error
sm_pred_df = results.get_prediction().summary_frame(alpha=0.05)

#%%
# NLR example


# Create the model.
def model(X_arr, coefs_):
    """ """
    return coefs_[0] + coefs_[1] * X_arr[:, 0] + coefs_[2] * X_arr[:, 0] ** 2


X_arr = x.reshape((-1, 1))
y_model = model(X_arr, beta)

# Fit using NLR
nlr = NonLinearRegression(model=model, p0_length=beta.shape[0], fit_intercept=True)

nlr.fit(X_arr, y)

inf = ParametricModelInference()

inf.set_up_model_inference(X_arr, y, nlr)

# Get prediction intervals on the features.
df_feature_imp = inf.get_parameter_errors()

# Get prediction intervals.
# Getting prediction intervals for X_expt values.
df_int = inf.get_intervals(X_arr, confidence_level=95.0, distribution="t")

# Compare sm_best_fit_params with nlr.coef_
assert np.linalg.norm(sm_params - nlr.coef_) < 1e-3, "Best fit parameters not equal"

# Compare pred_sm_df with df_feature_imp
assert (
    np.linalg.norm(sm_param_se - df_feature_imp["std"].values) < 1e-3
), "Stanard error of parameter values are different"

# Compare pred_sm_df with df_int
assert (
    np.linalg.norm(
        (
            sm_pred_df[["obs_ci_lower", "obs_ci_upper"]]
            - df_int[["lower_bound", "upper_bound"]].values
        )
        / df_int.shape[0]
    )
    < 1e-3
), "Prediction intervals predicted are different"


# %%
