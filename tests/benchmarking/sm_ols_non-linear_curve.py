"""Benchmark with OLS linear model but non-linear curve.
Source website: https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html

NOTE: This benchmarking example has not been included as a test.
This is because it is the exact same concept as the sm_linear_models.py case.
"""

#%%
# Statsmodels example: OLS non-linear curve but linear in parameters.

import autograd.numpy as np
import statsmodels.api as sm
from ml_uncertainty.non_linear_regression import NonLinearRegression
from ml_uncertainty.model_inference import ParametricModelInference

np.random.seed(9876789)

# Statsmodels example
nsample = 50
sig = 0.5
x = np.linspace(0, 20, nsample)
X = np.column_stack((x, np.sin(x), (x - 5) ** 2, np.ones(nsample)))
beta = [0.5, 0.5, -0.02, 5.0]

y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

res = sm.OLS(y, X).fit()
# print(res.summary())

# Get prediction intervals.
pred_ols = res.get_prediction()

sm_params = res.params
sm_param_se = np.sqrt(np.diag(res.cov_params()))  # parameter error
sm_pred_df = pred_ols.summary_frame()  # alpha=0.05 (default)

# # Plot
# iv_l = pred_ols.summary_frame()["obs_ci_lower"]
# iv_u = pred_ols.summary_frame()["obs_ci_upper"]

# fig, ax = plt.subplots(figsize=(8, 6))

# ax.plot(x, y, "o", label="data")
# ax.plot(x, y_true, "b-", label="True")
# ax.plot(x, res.fittedvalues, "r--.", label="OLS")
# ax.plot(x, iv_u, "r--")
# ax.plot(x, iv_l, "r--")
# ax.legend(loc="best")

# Recreate this with NLR.


def model(X_arr, coefs_):
    r"""
    $$ y = \beta_0 x + \beta_1 \sin(x) + \beta_2(x-5)^2 + \beta_3
    """
    x = X_arr[:, 0]
    beta0, beta1, beta2, beta3 = coefs_
    y = beta0 * x + beta1 * np.sin(x) + beta2 * (x - 5) ** 2 + beta3
    return y


X_arr = x.reshape((-1, 1))  # Must be 2D array.
beta_arr = np.array(beta)
y_model = model(X_arr, np.array(beta_arr))

# Fit using NLR
nlr = NonLinearRegression(model=model, p0_length=beta_arr.shape[0], fit_intercept=True)

nlr.fit(X_arr, y)

inf = ParametricModelInference()

inf.set_up_model_inference(X_arr, y, nlr)

# Get prediction intervals on the features.
df_feature_imp = inf.get_parameter_errors(confidence_level=95.0)

# Get prediction intervals.
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
