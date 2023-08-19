"""Benchmark with statsmodels weighted least squares regression.

Source: https://www.statsmodels.org/dev/examples/notebooks/generated/wls.html
"""
#%%
import numpy as np
import statsmodels.api as sm
from ml_uncertainty.non_linear_regression import NonLinearRegression
from ml_uncertainty.model_inference import ParametricModelInference

np.random.seed(1024)

# Statsmodels examples.
# Artificial data: Heteroscedasticity 2 groups

nsample = 50
x = np.linspace(0, 20, nsample)
X = np.column_stack((x, (x - 5) ** 2))
X = sm.add_constant(X)
beta = [5.0, 0.5, -0.01]
sig = 0.5
w = np.ones(nsample)
break_int = nsample * 6 // 10
w[break_int:] = 3
y_true = np.dot(X, beta)
e = np.random.normal(size=nsample)
y = y_true + sig * w * e
X = X[:, [0, 1]]  # The squared term has been left out of the model.

mod_wls = sm.WLS(y, X, weights=1.0 / (w ** 2))
res_wls = mod_wls.fit()
# print(res_wls.summary())
sm_params = res_wls.params
sm_param_se = np.sqrt(np.diag(res_wls.cov_params()))  # parameter error
sm_pred_df = res_wls.get_prediction().summary_frame(alpha=0.05)

# Repeat this analysis with NLR model.


def model_true(X_arr, coef_):
    """
    $$y = \beta_0 x + \beta_1 x + beta_2 (x-5)^2$$
    """
    x = X_arr[:, 0]
    beta0, beta1, beta2 = coef_
    y = beta0 + beta1 * x + beta2 * (x - 5) ** 2
    return y


# In this example, they have used an abridged version on this
# model for fitting, but they have used the true model for generating y.
def model(X_arr, coef_):
    """
    $$y = \beta_0 x + \beta_1 x $$
    """
    x = X_arr[:, 0]
    beta0, beta1 = coef_
    y = beta0 + beta1 * x
    return y


# Fit using NLR
X_arr = x.reshape((-1, 1))
beta_arr = np.array(beta[:-1])
# Quantify weights
weights = 1.0 / w ** 2

nlr = NonLinearRegression(model=model, p0_length=beta_arr.shape[0], fit_intercept=True)

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
assert np.linalg.norm(sm_params - nlr.coef_) < 1e-1, "Best fit parameters not equal"

# Compare pred_sm_df with df_feature_imp
assert (
    np.linalg.norm(sm_param_se - df_feature_imp["std"].values) < 1e-3
), "Standard error of parameter values are different"

# Compare pred_sm_df with df_int
assert (
    np.linalg.norm(
        (
            sm_pred_df[["obs_ci_lower", "obs_ci_upper"]].values
            - df_int[["lower_bound", "upper_bound"]].values
        )
        / df_int.shape[0]
    )
    < 1e-1
), "Prediction intervals predicted are different"


#%%
# Write out the inputs and outputs to a file.
# np.savetxt("sm_wls_outputs/x.csv", x, fmt="%.10f", delimiter=",")
# np.savetxt("sm_wls_outputs/y.csv", y, fmt="%.10f", delimiter=",")
# np.savetxt("sm_wls_outputs/w.csv", w, fmt="%.10f", delimiter=",")
# np.savetxt("sm_wls_outputs/sm_params.csv", sm_params, fmt="%.10f", delimiter=",")
# np.savetxt("sm_wls_outputs/sm_params_se.csv", sm_param_se, fmt="%.10f", delimiter=",")
# np.savetxt(
#     "sm_wls_outputs/sm_pred_df_bounds.csv",
#     sm_pred_df[["obs_ci_lower", "obs_ci_upper"]],
#     fmt="%.10f",
#     delimiter=",",
# )

# %%
