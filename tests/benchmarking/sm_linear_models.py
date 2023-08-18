"""Benchmarks the non-linear regression against statsmodels results.

This benchmark example creates files that is used in the `test_benchmark_sm_ols.py`
test.
"""

#%%
# Import libraries.
import statsmodels.api as sm
import autograd.numpy as np
from ml_uncertainty.non_linear_regression import NonLinearRegression
from ml_uncertainty.model_inference import ParametricModelInference

np.random.seed(1)


def linear_model(X, beta):
    """ """
    return X @ beta


# Case 1: 1D X. Generate X and y variables
X_expt = np.linspace(0, 10, 1000).reshape((-1, 1))
true_params = np.array([1.0])

# Case 2: 2D X
# X_expt, _ = make_regression(n_samples=500,
#                             n_features=2,
#                             n_informative=2,
#                             random_state=1
#                                  )
# true_params = np.array([1., 1.])


X_expt = X_expt + 500


y_expt = linear_model(X_expt, true_params) + np.random.normal(
    loc=0, scale=1, size=X_expt.shape[0]
)

model = sm.OLS(y_expt, X_expt)

results = model.fit()
sm_best_fit_params = results.params
param_std_error_sm = np.sqrt(np.diag(results.cov_params()))  # parameter error
pred_sm_df = results.get_prediction(X_expt).summary_frame(alpha=0.05)[
    ["obs_ci_lower", "obs_ci_upper"]
]


# Fit the same data using NLR
nlr = NonLinearRegression(model=linear_model, p0_length=true_params.shape[0])

nlr.fit(X_expt, y_expt)

# Predicted with fitted parameters.
y_pred = nlr.predict(X_expt)

inf = ParametricModelInference()

inf.set_up_model_inference(X_expt, y_expt, nlr)

# Get prediction intervals on the features.
df_feature_imp = inf.get_parameter_errors()

# Getting prediction intervals for X_expt values.
df_int = inf.get_intervals(X_expt, confidence_level=95.0, distribution="t")

# Compare sm_best_fit_params with nlr.coef_
assert (
    np.linalg.norm(sm_best_fit_params - nlr.coef_) < 1e-3
), "Best fit parameters not equal"

# Compare pred_sm_df with df_feature_imp
assert (
    np.linalg.norm(param_std_error_sm - df_feature_imp["std"].values) < 1e-3
), "Stanard error of parameter values are different"

# Compare pred_sm_df with df_int
assert (
    np.linalg.norm(
        (pred_sm_df.values - df_int[["lower_bound", "upper_bound"]].values)
        / df_int.shape[0]
    )
    < 1e-3
), "Prediction intervals predicted are different"


# Write out these arrays to a file.
# 1D case
# np.savetxt("sm_outputs/1D_best_fit_params.csv", sm_best_fit_params, fmt="%.10f",
#  delimiter=",")
# np.savetxt("sm_outputs/1D_param_errors.csv", param_std_error_sm, fmt="%.10f",
# delimiter=",")
# np.savetxt("sm_outputs/1D_pred_bounds.csv", pred_sm_df, fmt="%.10f", delimiter=",")

# 2D case
# np.savetxt("sm_outputs/2D_best_fit_params.csv", sm_best_fit_params, fmt="%.10f",
# delimiter=",")
# np.savetxt("sm_outputs/2D_param_errors.csv", param_std_error_sm, fmt="%.10f",
# delimiter=",")
# np.savetxt("sm_outputs/2D_pred_bounds.csv", pred_sm_df, fmt="%.10f", delimiter=",")
