"""Benchmarks the various methods for ensemble models.
"""

#%%
import numpy as np
from ml_uncertainty.model_inference import EnsembleModelInference
from sklearn.ensemble import RandomForestRegressor


# Set up regression.
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


# Set up function to compute coverage as necessary.
def compute_coverage(df_int, y):
    """Gets coverage from the dataframe."""

    lower_bound = df_int["lower_bound"].values
    upper_bound = df_int["upper_bound"].values

    coverage = ((lower_bound < y) & (upper_bound > y)).sum() / y.shape[0]

    return coverage


# Marginal method.

inf_mar = EnsembleModelInference()

inf_mar.set_up_model_inference(X_train, y_train, regr, variance_type_to_use="marginal")

# Gets prediction intervals NOT using SD.
df_int_list_mar_1 = inf_mar.get_intervals(
    X_train,
    is_train_data=True,
    type_="prediction",
    estimate_from_SD=False,
    confidence_level=90.0,
    return_full_distribution=False,
)

df_int_mar_1 = df_int_list_mar_1[0]
coverage_mar_1 = compute_coverage(df_int_mar_1, y_train)

# Test when prediction  intervals are from SD.
df_int_list_mar_2 = inf_mar.get_intervals(
    X_train,
    is_train_data=True,
    type_="prediction",
    estimate_from_SD=True,
    confidence_level=90.0,
    return_full_distribution=False,
)
df_int_mar_2 = df_int_list_mar_2[0]
coverage_mar_2 = compute_coverage(df_int_mar_2, y_train)

# Individual method.
inf_ind = EnsembleModelInference()

inf_ind.set_up_model_inference(
    X_train, y_train, regr, variance_type_to_use="individual"
)

# Gets prediction intervals NOT using SD.
df_int_list_ind_1 = inf_ind.get_intervals(
    X_train,
    is_train_data=True,
    type_="prediction",
    estimate_from_SD=False,
    confidence_level=90.0,
    return_full_distribution=False,
)
df_int_ind_1 = df_int_list_ind_1[0]
coverage_ind_1 = compute_coverage(df_int_ind_1, y_train)

# Test when prediction  intervals are from SD.
df_int_list_ind_2 = inf_ind.get_intervals(
    X_train,
    is_train_data=True,
    type_="prediction",
    estimate_from_SD=True,
    confidence_level=90.0,
    return_full_distribution=False,
)
df_int_ind_2 = df_int_list_ind_2[0]
coverage_ind_2 = compute_coverage(df_int_ind_2, y_train)

# Print the values
print(f"Coverage: Marginal + without SD = {coverage_mar_1}")
print(f"Coverage: Marginal + with SD = {coverage_mar_2}")
print(f"Coverage: Individual + without SD = {coverage_ind_1}")
print(f"Coverage: Individual + with SD = {coverage_ind_2}")

# Conclusions:
# 1. The coverages for marginal method match the confidence level while
# those obtained through the individual method overestimate them.
# 2. The coverages hardly change much based on whether or not "estimate_with_SD" was
# set to True or not.
# %%
