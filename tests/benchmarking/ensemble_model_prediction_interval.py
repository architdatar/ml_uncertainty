"""Benchmarks the claims for ensemble models.

Checks if this method indeed yields prediction and confidence intervals.

Procedure: 
1. Create X for train and tests for a given model. 
2. For given X, get y from an underlying model with some variance.
3. For this data, fit a random forest model and get y values and y prediction intervals.
5. Look at distribution of y predicted and compare it with y expected.
6. Repeat for different models.
"""

#%%
import numpy as np
from sklearn.datasets import make_regression
from ml_uncertainty.model_inference import EnsembleModelInference
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.random.seed(1)

# X, _ = make_regression(
#     n_samples=10000,
#     n_features=1,
#     n_informative=1,
#     random_state=10,
#     shuffle=True
# )

X = (np.random.random(size=10000) * 10 + 5).reshape((-1, 1))

int_for_split = int(0.8 * X.shape[0])
X_train = X[:int_for_split, :]
X_test = X[int_for_split:, :]


def model(X, coef_):
    """ """
    return np.dot(X, coef_)


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

# Run ensemble model inference
inf = EnsembleModelInference()

# Get feature importances and their uncertainties for the first variable.
df_imp = inf.get_feature_importance_intervals(
    regr,
    return_full_distribution=False,
    confidence_level=90.0,
)[0]

# Compute prediction intervals for the first target variable.
df_int_list_train = inf.get_intervals(
    X_train,
    regr,
    is_train_data=True,
    confidence_level=90.0,
    return_full_distribution=False,
)
df_int_train = df_int_list_train[0]

df_int_list_test, oob_pred, n_oob_pred, means_array, std_array = inf.get_intervals(
    X_test,
    regr,
    is_train_data=False,
    confidence_level=90.0,
    return_full_distribution=True,
)
df_int_test = df_int_list_test[0]

#%%
# Let's plot everything.
plt.figure()
sns.barplot(
    x=df_imp["mean"],
    y=df_imp.index.astype("category"),
    xerr=(
        df_imp["mean"] - df_imp["lower_bound"],
        df_imp["upper_bound"] - df_imp["mean"],
    ),
    color="gray",
    capsize=10,
)

# Intervals
print(f"R2 train= {r2_score(y_train, y_pred_train)}")
print(f"R2 test = {r2_score(y_test, y_pred_test)}")

# With predicted error values.
plt.figure()
plt.errorbar(
    y_train,
    y_pred_train,
    yerr=(
        df_int_train["mean"].values - df_int_train["lower_bound"].values,
        df_int_train["upper_bound"].values - df_int_train["mean"].values,
    ),
    marker="o",
    markersize=8,
    ls="none",
    zorder=0,
    color="red",
    label="Train",
)

# Test: With predicted error values.
plt.errorbar(
    y_test,
    y_pred_test,
    yerr=(
        df_int_test["mean"].values - df_int_test["lower_bound"].values,
        df_int_test["upper_bound"].values - df_int_test["mean"].values,
    ),
    marker="o",
    markersize=8,
    ls="none",
    zorder=0,
    color="blue",
    label="Test",
)

#%%
# Let's pick the first 3 points from the test set and
# see what their distribution looks like.

for point_ind in range(3):
    X_point = X_test[[point_ind], :]
    y_point = y_pred_test[point_ind]

    plt.figure()
    plt.title(f"Test point: {point_ind}")
    plt.hist(noise, bins=10, color="green", zorder=0)
    plt.hist(oob_pred[point_ind, :, 0] - y_point, color="red", zorder=0.1)

    print(
        f"Point {point_ind}: Prediction SE: {(oob_pred[point_ind, :, 0] - y_point).std()}"
    )


# It seems that the distribution we get here is the confidence interval
# after all. So, we need to add the model $\sigma^2$ to it to get
# the prediction interval.


# Benchmarking: If we use the simple method shown in
# https://github.com/haozhestat/rfinterval/blob/master/R/rfinterval.R
# and in DOI: 10.1080/00031305.2019.1585288
# we benchmark it by computing the coverage of the interval.
# P(lower_bound < y < upper_bound) \approx (1-\alpha)
# For the example considered here, it works.

# D_true = y_pred_test - y_test
# ((y_pred_test + np.quantile(D_true, 0.025) < y_test) & ((y_pred_test + np.quantile(D_true, 0.975) > y_test))).sum()

# This also provides an impetus to show that the method of
# getting y_hat - y for each tree, then getting its variance through
# MSE and then adding those variances could work.
# That way, we would be able to get the confidence intervals as well.
# To test that it works, benchmark it this way.


# %%
