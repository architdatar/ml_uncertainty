"""Benchmark with the R example 
Source: https://stats.stackexchange.com/questions/56895/do-the-predictions-of-a-random-forest-model-have-a-prediction-interval

Basically, the intervals we obtain through our method are essentially
confidence intervals. To get to prediction intervals, we need to make
assumptions about the data and get the $\hat{\sigma}^2$.

According to ESL Pg 262, for bootstrap methods, it is estimated as
$$ \hat{\sigma}^2 = \sum_{i=1}^{N} (y_i - \hat{\mu}(x_i))^2 / N $$

But, we further benchmark our code with the R example given in 
Stackoverflow.

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


x1 = np.zeros((1000,))
x1[500:] = 1

x2 = np.zeros((1000,))
x2[250:500] = 1
x2[750:1000] = 1

X = np.stack((x1, x2), axis=1)

y = 10 + 5 * x1 + 10 * x2 - 3 * x1 * x2 + np.random.normal(size=1000)

# Newdata
newdat = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])


# Fit a random forest and get CI

regr = RandomForestRegressor(n_estimators=1001)

regr.fit(X, y)

# Run ensemble model inference
inf = EnsembleModelInference()
#%%
df_int_list_test, oob_pred, n_oob_pred, means_array, std_array = inf.get_intervals(
    newdat,
    regr,
    is_train_data=False,
    type="confidence",
    distribution="default",
    confidence_level=95.0,
    return_full_distribution=True,
)
df_int_test = df_int_list_test[0]


# Does this contain 95% of the samples?

# %%
