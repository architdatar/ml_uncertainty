#!/usr/bin/env python'

"""Tests for `ensemble_model_inference` package"""

import pytest
import pandas as pd
from io import StringIO
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from ml_uncertainty.model_inference.ensemble_model_inference import (
    EnsembleModelInference,
)

np.random.seed(1)


# Fit a regression model.
@pytest.fixture
def model_fit():
    """Sample model fit"""

    X, y = make_regression(
        n_features=4,
        n_informative=2,
        random_state=0,
        shuffle=False,
        n_samples=100,
        n_targets=1,
    )
    regr = RandomForestRegressor(
        max_depth=2, random_state=0, max_samples=0.7, n_estimators=500
    )
    regr.fit(X, y)

    return [X, y, regr]


def test_feature_importance(model_fit):
    """Tests that feature importance can be obtained."""

    X, y, regr = model_fit

    inf = EnsembleModelInference()

    inf.set_up_model_inference(X, y, regr)

    df_imp_list = inf.get_feature_importance_intervals(
        confidence_level=90.0, return_full_distribution=False
    )

    # Getting the variables for dimensions.
    n_features = regr.feature_importances_.shape[0]
    n_outputs = regr.n_outputs_

    # Test the the length of this list is the n_outputs.
    assert (
        len(df_imp_list) == n_outputs
    ), "Length of the feature importance list returned \
                does not equal the number of outputs."

    # Test that the shape of the output df is correct
    assert (
        df_imp_list[0].shape[0] == n_features
    ), "Shape of the dataframe for the first output does not match \
            expected shape. The first dimension of the shape should equal \
            number of features, but that is not the case."

    # Test that the output dataframe is right.
    df_expected_string = ",mean,std,median,lower_bound,upper_bound\n\
        0,0.18222003617429855,0.12723530638421554,0.16748837281064877,0.0,0.3810085909109055\n\
        1,0.8155922827952207,0.1275286651060924,0.8291836478458151,0.6189914090890943,1.0\n\
        2,0.0013591394769462979,0.015180180181922681,0.0,0.0,0.0\n\
        3,0.0008285415535345963,0.008280122731785905,0.0,0.0,0.0\n"
    df_expected = pd.read_csv(StringIO(df_expected_string), index_col=0)

    pd.testing.assert_frame_equal(df_imp_list[0], df_expected, atol=1e-3)
