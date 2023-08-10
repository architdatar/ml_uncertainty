#!/usr/bin/env python'

"""Tests for `ensemble_model_inference` package"""

# TODO: Create tests for other kinds of models such as gradient boosting,
# and classification models.

import pytest
import pandas as pd
from io import StringIO
import numpy as np
from ml_uncertainty.model_inference.ensemble_model_inference import (
    EnsembleModelInference,
)

np.random.seed(1)

# Fit a regression model.
@pytest.fixture
def model_fit():
    """Sample model fit"""

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression

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

    _, _, regr = model_fit

    inf = EnsembleModelInference()

    df_imp_list = inf.get_feature_importance_intervals(
        regr, return_full_distribution=False
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

    pd.testing.assert_frame_equal(df_imp_list[0], df_expected)


def test_feature_importance_parametric(model_fit):
    """Tests that feature importance can be obtained.
    Here, we will require parametric intervals.
    """

    _, _, regr = model_fit

    inf = EnsembleModelInference()

    df_imp_list = inf.get_feature_importance_intervals(
        regr, return_full_distribution=False, distribution="parametric"
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


def test_intervals(model_fit):
    """Tests that intervals work properly."""

    X, y, regr = model_fit

    inf = EnsembleModelInference()

    pred_int_list = inf.get_intervals(
        X,
        regr,
        is_train_data=True,
        confidence_level=90.0,
    )

    # Getting the variables for dimensions.
    n_samples = X.shape[0]
    n_outputs = regr.n_outputs_

    # Test the the length of this list is the n_outputs.
    assert (
        len(pred_int_list) == n_outputs
    ), "Length of the prediction interval list returned \
                does not equal the number of outputs."

    # Test that the shape of the output df is correct
    assert (
        pred_int_list[0].shape[0] == n_samples
    ), "Shape of the dataframe does not match the number of samples."

    # Test that the output dataframe is right by verifying the first
    # 5 rows
    df_expected_string = """,mean,std,median,lower_bound,upper_bound\n
    0,20.514626422982776,26.73227351523108,11.107836549809928,-7.2673862345478275,75.70297124437309\n
    1,-15.368613442862479,15.682739238241636,-10.992424478595916,-44.76175165954471,3.4042684609686766\n
    2,10.86812975279185,19.488219008040645,12.309238938165487,-31.45347712964715,41.353964786612146\n
    3,5.443480417697841,17.221153086618703,1.764876555848281,-14.2276752925064,47.7225474631879\n
    4,-1.9141498751819868,18.015088444213436,-4.446160653192964,-20.727319540143643,32.12844331653147\n"""

    df_expected = pd.read_csv(StringIO(df_expected_string), index_col=0)

    pd.testing.assert_frame_equal(pred_int_list[0].head(), df_expected)


def test_intervals_parametric(model_fit):
    """Tests that intervals work properly.
    Here, we require that distribution be parametric.
    """

    X, y, regr = model_fit

    inf = EnsembleModelInference()

    pred_int_list = inf.get_intervals(
        X, regr, is_train_data=True, confidence_level=90.0, distribution="parametric"
    )

    # Getting the variables for dimensions.
    n_samples = X.shape[0]
    n_outputs = regr.n_outputs_

    # Test the the length of this list is the n_outputs.
    assert (
        len(pred_int_list) == n_outputs
    ), "Length of the prediction interval list returned \
                does not equal the number of outputs."

    # Test that the shape of the output df is correct
    assert (
        pred_int_list[0].shape[0] == n_samples
    ), "Shape of the dataframe does not match the number of samples."
