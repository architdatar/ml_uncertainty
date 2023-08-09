#!/usr/bin/env python'

"""Tests for `error_propagation` package"""

import numpy as np
from .arrhenius_model import arrhenius_model, T_expt, k_expt, fitted_params
from ml_uncertainty.non_linear_regression import NonLinearRegression
from ml_uncertainty.model_inference import ParametricModelInference

np.random.seed(1)


def test_quadratic_model():
    """Creates a simple non-linear model and tests it.
    """

    def model(X, params):
        return params[0] + X[:, 0] * params[1] + X[:, 0]**2 * params[2]
    
    # Generate X and y data.
    X_expt = np.linspace(0, 10, 100).reshape((-1, 1))
    true_params = np.array([1., 1., 1.])
    y_expt = model(X_expt, true_params) + np.random.random(X_expt.shape[0])*0.01

    # Non-linear regression.
    nlr = NonLinearRegression(model, p0_length=3)

    nlr.fit(X_expt, y_expt)

    # Predict y from the model.
    y_pred = nlr.predict(X_expt, nlr.coef_)

    assert y_pred.shape == y_expt.shape, "Shape of y pred and y expt is different."
    assert nlr.coef_.shape == true_params.shape,  "Shape of fitted coefficients \
                    does not match the shape of true coeffients."

    # Compare the parameter values to the known parameter values. 
    np.testing.assert_almost_equal(nlr.coef_, true_params, decimal=2, 
                err_msg="For quadratic model, the fitted parameters are \
                      different from the expected parameters.")


def test_arrhenius_model():
    """Tests the non-linear regression for an Arrhenius model.
    We deliberately do not linearize this model so as to try it
    out for a non-linear case.
    """

    # Instantiate the class
    nlr = NonLinearRegression(model=arrhenius_model, p0_length=2)

    # Ensure that the shape of T_expt is (m,1) rather than (m,).
    T_expt_mat = T_expt.reshape((-1, 1))

    # Fit the model and get the parameters.
    nlr.fit(T_expt_mat, k_expt, p0=fitted_params*0.5)

    k_pred = nlr.predict(T_expt_mat, nlr.coef_)
    
    assert k_pred.shape == k_expt.shape, "Shape of k pred and k expt is different."
    assert nlr.coef_.shape == fitted_params.shape,  "Shape of fitted coefficients \
                    does not match the shape of true parameters."


def test_arrhenius_model_inference():
    """Test that the non-linear regression function can work with 
    parametric model inference to do model inference.
    """

    # Instantiate the class
    nlr = NonLinearRegression(model=arrhenius_model, p0_length=2)

    # Ensure that the shape of T_expt is (m,1) rather than (m,).
    T_expt_mat = T_expt.reshape((-1, 1))

    # Fit the model and get the parameters.
    nlr.fit(T_expt_mat, k_expt, p0=fitted_params*0.5)

    k_pred = nlr.predict(T_expt_mat, nlr.coef_)

    # Model inference 
    inf = ParametricModelInference()

    inf.set_up_model_inference(X_train=T_expt_mat,
                               y_train=k_expt,
                               estimator=nlr)
    
    df_feature_imp = inf.get_parameter_errors()

    # Compute confidence intervals for new T values.
    T_new = np.linspace(T_expt.min()*0.8, T_expt.max()*1.2, 50).reshape((-1, 1))

    df_int = inf.get_intervals(T_new)

    pass