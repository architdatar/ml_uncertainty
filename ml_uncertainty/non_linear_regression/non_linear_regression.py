"""
Wrapper around scipy.optimize.least_squares to provide easy fit for 
non-linear functions. 

Future: 
    1. Add capabilities for regularization (using scipy.optimize.minimize)
    2. Non-explicit constraints (using scipy.optimize.minimize)
    3. Accurate degree of freedom calculations for non-linear models 
    (develop code using simulations, see documentation for ensemble model 
    inference)
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import least_squares
from typing import Callable
from copy import deepcopy
from sklearn.exceptions import DataDimensionalityWarning
from ..error_propagation.error_propagation import ErrorPropagation
import inspect
from ..model_inference.common_model_functions import ordinary_residual

# Reusing function to check that arrays have the right type and dtype.
check_type_and_dtype = ErrorPropagation.check_type_and_dtype


class NonLinearRegression(RegressorMixin, BaseEstimator):
    """
    Scikit-learn-like fitting utility for non-linear models.

    Can be used with sklearn pipelines, etc. to provide an easy interface for
    non-linear model fitting and inference.

    Parameters:
    -----------
    residual: {callable, "ordinary"}
        Defines how to compute residuals. If a function is specified,
            it must have its arguments as y_pred, y (each 1D arrays) and return a 1-D
            vector of the same dimension.
        If "ordinary", residual = y_hat - y


    Limitations:
    ------
    1. Only supports single target variable.
    2. Uses only a single processor to fit. Parallelization not implemented.
    3. Handles dense matrices only; no support for sparse matrices.
    4. Each instantiation can handle only one set of kwargs for the model.
    5. No regularization available. This is intuitive because non-linear models
        are generally not used for feature selection, etc.
    6. Only bounded constraints mentioned in scipy.optimize.least_squares are allowed.

    Caution:
    1. fit_intercept = True method computes intercept by taking the mean of the
        reponse variable.
        This is NOT accurate for non-linear regression. For such cases,
        set up the model to incorporate intercept_ as a model parameters and
        not as a special kwarg. ..Will be removed in future.
    """

    def __init__(
        self,
        model: Callable,
        p0_length=None,
        model_kwargs_dict={},
        residual="ordinary",
        residual_kwargs_dict={},
        fit_intercept=False,
        least_sq_kwargs_dict={},
        copy_X=True,
    ):
        self.model = model
        self.p0_length = p0_length
        self.model_kwargs_dict = model_kwargs_dict
        self.fit_intercept = fit_intercept
        self.least_sq_kwargs_dict = least_sq_kwargs_dict
        self.copy_X = copy_X

        # Initialize a function modifying the model as required.
        self.func = self.model

        # If fit_intercept is set to True, the model must have an
        # intercept_ arg. In this case, the intercept will be computed
        # from the y values and y_sample_weights.
        # Else, intercept_ will be treated as any other kwarg that the user
        # must correctly specify.
        if self.fit_intercept:
            assert (
                "intercept_" in self.func.__code__.co_varnames
            ), "If fit_intercept was set to true, \
                    model function must have 'intercept_' as argument."

        # Set residual.
        if residual == "ordinary":
            self.residual = ordinary_residual
        else:
            self.residual = residual

        self.residual_kwargs_dict = residual_kwargs_dict

    def fit(self, X, y, p0=None, sample_weight=None):
        """
        Fit the non-linear model.
        """

        # Validate data
        # Validate X: 2-D array with float dtype.
        if self.copy_X:
            X = deepcopy(X)

        self._validate_X(X)

        # Validate y: 1-D array
        self._validate_y(y, X)

        # Validate p0:
        #   1. If p0 is None, initialize randomly using p0_length: if that is
        # also None, raise error.
        if p0 is None:
            p0 = self._initialize_p0()

        self._validate_p0(p0)

        # Validate sample_weight. Check for the same things as y.
        if sample_weight is not None:
            self._validate_y(sample_weight)

        # Here, we first compute the intercept before validating the
        # function. This might seem counterintuitive, but it is right
        # since we do not "fit" the intercept like we do other parameters.
        self._set_intercept(y, sample_weight)

        # Create a new model kwarg dict for fitting in this function only.
        # Modify the model_kwargs dict provided with the
        # new value of the intercept_ if that is an argument of the
        # function.
        model_kwargs_dict = deepcopy(self.model_kwargs_dict)

        # The second condition is redundant since if the self.fit_intercept is
        # True, the function must have an 'intercept_' parameter as mandated in
        # the __init__ function. It is however added here as an extra precaution.
        if self.fit_intercept and "intercept_" in self.func.__code__.co_varnames:
            model_kwargs_dict.update({"intercept_": self.intercept_})

        # Validate the model by computing the y_hat for the given X and ensuring
        # that is the same shape as y. Also test the predict method.
        y_pass = self.predict(X, p0, **model_kwargs_dict)

        self._validate_y(y_pass, X)

        # Build a residual function based off of user preference and fit_intercept
        #   If fit_intercept is True, make intercept the first argument of the
        # parameters and construct the residual function accordingly.
        def _model_residuals(params, X, y, sample_weight, model_kwargs_dict):

            y_hat = self.predict(X, params, **model_kwargs_dict)
            residuals = self.residual(y_hat, y, **self.residual_kwargs_dict)

            if sample_weight is None:
                return residuals
            else:
                sample_weight_sqrt = np.sqrt(sample_weight)
                return residuals @ sample_weight_sqrt

        res_ls = least_squares(
            _model_residuals,
            x0=p0,
            args=(X, y, sample_weight, model_kwargs_dict),
            **self.least_sq_kwargs_dict
        )

        self.fitted_object = res_ls
        self.coef_ = res_ls.x

    def _validate_X(self, X):
        """Checks that X is an ndarray of 2 dimensions and shape (m, n ).

        Checks that m and n > 0 .
        """

        X = check_type_and_dtype(X)

        if X.ndim == 2 and X.shape[0] > 0 and X.shape[1] > 0:
            pass
        else:
            raise DataDimensionalityWarning(
                "X should be a np.ndarray object with 2 dimensions and shape\
                (m, n) where m>0 and n>0.\
                Please correct the format."
            )

    def _validate_y(self, y, X):
        """Checks that the supplied y_hat is right."""

        # Check type and dtype
        check_type_and_dtype(y)

        # Ensure y_hat has the correct dimension.
        assert y.ndim == 1, "y is not a 1-D vector. Please supply a 1-D vector."
        assert (
            y.shape[0] == X.shape[0]
        ), "Size of y is different from that of X. Please make sure that they match."

    def _initialize_p0(self):
        """"""

        if self.p0_length is not None and type(self.p0_length) == int:
            p0 = np.random.random(self.p0_length)
        else:
            raise ValueError(
                "p0 not supplied and length of p0 could not inferred.\
                                Please supply p0 or supply p0_length value \
                                as an integer."
            )
        return p0

    def _validate_p0(self, p0):
        """Checks that the supplied p0 value is correct."""

        # Check type and dtype
        check_type_and_dtype(p0)

        assert p0.ndim == 1, "p0 is not a 1-D vector. Please supply a 1-D vector."

    def _set_intercept(self, y, sample_weight):
        """Sets intercept for the given problem."""

        # If we are required to fit intercept, we simply set it
        # as weight-averaged values of y.
        # NOTE: This value would have changed had we considered centering on X.
        # But since this is non-linear modeling, we do not allow for that internally.
        if self.fit_intercept:
            # Compute the intercept.
            if sample_weight is not None:
                sample_weight_sqrt = np.sqrt(sample_weight)
                self.intercept_ = np.average(y, weights=sample_weight_sqrt)
            else:
                self.intercept_ = np.average(y)

        else:
            # Get the default value of the intercept.
            # If specified in model kwargs, use that.
            # Else, use the default value of the function kwarg.
            if "intercept_" in list(self.model_kwargs_dict.keys()):
                self.intercept_ = self.model_kwargs_dict.get("intercept_")
            else:
                # Get default value of this from the function's signature.
                intercept_value_list = [
                    v.default
                    for k, v in inspect.signature(self.func).parameters.items()
                    if k == "intercept_"
                ]

                if intercept_value_list == []:  # Empty list signifies that
                    # the user has not set a value for intercept_
                    self.intercept_ = 0
                else:
                    self.intercept_ = intercept_value_list[0]

    def predict(self, X, params=None, **model_kwargs):
        """ """

        if params is None:
            params = self.coef_

        if model_kwargs == {}:
            model_kwargs = self.model_kwargs_dict

        return self.func(X, params, **model_kwargs)
