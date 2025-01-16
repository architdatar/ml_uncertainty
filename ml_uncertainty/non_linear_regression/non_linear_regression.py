"""
Wrapper around scipy.optimize.least_squares to provide easy fitting utility for
non-linear functions.
"""

import numpy as np
from typing import Callable
from copy import deepcopy
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import DataDimensionalityWarning
from ..error_propagation.error_propagation import ErrorPropagation
from ..model_inference.common_model_functions import ordinary_residual

# Reusing function to check that arrays have the right type and dtype.
check_type_and_dtype = ErrorPropagation.check_type_and_dtype


class NonLinearRegression(RegressorMixin, BaseEstimator):
    r"""
    Scikit-learn-like fitting utility for non-linear models.

    Can be easily used to get model inference such as parameter significance tests
    and prediction intervals.

    Parameters:
    -----------
    model: callable with inputs (X, coefs_, **model_kwargs) and output y
        Function defining how the predictor variables relate to the response.
        X: numpy array of shape (n_examples, n_dimensions)
        coefs_: numpy array of shape (n_parameters, )
        **model_kwargs: additional keyword arguments for the model.
        y: numpy array of shape (n_examples,)
    p0_length: {None, int}, optional
        If None: User must provide an initial guess for the model parameters when
            calling 'fit' method.
        If int: Number of model parameters (n_parameters) including the intercept.
    model_kwargs_dict: dict, optional, default={}
        Dictionary of model keyword arguments.
    residual: {callable, str}, optional, default: "ordinary"
        Defines how to compute residuals.
            If callable (function): It must have its arguments as y_pred, y and
            return residuals. Each of y_pred, y, and residuals must be a 1-D array of
            shape (n_examples,).
            If str: specifies what form of residual function to use.
            As of now, we only support "ordinary" which defines the residual as
            residual = y_hat - y.
    residual_kwargs_dict: dict, optional, default={}
        Dictionary of residual keyword arguments.
    least_sq_kwargs_dict: dict, optional, default={}
        Dictionary of optional keyword arguments to scipy.optimize.least_squares
          function. See scipy.optimize.least_squares documentation.
    copy_X: bool, optional, default=True
        Specified if X should be deep-copied during 'fit', This prevents X input to the
        function from undergoing any change during the fit.
    fit_intercept: bool, optional, default=False
        Whether the first term fit is the model intercept.


    Attributes:
    -----------
    coef_: array of shape (n_parameters, )
        Best parameters of the fit including the intercept where applicable.

    Notes:
    ------
    1. The fit_intercept=True behavior is slightly different from sklearn.
        Here, the self.coef_ variable includes the intercept if it is present.
    2. Only supports single target variable.
    3. Uses only a single processor to fit. Parallelization not implemented.
    4. Handles dense matrices only; no support for sparse matrices.
    5. No regularization available.
    6. Only bounded constraints allowed in scipy.optimize.least_squares are allowed.
        See documentation.

    Examples:
    --------
    examples/non_linear_regression_quadratic.py
    examples/non_linear_regression_arrhenius.py
    """

    def __init__(
        self,
        model: Callable,
        p0_length=None,
        model_kwargs_dict={},
        residual="ordinary",
        residual_kwargs_dict={},
        least_sq_kwargs_dict={},
        copy_X=True,
        fit_intercept=False,
    ):
        self.model = model
        self.p0_length = p0_length
        self.model_kwargs_dict = model_kwargs_dict
        self.least_sq_kwargs_dict = least_sq_kwargs_dict
        self.copy_X = copy_X
        self.fit_intercept = fit_intercept

        # Initialize a function modifying the model as required.
        self.func = self.model

        self._validate_p0_length(self.p0_length)

        # Set residual.
        if residual == "ordinary":
            self.residual = ordinary_residual
        else:
            self.residual = residual

        self.residual_kwargs_dict = residual_kwargs_dict

    def _validate_p0_length(self, p0_length):
        """p0 length must be None of type int"""

        if p0_length is not None:
            if type(p0_length) == int:
                pass
            else:
                raise ValueError(
                    f"p0_length type detected as {type(p0_length)}. \
                                 It must be either None or type int."
                )

    def fit(self, X, y, p0=None, sample_weight=None):
        """
        Fits the non-linear model.

        Wrapper around scipy.optimize.least_squares to implement optimization.

        Parameters:
        -----------
        X: array of shape (n_examples, n_dimensions)
            Training data
        y: array of shape (n_examples, )
            Target values.
        p0: {None, array of shape (n_parameters,)}, optional, default: None
            Array specifying initial estimate of the best fit parameters
            including the intercept.
            If None, an array of shape (p0_length,) of random values between
            0 and 1 from a uniform distribution will be generated and use as the
            initial guess.
        sample_weight: {None, array of shape (n_samples,)}, optional, default: None
            Individual weights for each sample.

        Returns:
        --------
        self: object
            Fitted estimator

        Notes:
        -----
        1. If p0 is None and p0_length is also None, an error will be raised.
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
            self._validate_y(sample_weight, X)

        # Validate the model by computing the y_hat for the given X and ensuring
        # that is the same shape as y. Also test the predict method.
        y_pass = self.predict(X, p0, **self.model_kwargs_dict)

        self._validate_y(y_pass, X)

        # Build a residual function.
        def _model_residuals(params, X, y, sample_weight, model_kwargs_dict):

            y_hat = self.predict(X, params, **model_kwargs_dict)
            residuals = self.residual(y_hat, y, **self.residual_kwargs_dict)

            if sample_weight is None:
                return residuals
            else:
                sample_weight_sqrt = np.sqrt(sample_weight)
                return residuals * sample_weight_sqrt

        res_ls = least_squares(
            _model_residuals,
            x0=p0,
            args=(X, y, sample_weight, self.model_kwargs_dict),
            **self.least_sq_kwargs_dict,
        )

        self.fitted_object = res_ls

        self.coef_ = res_ls.x

        if self.fit_intercept:
            self.intercept_ = res_ls.x[0]
        else:
            self.intercept_ = None

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

    def predict(self, X, params=None, **model_kwargs):
        """
        Predict using the given model.

        Parameters:
        -----------
        X: array of shape (n_samples, n_dimensions)
            Samples
        params: {None, array of shape (n_parameters,)}, optional, default=None
            Model parameters.
            If None:
                Will use fitted model parameters.
                If estimator is not fitted, an error will be raised.
        **model_kwargs:
            Optional keyword arguments to be passed to the model.

        Returns:
        --------
        y: array of shape (n_samples,)
            Returns predicted values.
        """

        if params is None:
            params = self.coef_

        if model_kwargs == {}:
            model_kwargs = self.model_kwargs_dict

        return self.func(X, params, **model_kwargs)
