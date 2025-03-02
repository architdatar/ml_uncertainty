"""Parametric model inference class"""

import autograd.numpy as np
from autograd import jacobian
from autograd import elementwise_grad as egrad
import pandas as pd
import sklearn
from sklearn.exceptions import DataDimensionalityWarning
from ..error_propagation.error_propagation import ErrorPropagation
from .common_model_functions import (
    linear_model,
    linear_model_with_intercept,
    ordinary_residual,
    least_squares_loss,
)
from ..error_propagation.statistical_utils import compute_intervals
from ..non_linear_regression import NonLinearRegression

# Reusing function to check that arrays have the right type and dtype.
check_type_and_dtype = ErrorPropagation.check_type_and_dtype


class ParametricModelInference:
    """
    Utility to get model inference for parametric models.

    Can be used to get parameter standard errors and prediction/
    confidence intervals for samples.

    Examples:
    ---------
    examples/non_linear_regression_quadratic.py
    examples/non_linear_regression_arrhenius.py
    examples/parametric_model.py

    References:
    -----------
    1. The functions get_J, get_H are adopted from
        https://github.com/sriki18/adnls/blob/master/adnls.py#L92.
    2. The formulae are taken from Niclas Borlin's lecture slides.
        https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts.pdf
    """

    def __init__(self):
        """
        Initialize the class
        """

        self.__estimators_implemented = [
            sklearn.linear_model._base.LinearRegression,
            sklearn.linear_model.Lasso,
            sklearn.linear_model.Ridge,
            sklearn.linear_model._coordinate_descent.ElasticNet,
            NonLinearRegression,
        ]

    def set_up_model_inference(
        self,
        X_train,
        y_train,
        estimator=None,
        y_train_weights=None,
        model=None,
        model_kwargs=None,
        intercept=None,
        best_fit_params=None,
        residual=None,
        residual_kwargs=None,
        loss=None,
        loss_kwargs=None,
        regularization=None,
        l1_penalty=None,
        l2_penalty=None,
        custom_reg=None,
        custom_reg_kwargs=None,
        model_dof=None,
    ):
        """Sets up model inference for the fitted model.

        Validates models, corresponding inputs and raises errors if necessary.
        Computes degrees of freedom, resdual mean squared error, and
        the variance-covariance matrix which are important to analyzing the model.

        Parameters
        ----------
        X_train: array of shape (n_training_examples, n_dimensions)
            Training data
        y_train: array of shape (n_training examples, )
            Training targets
        estimator: {None,
                    sklearn.linear_model.LinearRegression,
                    sklearn.linear_model.ElasticNet,
                    ml_uncertainty.non_linear_regression.NonLinearRegression,
                    }, optional, default=None

            If the model is fit through one of the estimators in
                self.__estimators_implemented, the fitted object can be passed
                directly. Several required attributes such as model-model_dof.
                will be inferred automatically.
                Else, these will have to be explicity provided.

            NOTE: Scikit-learn linear models treats the intercept are a different
            parameter from the rest of the model coefficients. It can be fitted
            via the fit_incetercept_=True argument.
            But, for accurate error propagation, the intercept_ must be treated
            as a parameter (as done in statsmodels).

            Thus, for sklearn models, if fit_intercept_=True, we append the
            "intercept_" parameter to the "coefs_" parameter to construct the
            "best_fit_params" and perform error analysis accordingly.
        y_train_weights: array of shape (n_training examples, )
            Weights of training targets. Often used when the variances of the
             distributions from which they are drawn substantially different.
        model: {None, callable}, optional, default=None
            If callable: callable with inputs (X, coefs_, **model_kwargs) and output y
                Function defining how the predictor variables relate to the response.
                X: numpy array of shape (n_examples, n_dimensions)
                coefs_: numpy array of shape (n_parameters, )
                **model_kwargs: additional keyword arguments for the model.
                y: numpy array of shape (n_examples,)
        model_kwargs: {None, dict}, optional, default=None
            If dict: Dictionary of model keyword arguments.
        intercept: {None, float}, optional, default=None
            If the model has an intercept.
        best_fit_params: {None, array}, optional, default: None
            Best fit parameters of the model fit.
            If array:
                Array of shape (n_parameters,).
        residual: {None, callable, str}, optional, default:None
            Defines how to compute residuals.
                If callable (function): It must have its arguments as y_pred, y and
                    return residuals. Each of y_pred, y, and residuals must be a 1-D
                    array of shape (n_examples,).
                If str: specifies what form of residual function to use.
                    As of now, we only support "ordinary" which defines the residual
                    as residual = y_hat - y.
        residual_kwargs: dict, optional, default:None
            Dictionary of residual keyword arguments.
        loss: {None, callable}, optional, default=None
            Defines how loss term in loss function is computed.
            If callable:
                Function with input arguments  residuals (array of shape (n_examples,))
                and other keyword arguments and return float.
                    NOTE: If there is regularization in the loss term, please specify
                    it through the regularization parameters below. Else, the degrees
                    of freedom computation will be incorrect.
        loss_kwargs: {None, dict}, optional, default:None
            Keyword arguments for the loss function.
        regularization: {None, str}, optional, default: None
            Specifies regularization used in the model.
            Used to compute degrees of freedom and variance-covariance matrix.
            If str: specify one of
                "none": No regularization
                "l1": L1 regularization (LASSO)
                "l2": L2 regularization (Ridge)
                "l1+l2": L1 + L2 regularization (elastic net)
                "custom": custom regularization (see custom regulazrization args below)
                    Warning: In this case, please specify model degrees of freedom
                    explicity as that calculation is not implemented here.
        l1_penalty: {None, float}, optional, default:None
            Specifies l1_penalty to be used. See Notes below.
            Only considered if regularization is among {"l1", "l1+l2"}.
        l2_penalty: {None, float}, optional, default:None
            Specifies l2_penalty to be used. See Notes below.
            Only considered if regularization is among {"l2", "l1+l2"}.
        custom_reg: {None, callable}, optional, default: None
            Only considered if regularization=="custom". Else, ignored.
            If callable, function with inputs as (coefs_, **custom_reg_kwargs)
            and outputs a float.
                coefs_: array of shape (n_paramters,) (same as best_fit_params)
        custom_reg_kwargs: {None, dict}, optional, default:None
            Only considered if regularization=="custom". Else, ignored.
            Optional arguments for the function custom_reg.
        model_dof: {None, float}, optional, default: None
            Model degrees of freedom.
            If None: Computed interally using class functions.
            If float: Used directly.

        Notes:
        -------
        1. Loss function is constructed as
                loss = loss_term + L1_penalty * L1_norm(coefs_)
                        + l2_penalty * L2_norm(coefs_)
            If regularization == "custom":
                loss = loss_term + custom_reg(coefs_, **custom_reg_kwargs)
        2. For L1 loss: The derivatives are an approximation.
            Autograd does not differenciate L1 norm, so, for the purposes of
            computing the derivative, we approximate it as L2 norm.
            The errors between the two are small. See benchmarking/L1_v_L2_loss.py
        """

        self.X_train = X_train
        self.y_train = y_train
        self.y_train_weights = y_train_weights
        self.estimator = estimator
        self.model = model
        self.model_kwargs = model_kwargs
        self.intercept = intercept
        self.best_fit_params = best_fit_params
        self.residual = residual
        self.residual_kwargs = residual_kwargs
        self.loss = loss
        self.loss_kwargs = loss_kwargs
        self.regularization = regularization
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.custom_reg = custom_reg
        self.custom_reg_kwargs = custom_reg_kwargs
        self.model_dof = model_dof

        # Validate that the estimators, etc. are correctly specified.
        self._validate_input_arguments()

        # If estimator is not None, populate the required model args.
        if self.estimator is not None:
            self._populate_args_for_known_estimators()

        # Compute model and error degrees of freedom.
        self._set_model_dof()
        self._set_error_dof()

        self._set_sigma()  # Square root of RSS

        # Compute parameter standard deviations.
        self._set_parameter_errors()

    def model_function(self, X, coefs_, **model_kwargs):
        return self.model(X, coefs_, **model_kwargs)

    def residual_function(self, X, coefs_, y, model_kwargs, **residual_kwargs):
        """ """
        y_pred = self.model_function(X=X, coefs_=coefs_, **model_kwargs)
        return self.residual(y_pred, y, **residual_kwargs)

    def loss_function(self, X, coefs_, y, model_kwargs, residual_kwargs, **loss_kwargs):
        """ """
        residuals = self.residual_function(
            X, coefs_, y, model_kwargs, **residual_kwargs
        )

        loss_term = self.loss(residuals, **loss_kwargs)

        if self.regularization == "none":
            loss = loss_term
        elif self.regularization == "l1":
            """NOTE: This is an approximation of the L1 norm so that it can
            be differenciated. Autograd does not differenciate L1 norm,
            so this is a stopgap arrangement.
            """
            loss = loss_term + self.l1_penalty * np.linalg.norm(coefs_, ord=2)
        elif self.regularization == "l2":
            loss = loss_term + self.l2_penalty * np.linalg.norm(coefs_, ord=2) ** 2
        elif self.regularization == "l1+l2":
            loss = (
                loss_term
                + self.l1_penalty * np.linalg.norm(coefs_, ord=2)
                + self.l2_penalty * np.linalg.norm(coefs_, ord=2) ** 2
            )
        elif self.regularization == "custom":
            loss = loss_term + self.custom_reg(coefs_, **self.custom_reg_kwargs)
        return loss

    def _validate_input_arguments(self):
        """Validates that the correct input arguments are provided."""

        # Validate X.
        self._validate_X(self.X_train)

        # Validate y.
        self._validate_y(self.y_train)

        # Validate estimator.
        if self.estimator is not None:
            # Check that the estimator is fitted.
            sklearn.utils.validation.check_is_fitted(self.estimator)

            # If estimator is of known type, populate the arguments for it.
            if not type(self.estimator) in self.__estimators_implemented:
                raise ValueError(
                    f"Estimator of type {type(self.estimator).__module__} \
                                 cannot be used to infer the desired properties for \
                                 error analysis. Please supply the required functions \
                                 externally."
                )
        else:
            # Check that all the required functions are provided.
            if any(
                [
                    self.model is None,
                    self.residual is None,
                    self.loss is None,
                    self.best_fit_params is None,
                ]
            ):
                raise ValueError(
                    "At least one of model_function,\
                        residual_function, loss_function, \
                        best fit params, regularization \
                        is not defined.\
                        Please provide all the above as it is necessary for \
                        model inference."
                )

        # Validate y_train_weights
        self._validate_y_train_weights(self.y_train_weights, self.y_train)

        # If custom regularization, make sure that the model DOFs are provided.
        if self.regularization == "custom":
            self._validate_custom_reg(
                self.custom_reg, self.custom_reg_kwargs, self.model_dof
            )
        # Validate model_dof
        self._validate_model_dof(self.model_dof)

    def _validate_X(self, X):
        """Checks that X is an ndarray of 2 dimensions and shape (m, n ).

        Checks that m and n > 0 .
        """

        X = check_type_and_dtype(X)

        if X.ndim == 2 and X.shape[0] > 0 and X.shape[1] > 0:
            return X
        else:
            raise DataDimensionalityWarning(
                "X should be a np.ndarray object with 2 dimensions and shape\
                (m, n) where m>0 and n>0.\
                Please correct the format."
            )

    def _validate_y(self, y):
        """Validates y variable."""

        y = check_type_and_dtype(y)

        if y.ndim != 1:
            raise DataDimensionalityWarning(
                "y should be a 1-D array.\
                    Please provide appropriately. \
                    Support for multiple targets is not available \
                    as of this version. Please treat each target as a separate model."
            )

    def _populate_args_for_known_estimators(self):
        """Populates required arguments for known estimators.

        For instance, If we know that the estimator is a sklearn class,
        we automatically assign it the required functions and get the
        required parameters.
        """

        if type(self.estimator) == sklearn.linear_model._base.LinearRegression:

            self.__populate_sklearn_linear_attributes()

            # Get regularization info.
            self.regularization = "none"
            self.l1_penalty = None
            self.l2_penalty = None

            self.model_kwargs = dict()
            self.residual_kwargs = dict()
            self.loss_kwargs = dict(sample_weight=self.y_train_weights)

        elif type(self.estimator) == sklearn.linear_model.Lasso:
            self.__populate_sklearn_linear_attributes()

            # Get regularization info.
            self.regularization = "l1"
            self.l1_penalty = self.estimator.alpha
            self.l2_penalty = None

            self.model_kwargs = dict()
            self.residual_kwargs = dict()
            self.loss_kwargs = dict(sample_weight=self.y_train_weights)

        elif type(self.estimator) == sklearn.linear_model.Ridge:
            self.__populate_sklearn_linear_attributes()

            self.residual = ordinary_residual
            self.loss = least_squares_loss

            # Get regularization info.
            self.regularization = "l2"
            self.l1_penalty = self.estimator.alpha
            self.l2_penalty = None

            self.model_kwargs = dict()
            self.residual_kwargs = dict()
            self.loss_kwargs = dict(sample_weight=self.y_train_weights)

        elif (
            type(self.estimator) == sklearn.linear_model._coordinate_descent.ElasticNet
        ):
            self.__populate_sklearn_linear_attributes()

            # Get regularization info.
            self.regularization = "l1+l2"
            self.l1_penalty = self.estimator.alpha * self.estimator.l1_ratio
            self.l2_penalty = 0.5 * self.estimator.alpha * (1 - self.estimator.l1_ratio)

            self.model_kwargs = dict()
            self.residual_kwargs = dict()
            self.loss_kwargs = dict(sample_weight=self.y_train_weights)

        elif type(self.estimator) == NonLinearRegression:
            if self.estimator.fit_intercept:
                self.intercept = self.estimator.intercept_
                # In this case, the coef_includes the intercept.
                self.best_fit_params = self.estimator.coef_
            else:
                self.intercept = None
                self.best_fit_params = self.estimator.coef_

            self.model = self.estimator.model
            self.residual = self.estimator.residual
            self.loss = (
                least_squares_loss  # For non-linear, currently, we only support this.
            )

            # Get regularization info.
            self.regularization = "none"
            self.l1_penalty = None
            self.l2_penalty = None

            self.model_kwargs = self.estimator.model_kwargs_dict
            self.residual_kwargs = self.estimator.residual_kwargs_dict
            self.loss_kwargs = dict(sample_weight=self.y_train_weights)

    def __populate_sklearn_linear_attributes(self):
        """Populate for sklearn linear attributes."""

        # If model if fitted with intercept, add intercept to best_fit_params
        # as the first argument.
        if self.estimator.fit_intercept:
            self.intercept = self.estimator.intercept_
            self.best_fit_params = np.concatenate(
                (np.array([self.estimator.intercept_]), self.estimator.coef_)
            )
            self.model = linear_model_with_intercept
        else:
            self.intercept = None
            self.best_fit_params = self.estimator.coef_
            self.model = linear_model

        self.residual = ordinary_residual
        self.loss = least_squares_loss

    def _validate_y_train_weights(self, y_train_weights, y_train):
        """Runs the same validation protocol as y and then checks if they are the same
        dimension.
        """

        if y_train_weights is not None:
            self._validate_y(y_train_weights)

            assert (
                y_train_weights.shape == y_train.shape
            ), "y_train_weights must be None or be the same shape as y_train."

    def _validate_custom_reg(self, custom_reg, custom_reg_kwargs, model_dof):
        """
        1. If regularization is custom: check that the custom reg function
            works correctly.
        2. Check that the model dofs are specified correctly.
        """

        intercept = self.intercept
        best_fit_params = self.best_fit_params

        if intercept is None:  # Intercept wasn't fit separately.
            coef_ = best_fit_params
        else:
            coef_ = best_fit_params[1:]

        # Check that the custom reg function works correctly.
        reg_term = custom_reg(coef_, **custom_reg_kwargs)

        if type(reg_term) != np.float64:
            raise ValueError(
                "custom_reg call did not yield float.\
                             Please provide correct custom_reg function."
            )

        if model_dof is None:
            raise ValueError(
                "Regularization is set to 'custom' but \
                             'model_dof' has not been specified. \
                             Please specify it explicitly."
            )

    def _validate_model_dof(self, model_dof):
        """Either None or float or int"""
        if model_dof is not None:
            if type(model_dof) not in [int, float]:
                raise ValueError(
                    "model_dof does not have correct type.\
                                 Either it should be None or float or int."
                )

    def _set_model_dof(self):
        """Computes model degrees of freedom.

        References:
        1. https://online.stat.psu.edu/stat462/node/131/
        2. https://www.statsmodels.org/dev/generated/statsmodels.
            regression.linear_model.RegressionResults.html#statsmodels.
            regression.linear_model.RegressionResults
        """

        model_dof = self.model_dof
        regularization = self.regularization

        intercept = self.intercept
        best_fit_params = self.best_fit_params

        if model_dof is not None:
            # Model degrees of freedom externally computed and specified.
            return model_dof
        else:  # Compute model degrees of freedom internally.
            if intercept is None:  # Intercept wasn't fit separately.
                coef_ = best_fit_params
            else:
                coef_ = best_fit_params[1:]

            if regularization == "none":
                model_dof = coef_.shape[0]
            elif regularization == "l1":
                non_zero_mask = coef_ != 0
                model_dof = coef_[non_zero_mask].shape[0]
            elif regularization == "l2":
                """Source: Elements of Statistical Learning, Ed. 2, Pg 68"""
                d_values = np.linalg.svd(self.X_train)[1]
                l2_penalty = self.l2_penalty
                model_dof = (d_values ** 2 / (d_values ** 2 + l2_penalty)).sum()
            elif regularization == "l1+l2":
                """Elastic net-type.
                In this case, we refer to the talk by Hui Zou
                (https://hastie.su.domains/TALKS/enet_talk.pdf)
                """
                non_zero_mask = coef_ != 0
                d_values = np.linalg.svd(self.X_train)[1]

                # Since there is L1 regualarization, apply non-zero mask
                # to singular values of X_train.
                d_values = d_values[non_zero_mask]
                l2_penalty = self.l2_penalty
                model_dof = (d_values ** 2 / (d_values ** 2 + l2_penalty)).sum()

        # The intercept value is not penalized during regularization.
        # # So, it should be added as +1 to the model degrees of freedom if it is
        # # part of the model kwargs.
        # if intercept in self.model_kwargs.keys() and intercept is not None:
        #     model_dof += 1

        # Set model dof.
        self.model_dof = model_dof

    def _set_error_dof(self):
        """Error dof = n-p-1 when intercept is present.
        Error dof = n-p when intercept is absent.
        This is in accordance with statsmodels.
        See docs for RegressionResults.
        """
        if self.intercept:
            self.error_dof = self.X_train.shape[0] - 1 - self.model_dof
        else:
            self.error_dof = self.X_train.shape[0] - self.model_dof

    def _set_sigma(self):
        """Computes residual mean squared error of the fit."""

        res = self.residual_function(
            self.X_train,
            self.best_fit_params,
            self.y_train,
            self.model_kwargs,
            **self.residual_kwargs,
        )

        # Also allow for inference from weighted least
        # squares, ultimately allowing for heteroskedasticity in y.
        if self.y_train_weights is not None:
            res = np.sqrt(self.y_train_weights) * res

        sigma = np.sqrt(np.matmul(res.transpose(), res) / self.error_dof)
        self.sigma = sigma

    def get_J(
        self,
        residual_function,
        best_fit_params,
        X,
        y,
        model_kwargs,
        residual_kwargs,
    ) -> np.ndarray:
        """Jacobian of residuals.

        Parameters
        ----------
        residual_function: callable
            Defines how to compute residuals.
            Arguments: (X, best_fit_params, y, model_kwargs, **residual_kwargs)
                and return residuals.
                Each of y, and residuals must be a 1-D
                array of shape (n_examples,).
                The specifications of other input parameters are described below.
        best_fit_params: arrray of shape (n_parameters,)
            Best fit parameters of the model fit.
        X: array of shape (n_samples, n_dimensions)
            Input array
        y: array of shape (n_samples,)
            Targets
        model_kwargs: dict
            Keyword arg dict for the model to relating y to X.
        residual_kwargs:
            Keyword arguments for residual_function.

        Returns
        -------
        J: array of shape (n_samples, n_parameters)
            Jacobian of residuals.
        """

        coefs_ = best_fit_params

        J = jacobian(
            lambda coefs_: residual_function(
                X, coefs_, y, model_kwargs, **residual_kwargs
            )
        )(coefs_)
        return J

    def get_H(
        self,
        loss_function,
        best_fit_params,
        X,
        y,
        model_kwargs,
        residual_kwargs,
        loss_kwargs,
    ) -> np.ndarray:
        """Hessian of objective function.

        Hessian of the objective function is the Jacobian of the gradient
        of the objective funciton.

        Parameters
        ----------
        loss_function: callable with arguments (X, coefs_, y, model_kwargs,
                                                residual_kwargs, loss_kwargs).
                        Returns float
                    X: see below
                    coefs_: same as best_fit_params below
                    y, model_kwargs, model_kwargs, residual_kwargs: see below
        X: array of shape (n_samples, n_dimensions)
            Input array
        best_fit_params: arrray of shape (n_parameters,)
            Best fit parameters of the model fit.
        y: array of shape (n_samples,)
            Targets
        model_kwargs: dict
            Keyword arg dict for the model to relating y to X.
        residual_kwargs: dict
            Keyword arguments for residual_function.
        loss_kwargs:
            Keyword arguments for loss function.

        Returns
        -------
        H: array of shape (n_parameters, n_parameters)
            Hessian of the objective function.
        """

        coefs_ = best_fit_params

        H = jacobian(
            egrad(
                lambda coefs_: loss_function(
                    X,
                    coefs_,
                    y,
                    model_kwargs,
                    residual_kwargs,
                    **loss_kwargs,
                )
            )
        )(coefs_)

        return H

    def get_vcov(self) -> np.ndarray:
        r"""Variance-covariance matrix of parameters.

        Estimate variance-covariance matrix of the provided parameters.
        The formula used is $$ D = \sigma^2 (\nabla^2 f(x^*))^{-1}$$ as described in
        the lecture notes of Niclas Börlin.
        https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts.pdf

        Returns
        -------
        vcov
            Variance-covariance matrix of the provided parameters.
        """
        sigma = self.sigma
        H = self.H

        # Check if Hessian is invertible else raise error.
        try:
            Hinv = np.linalg.inv(H)
        except Exception:
            raise DataDimensionalityWarning(
                "The computed Hessian is not invertible.\
                    The variance-covariance matrix for the parameters cannot be\
                    computed."
            )

        vcov = (sigma ** 2) * Hinv
        return vcov

    def get_sd_bf(self):
        """Standard deviation of best-fit parameters.

        Get the standard deviation of the best-fit parmeters.

        Returns
        -------
        sd_bf
            Standard deviations of the best-fit parameters.
        """
        vcov = self.vcov
        sd_bf = np.sqrt(np.diag(vcov))
        return sd_bf

    def _set_parameter_errors(self):
        """Compute the standard errors of the parameters."""

        # Compute the Hessian of the loss function.
        # TODO: For linear and ridge models, implement the closed
        # form solution for Hessian.
        self.H = self.get_H(
            self.loss_function,
            self.best_fit_params,
            self.X_train,
            self.y_train,
            self.model_kwargs,
            self.residual_kwargs,
            self.loss_kwargs,
        )

        # Get variance-covariance matrix.
        self.vcov = self.get_vcov()
        self.sd_coef = self.get_sd_bf()

    def get_parameter_errors(
        self,
        distribution="parametric",
        lsa_assumption=False,
        interval_stat=None,
        confidence_level=90.0,
        side="two-sided",
    ):
        """Gets shows parameter standard errors with appropriate intervals.

        Parameters:
        -----------
        distribution: str, optional, {"parametric", "non-parametric"},
                        default: "parametric"
            Distribution of the parameter standard errors.
        lsa_assumption: bool, optional, default=False
            Whether or not to invoke the large sample approximation.
            If True, assumes the distribution to be normal.
            Else, assumes the distribution to be t distribution with error_dof
            degrees of freedom.
        interval_stat: {None, str}, optional, {"normal", "t"}, default: None
            What kind of interval to return. If None: infers from the lsa_assumption
            as described above.
        confidence_level: float, optional, default: 90.0
            Confidence level (0-100) of the interval returned.
        side: str, optional, {"two-sided", "upper", "lower"}, default: "two-sided"
            Defines the type of interval to be computed.
        """

        means_array = self.best_fit_params
        std_array = self.sd_coef

        if distribution == "non-parametric":
            raise NotImplementedError(
                "Non-parametric distribution of \
                of prediction intervals is not yet implemented."
            )
        elif distribution == "parametric":
            # We choose what kind of interval we wish to
            # compute. Thsi is equivalent to the 'distribution' argument in
            # error_propagation. But, we refrain from using that here to
            # avoid confusion. Instead, we use "interval_stat". So,
            # if it is not specifically mentioned, we use 'normal' if LSA is
            # taken to be true, else we use "t" test.
            if interval_stat is None:
                if lsa_assumption:
                    interval_stat = "normal"
                else:
                    interval_stat = "t"

            interval_array = compute_intervals(
                means_array,
                std_array,
                side=side,
                confidence_level=confidence_level,
                distribution=interval_stat,
                dfe=self.error_dof,
            )

        else:
            raise NotImplementedError(
                "'distribution' argument not permissible. \
                    Please provde it as 'parameteric' or\
                    'non-parametric'."
            )

        # Output dictionary.
        param_err_dict = {}
        param_err_dict["mean"] = means_array
        param_err_dict["std"] = std_array
        param_err_dict["lower_bound"] = interval_array[0]
        param_err_dict["upper_bound"] = interval_array[1]

        # Convert into dataframe
        param_err_df = pd.DataFrame.from_dict(param_err_dict)

        return param_err_df

    def get_intervals(
        self,
        X,
        X_err=None,
        X_err_denotes_feature_correlation=False,
        y_weights=None,
        type_="prediction",
        distribution=None,
        lsa_assumption=True,
        confidence_level=90.0,
        side="two-sided",
        dfe=None,
    ):
        r"""Gets intervals for sample data.

        Uses model parameter errors as well as uncertainties in input data
        to compute the desired intervals.

        Parameters:
        -----------
        X: array of shape (n_samples, n_dimensions)
            Input array
        X_err: {None, np.ndarray}, optional, default: None
            Herer, m : number of samples,
            n : number of features
            p: number of parameters

            If X_err is array:
                1. If X_err is a 1-D vector of shape (n,)
                Standard errors for features in X are specified.
                Errors are equal for all examples and are uncorrelated with each other.
                2. If X_err is a 2-D vector of shape (m, n)
                    Standard errors are specified.
                    These are different across samples but assumed to be
                    uncorrelated across features.
                3. If X_err is a 2-D array of shape (n, n)
                    Covariance matrix for errors are specified.
                    Errors are equal for each example but those for different
                    features might be correlated. The $i, j$ th element of matrix
                    represents $cov(X_err_i, X_err_j)$ representing the covariance
                    of the errors $i, j$ th feature of the data.
                4. If X_err is a tensor of shape (m, n, n)
                    For each example, the nxn matrix denotes the covariance matrix of
                    errors in X.
        X_err_denotes_feature_correlation: bool, optional, default: False
            In the rare case that m=n, X_err will computed according to
            case 2 unless this parameter is
            explicity set to True.
        y_weights: {None, array}, optional, default: None
            If array: must have shape (n_samples, )
            Weights of targets. Often used when the variances of the
             distributions from which they are drawn substantially different.
             Used to compute uncertainty of the prediction through
             $\sigma_i = \sigma / weight_i $.
        type_: str, {"confidence", "prediction"}, default: "confidence"
            Defines the type of interval to compute.
        distribution: str, {"normal", "t"}, default: "normal"
            The type of distribution that the desired interval is from.
        lsa_assumption: bool, optional, default=False
            Whether or not to invoke the large sample approximation.
            If True, assumes the distribution to be normal.
            Else, assumes the distribution to be t distribution with error_dof
            degrees of freedom.
        confidence_level: float, default: 90.0
            Percentage of the distribution to be enveloped by the interval.
            Example: A value of 90 means that you wish to compute a 90% interval.
        side: str, "two-sided", "lower", "upper", default: "two-sided"
            Defines the type of interval to be computed.
        dfe: {None, float, int}, optional, default:None
            Error degrees of freedom of the fit.
            Only needed to compute prediction intervals. Ignored for "confidence".
            If None, uses the self.error_dof attribute; i.e., the computed
            error degrees of the model fit.

        Returns:
        --------
        df_int: pandas.DataFrame
            Dataframe consisting of mean (predicted values),
                std (standard error of the interval (prediction / confidence))
                lower_bound, upper_bound: lower, upper bound of the desired intervals
        """

        # Check that the parameter SE have been predicted.
        if not hasattr(self, "sd_coef"):
            raise ValueError(
                "SD coefficients have not been predicted.\
                             Please fit those first using the 'get_paramter_errors' \
                             function."
            )

        if dfe is None:
            dfe = self.error_dof

        eprop = ErrorPropagation()

        df_int = eprop.get_intervals(
            function_=self.model_function,
            X=X,
            params=self.best_fit_params,
            X_err=X_err,
            params_err=self.vcov,
            X_err_denotes_feature_correlation=X_err_denotes_feature_correlation,
            sigma=self.sigma,
            y_weights=y_weights,
            type_=type_,
            side=side,
            confidence_level=confidence_level,
            distribution=distribution,
            lsa_assumption=lsa_assumption,
            dfe=dfe,
            model_kwarg_dict=self.model_kwargs,
        )

        return df_int
