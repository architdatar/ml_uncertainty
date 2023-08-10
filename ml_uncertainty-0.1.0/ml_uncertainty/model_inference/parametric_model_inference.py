"""
DOF calculation: Talk by Hui Zou (https://hastie.su.domains/TALKS/enet_talk.pdf)

1. Create tests for elastic net model. 
    Degrees of freedom: ESL Pg 64 and 68
2. Create functions correspondingly.
3. Test that the parameter errors are being fit correctly.
4. Test that the prediction intervals are computed correctly.
5. Compare with statsmodels.

# TODO: Write proper documentation.
# TODO: Enable other non-parametric tests for cases where
        large sample approximation does not hold.
        easily.
# TODO: Enable model signficance tests.
"""

import autograd.numpy as np
# import numpy as np
from autograd import jacobian
from autograd import elementwise_grad as egrad
import pandas as pd
import sklearn
from sklearn.exceptions import DataDimensionalityWarning
from ..error_propagation.error_propagation import ErrorPropagation
from .common_model_functions import linear_model, ordinary_residual, least_squares_loss
from ..error_propagation.statistical_utils import compute_intervals
from ..non_linear_regression import NonLinearRegression

# Reusing function to check that arrays have the right type and dtype.
check_type_and_dtype = ErrorPropagation.check_type_and_dtype


class ParametricModelInference:
    """
    Contains a group of methods to get model inference from models in scikit-learn
    NOTE: The functions get_J, get_H are adopted from
    https://github.com/sriki18/adnls/blob/master/adnls.py#L92.
    """

    """
    Procedure:
    1. Initialize this with a sklearn fitted object, and training data. 
    2. Check that the object is fitted. Because we will not refit the model.
    3. For now, try this for linear regression and elastic net models. We can expand this later.
    4. For each such instance, obtain the jacobian matrix using the autograd library. 
    5. Once the Jacobian is obtained, we can get parameter standard errors and CIs.
    6. We can get the confidence / prediction intervals. 
    7. Extend this to other linear models in sklearn as need requires. 
    """

    def __init__(self):
        """
        Initialize the class
        """

        self.__estimators_implemented = [
            sklearn.linear_model._base.LinearRegression,
            sklearn.linear_model._coordinate_descent.ElasticNet,
            NonLinearRegression,
        ]

    def set_up_model_inference(
        self,
        X_train,
        y_train,
        estimator=None,
        model=None,
        model_kwargs=None,
        residual=None,
        residual_kwargs=None,
        loss=None,
        loss_kwargs=None,
        regularization=None,
        intercept=None,
        best_fit_params=None,
        l1_penalty=None,
        l2_penalty=None,
        y_train_weights=None,
        model_dof=None,
    ):
        """Sets up model inference for the fitted model.

        Validates models, inputs and raises errors if necessary.

        Parameters
        ----------
        y_train_weights: 
            NOTE: If this is not set to None, to get accurate results,
            you must set up the loss function correctly. Please refer 
            to the example examples/weighted_least_squares_regression.
        """

        self.X_train = X_train
        self.y_train = y_train
        self.estimator = estimator
        self.model = model
        self.model_kwargs = model_kwargs
        self.residual = residual
        self.residual_kwargs = residual_kwargs
        self.loss = loss
        self.loss_kwargs = loss_kwargs
        self.regularization = regularization
        self.intercept = intercept
        self.best_fit_params = best_fit_params
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.y_train_weights = y_train_weights
        self.model_dof = model_dof

        # Validate that the estimators, etc. are correctly specified.
        self._validate_input_arguments()

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

        loss_term = self.loss(residuals)

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
            # TODO: Enable custom regularization
            raise NotImplementedError(
                "Custom regularization not implemented as of \
                                      this version. Will be implemented in future. \
                                      Please reconstruct this function externally as \
                                      shown in examples."
            )

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
            if type(self.estimator) in self.__estimators_implemented:
                self._populate_args_for_known_estimators()
            else:
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
                        residual_function, loss_function, best fit params \
                        is not defined.\
                        Please provide all the above as it is necessary for \
                        model inference."
                )

            # TODO: Further validate these functions by checking that they yield the
            # desired values in the desired formats.
            # TODO: Also validate other parameters related to best fit and regularization.

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

        # Get the Jacobian matrix for this model.
        if type(self.estimator) == sklearn.linear_model._base.LinearRegression:
            self.intercept = self.estimator.intercept_
            self.best_fit_params = self.estimator.coef_
            self.model = linear_model
            self.residual = ordinary_residual
            self.loss = least_squares_loss

            # Get regularization info.
            self.regularization = "none"
            self.l1_penalty = None
            self.l2_penalty = None

            self.model_kwargs = dict(intercept_=self.intercept)
            self.residual_kwargs = dict()
            self.loss_kwargs = dict()

        elif (
            type(self.estimator) == sklearn.linear_model._coordinate_descent.ElasticNet
        ):
            self.intercept = self.estimator.intercept_
            self.best_fit_params = self.estimator.coef_
            self.model = linear_model
            self.residual = ordinary_residual
            self.loss = least_squares_loss

            # Get regularization info.
            self.regularization = "l1+l2"
            self.l1_penalty = self.estimator.alpha * self.estimator.l1_ratio
            self.l2_penalty = 0.5 * self.estimator.alpha * (1 - self.estimator.l1_ratio)

            self.model_kwargs = dict(intercept_=self.intercept)
            self.residual_kwargs = dict()
            self.loss_kwargs = dict()
        elif type(self.estimator) == NonLinearRegression:
            self.intercept = self.estimator.intercept_
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
            self.loss_kwargs = dict()

    def _set_model_dof(self):
        """Computes model degrees of freedom."""

        model_dof = self.model_dof
        regularization = self.regularization

        intercept = self.intercept
        best_fit_params = self.best_fit_params

        if model_dof is not None:
            # Model degrees of freedom externally computed and specified.
            return model_dof
        else:  # Compute model degrees of freedom internally.
            if regularization == "none":
                model_dof = best_fit_params.shape[0]
            elif regularization == "l1":
                non_zero_mask = best_fit_params != 0
                model_dof = best_fit_params[non_zero_mask].shape[0]
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
                non_zero_mask = best_fit_params != 0
                d_values = np.linalg.svd(self.X_train)[1]

                # Since there is L1 regualarization, apply non-zero mask
                # to singular values of X_train.
                d_values = d_values[non_zero_mask]
                l2_penalty = self.l2_penalty
                model_dof = (d_values ** 2 / (d_values ** 2 + l2_penalty)).sum()

        # The intercept value is not penalized during regularization.
        # So, it should be added as +1 to the model degrees of freedom if it is
        # part of the model kwargs.
        if intercept in self.model_kwargs.keys() and intercept is not None:
            model_dof += 1

        # Set model dof.
        self.model_dof = model_dof

    def _set_error_dof(self):
        """Error dof = n-p"""
        self.error_dof = self.X_train.shape[0] - self.model_dof

    def _set_sigma(self):
        """Standard deviation of the fit.

        Estimate standard deviation from the residual vector.

        Parameters
        ----------
        residual_function: callable
            Function to compute residuals for the training data.
        error_dof: float
            Error / residual degrees of freedom for the fit.
        Returns
        -------
        sig
            Estimated standard deviation.
        """

        res = self.residual_function(
            self.X_train,
            self.best_fit_params,
            self.y_train,
            self.model_kwargs,
            **self.residual_kwargs,
        )

        # Also allow for inference from weighted least
        # squares, ultimately allowing for heteroskedasticity in y.
        # NOTE: Here, the weights are normalized by their mean,
        # So, if they all have the same weights, they drop out.
        if self.y_train_weights is not None:
            res = self.y_train_weights / self.y_train_weights.mean() * res

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

        Residuals (prediction - data) of the fit of interest.

        Parameters
        ----------

        X
            The `np.ndarray` of paramaters used to compute the Jacobian.

        Returns
        -------
        J
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
        x
            The `np.ndarray` of paramaters used to compute the Hessian.

        Returns
        -------
        H
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

        # TODO:1. For linear models and ridge, there is a closed form
        # solution. Var[b]=σ2(X′X)−1.
        # https://stats.stackexchange.com/questions/68151/how-to-derive-variance-covariance-matrix-of-coefficients-in-linear-regression
        # Ridge: https://online.stat.psu.edu/stat857/node/155/ under Properties of Ridge estimator.
        # 2. Allow users to compute Hessian through a first-order approximation for
        # functions that might not be 2-times differenciable. Refer to Niclas
        # Borgin's lectures.

        return H

    def get_vcov(self) -> np.ndarray:
        r"""Variance-covariance matrix of parameters.

        Estimate variance-covariance matrix of the provided parameters.
        The formula used is $$ D = \sigma^2 (\nabla^2 f(x^*))^{-1}$$ as described in the
        lecture notes of Niclas Börlin.
        https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts.pdf

        Parameters
        ----------
        x
            The `np.ndarray` of parameters used to compute the Hessian.

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
        return_full_distribution=False,
    ):
        """Gets model inference for the selected fitted estimator."""

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

            if lsa_assumption:
                interval_stat = "normal"
            else:
                interval_stat = "t"
            
            interval_array = compute_intervals(means_array, 
                                               std_array,
                                               side=side,
                                               confidence_level=confidence_level,
                                               distribution=interval_stat,
                                               dfe=self.error_dof)
            
        else:
            raise NotImplementedError("'distribution' argument not permissible. \
                                      Please provde it as 'parameteric' or\
                                      'non-parametric'.")

        # Output dictionary.
        param_err_dict = {}
        param_err_dict["mean"] = means_array
        param_err_dict["std"] = std_array
        param_err_dict["lower_bound"] = interval_array[0]
        param_err_dict["upper_bound"] = interval_array[1]

        # Convert into dataframe
        param_err_df = pd.DataFrame.from_dict(param_err_dict)

        if return_full_distribution:
            return param_err_df, means_array, std_array
        else:
            return param_err_df

    def get_intervals(
        self,
        X,
        X_err=None,
        X_err_denotes_feature_correlation=False,
        type_="prediction",
        distribution=None,
        lsa_assumption=True,
        confidence_level=90.0,
        side="two-sided",
        dfe=None,
        center_X=True,
    ):

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
            params=self.estimator.coef_,
            X_err=X_err,
            params_err=self.sd_coef,
            X_err_denotes_feature_correlation=X_err_denotes_feature_correlation,
            sigma=self.sigma,
            type_=type_,
            side=side,
            confidence_level=confidence_level,
            distribution=distribution,
            lsa_assumption=lsa_assumption,
            dfe=dfe,
            model_kwarg_dict=self.model_kwargs,
            center_X=center_X,
        )

        return df_int
