"""
Defines functions for forward propagation of errors.
"""

from typing import Callable
import autograd.numpy as np
import pandas as pd
from autograd import jacobian
from sklearn.exceptions import DataDimensionalityWarning
from .statistical_utils import compute_intervals


class ErrorPropagation:
    r"""
    Performs forward propagation of error for any parametric function
    based on the errors of parameters and its input uncerainties.

    Given $ y = f(\bf{X}, \bf{\beta})  $, computes

    $$ \delta y = \delta f(\bf{X}, \bf{\beta}) $$
    $$ \delta y= \sqrt{(\nabla_{\textbf{X}}f) \delta\textbf{X} (\nabla_{\textbf{X}}f)^T
        + (\nabla_{\bm{\beta}}f) \delta\bm{\beta} (\nabla_{\bm{\beta}}f)^T} $$

    Can be used to compute confidence as well as prediction intervals.

    Confidence intervals ~ Interval of mean of the predicted value.
            $$  \mathrm{Var}( \mathbb{E}(\hat{y} | \mathbf{X}, \bm{\beta})) $$
    Prediction intervals ~ Interval of the predicted value.
            $$ \mathrm{Var}(\hat{y} | \bf{X}, \bm{\beta}) $$

    We know that
            $$ \mathrm{Var}(\hat{y} | \bf{X}, \bm{\beta}) =
            \mathrm{Var}( \mathbb{E}(\hat{y} | \mathbf{X}, \bm{\beta})) +
                \sigma^2 $$
    So, we need some estimate of $\sigma^2$ to calculate prediction intervals.


    Assumptions:
    ------------
    1. All samples in X are i.i.d. (i.e., uncorrelated)

    References:
    -----------
    1.https://www.stat.cmu.edu/~cshalizi/36-220/lecture-11.pdf
    """

    def __init__(self):
        """Initialize the function and set parameters as required."""

    def get_intervals(
        self,
        function_: Callable,
        X,
        params,
        X_err=None,
        params_err=None,
        X_err_denotes_feature_correlation=False,
        sigma=None,
        y_weights=None,
        type_="confidence",
        side="two-sided",
        confidence_level=90.0,
        distribution=None,
        lsa_assumption=False,
        dfe=None,
        model_kwarg_dict={},
    ):
        r"""
        Parameters
        ----------
        function_: callable
            Function with respect to which we want to propagate errors.
            The arguments of the this function MUST be of the form:
            f(X, params, **kwargs) otherwise, errors will be raised.
        X: np.ndarray of shape (m, n)
            m : number of examples,
            n : number of features
        params: np.ndarray of shape (p,)
            The parameters of the function.
            p: Number of parameters
        X_err: {None, array}, optional, default: None
            1. If X_err is a 1-D vector of shape (n,)
             Standard errors for features in X are specified.
             Errors are equal for all examples and are uncorrelated with each other.
            2. If X_err is a 2-D vector of shape (m, n)
                Standard errors are specified.
                These are different across samples but assumed to be uncorrelated
                across features.
            3. If X_err is a 2-D array of shape (n, n)
                Covariance matrix for errors are specified.
                Errors are equal for each example but those for different
                features might be correlated. The $i, j$ th element of matrix
                represents $cov(X_err_i, X_err_j)$ representing the covariance of the
                errors $i, j$ th feature of the data.
            4. If X_err is a tensor of shape (m, n, n)
                For each example, the nxn matrix denotes the covariance matrix of
                errors in X.
        params_err: {None, array}, optional, default: None
            If array-type: shape must be (p,) or (p, p)
            1. If a 1-D vector of shape (p,), treats it as standard error for each
                parameter.
                It will be assumed that the various parameters in $\beta$ are
                mutually uncorrelated.
            2. If a 2-D vector of shape (p, p), it will be assumed that this matrix is
            the variance-covariance matrix for the parameters
            $\mathrm{cov}(\bm{\beta})$.
        X_err_denotes_feature_correlation: bool, optional, default: False
            In the rare case that m=n, X_err will computed according to
            case 2 unless this parameter is
            explicity set to True.
        sigma: {None, float}, default: None
            Estimate of the standard deviation of the distribution of $y$
            at given $\bf{X}, \bm{beta}$.
            Only required if type_=="prediction".

            Mathematically,
            $$ y = f(\bf{X}, \bm{\beta}) + \epsilon $$
            Where $ \epsilon$ follows some distribution given by:
            $$ \epsilon \sim \mathcal{N}(\mu, \sigma^2) $$.
            var_y is an estimate of $\sigma_2$.

            Practically, if this function has been fitted earlier,
            the residual sum of squares (RSS) is a commonly used estimator.

            $$ r = y - \hat{y} $$
            $$ \sigma^2 = MSE = \frac{r^Tr}{m-n} $$

            Alternatively, if there is some prior knowledge / intuition
            about $ \sigma^2 $, it can also be used.
        y_weights: {None, array}, optional, default: None
            If array: must have shape (n_samples, )
            Weights of targets. Often used when the variances of the
             distributions from which they are drawn substantially different.
             Used to compute uncertainty of the prediction through
             $\sigma_i = \sigma / weight_i $.
        type_: str, {"confidence", "prediction"}, default: "confidence"
            Defines the type of interval to compute.
        side: str, "two-sided", "lower", "upper", default: "two-sided"
            Defines the type of interval to be computed.
        confidence_level: float, default: 90.0
            Percentage of the distribution to be enveloped by the interval.
            Example: A value of 90 means that you wish to compute a 90% interval.
        distribution: str, {"normal", "t"}, default: "normal"
            The type of distribution that the desired interval is from.
        lsa_assumption: bool, optional, default=False
            Whether or not to invoke the large sample approximation.
            If True, assumes the distribution to be normal.
            Else, assumes the distribution to be t distribution with error_dof
            degrees of freedom.
        dfe: {None, float, int}, optional, default:None
            Error degrees of freedom of the fit.
            Only needed to compute prediction intervals. Ignored for "confidence".
            If None, uses the self.error_dof attribute; i.e., the computed
            error degrees of the model fit.
        model_kwarg_dict: dict, optional, default:{}
            Keyword arguments to be passed to "function_"

        Returns
        -------
        df_int: pandas.DataFrame
            Dataframe consisting of mean (predicted values),
                std (standard error of the interval (prediction / confidence))
                lower_bound, upper_bound: lower, upper bound of the desired intervals.
        """

        # Validate and transform the inputs
        func, X, params, X_err, params_err = self._validate_and_transform_inputs(
            function_,
            X,
            params,
            X_err=X_err,
            params_err=params_err,
            X_err_denotes_feature_correlation=X_err_denotes_feature_correlation,
            sigma=sigma,
            type_=type_,
            side=side,
            confidence_level=confidence_level,
            distribution=distribution,
            lsa_assumption=lsa_assumption,
            dfe=dfe,
            y_weights=y_weights,
        )

        # Ensure that the function meets the required demands by
        # computing y_hat. If it raises errors, we can simply stop here.
        y_hat = self._compute_y_hat(func, X, params, model_kwarg_dict)

        # Validate y_hat to make sure that it is a 1-D array
        self.__validate_y_hat(y_hat, X)

        # Calculate error propagation.
        SE_on_mean, SE_on_prediction = self._propagate_errors(
            func, X, params, X_err, params_err, model_kwarg_dict, sigma, y_weights
        )

        # Determine appropriate SD and compute interval.
        if type_ == "confidence":
            se = SE_on_mean
        if type_ == "prediction":
            if SE_on_prediction is None:
                raise ValueError(
                    "The prediction error cannot be computed because \
                        'sigma' was not supplied. This parameter is needed to \
                        compute prediction intervals. \
                        Please supply this parameter and recompute.\
                        Or, please request for confidence intervals instead."
                )
            se = SE_on_prediction

        # If LSA assumption is true, we need not compute the t distribution
        # since it required model degrees of freedom which we may not always have.
        # Thus, if distribution is set to "None", we select which one to use
        # considering LSA assumption.
        if distribution is None:
            if lsa_assumption:
                distribution = "normal"
            else:
                distribution = "t"

        # Computes the required interval.
        lower_bound, upper_bound = compute_intervals(
            y_hat,
            se,
            side=side,
            confidence_level=confidence_level,
            distribution=distribution,
            dfe=dfe,
        )

        # Return dataframe.
        df = pd.DataFrame()
        df["mean"] = y_hat
        df["se"] = se
        df["lower_bound"] = lower_bound
        df["upper_bound"] = upper_bound

        return df

    def _validate_and_transform_inputs(
        self,
        function_,
        X,
        params,
        X_err,
        params_err,
        X_err_denotes_feature_correlation,
        sigma,
        type_,
        side,
        confidence_level,
        distribution,
        lsa_assumption,
        dfe,
        y_weights,
    ):
        """Validates and transforms inputs for the function"""

        # Validates and transforms the function.
        # As such, there is nothing to validate, so we simply assign it.
        func = function_

        # Validates and transforms X.
        X = self._validate_X(X)

        # Validates and transforms params.
        params = self._validate_params(params)

        # Validates and transforms params_err and X_err
        params_err = self._validate_and_transform_params_err(params_err)
        X_err = self._validate_and_transform_X_err(X_err, X)

        # Check that at least one of params_err and X_err is not None
        if any([X_err is not None, params_err is not None]):
            pass
        else:
            raise Exception(
                "At least one of X_err, params_err should be supplied\
                for error propagation."
            )

        # Validates X_err_denotes_feature_correlation
        assert isinstance(
            X_err_denotes_feature_correlation, bool
        ), "X_err_denotes_feature_correlation not of type bool. \
                Please supply correct value"

        # Validates sigma
        assert any(
            [sigma is None, isinstance(sigma, float)]
        ), "sigma not of type None or float. \
                Please supply value appropriately."

        # Validates type_
        assert isinstance(
            type_, str
        ), "type_ not of type string. \
                Please supply appropriately."

        # Validates side
        assert isinstance(
            side, str
        ), "side not of type string. \
                Please supply appropriately."

        # Validates confidence_level
        assert isinstance(
            confidence_level, float
        ), "confidence_level not of type float. \
                Please supply appropriately."

        # Validates distribution
        assert any(
            [isinstance(distribution, str), distribution is None]
        ), "distribution not of type string.\
                Please supply appropriately."

        # Validates lsa_assumption
        assert isinstance(
            lsa_assumption, bool
        ), "lsa_assumption not of type bool. \
            Please supply correct value"

        # Validates dfe
        assert any(
            [dfe is None, isinstance(dfe, float), isinstance(dfe, int)]
        ), "dfe is not None, float, or int.\
                   Please supply appropriate values."

        # Validates y_weights
        self._validate_y_weights(y_weights, X)

        return func, X, params, X_err, params_err

    @staticmethod
    def check_type_and_dtype(array):
        """Checks that the given object is a numpy array,
        determines it dtype, and if it is int, changes it to float.
        This avoids problems when using autograd functions.
        """

        if type(array) == np.ndarray:
            if array.dtype.name == "float64":  # This passes the test.
                return array

            else:  # Try to convert it to float 64, else give raise error.
                try:
                    array = array.astype("float64")
                    return array
                except Exception:
                    raise ValueError(
                        "The array could not be converted to\
                        dtype 'float64'. Please try to convert it externally\
                        and ensure that the array.dtype == 'float64'. "
                    )

        else:
            raise ValueError(
                "The array is not of type ndarray. Please supply an\
                array of type ndarray"
            )

    def _validate_params(self, params):
        """Checks that params is a 1-D array or can be cast as one."""

        params = self.check_type_and_dtype(params)

        if params.ndim == 1 and params.shape[0] > 0:
            return params
        else:
            raise DataDimensionalityWarning(
                "params is not passed as a vector of shape (p,).\
                Please correct the format."
            )

    def _validate_X(self, X):
        """Checks that X is an ndarray of 2 dimensions and shape (m, n ).

        Checks that m and n > 0 .
        """

        X = self.check_type_and_dtype(X)

        if X.ndim == 2 and X.shape[0] > 0 and X.shape[1] > 0:
            return X
        else:
            raise DataDimensionalityWarning(
                "X should be a np.ndarray object with 2 dimensions and shape\
                (m, n) where m>0 and n>0.\
                Please correct the format."
            )

    def _validate_and_transform_params_err(self, params_err):
        """
        Check and validate params_err.
        """

        if params_err is not None:  # If it has been defined and is not None.
            params_err = self.check_type_and_dtype(params_err)
            p = params_err.shape[0]
            if params_err.ndim == 1 and p > 0:
                # checks that the params_err is a 1-D non-empty vector.
                # This means that these are the standard errors of the parameters.
                return np.diag(params_err ** 2)

            elif params_err.ndim == 2 and params_err.shape == (p, p):
                # params_err is a pxp matrix specifying the covariance matrix of the
                # parameters.
                return params_err
            else:
                raise DataDimensionalityWarning(
                    "params_err not in one of the required shapes.\
                    Please read the docs."
                )
        else:
            return params_err

    def _validate_and_transform_X_err(self, X_err, X):
        """
        Check and validate X_err
        """

        if X_err is not None:  # If X_err is not None.
            X_err = self.check_type_and_dtype(X_err)
            m, n = X.shape

            if X_err.ndim == 1 and X_err.shape[0] == n:
                # if X_err is a 1-D vector of shape (n,), it means
                # error are the same for all examples and uncorrelated with each other.
                X_err_transformed = np.tile(np.diag(X_err ** 2), (m, 1, 1))
            elif X_err.ndim == 2:
                if X_err.shape == (m, n):
                    X_err_transformed = np.apply_along_axis(
                        np.diag, axis=1, arr=X_err ** 2
                    )

                elif X_err.shape == (n, n):
                    # if X_err is a 2-D matrix with (n,n) dimensions, it means that the
                    # errors are same across all examples, but the features
                    # are correlated and their covariance matrix is specified.
                    X_err_transformed = np.tile(X_err, (m, 1, 1))

                if (
                    X_err.shape(n, n)
                    and m == n
                    and self.X_err_denotes_feature_correlation
                ):
                    # Force the program to treat the matrix as (n,n) rather than (m,n).
                    # In this case, assume that the covariance matrix is specified.
                    X_err_transformed = np.tile(X_err, (m, 1, 1))
            elif X_err.ndim == 3 and X_err.shape == (m, n, n):
                # This means that there are different errors in X for different
                # examples.
                # And assumed that the covariance matrix is specified.
                X_err_transformed = X_err
            else:
                # All other cases
                raise DataDimensionalityWarning(
                    "X_err is not correctly specified. Plesae refer to the docs for \
                    the correct specification"
                )

            return X_err_transformed

        else:  # Simple store it as a None object.
            return X_err

    def _compute_y_hat(self, func, X, params, model_kwarg_dict):
        """Computes y_hat for the X, params, args and kwargs provided."""

        # Get the desired y_hat for the prediction.
        y_hat = func(X, params, **model_kwarg_dict)
        return y_hat

    def __validate_y_hat(self, y_hat, X):
        """Checks that the supplied y_hat is right."""

        # Ensure y_hat has the correct dimension.
        assert y_hat.ndim == 1, "y_hat is not a 1-D vector. Please supply a 1-D vector."
        assert (
            y_hat.shape[0] == X.shape[0]
        ), "Size of y_hat is different from that of X. Please make sure that\
            they match."

    def _validate_y_weights(self, y_weights, X):
        """Checks that y_weights is either NoneType or an array of shape
        (X.shape[0],)"""

        if y_weights is not None:
            assert y_weights.shape == (
                X.shape[0],
            ), "y_weights is not None or array of shape \
                (X.shape[0],). Please provide appropriate values."

    def get_grad_matrix_for_params(self, func, X, params, model_kwarg_dict):

        grad_matrix = jacobian(lambda params: func(X, params, **model_kwarg_dict))(
            params
        )

        m = X.shape[0]
        p = params.shape[0]

        assert grad_matrix.shape == (m, p), DataDimensionalityWarning(
            "Grad matrix for params is of an unexpected dimension."
        )

        return grad_matrix

    def get_grad_matrix_for_X(self, func, X, params, model_kwarg_dict):
        """ """

        m, n = X.shape

        mat = jacobian(lambda X: func(X, params, **model_kwarg_dict))(X)

        # Upon taking the Jacobian,
        # We get a matrix of the shape (m, m, n). This also shows the correlations
        # between various samples. As of now, i dont know how to multiply such a matrix
        # or handle the information about the sample-sample correlation.
        # So, we remove that information and only consider the elements along the
        # diagonal.

        grad_matrix = np.diagonal(mat, axis1=0, axis2=1).T  # grad matrix of shape mxn

        assert grad_matrix.shape == (m, n), DataDimensionalityWarning(
            "Grad matrix for X is of an unexpected dimension."
        )

        return grad_matrix

    def get_variances_based_on_params(self, grad_matrix_for_params, params_err):
        r"""Computes $ \mathrm{var}(E(y|X,\beta)) $ based on $ \beta $ alone."""

        m = grad_matrix_for_params.shape[0]

        # Multiply the two.
        mult = grad_matrix_for_params @ params_err @ grad_matrix_for_params.T

        # Take the diagonal elements
        variances_params = np.diagonal(mult)  # This should have a shape of (m, )

        assert variances_params.shape == (m,), DataDimensionalityWarning(
            "Variances based on params have the wrong dimensions."
        )

        return variances_params

    def get_variances_based_on_X(self, grad_matrix_for_X, X_err):

        r"""Computes the $ \mathrm{var}(E(y|X,\beta)) $ based on $ X $ alone."""

        X = self.X
        m = X.shape[0]

        def compute_y_err_for_x(i):
            """
            Computes the error based on the grad matrix and X_err matrix for each \
            example.
            """

            # grad_X_example: 1xn, X_err_example: nxn, grad_X_example.T: nx1
            # total value: 1x1

            grad_X_example = grad_matrix_for_X[i, :].reshape(1, -1)  # 1xn dimensions
            X_err_example = X_err[i, ...]

            output = grad_X_example @ X_err_example @ grad_X_example.T

            assert output.shape == (1, 1), DataDimensionalityWarning(
                "Variance based on X: wrong shape."
            )

            return output[0][0]

        vfunc = np.vectorize(compute_y_err_for_x)
        variances_X = vfunc([i for i in range(m)])

        return np.array(variances_X)

    def _propagate_errors(
        self, func, X, params, X_err, params_err, model_kwarg_dict, sigma, y_weights
    ):
        """This function will perform the computations to compute
        the required properties which can be accessed as required.
        """

        if X_err is not None:
            grad_matrix_for_X = self.get_grad_matrix_for_X(
                func, X, params, model_kwarg_dict
            )
            variances_X = self.get_variances_based_on_X(grad_matrix_for_X, X_err)
        else:
            variances_X = None

        if params_err is not None:

            grad_matrix_for_params = self.get_grad_matrix_for_params(
                func, X, params, model_kwarg_dict
            )
            variances_params = self.get_variances_based_on_params(
                grad_matrix_for_params, params_err
            )

        else:
            variances_params = None

        # Compute the SE on mean.
        SE_on_mean = self._compute_SE_on_mean(
            X_err, params_err, variances_X, variances_params
        )

        # Get prediction intervals.
        SE_on_prediction = self._compute_SE_on_prediction(SE_on_mean, sigma, y_weights)

        return SE_on_mean, SE_on_prediction

    def _compute_SE_on_mean(self, X_err, params_err, variances_X, variances_params):
        r"""Computes the standard error of the mean prediction SE(E(y|X, \beta))"""

        # Add all variances and take a square root.
        if params_err is not None and X_err is None:  # Only error prop from params.
            SE_on_mean = np.sqrt(variances_params)
        elif params_err is None and X_err is not None:  # Only error prop from X.
            SE_on_mean = np.sqrt(variances_X)
        elif params_err is not None and X_err is not None:  # Both are present.
            SE_on_mean = np.sqrt(variances_params + variances_X)

        return SE_on_mean

    def _compute_SE_on_prediction(self, SE_on_mean, sigma, y_weights):
        r"""Uses SE_on_mean to get SE_on_prediciton.

        $$ SE(y|X, \beta) = \sqrt(SE(E(y|X, \beta))^2 + \sigma^2) $$
        """

        if sigma is not None:
            if y_weights is None:  # Equal weights for all samples.
                SE_on_prediction = np.sqrt((SE_on_mean ** 2 + sigma ** 2))
            if y_weights is not None:
                SE_on_prediction = np.sqrt((SE_on_mean ** 2 + sigma ** 2 / y_weights))

        else:
            SE_on_prediction = None

        return SE_on_prediction
