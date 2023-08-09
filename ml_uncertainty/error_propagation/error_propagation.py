"""
Defines functions for forward propagation of errors.

# TODO: reorganize documentation
"""

from typing import Callable
import autograd.numpy as np
from autograd import jacobian
from sklearn.exceptions import DataDimensionalityWarning
from scipy.stats import norm, t as t_dist
import pandas as pd


def get_significance_levels(confidence_level, side):
    """Computes the significance level for the given side

    Effectively, returns the percentiles of the distributions which we wish to
    evaluate.

    In other words, it is the stat at which the percent of distribution
    less than the desired value is the desired probability.

    Examples:
    --------
    1. For a 2-sided 90% confidence interval, we would wish to obtain the stats
        x1* and x2* such that P(x1* < x < x2*) = 90%..
        Considering a symmetric distribution: we would want
        P(x < x1*) = 5% and P(x > x2*) = 5% => P(x < x2*) = 95%
    2. For a lower 90%, we would wish to obtain the stat
        x* such that P(x >= x*) = 90%, alternatively, P(x < x*) = 10%
    3. For upper 90% CI, we would wish to obtain the stat
        x* such that P(x < x*) = 90%

    """

    if side == "two-sided":
        significance_levels = [
            (1 - confidence_level / 100) / 2,
            1 - (1 - confidence_level / 100) / 2,
        ]
    elif side == "upper":
        significance_levels = [1 - confidence_level / 100, None]
    elif side == "lower":
        significance_levels = [None, confidence_level / 100]
    else:
        raise ValueError(
            "Side should be one of \
            two-sided, lower, and upper. Please provided one of these."
        )

    return significance_levels


def get_z_values(significance_levels):
    # Gets z values for the required significance_levels
    z_values = []
    for level in significance_levels:
        if level is None:
            z_values.append(None)
        else:
            stat = norm.ppf(level)
            z_values.append(stat)

    return z_values


def get_t_values(significance_levels, dfe):
    """Computes t values"""

    if not dfe:
        # dfe is not supplied.
        raise ValueError(
            "dfe is not supplied\
                due to which t-stats cannot be computed.\
                If you wish to make a large sample approximation (LSA)\
                that your sample standard deviation is based on enough samples to\
                be close to the population standard deviation, you may consider using\
                the normal distribution instead."
        )
    else:
        # For each level: get the desired t-stat.
        t_values = []
        for level in significance_levels:
            if level is None:
                t_values.append(None)
            else:
                stat = t_dist.ppf(q=level, df=dfe)
                t_values.append(stat)

        return t_values


class ErrorPropagation:
    r"""
    Performs forward propagation of error for any parametric function
    based on the errors of parameters and its inputs.

    Given $ y = f(\bf{X}, \bf{\beta})  $, computes

    $$ (\delta y) = (\delta f(\bf{X}, \bf{\beta})) $$
    $$ = \sqrt{(\nabla_{\textbf{X}}f) \delta\textbf{X} (\nabla_{\textbf{X}}f)^T  +
        (\nabla_{\bm{\beta}}f) \delta\bm{\beta} (\nabla_{\bm{\beta}}f)^T} $$

    Assumptions:
    1. All samples in X are i.i.d. (i.e., uncorrelated)

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
        type_="confidence",
        side="two-sided",
        confidence_level=90.0,
        distribution="normal",
        lsa_assumption=True,
        dfe=None,
        model_kwarg_dict={},
    ):
        r"""
        Parameters
        ----------
        func: Callable
            Function with respect to which we want to propagate errors.
            The arguments of the this function MUST be of the form:
            f(X, params, *args, **kwargs) otherwise, errors will be raised.

        X: np.ndarray of shape (m, n)
            m : number of examples,
            n : number of features

        params: np.ndarray of shape (p,)
            The parameters of the function.
            p: Number of parameters

        X_err: np.ndarray
            1. If X_err is a 1-D vector of shape (n,)
             Standard errors for features in X are specified.
             Errors are equal for all examples and are uncorrelated with each other.
            2. If X_err is a 2-D vector of shape (m, n)
                Standard errors are specified.
                These are different across samples but assumed to be uncorrelated across
                features.
            3. If X_err is a 2-D array of shape (n, n)
                Covariance matrix for errors are specified.
                Errors are equal for each example but those for different
                features might be correlated. The $i, j$ th element of matrix represents
                $cov(X_err_i, X_err_j)$ representing the covariance of the errors $i, j$ th feature of the data.
            4. If X_err is a tensor of shape (m, n, n)
                For each example, the nxn matrix denotes the covariance matrix of errors in X.

        X_err_denotes_feature_correlation: bool, optional, default: False
            In the rare case that m=n, X_err will computed according to
            case 2 unless this parameter is
            explicity set to True.

        params_err: np.ndarray of shape (p,) or (p, p)
            1. If a 1-D vector of shape (p,), treats it as standard errors for each parameter.
            It will be assumed that the various parameters in $\beta$ are mutually uncorrelated.
            2. If a 2-D vector of shape (p, p), it will be assumed that this matrix is
            the covariance matrix for the parameters $ \mathrm{cov}(\bm{\beta}) $.

        var_y: float64, default: None
            Estimate of the variance of the distribution of $y$
            at given $\bf{X}, \bm{beta}$.

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

            NOTE: var_y is used to compute prediction intervals.
            If it is not specified, prediction intervals will NOT be computed.
            Only confidence intervals will be computed.

            Confidence intervals ~ Interval of mean of the predicted value.
                    $$  \mathrm{Var}( \mathbb{E}(\hat{y} | \mathbf{X}, \bm{\beta})) $$
            Prediction intervals ~ Interval of the predicted value.
                    $$ \mathrm{Var}(\hat{y} | \bf{X}, \bm{\beta}) $$

            We know that
                    $$ \mathrm{Var}(\hat{y} | \bf{X}, \bm{\beta}) =
                    \mathrm{Var}( \mathbb{E}(\hat{y} | \mathbf{X}, \bm{\beta})) +
                     \sigma^2 $$
            So, we need some estimate of $\sigma^2$ to calculate prediction intervals.

        *args, **kwargs: Extra args and kwargs to be passed to the function.

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
        )

        # Ensure that the function meets the required demands by
        # computing y_hat. If it raises errors, we can simply stop here.
        y_hat = self._compute_y_hat(func, X, params, model_kwarg_dict)

        # Validate y_hat to make sure that it is a 1-D array
        self.__validate_y_hat(y_hat, X)

        # Calculate error propagation.
        SE_on_mean, SE_on_prediction = self._propagate_errors(
            func, X, params, X_err, params_err, model_kwarg_dict, sigma
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

        # Computes the required interval.
        lower_bound, upper_bound = self._compute_intervals(
            y_hat,
            se,
            side=side,
            confidence_level=confidence_level,
            lsa_assumption=lsa_assumption,
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
        assert isinstance(
            distribution, str
        ), "distribution not of type string. \
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
                # params_err is a pxp matrix specifying the covariance matrix of the parameters.
                return params_err
            else:
                raise DataDimensionalityWarning(
                    "params_err not in one of the required shapes. Please read the docs."
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
                    # if X_err is a 2-D matrix with (n,n) dimensions, it means that the errors are
                    # same across all examples, but the features are correlated and their
                    # covariance matrix is specified.
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
                # This means that there are different errors in X for different examples.
                # And assumed that the covariance matrix is specified.
                X_err_transformed = X_err
            else:
                # All other cases
                raise DataDimensionalityWarning(
                    "X_err is not correctly specified. Plesae refer to the docs for the correct specification"
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
        ), "Size of y_hat is different from that of X. Please make sure that they match."

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
        # So, we remove that information and only consider the elements along the diagonal.

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
            Computes the error based on the grad matrix and X_err matrix for each example.
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
        self, func, X, params, X_err, params_err, model_kwarg_dict, sigma
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
        SE_on_prediction = self.compute_SE_on_prediction(SE_on_mean, sigma)

        return SE_on_mean, SE_on_prediction

    def _compute_SE_on_mean(self, X_err, params_err, variances_X, variances_params):
        """computes the standard error of the mean prediction SE(E(y|X, \beta))"""

        # Add all variances and take a square root.
        if params_err is not None and X_err is None:  # Only error prop from params.
            SE_on_mean = np.sqrt(variances_params)
        elif params_err is None and X_err is not None:  # Only error prop from X.
            SE_on_mean = np.sqrt(variances_X)
        elif params_err is not None and X_err is not None:  # Both are present.
            SE_on_mean = np.sqrt(variances_params + variances_X)

        return SE_on_mean

    def compute_SE_on_prediction(self, SE_on_mean, sigma):
        """Uses SE_on_mean to get SE_on_prediciton."""

        if sigma is not None:
            SE_on_prediction = np.sqrt((SE_on_mean ** 2 + sigma ** 2))
        else:
            SE_on_prediction = None

        return SE_on_prediction

    def _compute_intervals(
        self,
        y_hat,
        se,
        side="two-sided",
        confidence_level=95.0,
        lsa_assumption=True,
        distribution="normal",
        dfe=None,
    ):
        """Computes the desired confidence / prediction intervals.

        Uses the y_hat, se, and other properties to compute the desired intervals.

        Parameters:
        ------------

        type: str, "confidence" or "prediction", default: "confidence"
            Defines the type of interval you wish to compute.
        side: str, "two-sided", "lower", "upper", default: "two-sided"
            Defined the type of interval to be computed.
        confidence_level: float, default: 90
            Percentage of the distribution to be enveloped by the interval.
            Example: A value of 90 means that you wish to compute a 90% interval.
        distribution: str, "normal" or "t", default: "normal"
            The type of distribution that the desired interval is from.
            For now, only normal and t distributions are supported.
            If you wish to compute an interval on a different distribution, please compute
            the appropriate interval externally.
        y_hat: None or np.ndarray of shape (m,), default: None
            The central value of the interval you wish to compute.
            If None, a value of y_hat will be computed for the function and its arguments
            supplied earlier.
            Alternatively, this can be externally computed and supplied here.
        se: None or float, default: None
            If None, will be computed from the function, its parameters,
            and errors supplied.
            If float, will be used to compute the interval.
        dfe: None or float, default: None
            Only used if distribution == "t" to supply the error degrees of freedom.

        Returns
        ----------
        list of 2 arrays each of shape (m,) containing the lower and upper bounds of the
        desired interval
        """

        # Workflow: Compute confidence intervals.
        # 1. Gets y_hat for the prediction.
        # 2. Gets SE for the appropriate interval based on type.
        # 3. Gets significance level.
        # 4. Gets appropriate stat based on distribution.
        # 5. Computes the desired confidence interval.

        # Gets significance level.
        significance_levels = get_significance_levels(confidence_level, side)

        # Currently, we only provide support for
        # normal and t-distributions.
        if distribution == "normal":
            stats = get_z_values(significance_levels)
        elif distribution == "t":
            stats = get_t_values(significance_levels, dfe)
        else:
            raise ValueError(
                "Currently intervals can only be obtained for normal\
                and t-distributions. In case your data come from a difference distribution,\
                Please compute the appropriate intervals externally using \
                y_hat and se."
            )

        # Finally, return desired interval.
        if side == "two-sided":
            return [y_hat + stats[0] * se, y_hat + stats[1] * se]
        if side == "upper":
            return [y_hat + stats[0] * se, np.repeat(np.inf, y_hat.shape[0])]
        if side == "lower":
            return [-np.repeat(np.inf, y_hat.shape[0]), y_hat + stats[1] * se]
