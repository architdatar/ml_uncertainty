"""
Defines functions for forward propagation of errors.
"""

#%%

from typing import Callable
import autograd.numpy as np
from autograd import jacobian
from sklearn.exceptions import DataDimensionalityWarning
from scipy.stats import norm, t as t_dist


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
        significance_levels = [(1 - confidence_level/100)/2 , 
                            1-(1 - confidence_level/100)/2]
    elif side == "upper":
        significance_levels = [1 - confidence_level/100, None]
    elif side == "lower":
        significance_levels = [None, confidence_level/100] 
    else:
        raise ValueError("Side should be one of \
            two-sided, lower, and upper. Please provided one of these.")

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
    """Computes t values
    """

    if not dfe:
        # dfe is not supplied.
        raise ValueError("dfe is not supplied\
                due to which t-stats cannot be computed.\
                If you wish to make a large sample approximation (LSA)\
                that your sample standard deviation is based on enough samples to\
                be close to the population standard deviation, you may consider using\
                the normal distribution instead.")
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

    def __init__(self, 
                func:Callable, 
                X,
                params, 
                *args,
                X_err=None,
                params_err=None,
                X_err_denotes_feature_correlation=False,
                var_y=None,
                **kwargs):
        """
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
            $$ RSS = \frac{r^Tr}{m-n} $$

            Alternatively, if there is some prior knowledge / intuition 
            about $ \sigma^2 $, it can also be used. 

            NOTE: var_y is used to compute prediction intervals. 
            If it is not specified, prediction intervals will be computed.
            Only confidence intervals will be computed.

            Confidence intervals ~ Interval of mean of the predicted value. 
                    $$  \mathrm{Var}( \mathbb{E}(\hat{y} | \mathbf{X}, \bm{\beta})) $$
            Prediction intervals ~ Interval of the predicted value. 
                    $$ \mathrm{Var}(\hat{y} | \bf{X}, \bm{\beta}) $$

        *args, **kwargs: Extra args and kwargs to be passed to the function.

        """

        self.func = func
        self.params = self._validate_params(params)
        self.X = self._validate_X(X)
        
        self.params_err = self._validate_and_transform_params_err(params_err)
        self.X_err = self._validate_and_transform_X_err(X_err)
        self.X_err_denotes_feature_correlation=X_err_denotes_feature_correlation

        self.var_y = var_y

        self.args = args
        self.kwargs = kwargs        

        # Ensure that the function meets the required demands by 
        # computing y_hat. If it raises errors, we can simply stop here.
        self.y_hat = self._compute_y_hat()

        # Check that at least one of params_err and X_err is not None
        if any([self.params_err is not None, self.X_err is not None]):
            pass
        else:
            raise Exception("At least one of params_err, X_err should be\
                defined for error propagation.")

        self.propagate_errors()


    @staticmethod
    def check_type_and_dtype(array):
        """Checks that the given object is a numpy array,
        determines it dtype, and if it is int, changes it to float.
        This avoids problems when using autograd functions. 
        """

        if type(array) == np.ndarray:
            if array.dtype.name == "float64": # This passes the test.
                return array
            
            else: #try to convert it to float 64, else give raise error.
                try:
                    array = array.astype("float64")
                    return array
                except:
                    raise ValueError("The array could not be converted to\
                        dtype 'float64'. Please try to convert it externally\
                        and ensure that the array.dtype == 'float64'. ")
        
        else:
            raise ValueError("The array is not of type ndarray. Please supply an\
                array of type ndarray")

    def _validate_params(self, params):
        """Checks that params is a 1-D array or can be cast as one.
        """

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

        if params_err is not None: # If it has been defined and is not None.
            params_err = self.check_type_and_dtype(params_err)
            p = params_err.shape[0]
            if params_err.ndim == 1 and p > 0:
                # checks that the params_err is a 1-D non-empty vector.
                # This means that these are the standard errors of the parameters. 
                return np.diag(params_err**2)
            
            elif params_err.ndim == 2 and params_err.shape == (p, p):
                # params_err is a pxp matrix specifying the covariance matrix of the parameters.  
                return params_err
            else:
                raise DataDimensionalityWarning(
                    "params_err not in one of the required shapes. Please read the docs."
                )
        else:
            return params_err


    def _validate_and_transform_X_err(self, X_err):
        """
        Check and validate X_err
        """

        if X_err is not None: # If X_err is not None. 
            X_err = self.check_type_and_dtype(X_err)
            m, n = self.X.shape

            if X_err.ndim == 1 and X_err.shape[0]==n:
                # if X_err is a 1-D vector of shape (n,), it means 
                # error are the same for all examples and uncorrelated with each other.
                X_err_transformed = np.tile(np.diag(X_err**2), (m,1,1))
            elif X_err.ndim ==2:
                    if X_err.shape == (m, n):
                        X_err_transformed = np.apply_along_axis(np.diag, axis=1, arr=X_err**2)                  
                        
                    elif X_err.shape == (n, n):
                        # if X_err is a 2-D matrix with (n,n) dimensions, it means that the errors are 
                        # same across all examples, but the features are correlated and their 
                        # covariance matrix is specified. 
                        X_err_transformed = np.tile(X_err, (m, 1, 1))

                    if X_err.shape(n, n) and m == n \
                            and self.X_err_denotes_feature_correlation:
                        # Force the program to treat the matrix as (n,n) rather than (m,n).
                        # In this case, assume that the covariance matrix is specified.
                        X_err_transformed = np.tile(X_err, (m, 1, 1))
            elif X_err.ndim == 3 and X_err.shape == (m, n, n):
                # This means that there are different errors in X for different examples. 
                # And assumed that the covariance matrix is specified.
                X_err_transformed = X_err
            else:
                # All other cases
                raise DataDimensionalityWarning("X_err is not correctly specified. Plesae refer to the docs for the correct specification")

            return X_err_transformed

        else: # Simple store it as a None object. 
            return X_err


    def get_grad_matrix_for_params(self, func=None, X=None, 
                    params=None, *args, **kwargs):

        if func is None:
            func = self.func
        
        if X is None:
            X = self.X

        if params is None:
            params = self.params

        if args == ():
            args = self.args

        if kwargs == {}:
            kwargs = self.kwargs

        grad_matrix = jacobian(
                lambda params: func(X, params, *args, **kwargs))(params)

        m = X.shape[0]
        p = params.shape[0]

        assert grad_matrix.shape==(m,p), DataDimensionalityWarning(
                        "Grad matrix for params is of an unexpected dimension.")

        return grad_matrix


    def get_grad_matrix_for_X(self, func=None, X=None,
                params=None, *args, **kwargs):
        """
        """

        if func is None:
            func = self.func
        
        if X is None:
            X = self.X

        if params is None:
            params = self.params

        if args == ():
            args = self.args

        if kwargs == {}:
            kwargs = self.kwargs

        m, n = X.shape
        p = params.shape[0]

        mat = jacobian(
                lambda X: func(X, params, *args, **kwargs))(X)

        # Upon taking the Jacobian,
        # We get a matrix of the shape (m, m, n). This also shows the correlations 
        # between various samples. As of now, i dont know how to multiply such a matrix
        # or handle the information about the sample-sample correlation. 
        # So, we remove that information and only consider the elements along the diagonal. 
        
        grad_matrix = np.diagonal(mat, axis1=0, axis2=1).T # grad matrix of shape mxn

        assert grad_matrix.shape==(m,n), DataDimensionalityWarning(
                        "Grad matrix for X is of an unexpected dimension.")

        return grad_matrix


    def get_variances_based_on_params(self, 
        grad_matrix_for_params=None, 
        params_err=None):
        """Computes $ \mathrm{var}(E(y|X,\beta)) $ based on $ \beta $ alone.
        """
  
        if grad_matrix_for_params is None:
            grad_matrix_for_params = self.grad_matrix_for_params
        
        if params_err is None:
            params_err = self.params_err

        m = self.X.shape[0]

        # Multiply the two.
        mult = grad_matrix_for_params @ params_err @ grad_matrix_for_params.T

        # Take the diagonal elements
        variances_params = np.diagonal(mult) # This should have a shape of (m, )

        assert variances_params.shape == (m,), DataDimensionalityWarning(
            "Variances based on params have the wrong dimensions.")

        return variances_params


    def get_variances_based_on_X(self,
            grad_matrix_for_X=None,
            X_err=None):
        
        """Computes the $ \mathrm{var}(E(y|X,\beta)) $ based on $ X $ alone.
        """
        
        if grad_matrix_for_X is None:
            grad_matrix_for_X = self.grad_matrix_for_X
        
        if X_err is None:
            X_err = self.X_err

        X = self.X
        m = X.shape[0]


        def compute_y_err_for_x(i):
            """
            Computes the error based on the grad matrix and X_err matrix for each example.
            """

            # grad_X_example: 1xn, X_err_example: nxn, grad_X_example.T: nx1
            # total value: 1x1
                        
            grad_X_example = grad_matrix_for_X[i, :].reshape(1, -1) # 1xn dimensions
            X_err_example = X_err[i, ...]

            output = grad_X_example @ X_err_example @ grad_X_example.T

            assert output.shape == (1,1), DataDimensionalityWarning("Variance based on X: wrong shape.")
                    
            return output[0][0]

        vfunc = np.vectorize(compute_y_err_for_x)
        variances_X = vfunc([i for i in range(m)])

        return np.array(variances_X)


    def compute_SE_on_mean(self):
        """computes the standard error of the mean prediction SE(E(y|X, \beta))
        """

        # Add all variances and take a square root.
        if self.params_err is not None and self.X_err is None: #Only error prop from params.
            SE_on_mean = np.sqrt(self.variances_params)
        elif self.params_err is None and self.X_err is not None: # Only error prop from X.
            SE_on_mean = np.sqrt(self.variances_X)
        elif self.params_err is not None and self.X_err is not None: # Both are present.
            SE_on_mean = np.sqrt(self.variances_params + self.variances_X)

        self.SE_on_mean = SE_on_mean


    def compute_SE_on_prediction(self):
        """Uses SE_on_mean to get SE_on_prediciton.
        """
        
        if self.var_y is not None:
            self.SE_prediction = np.sqrt((self.SE_on_mean**2 + self.var_y))
        else:
            self.SE_prediction = None


    def propagate_errors(self):
        """This function will perform the computations to compute 
        the required properties which can be accessed as required.
        """

        if self.params_err is not None:
            self.grad_matrix_for_params = self.get_grad_matrix_for_params()
            self.variances_params = self.get_variances_based_on_params()

        if self.X_err is not None:
            self.grad_matrix_for_X = self.get_grad_matrix_for_X()
            self.variances_X = self.get_variances_based_on_X()

        # Compute the SE on mean.
        self.compute_SE_on_mean()

        # Get prediction intervals.
        self.compute_SE_on_prediction()


    def _compute_y_hat(self):
        """Computes y_hat for the X, params, args and kwargs provided. 
        """

        # Get the desired y_hat for the prediction.
        y_hat = self.func(self.X, self.params, *self.args, **self.kwargs)
        return y_hat


    def __validate_supplied_y_hat(self, y_hat):
        """Checks that the supplied y_hat is right.
        """

        # Ensure y_hat has the correct dimension. 
        assert y_hat.ndim == 1, "y_hat is not a 1-D vector. Please supply a 1-D vector."
        assert y_hat.shape[0] == self.X.shape[0], "Size of y_hat is different from that of X. Please make sure that they match."


    def compute_interval(self,
                type="confidence", 
                side="two-sided", 
                confidence_level=95.,
                distribution="normal",
                y_hat=None,
                se=None,
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

        # Compute y_hat
        if y_hat is None:
            y_hat = self.y_hat
        else:
            self.__validate_supplied_y_hat(y_hat)

        if not se:
            # Gets SE for the appropriate interval type.
            if type=="confidence":
                se = self.SE_on_mean
            if type=="prediction":
                if self.SE_prediction is None:
                    raise ValueError("The prediction error has not been computed.\
                            This is likely due to the fact the var_y parameter was\
                            not supplied. Please supply this parameter and recompute.\
                            Or, please not request for prediction intervals.")
                
                se = self.SE_prediction
        
        # Gets significance level. 
        significance_levels = get_significance_levels(confidence_level, side)
    
        # Currently, we only provide support for 
        # normal and t-distributions.
        if distribution=="normal":
            stats = get_z_values(significance_levels)
        elif distribution=="t":
            stats = get_t_values(significance_levels, dfe)
        else:
            raise ValueError("Currently intervals can only be obtained for normal\
                and t-distributions. In case your data come from a difference distribution,\
                Please compute the appropriate intervals externally using \
                y_hat and se.")

        # Finally, return desired interval.
        if side == "two-sided":
            return [y_hat + stats[0] * se, y_hat + stats[1] * se]
        if side == "upper":
            return [y_hat + stats[0] * se, np.repeat(np.inf, y_hat.shape[0])]
        if side == "lower":
            return [-np.repeat(np.inf, y_hat.shape[0]), y_hat + stats[1] * se]
        


if __name__ == "__main__":

    # Calculating grad matrix for predictions
    def multivariate_linear(X, coefs_, intercept=1):
        return X @ coefs_ + intercept

    def quadratic(X, coefs_, intercept=1):
        return X**2 @ coefs_ + intercept

    #X = np.array([[0., 1.], [1., 0.], [0., 0.]])
    X = np.array([[2.], [3.], [6]])

    params = np.array([1.])

    params_err = np.array([2])

    # Trying our the class.
    error_prop = ErrorPropagation(
            func=quadratic,
            X=X,
            params=params,
            X_err=None,
            params_err=params_err,
            intercept=2
            )

    error_prop.propagate_errors()

    assert error_prop.SE_on_mean.ndim == 1, "SE on mean dimension is wrong"
    
    #assert error_prop.SE_on_mean == np.array([[1.]]), "Wrong value for multivariate linear regression"
    confidence_interval = error_prop.compute_interval()

# %%
