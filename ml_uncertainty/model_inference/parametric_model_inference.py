#%%
import autograd.numpy as np
#import numpy as np
from autograd import grad, jacobian
from autograd import elementwise_grad as egrad
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
import sys
import os
import scipy
from scipy.optimize import least_squares
import sklearn
from sklearn.exceptions import NotFittedError, DataDimensionalityWarning

sys.path.append(os.path.dirname(__file__))

from ml_uncertainty.error_propagation.error_propagation import ErrorPropagation


# Defining some commonly used utils.

def linear_model(X, coefs_, intercept_=0):
    return X @ coefs_ + intercept_

def linear_residual_function(X, coefs_, y, intercept_=0):
    y_pred = linear_model(X, coefs_, intercept_=intercept_)
    residuals = y_pred - y
    return residuals

def linear_loss_function(X, coefs_, y, intercept_=0):
    """
    """
    residuals = linear_residual_function(X, coefs_, y, intercept_=intercept_)
    loss = 1 / (2 * residuals.shape[0]) * (residuals @ residuals)
    return loss

def elastic_net_loss_function(X, coefs_, y, intercept_=0, alpha=1, l1_ratio=0.5):
    """
    Warning: This function doesn't compute the L1
    norm because the error cannot be computed with it. 
    """

    residuals = linear_residual_function(X, coefs_, y, intercept_=intercept_)
    loss =  1 / (2 * residuals.shape[0]) * (residuals @ residuals) +\
            alpha * l1_ratio * np.linalg.norm(coefs_, ord=2) +\
            0.5  * alpha * (1-l1_ratio) * np.linalg.norm(coefs_, ord=2)**2

    return loss


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
        
    def get_model_inference(self,
                            X, 
                            y, 
                            y_weights=None,
                            estimator=None, 
                            model_function=None,
                            model_function_kwargs=None,
                            residual_function=None,
                            residual_function_kwargs=None,
                            loss_function=None,
                            loss_function_kwargs=None):
        """Gets model inference for the selected fitted estimator.
        
        
        """

        # Assigns variables to class attributes.        
        self.X = X
        self.y = y
        self.y_weights = y_weights
        self.estimator = estimator
        self.model_function = model_function
        self.model_function_kwargs = model_function_kwargs
        self.residual_function = residual_function
        self.residual_function_kwargs = residual_function_kwargs
        self.loss_function = loss_function
        self.loss_function_kwargs = loss_function_kwargs

        # Validate the inputs are right. 
        if self.y.ndim != 1:
            raise DataDimensionalityWarning("y should be a 1-D array.\
                                            Please provide appropriately.")

        # Validate that the estimators, etc. are correctly specified.
        self._validate_input_arguments()

        # Compute the Jacobian of the residuals.
        J = self.get_J(self.residual_function)
        H = self.get_H(self.loss_function)
        sigma = self.get_sig(self.residual_function) # Square root of RSS

        # Allows for inference from weighted least squares, ultimately allowing for
        # errors in Y.
        if self.y_weights is None:
            self.sigma = sigma

        self.H = H

        # Get variance-covariance matrix.
        self.vcov = self.get_vcov()
        self.sd_coef = self.get_sd_bf()


    def _validate_input_arguments(self):
        """Validates that the correct input arguments are provided.
        """

        if self.estimator is not None: # inputs to be extracted through estimator object.
            # Check that the estimator is fitted.    
            sklearn.utils.validation.check_is_fitted(self.estimator)

            # Get the Jacobian matrix for this model. 
            if type(self.estimator) == sklearn.linear_model._base.LinearRegression:
                self.model_function = linear_model
                self.residual_function = linear_residual_function
                self.loss_function = linear_loss_function

                self.model_function_kwargs = dict(intercept_=self.estimator.intercept_)
                self.residual_function_kwargs=dict()
                self.loss_function_kwargs=dict()

            elif type(self.estimator) == sklearn.linear_model._coordinate_descent.ElasticNet:
                self.model_function = linear_model
                self.residual_function = linear_residual_function
                self.loss_function = elastic_net_loss_function
                
                self.model_function_kwargs = dict(intercept_=self.estimator.intercept_)
                self.residual_function_kwargs = dict(intercept_=self.estimator.intercept_)
                self.loss_function_kwargs = dict(alpha=self.estimator.alpha,
                                l1_ratio=self.estimator.l1_ratio)

        else:
            # Check that all the required functions are provided. 
            if any([self.model_function is None, 
                    self.residual_function is None, 
                    self.loss_function is None]):
                raise NotImplementedError("One of model_function,\
                            residual_function, loss_function is not defined.\
                            Please provide all three.")
            
            # Further validate these functions by checking that they yield the
            # desired values in the desired formats. 


    def get_J(self, residual_function) -> np.ndarray:
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
        coefs_ = self.estimator.coef_
        J = jacobian(lambda coefs_: residual_function(self.X, coefs_, 
                     self.y, **self.residual_function_kwargs))(coefs_)
        return J


    def get_H(self, loss_function) -> np.ndarray:
        """Hessian of objective function.

        Hessian of the objective function is the Jacboian of the gradient
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
        coefs_ = self.estimator.coef_

        H = jacobian(
            egrad(lambda coefs_: loss_function(self.X, coefs_, 
                self.y, **self.loss_function_kwargs))
                )(coefs_)
        return H


    def get_sig(self, residual_function) -> float:
        """Standard deviation of the fit.

        Estimate standard deviation from the residual vector.

        Returns
        -------
        sig
            Estimated standard deviation.
        """
        coef_ = self.estimator.coef_
        m = self.X.shape[0]
        n = coef_.shape[0]
        res = residual_function(self.X, coef_, self.y, **self.residual_function_kwargs)
        sig = np.sqrt(np.matmul(res.transpose(), res) / (m - n))
        return sig


    def get_vcov(self) -> np.ndarray:
        """Variance-covariance matrix of parameters.
        
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
        sig = self.sigma
        H = self.H

        # Check if Hessian is invertible else raise error.
        try:
            Hinv = np.linalg.inv(H)
        except:
            raise DataDimensionalityWarning("The computed Hessian is not invertible.\
                    The variance-covariance matrix for the parameters cannot be\
                    computed.")
        
        vcov = (sig ** 2) * Hinv
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


    def error_propagation(self):
        """Initializes the ErrorPropagation class with the 
        info from this model and comptues required properties.
        """
        
        error_prop = ErrorPropagation(
                func=self.model_function,
                X=self.X,
                params=self.estimator.coef_,
                X_err=None,
                params_err=self.sd_coef,
                var_y=self.sigma**2,
                intercept_=self.estimator.intercept_,                
            )

        self.error_prop = error_prop
        self.SE_on_mean = error_prop.SE_on_mean
        self.SE_prediction = error_prop.SE_prediction

        # We simply assign the function of the error prop class here.
        self.compute_interval = error_prop.compute_interval


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import time

    from sklearn.datasets import make_regression
    from sklearn.linear_model._base import LinearRegression
    from sklearn.linear_model import ElasticNet


    pd.set_option('display.max_rows', 600)
    pd.set_option('display.expand_frame_repr', False)

    if sys.platform == 'win32':
        home = 'D:\\'
    else:
        home=os.path.expanduser('~')

    #plt.style.use(os.path.join(home, "mplstyles", "mypaper.mplstyle"))

    np.random.seed(1)


    # Create a test case for linear regression and test the inference with
    # the created class.
    X, y = make_regression(n_samples=20, n_features=2, n_informative=2, noise=1)

    #regr = LinearRegression()
    regr = ElasticNet(alpha=.010)

    regr.fit(X, y)

    regr.coef_

    
    inf = ParametricModelInference()
    inf.get_model_inference(X, y, estimator=regr)
    inf.error_propagation()

    y_hat = inf.error_prop.y_hat #Alternatively, can also be done using sklearn predict

    intervals = inf.compute_interval(type="prediction", 
                         side="two-sided", 
                         confidence_level=90,
                         distribution="t", 
                         y_hat=regr.predict(X),  
                         dfe=X.shape[0]-regr.coef_.shape[0],
                         )
    
    print(f"Hessian matrix: \n {inf.H}")
    print(f"Hessian matrix inverse: \n {np.linalg.inv(inf.H)}")
    print(f"Residual: \n {inf.sigma}")
    print(f"SD coef: \n {inf.sd_coef}")


#Make the regression summary table. Find existing methods 
#Write a file to include tests. Write one with complicated exponential models.
#or ODEs, log models. 
#Test 1: types of things
#Test 2: compare with statsmodels. 
#Make figures. 
# %%
