import sys
import os
import warnings
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import least_squares

sys.path.append(os.path.dirname(__file__))


class NonLinearRegression(RegressorMixin, BaseEstimator):
    """
    Currently only supports single output variable and fitting on a single processor.
    Assuming: each training sample has equal variance. 
    Only handles dense matrices. 
    Each instantiation can handle only one set of kwargs for the model. 
    """

    def __init__(self, 
        model,
        p0_length,
        model_kwargs_dict = {},
        least_sq_kwargs_dict = {}, #optional arguments for least sq
        normalize_x = False, 
        copy_X = True
        ):
        self.normalize_x = normalize_x
        self.copy_X = copy_X
        self.model  = model
        self.p0_length = p0_length
        self.model_kwargs_dict = model_kwargs_dict
        self.least_sq_kwargs_dict = least_sq_kwargs_dict


    def fit(self, X, y, p0 = None):
        """
        Fit the non-linear model. 
        """

        def _model_residuals(params, X, y):
            return y - self.model(X, params, **self.model_kwargs_dict)

        if p0 is None:
            p0 = np.repeat(1, self.p0_length)

        res_ls = least_squares(_model_residuals, x0=p0, 
             args=(X, y), 
            kwargs=self.model_kwargs_dict, 
            **self.least_sq_kwargs_dict)

        self.coef_ = res_ls.x
        self.jac = res_ls.jac
        self.fitted_lsq_object = res_ls
        self.dfe = res_ls.fun.shape[0] - res_ls.x.shape[0]
        self.RSS = res_ls.fun.T @ res_ls.fun / self.dfe

        try:
            self.get_parameter_errors()
        except:
            warnings.warn("Parameter errors could not be estimated. \
            Methods depending on these will not work.")
            pass
