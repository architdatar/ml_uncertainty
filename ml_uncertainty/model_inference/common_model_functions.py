"""Provides commonly used functions for models.
"""
import numpy as np


def linear_model(X, coefs_):
    """
    Linear model without intercept.
    """
    return X @ coefs_


def linear_model_with_intercept(X, coefs_):
    """Linear model where first parameter is the intercept."""
    return coefs_[0] + X @ coefs_[1:]


def ordinary_residual(y_pred, y):
    return y_pred - y


def least_squares_loss(residuals, sample_weight=None):
    r"""
    For ordinary least squares (sample weight is None):
    $$ \mathcal{L}(r, W) = 1/2 r^Tr $$
    For weight least squares (samle weight is not None):
        It must be an array of the shape of residuals.
    $$ \mathcal{L}(r, W) = 1/2 r^TWr $$
    """
    if sample_weight is None:
        loss = 1 / 2 * residuals @ residuals
    else:
        W_mat = np.diag(sample_weight)
        loss = 1 / 2 * residuals @ W_mat @ residuals
    return loss


# More direct way of specifying functions. Can be directly
# defined later.


def linear_residual_function(X, coefs_, y, intercept_=0):
    y_pred = linear_model(X, coefs_, intercept_=intercept_)
    residuals = y_pred - y
    return residuals


def linear_loss_function(X, coefs_, y, intercept_=0):
    """ """
    residuals = linear_residual_function(X, coefs_, y, intercept_=intercept_)
    loss = 1 / (2 * residuals.shape[0]) * (residuals @ residuals)
    return loss


def elastic_net_loss_function(X, coefs_, y, intercept_=0, alpha=1, l1_ratio=0.5):
    """
    Warning: This function doesn't compute the L1
    norm because the error cannot be computed with it.
    """

    residuals = linear_residual_function(X, coefs_, y, intercept_=intercept_)
    loss = (
        1 / (2 * residuals.shape[0]) * (residuals @ residuals)
        + alpha * l1_ratio * np.linalg.norm(coefs_, ord=2)
        + 0.5 * alpha * (1 - l1_ratio) * np.linalg.norm(coefs_, ord=2) ** 2
    )

    return loss
