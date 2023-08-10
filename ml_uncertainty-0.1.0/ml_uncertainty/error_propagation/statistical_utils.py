"""Contains statisical utils utilized by several classes"""
from scipy.stats import norm, t as t_dist
import numpy as np


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


def compute_intervals(
    mean,
    se,
    side="two-sided",
    confidence_level=95.0,
    distribution="normal",
    dfe=None,
):
    """Computes the desired intervals given the mean and standard error (SE) values.

    Uses the mean, SE, and other properties to compute the desired intervals.

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
            and t-distributions. In case your data come from a difference \
            distribution, please compute the appropriate intervals externally \
            using y_hat and se."
        )

    # Finally, return desired interval.
    if side == "two-sided":
        return [mean + stats[0] * se, mean + stats[1] * se]
    if side == "upper":
        return [mean + stats[0] * se, np.repeat(np.inf, mean.shape[0])]
    if side == "lower":
        return [-np.repeat(np.inf, mean.shape[0]), mean + stats[1] * se]
