"""Contains statisical utils utilized by several classes"""
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
