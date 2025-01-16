"""Describes the arrhenius model as a function which is useful
in several tests.
"""

# Regular numpy doesn't work for error propagation since there are
# issues with respect to taking a derivative.
import autograd.numpy as np


def arrhenius_model(T, coefs_):
    r"""Arrhenius model function

    $$ k = Ae^{-Ea/RT}$$

    Parameters:
    -----------
    T: np.ndarray of dimension 1
        Temperatures in $\degree C$
    coefs_: np.ndarray of shape (2,)
        Denotes [A, Ea].
        A (units: same as those of rate constant k),
        Ea (units: J/mol).

    Returns:
    --------
    np.ndarray of dimension 1.
        Corresponding to k values for each T value.
    """

    X = T[:, 0] + 273  # transforming to K

    R = 8.314  # J/mol/K
    A, Ea = coefs_

    k = A * np.exp(-Ea / (R * X))

    k = k.reshape((-1,))

    return k


# For reaction from cyclopropane to propene T, k, and best fit parameter
# values are shown below.
# https://chem.libretexts.org/Bookshelves/
# Physical_and_Theoretical_Chemistry_Textbook_Maps/
# Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/
# Kinetics/06%3A_Modeling_Reaction_Kinetics/
# 6.02%3A_Temperature_Dependence_of_Reaction_Rates/
# 6.2.03%3A_The_Arrhenius_Law/6.2.3.01%3A_Arrhenius_Equation

# Temperature values ($\degree C$)
T_expt = np.array([477, 523, 577, 623])

# Rate constant (k) values in 1/s
k_expt = np.array([0.0018, 0.0027, 0.030, 0.26])

# Calculated from math shown in source. Ea: provided, A: calculated accordingly.
# A: 1/s, Ea: J/mol
fitted_params = np.array([1.39406453358858e15, 271.867e3])
