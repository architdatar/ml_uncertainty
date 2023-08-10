#!/usr/bin/env python'

"""Tests for `error_propagation` package"""

import autograd.numpy as np
import pandas as pd
from io import StringIO
from ml_uncertainty.error_propagation.error_propagation import ErrorPropagation
from .arrhenius_model import arrhenius_model, T_expt, fitted_params


def test_error_propagation():
    """Tests that errors can be propagated correctly."""

    T = T_expt.reshape((-1, 1))

    # Some parameters and their standard deviations.
    # For reaction from cyclopropane to propene:
    # https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Kinetics/06%3A_Modeling_Reaction_Kinetics/6.02%3A_Temperature_Dependence_of_Reaction_Rates/6.2.03%3A_The_Arrhenius_Law/6.2.3.01%3A_Arrhenius_Equation
    # Calculated from math shown in source. Ea: provided, A: calculated accordingly.
    # A: 1/s, Ea: J/mol
    best_fit_params = fitted_params

    # Let's assume the standard errors in these parameters to be 1% of their
    # value
    best_fit_err = best_fit_params * 1 / 100

    # Supply it an estimate of model RMSE (estimate of $\sigma$).
    sigma_hat = 0.0001

    # Computing the predictions and prediction intervals
    # Initialize the error propagation class
    eprop = ErrorPropagation()

    # Propagate errors
    df_int = eprop.get_intervals(
        arrhenius_model,
        T,
        best_fit_params,
        params_err=best_fit_err,
        sigma=sigma_hat,
        type_="prediction",
        side="two-sided",
        confidence_level=90.0,
        distribution="normal",
    )

    # Test that the output dataframe as expected shape and values.
    assert (
        df_int.shape[0] == T.shape[0]
    ), "Shape of the returned dataframe differs from the expected shape."

    # Assert that the errors found are correct.
    df_expected_string = ",mean,se,lower_bound,upper_bound\n\
    0,0.0001618449625364052,0.0001,-2.640400158742095e-06,0.0003263303252315524\n\
    1,0.0020106823707566135,0.0001,0.0018461970080614662,0.0021751677334517607\n\
    2,0.02733807030705305,0.0001,0.0271735849443579,0.027502555669748196\n\
    3,0.1970242774049084,0.0001,0.19685979204221327,0.19718876276760355\n"

    df_expected = pd.read_csv(StringIO(df_expected_string), index_col=0)

    pd.testing.assert_frame_equal(df_int, df_expected)
