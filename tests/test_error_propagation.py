#!/usr/bin/env python'

"""Tests for `error_propagation` package"""

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
    0,0.0001618449625364052,0.00012240067458761856,-3.9486231000346165e-05,0.00036317615607315645\n\
    1,0.0020106823707566135,0.0008322681801027813,0.0006417230361182521,0.0033796417053949743\n\
    2,0.02733807030705305,0.010521113280749972,0.010032378967644147,0.044643761646461944\n\
    3,0.1970242774049084,0.07193191139129122,0.07870681205939108,0.3153417427504257\n"
    df_expected = pd.read_csv(StringIO(df_expected_string), index_col=0)

    pd.testing.assert_frame_equal(df_int, df_expected)
