"""General utils to be used across functions.
"""


def validate_bool(term, term_name=""):
    """Validates if it is bool."""
    assert type(term) == bool, f"{term_name} is not of type bool."


def validate_str(term, allowed_vals=None, term_name=""):
    """Validates string"""

    assert type(term) == str, f"{term_name} is not type string."

    if allowed_vals is not None:
        assert (
            term in allowed_vals
        ), f"{term_name} is not among allowed values:\
                            {allowed_vals}"


def validate_float(term, term_name=""):
    assert type(term) == float, f"{term_name} is not type float."
