r"""
Develops utlities to get model inference for tree-based models in scikit-learn.

# TODO: Modify documentation.
"""

import numpy as np
from warnings import warn
import pandas as pd
from scipy.sparse import issparse
from sklearn.base import is_classifier
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble._forest import (
    _get_n_samples_bootstrap,
    _generate_unsampled_indices,
)
from ..error_propagation.statistical_utils import (
    get_significance_levels,
    get_z_values,
)
from .gen_utils import validate_str, validate_bool, validate_float
from copy import deepcopy


class EnsembleModelInference:
    """Provides utilities to perform inference for ensemble models.

    Examples:
    ---------
    examples/ensemble_model.py
    examples/random_forest_non-normal_distribution.py

    References:
    -----------
    1. Confidence interval calculations are performed using the answer
    provided by @Greg Snow in this Stackoverflow thread on the subject.
    https://stats.stackexchange.com/questions/56895/
    do-the-predictions-of-a-random-forest-model-have-a-prediction-interval
    2. Prediction intervals can be computed using two methods:
        "marginal": This is the 'OOB Prediction Interval' method proposed
        by Zhang et al. (2020).
            DOI: 10.1080/00031305.2019.1585288
        "individual": This is method proposed by @Greg Snow in Ref 1.
        The 'marginal' method is named here because it aggregates the
        predictions from the trees. On the other hand, 'individual' considers
        the distributions of the specific trees.

    """

    def __init__(self):
        """Initialize the function"""

    def set_up_model_inference(
        self,
        X_train,
        y_train,
        estimator,
        use_oob_pred=True,
        variance_type_to_use="marginal",
        copy_X_and_y=True,
        sigma=None,
        distribution="normal",
        distribution_kwargs={},
        stat_generating_function=None,
        stat_generating_function_kwargs={},
    ):
        """Sets up model inference and performs basic calculations.

        Validates estimators and functions, raises errors if anything
        is incorrect. Computes residual mean squared errors which are important
        to getting desired intervals.

        Parameters:
        -----------
        X_train: array of shape (n_training_examples, n_dimensions)
            Training data
        y_train: array of shape (n_training examples, )
            Training targets
        estimator : scikit-learn ensemble model of type RandomForestRegression,
            RandomForestClassification, etc.
            Must be fitted.
        use_oob_pred: bool, optional, default:True
            Whether to use out-of-bag predictions for train data to compute
            model mean squared errors.
        variance_type_to_use: str, optional, {"marginal", "individual"},
                                default="marginal"
            How to compute mean squared errors.
            "marginal": OOB Prediction intervals in Zhang et al. (Ref 2, class doc).
            "individual": Method proposed by Greg Snow (Ref 1, class docs).
        copy_X_and_y: bool, optional, default: True
            Whether to deepcopy X_train and y_train during analysis.
        sigma: {NoneType, float, array}, optional, default: None
            Allows user to externally specify root mean squared error (RMSE)
            for the fit.
            If NoneType: The code computes based on 'variance_type_to_use' argument.
            If float: Value interpreted as the RMSE of the fit.
            If array: Must have shape (n_estimators,), ith element corresponding
                to the RMSE of the ith tree.
        distribution: {NoneType, str, callable}, optional, default: "normal"
            Denotes assumed distribution of the residuals. Should be centered around 0.
            Ignored if variance_type_to_use=='marginal' and 'estimate_from_SD=False'
                in get_intervals.
            If NoneType:
                No distribution assumed. Only works if ignored.
            If str:
                Must be "normal". We provide functions internally to deal with
                normal distribution.
            If callable:
                Provide a function with signature identical to
                numpy.random.normal. See https://numpy.org/doc/stable/reference/random/
                generated/numpy.random.normal.html.
                Also, see ./examples/random_forest_non-normal_distribution.py
                Inputs: (loc, scale, size, **kwargs) and outputs: scalar or array
                scale and size should be able take and return numpy arrays.
        distribution_kwargs: dict, optional, default={}
            Key word arguments for the distribution function.
        stat_generating_function: {NoneType, callable}, optional, default: None
            Function to generate distribution stats for a list of quantile values.
            If NoneType:
                Constructed from 'distribution' using the
                self._stat_generator_function if distribution is not None.
            If callable:
                Must have signature (quantiles, **kwargs) -> list
                Where quantiles are a list and function must return a list of
                distribution statistics.
        stat_generating_function_kwargs: dict, optional, default: {}
            Key word arguments for the stat_generating_function.

        Notes:
        ------
        1.  Set 'use_oob_pred=True' only if there is a large number of trees
            ('n_estimators' > 100), boostrap sampling is used and proportion of samples
            used for each bootstrap is small (<0.8) to ensure that there are enough
            trees for which each sample is out-of-bag and a good distribution can
            be obtained.
        2. The "marginal" method yields prediction intervals for which the coverage
            is closer to the confidence level. So, it is used by default.
            See tests/benchmarking/ensemble_model_prediction_interval.py.
        2. This implementation works for multioutput regression; i.e.,
            y represents more than one target variable (shape (n_samples, n_targets)).
            However, this approach is not recommended. Instead, it is better to fit a
            separate random forest for each target and obtain inference accordingly.
            Reason: The RandomForestRegressor class outputs identical feature
            importances for all target variables leading to identical
            feature importance SD and intervals which is not ideal.
        """

        self.copy_X_and_y = copy_X_and_y
        self.sigma = sigma
        validate_bool(copy_X_and_y)

        self.X_train = self.__validate_and_transform_X(X_train, copy_X_and_y)
        self.y_train = self.__validate_y_and_transform_y(y_train, X_train, copy_X_and_y)

        # Ensure that the estimator is a tree.
        self._confirm_estimator_is_from_ensemble(estimator)
        # Check that the estimator is fitted.
        check_is_fitted(estimator)
        self.estimator = estimator

        validate_bool(use_oob_pred)
        self.use_oob_pred = use_oob_pred

        validate_str(variance_type_to_use, ["marginal", "individual"])
        self.variance_type_to_use = variance_type_to_use

        self.copy_X_and_y = copy_X_and_y

        self._validate_sigma(sigma)

        self._validate_and_assign_distribution_attributes(
            distribution,
            distribution_kwargs,
            stat_generating_function,
            stat_generating_function_kwargs,
        )

        # Further, validate if variance_type_to_use=="individual",
        # "distribution" cannot be None.
        if self.variance_type_to_use == "individual" and self.distribution is None:
            raise ValueError(
                "variance_type_to_use=='individual' \
                             requires that 'distribution' not be None.\
                             However, that value was passed."
            )

        # Validate the distribution and stat generation functions to make
        # sure that they run correctly. Pass a 3D numpy normal array to them and
        # make sure that they spit the right output.
        self.__validate_distribution(self.distribution, self.distribution_kwargs)
        self.__validate_stat_generating_function(
            self.stat_generating_function, self.stat_generating_function_kwargs
        )

        # Get the oob_pred and n_oob_pred arrays.
        train_ind_pred, train_ind_n_pred = self._generate_oob_array(
            X_train, estimator, is_train_data=self.use_oob_pred
        )

        # Compute means.
        train_means = np.nanmean(train_ind_pred, axis=1, keepdims=True)

        # If there are any nan values in train means, we raise a warning.
        if np.isnan(train_means).sum() > 0:
            warn(
                "Out-of-bag (OOB) estimate has been attempted but at least one sample \
                          hasn't been out-of-bag even once. This might lead to errors. \
                          Please reduce the 'max_samples' parameter in estimator or \
                          set 'use_oob_pred' to False."
            )

        if self.sigma is None:
            if self.variance_type_to_use == "marginal":
                # Compute RSS overall. (1, 1, n_outputs)
                marginal_dist = self.y_train - train_means
                MSE = np.nanmean(marginal_dist ** 2, axis=0, keepdims=True)
                std = np.sqrt(MSE)
                self.marginal_dist = marginal_dist

            elif self.variance_type_to_use == "individual":
                y_train_rep = np.tile(self.y_train, (1, self.estimator.n_estimators, 1))
                # Compute MSE by tree. (1, n_estimators, n_outputs)
                MSE = np.nanmean(
                    (train_ind_pred - y_train_rep) ** 2, axis=0, keepdims=True
                )
                std = np.sqrt(MSE)

            self.sigma = std

    def __validate_and_transform_X(self, X, copy_X):
        """Setting the right type of X."""

        if copy_X:
            X = deepcopy(X)

        if type(X) == pd.core.frame.DataFrame:
            X = X.values

        # If in case, ndim of X is 1, we reshape it.
        if X.ndim == 1:
            X = X.reshape((-1, 1))

        # Prediction requires X to be in CSR format
        if issparse(X):
            X = X.tocsr()

        assert X.ndim == 2, DataDimensionalityWarning(
            "X could not be converted to a 2D array"
        )

        return X

    def __validate_y_and_transform_y(self, y, X, copy_y):
        """ """

        if copy_y:
            y = deepcopy(y)

        if y.ndim == 1:
            y = y.reshape((-1, 1, 1))
        elif y.ndim == 2:
            y = y.reshape((y.shape[0], 1, y.shape[1]))
        else:
            raise ValueError("y is not of dimension 1 or 2 as required.")

        assert y.shape[0] == X.shape[0], "Shape of y different from corresponding X."

        return y

    def _confirm_estimator_is_from_ensemble(self, estimator):

        # Checks that the estimator is an ensemble model.
        module = getattr(estimator, "__module__")
        if module != "sklearn.ensemble._forest":
            raise TypeError(
                "Supplied estimator is not of type\
                            sklearn.ensemble._forest.\
                            Please ensure that it is of the right type."
            )

    def _stat_generator(self, significance_levels, **kwargs):
        """Stat generating function: Returns a list of stats given a list of
        significance levels."""

        stats = []
        for level in significance_levels:
            if level is None:
                stats.append(None)
            else:
                stat = np.quantile(self.distribution(size=int(1e6), **kwargs), level)
                stats.append(stat)

        return stats

    def _validate_sigma(self, sigma):
        # Validate sigma
        if sigma is not None:
            if type(sigma) == float:
                pass
            elif type(sigma) == np.ndarray:
                assert sigma.shape == (
                    self.estimator.n_estimators,
                ), f"Shape of sigma provided = {sigma.shape} does not equal the \
                        required shape of ({self.estimator.n_estimators},)."

    def _validate_and_assign_distribution_attributes(
        self,
        distribution,
        distribution_kwargs,
        stat_generating_function,
        stat_generating_function_kwargs,
    ):
        # If distribution is known, assign the correct parameters.
        if type(distribution) == str and distribution == "normal":
            self.distribution = np.random.normal
            self.distribution_kwargs = {}
            self.stat_generating_function = get_z_values
            self.stat_generating_function_kwargs = {}
        elif callable(distribution):
            self.distribution = distribution
            self.distribution_kwargs = distribution_kwargs
            if stat_generating_function is None:
                self.stat_generating_function = self._stat_generator
                self.stat_generating_function_kwargs = distribution_kwargs
            else:
                self.stat_generating_function = stat_generating_function
                self.stat_generating_function_kwargs = stat_generating_function_kwargs
        elif distribution is None:
            self.distribution = distribution
            self.distribution_kwargs = distribution_kwargs

            # In this case, stat_generating function must be None.
            # Else, raise error.
            assert (
                stat_generating_function is None
            ), "If distribution is None,\
                stat generating function must alse be None."

            self.stat_generating_function = stat_generating_function
            self.stat_generating_function_kwargs = stat_generating_function_kwargs
        else:
            raise ValueError(
                "'distribution' must either be None, \
                             function or 'normal', but some other value was passed."
            )

    def __validate_distribution(self, distribution, distribution_kwargs):
        """ """
        # Pass 3D array to scale parameter and make sure that it works.
        try:
            shape = (3, 4, 5)
            dist = distribution(
                size=shape, scale=np.random.random(size=shape), **distribution_kwargs
            )
            assert (
                dist.shape == shape
            ), "Distribution test failed. \
                Does not return the desired shape."
        except Exception:
            raise ValueError(
                "'distribution' validation failed. Please \
                             ensure that the size and scale parameter can take \
                             in 3D arrays like the numpy.random.normal function."
            )

    def __validate_stat_generating_function(
        self, stat_generating_function, stat_generating_function_kwargs
    ):
        """ """
        # Pass list to stat generating function and make sure that we get
        # a list of the same length.
        try:
            levels = [0.05, 0.95]
            stats = stat_generating_function(levels, **stat_generating_function_kwargs)

            assert len(stats) == len(
                levels
            ), "Stat generating function test failed.\
                Did not return the desired size."

        except Exception:
            raise ValueError(
                "Please ensure that the size and scale parameter can take\
                             in 3D arrays like the numpy.random.normal function."
            )

    def get_intervals(
        self,
        X,
        is_train_data=False,
        type_="prediction",
        estimate_from_SD=False,
        confidence_level=90.0,
        side="two-sided",
        lsa_assumption=True,
        return_full_distribution=False,
        copy_X=True,
    ):
        r"""
        Computes prediction / confidence intervals regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        is_train_data: bool, optional, default=False
            Indicates if the data is from the training set or not. If True,
            only out-of-bag estimates are used to compute the prediction
            interval.
        type_: str, optional, {"prediction", "confidence"}, default: "prediction"
            Type of the interval to be computed.
                Prediction: Refers to the spread of the predicted value.
                Confidence: Refers to the spread of the mean of the predicted values.
        estimate_from_SD: bool, optional, default: False
            How the interval is computed.
                If False, the appropriate quantiles are returned from the generated
                    distribution.
                If True, the parameters of the generated distribution such as
                    standard deviation (SD) are estimated and the appropriate
                    quantiles are returned.
        confidence_level: float, optional, default: 90.0
            Confidence level of the desired interval.
        side: str, optional, {"two-sided", "lower", "upper"}, default: "two-sided"
            Specifies the type of interval returned.
        lsa_assumption: bool, optional, default: True
            Whether large sample approximation can be considered to hold.
            Only used for confidence intervals if estimate_from_SD==True. Else ignored.
            Generally, valid if number of trees > 30.
            If False, confidence intervals to be estimated using the
            self.stat_generating_function.
        return_full_distribution: bool, optional, default=False
            Specifies whether to return some internal data during computation.
            If True, in addition to the normal output, returns:
                oob_pred:  array of dimensions (n_samples, n_estimators, n_outputs)
                    Provides prediction for each sample by each tree for each target.
                n_oob_pred: array of shape (n_samples, 1, n_outputs)
                    Provides counts for the number of trees for which each
                    sample is OOB.
                means_array: array of shape (n_samples, 1, n_outputs)
                    Mean values of the oob_pred over the estimators.
                std_array: array of shape (n_samples, 1, n_outputs)
                    Standard deviations of the predicted distributions.
        copy_X: bool, optional, default: True
            Whether to copy X so that the original will remain unchanged.

        Returns
        -------
        pred_int_list: List of dataframes of length (n_outputs).
            Each dataframe contains the mean, standard deviation, median,
            and the lower and upper bounds of the desired prediction / confidence
            intervals.
        """

        # Validate values.
        X = self.__validate_and_transform_X(X, copy_X)

        validate_bool(is_train_data, term_name="is_train_data")
        validate_str(
            type_, term_name="type_", allowed_vals=["prediction", "confidence"]
        )
        validate_bool(estimate_from_SD, term_name="estimate_from_SD")
        validate_float(confidence_level, term_name="confidence_level")
        validate_str(
            side, term_name="side", allowed_vals=["two-sided", "lower", "upper"]
        )
        validate_bool(lsa_assumption, term_name="lsa_assumption")
        validate_bool(return_full_distribution, term_name="return_full_distribution")
        validate_bool(copy_X, term_name="copy_X")

        estimator = self.estimator

        if is_classifier(estimator) and hasattr(estimator, "n_classes_"):
            raise ValueError(
                "Prediction intervals can only be computed for regressor models. \
                The model passed is a classifier."
            )

        # Validate others
        if (
            type_ == "confidence"
            and estimate_from_SD
            and not lsa_assumption
            and self.stat_generating_function is None
        ):
            raise ValueError(
                "For confidence intervals to be estimated using \
                             standard deviation, if LSA assumption does not hold,\
                             an appropriate 'stat generating function' must be\
                            supplied."
            )

        if (
            type_ == "prediction"
            and estimate_from_SD
            and self.stat_generating_function is None
        ):
            raise ValueError(
                "For prediction intervals to be computed \
                             standard deviation estimates, an appropriate \
                             'stat_generating_function' is necessary.\
                             Please provide this using 'set_up_model_inference' \
                             or set 'estimate_from_SD' to False."
            )

        # Get the oob_pred and n_oob_pred arrays.
        oob_pred, n_oob_pred = self._generate_oob_array(
            X, estimator, is_train_data=is_train_data
        )

        # Take the mean, median and std along the estimators axis.
        # Shape of these arrays
        # should be (n_samples x n_features)
        means_array = np.nanmean(oob_pred, axis=1, keepdims=True)
        median_array = np.nanmedian(oob_pred, axis=1, keepdims=True)
        std_array = np.nanstd(
            oob_pred, axis=1, ddof=1, keepdims=True
        )  # Sample std. dev.

        # Get significance levels and required percentiles.
        significance_levels = get_significance_levels(confidence_level, side)

        n_samples = X.shape[0]
        n_outputs = estimator.n_outputs_

        if type_ == "confidence":
            if not estimate_from_SD:
                interval_array = self._generate_interval_array_non_parametric(
                    side, significance_levels, n_samples, n_outputs, oob_pred
                )
            else:
                if lsa_assumption:
                    # Assume means to be normally distributed.
                    stat_generating_function = get_z_values
                    stat_generating_function_kwargs = {}
                else:
                    stat_generating_function = self.stat_generating_function
                    stat_generating_function_kwargs = (
                        self.stat_generating_function_kwargs
                    )

                interval_array = self._generate_interval_array_parametric(
                    side,
                    stat_generating_function,
                    stat_generating_function_kwargs,
                    significance_levels,
                    means_array,
                    std_array,
                    n_samples,
                    n_outputs,
                )
        elif type_ == "prediction":
            if self.variance_type_to_use == "marginal":
                if not estimate_from_SD:
                    # If marginal, simply use the distribution.
                    q1 = np.quantile(
                        self.marginal_dist,
                        significance_levels[0],
                        axis=0,
                        keepdims=True,
                    )
                    q2 = np.quantile(
                        self.marginal_dist,
                        significance_levels[1],
                        axis=0,
                        keepdims=True,
                    )

                    q1_rep = np.tile(q1, (n_samples, 1, 1))
                    q2_rep = np.tile(q2, (n_samples, 1, 1))

                    lower_bound = means_array + q1_rep
                    upper_bound = means_array + q2_rep

                    interval_array = np.concatenate((lower_bound, upper_bound), axis=1)
                else:
                    std_array = np.tile(self.sigma, (n_samples, 1, 1))

                    interval_array = self._generate_interval_array_parametric(
                        side,
                        self.stat_generating_function,
                        self.stat_generating_function_kwargs,
                        significance_levels,
                        means_array,
                        std_array,
                        n_samples,
                        n_outputs,
                    )

            elif self.variance_type_to_use == "individual":
                if not estimate_from_SD:
                    # Draw from distribution.
                    noise = np.concatenate(
                        [self.distribution(scale=self.sigma) for _ in range(n_samples)],
                        axis=0,
                    )

                    # Add this noise to the predictions from the trees.
                    pred = oob_pred + noise

                    # Now, get the required quantile from this distribution.
                    interval_array = self._generate_interval_array_non_parametric(
                        side, significance_levels, n_samples, n_outputs, pred
                    )

                else:
                    # sigma dims: (1, n_estimators, n_outputs)
                    # std array: (n_samples, 1, n_outputs)
                    mean_var = np.nanmean(self.sigma ** 2, axis=1, keepdims=True)
                    mean_var_rep = np.tile(mean_var, (n_samples, 1, 1))
                    std_array_pred = np.sqrt(std_array ** 2 + mean_var_rep)

                    interval_array = self._generate_interval_array_parametric(
                        side,
                        self.stat_generating_function,
                        self.stat_generating_function_kwargs,
                        significance_levels,
                        means_array,
                        std_array_pred,
                        n_samples,
                        n_outputs,
                    )

        return self._return_dfs(
            n_outputs,
            means_array,
            std_array,
            median_array,
            interval_array,
            return_full_distribution,
            oob_pred,
            n_oob_pred,
        )

    def _generate_oob_array(self, X, estimator, is_train_data):
        """
        Computes the oob_pred and n_oob_pred arrays.
        The required statistics are then drawn from this array.

        Source: https://github.com/scikit-learn/scikit-learn/blob/
        364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/ensemble/_forest.py#L530
            Inspired from the _compute_oob_predictions() function with the
            difference being that that function only provides the
            mean values of the predictions while we can provide prediction
            intervals.

        Returns
        -------
        oob_pred : ndarray of shape (n_samples, n_estimators, n_outputs)

        """

        # Specifying dimensions
        n_samples = X.shape[0]
        n_outputs = estimator.n_outputs_
        n_estimators = estimator.n_estimators

        oob_pred_shape = (n_samples, n_estimators, n_outputs)

        # Initializing oob_pred and n_oob_pred arrays
        oob_pred = np.tile(np.float64(np.nan), oob_pred_shape)
        n_oob_pred = np.zeros((n_samples, 1, n_outputs), dtype=int)

        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples,
            estimator.max_samples,
        )

        if is_train_data:
            for estimator_ind, estimator in enumerate(estimator.estimators_):
                unsampled_indices = _generate_unsampled_indices(
                    estimator.random_state,
                    n_samples,
                    n_samples_bootstrap,
                )

                # Should yield output dimensions of n_unsampled_indices x n_outputs
                y_pred = estimator.predict(X[unsampled_indices, :])

                # If y_pred is an array, reshape into a matrix. This can happend for
                # n_output = 1
                if y_pred.ndim == 1:
                    y_pred = y_pred.reshape((-1, 1))

                oob_pred[unsampled_indices, estimator_ind, :] = y_pred
                n_oob_pred[unsampled_indices, 0, :] += 1
        else:
            for estimator_ind, estimator in enumerate(estimator.estimators_):
                # Should yield output dimensions of n_samples x n_outputs
                y_pred = estimator.predict(X)

                # If y_pred is an array, reshape into a matrix. This can happend for
                # n_output = 1
                if y_pred.ndim == 1:
                    y_pred = y_pred.reshape((-1, 1))

                oob_pred[:, estimator_ind, :] = y_pred
                n_oob_pred[:, 0, :] += 1

        # Check if at least one the sample has not been out of the bag even once
        # in the bootstraps used.
        for k in range(n_outputs):
            if (n_oob_pred[..., k] == 0).any():
                warn(
                    "Some inputs do not have OOB scores. This probably means "
                    "too few trees were used to compute any reliable OOB "
                    "estimates.",
                    UserWarning,
                )

        # For samples that dont have any reported OOB values, sets their number to
        # n=1 to avoid problems while calculating means.
        # n_oob_pred[n_oob_pred == 0] = 1

        return oob_pred, n_oob_pred

    def _generate_interval_array_non_parametric(
        self, side, significance_levels, n_samples, n_outputs, oob_pred
    ):
        """ """

        if side == "two-sided":
            # Getting the array of required percentiles.
            # Shape: (n_samples, len(percentiles_required), n_features)
            percentiles = [np.round(sig * 100, 1) for sig in significance_levels]
            interval_array = np.nanpercentile(
                oob_pred, np.array(percentiles), axis=1
            ).swapaxes(0, 1)
        elif side == "upper":
            percentiles = [np.round(significance_levels[0], 1)]
            finite_val_array = np.nanpercentile(
                oob_pred, np.array(percentiles), axis=1
            ).swapaxes(0, 1)
            inf_array = np.tile(np.float64(np.inf), (n_samples, 1, n_outputs))
            interval_array = np.concatenate((finite_val_array, inf_array), axis=1)
        elif side == "lower":
            percentiles = [np.round(significance_levels[1], 1)]
            finite_val_array = np.nanpercentile(
                oob_pred, np.array(percentiles), axis=1
            ).swapaxes(0, 1)
            neg_inf_array = np.tile(np.float64(-np.inf), (n_samples, 1, n_outputs))
            interval_array = np.concatenate((neg_inf_array, finite_val_array), axis=1)

        return interval_array

    def _generate_interval_array_parametric(
        self,
        side,
        stat_generating_function,
        stat_generating_function_kwargs,
        significance_levels,
        means_array,
        std_array,
        n_samples,
        n_outputs,
    ):

        stats = stat_generating_function(
            significance_levels, **stat_generating_function_kwargs
        )
        if side == "two-sided":
            interval_array = np.concatenate(
                (
                    means_array + stats[0] * std_array,
                    means_array + stats[1] * std_array,
                ),
                axis=1,
            )
        elif side == "upper":
            finite_val_array = means_array + stats[0] * std_array
            inf_array = np.tile(np.float64(np.inf), (n_samples, 1, n_outputs))
            interval_array = np.concatenate((finite_val_array, inf_array), axis=1)
        elif side == "lower":
            finite_val_array = means_array + stats[1] * std_array
            neg_inf_array = np.tile(np.float64(-np.inf), (n_samples, 1, n_outputs))
            interval_array = np.concatenate((neg_inf_array, finite_val_array), axis=1)
        return interval_array

    def _return_dfs(
        self,
        n_outputs,
        means_array,
        std_array,
        median_array,
        interval_array,
        return_full_distribution,
        oob_pred=None,
        n_oob_pred=None,
    ):
        """Returns results for feture importance as well as prediction intervals."""
        # Output list of dictionaries.
        pred_int_list = []
        for output_ind in range(n_outputs):
            # Dictionary of the outputs which we will ultimately tranform
            # into a dataframe.
            pred_int_dict = {}
            pred_int_dict["mean"] = means_array[:, 0, output_ind]
            pred_int_dict["std"] = std_array[:, 0, output_ind]
            pred_int_dict["median"] = median_array[:, 0, output_ind]
            pred_int_dict["lower_bound"] = interval_array[:, 0, output_ind]
            pred_int_dict["upper_bound"] = interval_array[:, 1, output_ind]

            # Convert into dataframe
            pred_int_df = pd.DataFrame.from_dict(pred_int_dict)
            pred_int_list.append(pred_int_df)

        # Return the required things.
        if return_full_distribution:
            if oob_pred is None and n_oob_pred is None:
                return pred_int_list, means_array, std_array
            else:
                return pred_int_list, oob_pred, n_oob_pred, means_array, std_array
        else:
            return pred_int_list

    def get_feature_importance_intervals(
        self,
        confidence_level=90.0,
        side="two-sided",
        return_full_distribution=False,
    ):
        r"""
        Computes intervals for the spread of feature importances.

        Parameters
        ----------
        confidence_level: float, optional, default: 90.0
            Confidence level of the interval desired.
        side: str, optional, {"two-sided", "lower", "upper"}, default: "two-sided"
            Specifies if the interval desired is two-sided, upper or lower.
        return_full_distribution: bool, optional, default=False
            Specifies whether to return some internal data during computation.
            If True, in addition to the normal output, returns:
                means_array: array of shape (n_features, 1, n_outputs)
                    Mean values of the feature_importance over the estimators.
                std_array: array of shape (n_samples, 1, n_outputs)
                    Standard deviations of the feauture importances over the
                    estimators.

        Returns
        -------
        feature_imp_int_list: List of dataframes of length (n_outputs).
            Each dataframe contains the mean, standard deviation, median,
            and the lower and upper bounds of the desired feature importance interval.
        """

        # Validates inputs.
        validate_float(confidence_level, term_name="confidence_level")
        validate_str(
            side, term_name="side", allowed_vals=["two-sided", "lower", "upper"]
        )
        validate_bool(return_full_distribution, term_name="return_full_distribution")

        estimator = self.estimator

        (
            feature_imp_array,
            n_feature_imp_array,
        ) = self._generate_feature_importance_array(estimator)

        # Take the mean and std along the estimators axis. Shape of these arrays
        # should be (n_samples x n_features)
        means_array = np.nanmean(feature_imp_array, axis=1, keepdims=True)
        median_array = np.nanmedian(feature_imp_array, axis=1, keepdims=True)

        # Computes prediction standard deviation.
        std_array = np.nanstd(
            feature_imp_array, axis=1, ddof=1, keepdims=True
        )  # Sample std. dev.

        # Get significance levels and required percentiles.
        significance_levels = get_significance_levels(confidence_level, side)

        n_features = estimator.feature_importances_.shape[0]
        n_outputs = estimator.n_outputs_

        # Since we do not know anything about the distributions of the
        # feature importances, we do not assume any distribution and
        # show only the distribution quantiles.
        interval_array = self._generate_interval_array_non_parametric(
            side, significance_levels, n_features, n_outputs, feature_imp_array
        )

        return self._return_dfs(
            n_outputs,
            means_array,
            std_array,
            median_array,
            interval_array,
            return_full_distribution,
        )

        # # Output list of dictionaries.
        # feature_imp_int_list = []
        # for output_ind in range(n_outputs):
        #     # Dictionary of the outputs which we will ultimately tranform
        #     # into a dataframe.
        #     feature_imp_int_dict = {}
        #     feature_imp_int_dict["mean"] = means_array[:, 0, output_ind]
        #     feature_imp_int_dict["std"] = std_array[:, 0, output_ind]
        #     feature_imp_int_dict["median"] = median_array[:, 0, output_ind]
        #     feature_imp_int_dict["lower_bound"] = interval_array[:, 0, output_ind]
        #     feature_imp_int_dict["upper_bound"] = interval_array[:, 1, output_ind]

        #     # Convert into dataframe
        #     feature_imp_int_df = pd.DataFrame.from_dict(feature_imp_int_dict)

        #     feature_imp_int_list.append(feature_imp_int_df)

        # # Return the required things.
        # if return_full_distribution:
        #     return (
        #         feature_imp_int_list,
        #         feature_imp_array,
        #         n_feature_imp_array,
        #         means_array,
        #         std_array,
        #     )
        # else:
        #     return feature_imp_int_list

    def _generate_feature_importance_array(self, estimator):
        """Computes feature importance array."""

        n_features = estimator.feature_importances_.shape[0]
        n_estimators = estimator.n_estimators
        n_outputs = estimator.n_outputs_

        # For each tree, store the feature importances.
        feature_imp_shape = (n_features, n_estimators, n_outputs)

        feature_imp_array = np.tile(np.float64(np.nan), feature_imp_shape)
        n_feature_imp_array = np.zeros((n_features, 1, n_outputs), dtype=int)

        # Compute the array for each tree.
        for estimator_ind, estimator in enumerate(estimator.estimators_):
            # Should yield output dimensions of n_features x n_outputs
            tree_imp = estimator.feature_importances_

            # If y_pred is an array, reshape into a matrix. This can happend for
            # n_output = 1
            if tree_imp.ndim == 1:
                tree_imp = tree_imp.reshape((-1, 1))

            feature_imp_array[:, estimator_ind, :] = tree_imp
            n_feature_imp_array[:, 0, :] += 1

        return feature_imp_array, n_feature_imp_array
