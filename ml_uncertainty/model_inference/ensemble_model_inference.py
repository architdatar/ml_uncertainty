r"""
Develops utlities to get model inference for tree-based models in scikit-learn.

# TODO: Create tests for model significance. Basically: is it better than 
    the null model? 
        1. Look at non-parametric course textbook and see which tests can
        be used for this. Non-parametric version of F-test. 
        2. Degrees of freedom of non-parametric models. The MC algorithm for 
        this has been discussed in DOI: 10.1080/01621459.1998.10474094. Full paper
        can be accessed from virtual library -> Reprints Desk. 
        Algorithm to do so is mentioned in Pg 122, algorithm 1
            1. Basic idea of the method is to create t perturbations in Y and 
            measure the efects on $\hat{y}$. 
        Results can be compared with URL: https://arxiv.org/pdf/1911.00190.pdf
        Pg 11 and 12. 
"""

# Imports
from scipy.sparse import issparse
from sklearn.base import is_classifier
import numpy as np
from sklearn.ensemble._forest import (
    _get_n_samples_bootstrap,
    _generate_unsampled_indices,
)
from warnings import warn
import pandas as pd
from sklearn.exceptions import DataDimensionalityWarning
from ..error_propagation.error_propagation import (
    get_significance_levels,
    get_z_values,
)


class EnsembleModelInference:
    """Provided utilities to perform inference for ensemble models."""

    def __init__(self):
        """Initialize the function"""

    def _confirm_estimator_is_from_ensemble(self, estimator):

        # Checks that the estimator is an ensemble model.
        module = getattr(estimator, "__module__")
        if module != "sklearn.ensemble._forest":
            raise TypeError(
                "Supplied estimator is not of type\
                            sklearn.ensemble._forest.\
                            Please ensure that it is of the right type."
            )

    def _generate_feature_importance_array(self, estimator):
        """Computes feature importance array."""

        n_features = estimator.feature_importances_.shape[0]
        n_estimators = estimator.n_estimators
        n_outputs = estimator.n_outputs_

        # For each tree, store the feature importances.
        feature_imp_shape = (n_features, n_estimators, n_outputs)

        feature_imp_array = np.tile(np.float64(np.nan), feature_imp_shape)
        n_feature_imp_array = np.zeros((n_features, 1, n_outputs), dtype=np.int64)

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

    def get_feature_importance_intervals(
        self,
        estimator,
        type="prediction",
        distribution="non-parametric",
        lsa_assumption=True,
        confidence_level=90.0,
        side="two-sided",
        return_full_distribution=False,
    ):
        r"""
        Computes intervals for the notional spread of the feature importance intervals.

        Parameters
        ----------
        estimator : scikit-learn ensemble model (fitted)
            Model using which we wish to compute prediction intervals.
            Must be fitted to data. Of type RandomForestRegression,
            RandomForestClassification, etc.
        type: str, {"prediction", "confidence"}, default: "prediction"
            Type of the interval to be computed.
                Prediction: Refers to the spread of the predicted value. $SE(\hat{\beta})$
                Confidence: Refers to the spread of the mean of the predicted values. $SE(E(\hat{\beta}))$
        distribution: str, {"non-parametric", "parametric", "default"}, default: "default"
            Distribution of the interval to be computed.
                If non-parametric, appropriate percentile values are returned.
                If parametric, normal distribution is assumed (as of this version).
                    And appropriate values depending on confidence level are returned.
        lsa_assumption: bool, default=True
            If distribution of the interval is considered "parametric",
            specified is large sample approximation (LSA) can be used to compute the
            mean and standard deviation.
            Note: Under LSA, if the number of samples is large enough (typically >=30),
            the mean of a sample is assumed to be normally
             distributed irrespective of the distribution of the sample itself.
             See docs for futher details.$
        confidence_level: float, default: 90.0
            Confidence level of the interval desired.
        side: str, {"two-sided", "lower", "upper"}, default: "two-sided"
            Specifies if the interval desired is 2-sided, upper or lower.
        return_full_distribution: bool, default=False
            Set to True only if special statistics which are not provided here are required.
            If True, returns the full distribution of the predicted data.
            Returns oob_pred, an array of dimensions
                (n_samples, n_estimators, n_outputs).
            And n_oob_pred, an array that counts the number of non-nan
            values for each sample in each estimator and each output value.
            For example: If parmetric prediction interval is required with a
                t-distribution, Poisson distribution, etc., instead of normal
                distribution. In this case, set to True, use the oob_pred and
                n_oob_pred arrays returned and externally compute required statistics.

        Returns
        -------
        feature_imp_int_list: List of dataframes with each dataframe with containing the mean,
            standard deviation, median, and the desired prediction /
            confidence intervals.
        feature_imp_array, n_feature_imp_array: Only if return_full_distribution is True
            Arrays of shape (n_features, n_estimators, n_outputs) and
            (n_features, 1, n_outputs), respectively.
            feature_imp_array provides predictions by each tree for each sample at each variable.
            n_feature_imp_array tracks the number of non-nan values predicted by each estimator at
            each output.
        """

        # Ensure that the estimator is a tree.
        self._confirm_estimator_is_from_ensemble(estimator)

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

        if type == "prediction":
            if distribution in ["default", "non-parametric"]:
                interval_array = self._generate_interval_array_non_parametric(
                    side, significance_levels, n_features, n_outputs, feature_imp_array
                )
            elif distribution == "parametric":
                interval_array = self._generate_interval_array_parametric(
                    side,
                    lsa_assumption,
                    significance_levels,
                    means_array,
                    std_array,
                    n_features,
                    n_outputs,
                )
        elif type == "confidence":
            # Compute standard error of mean of the features.
            # NOTE: We use large sample approximation by default since we have a
            # large number of samples. So, standard error of mean = \sigma / \sqrt(n)
            std_array /= np.sqrt(n_feature_imp_array)

            # Get the interval array for parametric and normal distribution.
            if distribution in ["default", "parametric"]:
                distribution == "parametric"
                interval_array = self._generate_interval_array_parametric(
                    side,
                    lsa_assumption,
                    significance_levels,
                    means_array,
                    std_array,
                    n_features,
                    n_outputs,
                )
            else:
                raise NotImplementedError(
                    "'distribution' value provided cannot be handled. \
                        For confidence intervals, please provide \
                        distribution value as 'parametric' or None. \
                        In either case, we assume that the mean value \
                        will be normally distributed. Non-parametric \
                        estimates of means aren't implemented as of this\
                        version. Please use the distributions \
                        to compute those values externaly."
                )
        else:
            raise ValueError(
                "'type' must either be 'prediction' or 'confidence'.\
                             Please provide the appropriate value for 'type'."
            )

        # Output list of dictionaries.
        feature_imp_int_list = []
        for output_ind in range(n_outputs):
            # Dictionary of the outputs which we will ultimately tranform
            # into a dataframe.
            feature_imp_int_dict = {}
            feature_imp_int_dict["mean"] = means_array[:, 0, output_ind]
            feature_imp_int_dict["std"] = std_array[:, 0, output_ind]
            feature_imp_int_dict["median"] = median_array[:, 0, output_ind]
            feature_imp_int_dict["lower_bound"] = interval_array[:, 0, output_ind]
            feature_imp_int_dict["upper_bound"] = interval_array[:, 1, output_ind]

            # Convert into dataframe
            feature_imp_int_df = pd.DataFrame.from_dict(feature_imp_int_dict)

            feature_imp_int_list.append(feature_imp_int_df)

        # Return the required things.
        if return_full_distribution:
            return (
                feature_imp_int_list,
                feature_imp_array,
                n_feature_imp_array,
                means_array,
                std_array,
            )
        else:
            return feature_imp_int_list

    def __validate_and_transform_X(self, X):
        """Setting the right type of X."""

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

    def _generate_oob_array(self, X, estimator, is_train_data):
        """
        Computes the oob_pred and n_oob_pred arrays.
        The required statistics are then drawn from this array.

        Source: https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/ensemble/_forest.py#L530
                inspired from the _compute_oob_predictions() function with the
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
        n_oob_pred = np.zeros((n_samples, 1, n_outputs), dtype=np.int32)

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
        lsa_assumption,
        significance_levels,
        means_array,
        std_array,
        n_samples,
        n_outputs,
    ):

        if lsa_assumption:  # Large-sample approximation is true.
            z_stats = get_z_values(significance_levels)
            if side == "two-sided":
                interval_array = np.concatenate(
                    (
                        means_array + z_stats[0] * std_array,
                        means_array + z_stats[1] * std_array,
                    ),
                    axis=1,
                )
            elif side == "upper":
                finite_val_array = means_array + z_stats[0] * std_array
                inf_array = np.tile(np.float64(np.inf), (n_samples, 1, n_outputs))
                interval_array = np.concatenate((finite_val_array, inf_array), axis=1)
            elif side == "lower":
                finite_val_array = means_array + z_stats[1] * std_array
                neg_inf_array = np.tile(np.float64(-np.inf), (n_samples, 1, n_outputs))
                interval_array = np.concatenate(
                    (neg_inf_array, finite_val_array), axis=1
                )
            return interval_array

        else:  # No LSA assumption
            raise NotImplementedError(
                "You have chosen to compute \
                    prediction intervals assuming a non-normal \
                    parametric model. Prediction intervals for \
                    such models haven't been implemented. \
                    Please get the means_array and std_arrays using \
                    the 'return_all_data' argument and compute the \
                    intervals by yourself."
            )

    def get_intervals(
        self,
        X,
        estimator,
        is_train_data=False,
        type="prediction",
        distribution="non-parametric",
        lsa_assumption=True,
        confidence_level=90.0,
        side="two-sided",
        return_full_distribution=False,
    ):
        r"""
        Computes intervals for the predictions of the ensemble model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.
        estimator : scikit-learn ensemble model (fitted)
            Model using which we wish to compute prediction intervals.
            Must be fitted to data. Of type RandomForestRegression, RandomForestClassification, etc.
        is_train_data: bool, default=False
            Indicates if the data is from the training set or not. If True,
            only out-of-bag samples are used to compute the prediction
            interval.
        type: str, {"prediction", "confidence"}, default: "prediction"
            Type of the interval to be computed.
                Prediction: Refers to the spread of the predicted value. $SE(\hat{y})$
                Confidence: Refers to the spread of the mean of the predicted values. $SE(E(\hat{y}))$
        distribution: str, {"non-parametric", "parametric", "default"}, default: "default"
            Distribution of the interval to be computed.
                If non-parametric, appropriate percentile values are returned.
                If parametric, normal distribution is assumed (as of this version).
                    And appropriate values depending on confidence level are returned.
        lsa_assumption: bool, default=True
            If distribution of the interval is considered "parametric",
            specified is large sample approximation (LSA) can be used to compute the
            mean and standard deviation.
            Note: Under LSA, if the number of samples is large enough (typically >=30),
            the mean of a sample is assumed to be normally
             distributed irrespective of the distribution of the sample itself.
             See docs for futher details.$
        confidence_level: float, default: 90.0
            Confidence level of the interval desired.
        side: str, {"two-sided", "lower", "upper"}, default: "two-sided"
            Specifies if the interval desired is 2-sided, upper or lower.
        return_full_distribution: bool, default=False
            Set to True only if special statistics which are not provided here are required.
            If True, returns the full distribution of the predicted data.
            Returns oob_pred, an array of dimensions
                (n_samples, n_estimators, n_outputs).
            And n_oob_pred, an array that counts the number of non-nan
            values for each sample in each estimator and each output value.
            For example: If parmetric prediction interval is required with a
                t-distribution, Poisson distribution, etc., instead of normal
                distribution. In this case, set to True, use the oob_pred and
                n_oob_pred arrays returned and externally compute required statistics.

        Returns
        -------
        pred_int_list: List of dataframes with each dataframe containing the mean,
            standard deviation, median, and the desired prediction
            / confidence intervals.

        oob_pred, n_oob_pred: Only if return_full_distribution is True
            Arrays of shape (n_samples, n_estimators, n_outputs) and
            (n_samples, 1, n_outputs), respectively.
            oob_pred provides predictions by each tree for each sample at each variable.
            n_oob_pred tracks the number of non-nan values predicted by each estimator at
            each output.
        """

        X = self.__validate_and_transform_X(X)

        self._confirm_estimator_is_from_ensemble(estimator)

        if is_classifier(estimator) and hasattr(estimator, "n_classes_"):
            # only limited to regresor model, so we raise an error.
            raise ValueError(
                "Function only defined for random forest regressor while the model \
                passed is a classifier"
            )

        # Get the oob_pred and n_oob_pred arrays.
        oob_pred, n_oob_pred = self._generate_oob_array(
            X, estimator, is_train_data=is_train_data
        )

        # Take the mean and std along the estimators axis. Shape of these arrays
        # should be (n_samples x n_features)
        means_array = np.nanmean(oob_pred, axis=1, keepdims=True)
        median_array = np.nanmedian(oob_pred, axis=1, keepdims=True)

        # Computes prediction standard deviation.
        std_array = np.nanstd(
            oob_pred, axis=1, ddof=1, keepdims=True
        )  # Sample std. dev.

        # Get significance levels and required percentiles.
        significance_levels = get_significance_levels(confidence_level, side)

        n_samples = X.shape[0]
        n_outputs = estimator.n_outputs_

        if type == "prediction":
            if distribution in ["default", "non-parametric"]:
                interval_array = self._generate_interval_array_non_parametric(
                    side, significance_levels, n_samples, n_outputs, oob_pred
                )
            elif distribution == "parametric":
                interval_array = self._generate_interval_array_parametric(
                    side,
                    lsa_assumption,
                    significance_levels,
                    means_array,
                    std_array,
                    n_samples,
                    n_outputs,
                )
        elif type == "confidence":
            # Compute standard error of mean
            # NOTE: We use large sample approximation by default since we have a
            # large number of samples. So, standard error of mean = \sigma / \sqrt(n)
            std_array /= np.sqrt(n_oob_pred)

            # Get the interval array for parametric and normal distribution.
            if distribution in ["default", "parametric"]:
                distribution == "parametric"
                interval_array = self._generate_interval_array_parametric(
                    side,
                    lsa_assumption,
                    significance_levels,
                    means_array,
                    std_array,
                    n_samples,
                    n_outputs,
                )
            else:
                raise NotImplementedError(
                    "'distribution' value provided is not in 'default' or 'parametric'. \
                        For confidence intervals, please provide \
                        distribution value as 'parametric' or 'default'. \
                        Non-parametric estimates of means aren't implemented as of \
                        this version. If required, please use the distributions \
                        to compute those values externaly."
                )
        else:
            raise ValueError(
                "'type' must either be 'prediction' or 'confidence'.\
                             Please provide the appropriate value for 'type'."
            )

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
            return pred_int_list, oob_pred, n_oob_pred, means_array, std_array
        else:
            return pred_int_list
