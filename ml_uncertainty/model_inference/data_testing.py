"""Includes utilities to check which points come from a different distribution
 than training data.
Creates function to test assumptions on residuals and identify potential high leverage
and influence points.
"""

import numpy as np
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings


def id_outlier(point, df_train, opt_vars):
    """ """

    is_out = "No"
    out_for = []

    for var in opt_vars:
        lower_lim, upper_lim = np.quantile(df_train[var], [0.025, 0.975])
        val = point[var]

        if val < lower_lim or val > upper_lim:
            is_out = "Yes"
            out_for.append(var)

    return is_out, out_for


def id_outliers(df, df_train, opt_vars):
    """
    I/d if each point is an outlier or not.
    Get which are the points for which the model extrapolates.
    """

    is_out_list = []
    out_for_list = []

    df = df.copy(deep=True)

    for point_ind, point in df.iterrows():
        is_out, out_for = id_outlier(point, df_train, opt_vars)
        is_out_list.append(is_out)
        out_for_list.append(out_for)

    df["Is_out"] = is_out_list
    df["Out_for"] = out_for_list

    return df


def compute_hat_matrix(X, mode="OLS", lambda_=None, coef_=None):
    """
    Hat matrix is very important in any statistical analysis of regression.
    We compute it for here using two modes: assuming OLS / Ridge.
    For these two, we have a closed-form solution for the hat matrix.
    X: must of dimensions m x (p+1). Must be such that $X\beta = Y $.

    # Source: works of Hastie and Tibshirani. 
    # Another source.
    Source: https://search.r-project.org/CRAN/refmans/lmridge/html/hatr.html
    """

    if mode == "OLS":
        gram = np.matmul(X.T, X)
        gram_inv = np.linalg.inv(gram)
        l = np.matmul(X, gram_inv)
        H = np.matmul(l, X.T)
    elif mode == "L2":
        gram = np.matmul(X.T, X) + lambda_ * np.identity(X.shape[1])
        gram_inv = np.linalg.inv(gram)
        l = np.matmul(X, gram_inv)
        H = np.matmul(l, X.T)
    elif mode == "L1":
        # In line with the method for computing model dof,
        # we also compute hat matrix here only considering those
        # coefficients which are non-zero.
        if hasattr(coef_, "__iter__"):
            pass
        else:
            raise ValueError("mode is L1, but coef_ is not iterable.")

        # Convert coef_ to array.
        coef_ = np.array(coef_)

        # Check that the shape is correct.
        assert coef_.shape[0] == X.shape[1], "Shape of coef_ does not match X.shape[1]."

        # Apply the non-zero mask.
        non_zero_mask = coef_ != 0

        X = X[:, non_zero_mask]

        gram = np.matmul(X.T, X)
        gram_inv = np.linalg.inv(gram)
        l = np.matmul(X, gram_inv)
        H = np.matmul(l, X.T)

    # Eventually add support for non-linear regression. Intuitively, we should be
    # able to linearize it around beta_0.
    # Also, think about adding support for L1 regularization. We can exploit the idea
    # that some the coefficients get thrown
    # to 0. Thus, we could simply drop those columns.
    return H


def compute_standardized_residual(resid, sigma, hat_matrix_diag):
    """
    Computes standardized residual.
    """

    # Compute standardized residuals. Ideally, it should be studentized, but for
    # performance reasons, we use standardized residuals.
    # This is okay since those are used in R as well.
    resid_standardized = resid / (sigma * hat_matrix_diag)
    return resid_standardized


def compute_cooks_distance(resid_standardized, hat_matrix_diag, model_dof):
    """
    Reference: https://online.stat.psu.edu/stat462/node/173/.
    Ideally, this should be computed for the studentized residual, but we do it here for
    standardized. The values should be taken as indicative.
    """
    cooks_distance = (
        1
        / (model_dof + 1)
        * resid_standardized ** 2
        * (hat_matrix_diag / (1 - hat_matrix_diag))
    )
    return cooks_distance


class residualAnalysis:
    """Create a class to analyze residuals of the model.
    Example:
    --------
    >> training_data = pipeline_before_model.transform(df_train[numerical_variables])
    >> training_data = np.hstack([np.ones((training_data.shape[0], 1)), training_data])
    >> l2_reg = 0.5 * model.alpha * (1. - model.l1_ratio) # Computed for Elastic net.
    >> ra = residualAnalysis(df_train[target_variable].values,
        df_train[target_variable_pred].values, target_variable, target_variable_pred,
        training_data, inf_model=inf, l2_regularization=l2_reg)

    >> ra.create_analysis()

    """

    def __init__(
        self,
        target_vals,
        target_pred_vals,
        target_variable="target_var",
        target_variable_pred="target_var_pred",
        training_data=None,
        inf_model=None,
        l1_regularization=None,
        coef_=None,
        l2_regularization=None,
        model_dof=None,
        sigma=None,
    ):
        """

        training_data:
            Training data must be array
        inf_model:
            Model inference object
        """

        self.target_vals = target_vals
        self.target_pred_vals = target_pred_vals
        self.target_variable = target_variable
        self.target_variable_pred = target_variable_pred
        self.resid = self.target_vals - self.target_pred_vals

        # Try to set the values for model_dof and sigma.
        if inf_model is not None:
            self.model_dof = inf_model.model_dof
            self.sigma = inf_model.sigma
        else:
            self.model_dof = model_dof
            self.sigma = sigma

        # Inputs for hat matrix.
        self.X_train = training_data
        self.l2_reg = l2_regularization
        self.l1_reg = l1_regularization

        self.coef_ = coef_

    def get_hat_matrix(self):
        """
        """
        if self.X_train is not None:
            if self.l1_reg is None and self.l2_reg is None:
                mode = "OLS"
                lambda_ = None
                coef_ = None
            elif self.l1_reg is None and self.l2_reg is not None:
                mode = "L2"
                lambda_ = self.l2_reg
                coef_ = None
            elif self.l1_reg is not None and self.l2_reg is None:
                mode = "L1"
                lambda_ = None
                coef_ = self.coef_

            try:
                hat_matrix = compute_hat_matrix(
                    self.X_train, mode=mode, lambda_=lambda_, coef_=coef_
                )
            except Exception as e:
                hat_matrix = None
                warnings.warn(f"Unable to compute hat matrix: {e}")
        else:
            hat_matrix = None
            warnings.warn(f"Unable to compute hat matrix: No training data")

        return hat_matrix


    def create_analysis(self):
        """This is the orchestrator function which creates the full analysis."""

        nrows = 1
        ncols = 4
        fig_out, axes_list = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(7 * ncols, 6 * nrows)
        )
        fig_out.subplots_adjust(wspace=0.3)
        axes_list = axes_list.ravel()

        # Create the test_residual_independence.
        self.test_residual_independence(ax=axes_list[0])
        self.test_homoskedasticity(ax=axes_list[1])

        # Compute hat matrix.
        if self.X_train is not None:
            self.hat_matrix = self.get_hat_matrix()

            # We also check that sigma and model_dof are defined.
            # If these are defined, only then do these tests make sense.
            if (
                self.sigma is not None
                and self.model_dof is not None
                and self.hat_matrix is not None
            ):
                # Get the standardized residuals and compute the QQ plot.
                self.test_normality(ax=axes_list[2])

                # Get the leverage plot.
                self.visualize_leverage(ax=axes_list[3])

    def test_residual_independence(self, ax=None):
        """
        Tests the independence of residuals assumption.
        Returns a plot of residuals versus target variables along with the
        Durbin-Watson test statistic and p-value.
        """

        # Independence of residuals assumption.: Durbin-Watson test.
        # https://www.geeksforgeeks.org/statsmodels-durbin_watson-in-python/
        dw_stat = durbin_watson(self.resid)
        text_ = (
            f"Durbin-Watson statistic: {dw_stat:.3f} \n<=1.5 / >-2.5"
            + " => strong serial \nnegative / positive correlation."
        )

        if ax is None:
            ax = plt.gca()

        ax.scatter(self.target_vals, self.resid)
        ax.set_xlabel(self.target_variable)
        ax.set_ylabel(f"{self.target_variable}_resid")
        ax.text(
            0.05, 0.95, text_, ha="left", va="top", transform=ax.transAxes, fontsize=11
        )
        ax.set_title("Residual independence assumption", fontsize=14)

    def test_homoskedasticity(self, ax=None):
        """
        Test the homoskedasticity assumption
        """

        ax.scatter(self.target_pred_vals, self.resid)
        ax.set_xlabel(self.target_variable_pred)
        ax.set_ylabel(f"{self.target_variable}_resid")
        ax.set_title("Homoskedasticity assumption", fontsize=14)

    def test_normality(self, ax=None):
        """
        Make a QQ plot of standardized residuals and perform the Shapiro-Wilk test.
        """

        # Perform the Shapiro-Wilk test.
        # https://www.hec.usace.army.mil/confluence/sspdocs/ssptutorialsguides/r-based-statistics-tutorials/multiple-linear-regression-using-r/phase-4-checking-assumptions-using-r#:~:text=The%20Shapiro-Wilk%20test%20is%20a%20normality%20test%20based,and%20the%20shapiro.test%28%29%20function%20runs%20the%20Shapiro-Wilk%20test.
        stat, p_value = shapiro(self.resid)
        text_ = (
            f"Shapiro-Wilk: {stat:.3f}, p-value: {p_value:.3f}\n"
            + "Significant indicated departure from normality"
        )

        resid_standardized = compute_standardized_residual(
            self.resid, self.sigma, np.sqrt(1 - np.diag(self.hat_matrix))
        )

        if ax is None:
            ax = plt.gca()

        # Checking for normality: QQ plot and Shapiro-wilk test
        sm.qqplot(resid_standardized, line="45", ax=ax)  # fit=False,

        ax.text(
            0.05, 0.95, text_, ha="left", va="top", transform=ax.transAxes, fontsize=11
        )
        ax.set_title("Normality assumption", fontsize=14)

    def visualize_leverage(self, ax=None, outlier_threshold=3.0):
        """
        Here, we look for and understand high leverage and high influence points.
        """

        h_vals = np.diag(self.hat_matrix)
        resid_standardized = compute_standardized_residual(
            self.resid, self.sigma, np.sqrt(1 - np.diag(self.hat_matrix))
        )

        if ax is None:
            ax = plt.gca()

        ax.scatter(h_vals, resid_standardized)

        # Identify outliers in terms of residuals. Limits of 2 / 3 for standardized
        # residuals on the leverage plot. (JMP standard)
        ax.axhline(y=outlier_threshold, color="black", linestyle="--")
        ax.axhline(y=-outlier_threshold, color="black", linestyle="--")

        # Identify high leverage points. High leverage if
        # hii > 3 ((model_dof  + 1)/n) = 3(mean(h_ii)).
        # https://online.stat.psu.edu/stat462/node/171/.
        leverage_threshold = 3 * ((self.model_dof + 1) / self.X_train.shape[0])
        ax.axvline(x=leverage_threshold, color="black", linestyle="--")

        # After this, we set the xlim and ylims.
        ax.set_xlim()
        ax.set_ylim()

        # Also add cooks distance.
        # Reference: https://online.stat.psu.edu/stat462/node/173/
        # Cooks d for each point.
        cooks_d = compute_cooks_distance(resid_standardized, h_vals, self.model_dof)
        cooks_d_threshold = 0.5  # From reference.

        # Further plot the contours to indicate potentially influential points.
        XX, YY = np.meshgrid(
            np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50),
            np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 50),
        )

        ZZ = compute_cooks_distance(YY, XX, self.model_dof)

        ax.contour(XX, YY, ZZ, [0.5, 1], colors=["yellow", "red"])

        # Finally, we annotate the leverage plot to see high leverage, outliers,
        # and potenially high influence points.
        # I/d points which are either high leverage, outliers, or likely to be
        # influential.
        mask = (
            (h_vals > leverage_threshold)
            + (resid_standardized > outlier_threshold)
            + (resid_standardized < -outlier_threshold)
            + (cooks_d > cooks_d_threshold)
        )  # This does the OR operation
        interesting_indices = np.where(mask)[0].tolist()

        if len(interesting_indices) > 0:
            for ind in interesting_indices:
                # Annotate the points accordingly.
                ax.annotate(ind, (h_vals[ind], resid_standardized[ind]))

        ax.set_title("Leverage plot", fontsize=14)
        ax.set_xlabel("Leverage; $h_{ii}$")
        ax.set_ylabel("Standardized residual")
