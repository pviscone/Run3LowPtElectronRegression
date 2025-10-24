import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def train_curvefit_calibrators(out, Y_val, p, eta, p_edges, eta_edges):
    """
    Train per-(p,eta) bin linear calibration using curve_fit between
    true quantiles of Y_val and predicted quantiles.

    Parameters
    ----------
    out : (N, 5)
        Predicted quantiles (q05, q16, q50, q84, q95)
    Y_val : (N,)
        True target variable.
    p, eta : (N,)
        Kinematic variables for binning.
    p_edges, eta_edges : arrays
        Bin edges for p and eta.

    Returns
    -------
    calibrators : dict[(i, j)] -> (a, b)
        Linear fit parameters for y = a*x + b per bin.
    errors : dict[(i, j)] -> (σ_a, σ_b)
        1σ uncertainties from curve_fit covariance matrix.
    """
    calibrators = {}
    errors = {}

    quantile_levels = [0.05, 0.16, 0.5, 0.84, 0.95]

    for i in range(len(p_edges) - 1):
        for j in range(len(eta_edges) - 1):
            mask = (
                (p >= p_edges[i])
                & (p < p_edges[i + 1])
                & (eta >= eta_edges[j])
                & (eta < eta_edges[j + 1])
            )
            if np.sum(mask) < 30:
                continue

            y_bin = Y_val[mask]
            out_bin = out[mask]  # shape (n_bin, 5)

            # Compute true quantiles of Y_val in this bin
            q_true = np.quantile(y_bin, quantile_levels)

            # Median predicted quantile across events
            q_pred = np.median(out_bin, axis=0)

            # 1σ errors from predicted 84–16 quantile interval
            q_err = 0.5 * (
                np.percentile(out_bin, 84, axis=0) - np.percentile(out_bin, 16, axis=0)
            )

            # Define linear model
            def lin_func(x, a, b):
                return a * x + b

            try:
                popt, pcov = curve_fit(
                    lin_func,
                    q_true,
                    q_pred,
                    sigma=q_err,
                    absolute_sigma=True,
                    p0=[1.0, 0.0],
                )
                a, b = popt
                σ_a, σ_b = np.sqrt(np.diag(pcov))
                calibrators[(i, j)] = (a, b)
                errors[(i, j)] = (σ_a, σ_b)
            except Exception:
                continue

            fig, ax = plt.subplots()
            ax.errorbar(
                q_true,
                q_pred,
                yerr=q_err,
                fmt="o",
                label="Data",
            )
            x_fit = np.linspace(min(q_true), max(q_true), 100)
            ax.plot(
                x_fit,
                lin_func(x_fit, *popt),
                "r-",
                label="Fit: y = {:.2f}x + {:.2f}".format(a, b),
            )
            ax.plot(x_fit, x_fit, "k--", label="y = x", color="gray")
            ax.set_xlabel("Predicted Quantiles")
            ax.set_ylabel("True Quantiles")
            ax.set_title(
                f"Bin p: [{p_edges[i]}, {p_edges[i + 1]}], eta: [{eta_edges[j]}, {eta_edges[j + 1]}]"
            )
            ax.legend()
            os.makedirs("BDTQuantile/plots/calo/calibrator", exist_ok=True)
            fig.savefig(
                f"BDTQuantile/plots/calo/calibrator/calibrator_bin_p{i}_eta{j}.pdf"
            )
            plt.show()

    return calibrators, errors


def kernel_weighted_apply(out, p, eta, p_edges, eta_edges, calibrators,
                                              sigma_p=10.0, sigma_eta=0.5):
    """
    Vectorized kernel-weighted calibration with inverted linear fits:
    each bin's fit is y_pred = a * q_true + b, so calibration is
    q_cal = (q_pred - b) / a.

    Parameters
    ----------
    out : (N, n_q)
        Raw predicted quantiles.
    p, eta : (N,)
        Kinematics of events.
    p_edges, eta_edges : arrays
        Bin edges in p and eta.
    calibrators : dict[(i,j)] -> (a, b)
        Linear calibration parameters per bin.
    sigma_p, sigma_eta : float
        Gaussian kernel widths.

    Returns
    -------
    out_cal : (N, n_q)
        Calibrated quantile predictions.
    """
    out = np.asarray(out)
    p = np.asarray(p)
    eta = np.asarray(eta)
    p_edges = np.asarray(p_edges)
    eta_edges = np.asarray(eta_edges)

    p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])
    eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])

    N, n_q = out.shape
    n_p, n_eta = len(p_centers), len(eta_centers)

    # Build grids of inverted a, b
    a_grid = np.zeros((n_p, n_eta))
    b_grid = np.zeros((n_p, n_eta))
    for (i, j), (a, b) in calibrators.items():
        if a != 0:
            a_grid[i, j] = 1.0 / a  # invert slope
            b_grid[i, j] = -b / a   # invert intercept
        else:
            a_grid[i, j] = 1.0
            b_grid[i, j] = 0.0

    # Compute Gaussian weights
    dp = p[:, None] - p_centers[None, :]        # (N, n_p)
    deta = eta[:, None] - eta_centers[None, :]  # (N, n_eta)

    dp3 = dp[:, :, None]                         # (N, n_p, 1)
    deta3 = deta[:, None, :]                      # (N, 1, n_eta)

    w = np.exp(-0.5 * ((dp3 / sigma_p) ** 2 + (deta3 / sigma_eta) ** 2))  # (N, n_p, n_eta)
    w_flat = w.reshape(N, -1)  # (N, n_p*n_eta)

    a_flat = a_grid.flatten()  # (n_p*n_eta,)
    b_flat = b_grid.flatten()

    w_sum = w_flat.sum(axis=1, keepdims=True)
    zero_mask = (w_sum.squeeze() == 0)
    w_sum[zero_mask] = 1.0
    w_flat /= w_sum

    # Weighted blended a and b per event
    a_blend = w_flat @ a_flat  # (N,)
    b_blend = w_flat @ b_flat  # (N,)

    # Apply inverted calibration
    out_cal = a_blend[:, None] * out + b_blend[:, None]  # (N, n_q)

    return out_cal

"""
calibrators, errors = train_curvefit_calibrators(out, Y_val, p, eta, p_edges, eta_edges)

# 2. Apply kernel-weighted smooth calibration
out_calibrated = kernel_weighted_apply(
    out, p, eta, p_edges, eta_edges, calibrators, sigma_p=10.0, sigma_eta=0.3
)
"""