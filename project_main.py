"""
Gaussian Process Regression (GPR) - Synthetic 1D example

This script:
    - generates synthetic data from y = sin(x) + Gaussian noise
    - fits a Gaussian Process Regressor (RBF + WhiteKernel) using scikit-learn
    - visualises:
        * GP prior samples (before seeing any data)
        * GP posterior mean and 95% confidence intervals
        * GP posterior samples (functions drawn from the posterior)
    - evaluates predictive performance (RMSE, CI coverage)
    - studies the effect of length_scale and noise_level hyperparameters

Mathematical model:

    Prior on latent function f(x):
        f(x) ~ GP(m(x), k(x, x'))  with m(x) = 0 and RBF kernel k.

    Given training data (X, y) with additive Gaussian noise:
        y = f(X) + ε,   ε ~ N(0, σ_n^2 I)

    Joint distribution of training targets y and test function values f_*:

        [ y ] ~ N( 0, [ K(X, X) + σ_n^2 I      K(X, X_*)     ] )
        [f_*]       [     K(X_*, X)           K(X_*, X_*)   ]

    Posterior of f_* | X, y, X_* is Gaussian with:

        mean:      μ_* = K(X_*, X) [K(X, X) + σ_n^2 I]^{-1} y
        covariance: Σ_* = K(X_*, X_*) −
                          K(X_*, X) [K(X, X) + σ_n^2 I]^{-1} K(X, X_*)

    scikit-learn's GaussianProcessRegressor computes μ_* and Σ_*
    when calling:
        gpr.fit(X, y)
        gpr.predict(X_*, return_std=True)       # diag(Σ_*)
        gpr.predict(X_*, return_cov=True)       # full Σ_*
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

RANDOM_SEED = 42
rng = np.random.RandomState(RANDOM_SEED)


# --------------------------------------------------------------------
# 1. Data generation
# --------------------------------------------------------------------
def generate_sine_data(n_train=30, noise_std=0.1, x_min=0.0, x_max=10.0):
    """
    Generate 1D training data from y = sin(x) + Gaussian noise.

    Returns:
        X_train: shape (n_train, 1)
        y_train: shape (n_train,)
    """
    X = np.linspace(x_min, x_max, n_train)
    y = np.sin(X) + rng.normal(0.0, noise_std, size=n_train)
    return X.reshape(-1, 1), y


# --------------------------------------------------------------------
# 2. Fit GPR model (with optional hyperparameter optimisation)
# --------------------------------------------------------------------
def fit_gpr_model(X_train,
                  y_train,
                  length_scale=1.0,
                  noise_level=0.1,
                  normalise_inputs=True,
                  optimize_hyperparams=True):
    """
    Fit a GaussianProcessRegressor with RBF + WhiteKernel.

    Mathematical meaning of the kernel choice:

        k_total(x, x') = k_RBF(x, x') + σ_n^2 δ(x, x')

    where k_RBF is the squared exponential (RBF) kernel

        k_RBF(x, x') = σ_f^2 * exp( - ||x - x'||^2 / (2 * ℓ^2) )

    and WhiteKernel represents observation noise with variance σ_n^2.

    After calling gpr.fit(X_train_scaled, y_train), the GP posterior
    is defined by:

        μ_* = K_*^T (K + σ_n^2 I)^{-1} y
        Σ_* = K_** − K_*^T (K + σ_n^2 I)^{-1} K_*

    where:
        K     = K(X_train, X_train)
        K_*   = K(X_train, X_*)
        K_**  = K(X_*, X_*)
    """
    if normalise_inputs:
        x_scaler = StandardScaler()
        X_train_scaled = x_scaler.fit_transform(X_train)
    else:
        x_scaler = None
        X_train_scaled = X_train

    # Kernel: RBF for smoothness + WhiteKernel for observation noise
    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)

    if optimize_hyperparams:
        # Default behaviour: hyperparameters are optimised by maximising
        # the log-marginal likelihood.
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
            random_state=RANDOM_SEED
        )
    else:
        # Disable optimisation: use exactly the length_scale and noise_level
        # given above (important for hyperparameter effect plots).
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
            random_state=RANDOM_SEED,
            optimizer=None
        )

    gpr.fit(X_train_scaled, y_train)
    return gpr, x_scaler


# --------------------------------------------------------------------
# 3. Posterior predictions and uncertainty
# --------------------------------------------------------------------
def gpr_posterior_predictions(gpr, x_scaler, x_min=0.0, x_max=10.0, n_test=500):
    """
    Compute posterior mean and standard deviation on a dense test grid.

    After fitting, the call:

        y_mean, y_std = gpr.predict(X_test_scaled, return_std=True)

    returns:
        y_mean ≈ μ_*            (posterior mean)
        y_std  ≈ sqrt(diag(Σ_*)) (square root of the diagonal of posterior covariance)
    """
    X_test = np.linspace(x_min, x_max, n_test).reshape(-1, 1)
    if x_scaler is not None:
        X_test_scaled = x_scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    y_mean, y_std = gpr.predict(X_test_scaled, return_std=True)
    y_true = np.sin(X_test).ravel()

    return X_test, y_true, y_mean, y_std


def plot_posterior_with_ci(X_train, y_train, X_test, y_mean, y_std,
                           title="GPR posterior mean and 95% CI"):
    """
    Plot training data, posterior mean, and 95% confidence intervals.

    The 95% CI is approximated by:
        μ_* ± 1.96 * σ_*
    """
    ci_factor = 1.96
    y_lower = y_mean - ci_factor * y_std
    y_upper = y_mean + ci_factor * y_std

    plt.figure(figsize=(8, 4))
    plt.scatter(X_train, y_train, c="k", s=30, label="Training data")
    plt.plot(X_test, y_mean, label="Posterior mean")
    plt.fill_between(X_test.ravel(), y_lower, y_upper, alpha=0.3,
                     label="95% confidence interval")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_predictions(y_true, y_mean, y_std):
    """
    Compute RMSE and empirical coverage of the 95% confidence intervals.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_mean))

    ci_factor = 1.96
    y_lower = y_mean - ci_factor * y_std
    y_upper = y_mean + ci_factor * y_std
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))

    return rmse, coverage


# --------------------------------------------------------------------
# 4. Visualisation of GP prior and posterior samples
# --------------------------------------------------------------------
def plot_gp_prior_samples(length_scale=1.0,
                          x_min=0.0,
                          x_max=10.0,
                          n_test=200,
                          n_samples=5):
    """
    Draw samples from the GP PRIOR:

        f(X_*) ~ N(0, K(X_*, X_*))

    where K is given by the RBF kernel with a fixed length_scale.
    """
    X_test = np.linspace(x_min, x_max, n_test).reshape(-1, 1)
    kernel_prior = RBF(length_scale=length_scale)

    # Covariance matrix K(X_*, X_*)
    K_xx = kernel_prior(X_test, X_test)

    # Add small jitter for numerical stability
    jitter = 1e-8
    K_xx = K_xx + jitter * np.eye(n_test)

    # Cholesky decomposition: K_xx = L L^T
    L = np.linalg.cholesky(K_xx)

    # Draw samples: f_prior = L @ z, where z ~ N(0, I)
    z = rng.normal(size=(n_test, n_samples))
    f_samples = L @ z  # shape (n_test, n_samples)

    plt.figure(figsize=(8, 4))
    for i in range(n_samples):
        plt.plot(X_test, f_samples[:, i], lw=1.5, alpha=0.8)

    plt.title(f"GP prior samples (RBF length_scale = {length_scale})")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.tight_layout()
    plt.show()


def plot_gp_posterior_samples(gpr,
                              x_scaler,
                              X_train,
                              y_train,
                              x_min=0.0,
                              x_max=10.0,
                              n_test=200,
                              n_samples=5):
    """
    Draw samples from the GP POSTERIOR over the targets at test points X_*.

    We use:
        y_mean, y_cov = gpr.predict(X_*, return_cov=True)
        f_post = μ_* + L z,   z ~ N(0, I),   with Σ_* = L L^T.
    """
    X_test = np.linspace(x_min, x_max, n_test).reshape(-1, 1)
    if x_scaler is not None:
        X_test_scaled = x_scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    y_mean, y_cov = gpr.predict(X_test_scaled, return_cov=True)

    # Ensure symmetry and numerical stability
    y_cov = (y_cov + y_cov.T) / 2.0
    jitter = 1e-8
    y_cov = y_cov + jitter * np.eye(n_test)

    # Cholesky decomposition of posterior covariance
    L = np.linalg.cholesky(y_cov)

    # Draw posterior samples: μ_* + L z
    z = rng.normal(size=(n_test, n_samples))
    f_samples = y_mean.reshape(-1, 1) + L @ z

    # Posterior std and 95% CI
    y_std = np.sqrt(np.diag(y_cov))
    ci_factor = 1.96
    y_lower = y_mean - ci_factor * y_std
    y_upper = y_mean + ci_factor * y_std

    plt.figure(figsize=(8, 4))
    # Training data
    plt.scatter(X_train, y_train, c="k", s=30, label="Training data")

    # Posterior mean and 95% CI
    plt.plot(X_test, y_mean, "C0", lw=2, label="Posterior mean")
    plt.fill_between(X_test.ravel(), y_lower, y_upper, color="C0", alpha=0.2,
                     label="95% CI")

    # Posterior samples
    for i in range(n_samples):
        plt.plot(X_test, f_samples[:, i], lw=1.0, alpha=0.7, linestyle="--")

    plt.title("GP posterior samples with mean and 95% CI")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# 5. Study hyperparameters: length_scale and noise_level
# --------------------------------------------------------------------
def study_length_scale_effect(length_scales, n_train=30, noise_std=0.1):
    """
    Visual study of the effect of the RBF length_scale on the posterior.

    Here we DISABLE hyperparameter optimisation so that each length_scale
    is really used as specified.
    """
    X_train, y_train = generate_sine_data(n_train=n_train, noise_std=noise_std)
    x_min, x_max = X_train.min(), X_train.max()

    plt.figure(figsize=(9, 3 * len(length_scales)))

    for i, l in enumerate(length_scales, start=1):
        gpr, x_scaler = fit_gpr_model(
            X_train,
            y_train,
            length_scale=l,
            noise_level=noise_std,
            normalise_inputs=True,
            optimize_hyperparams=False   # <--- FIXED length_scale here
        )
        X_test, y_true, y_mean, y_std = gpr_posterior_predictions(
            gpr, x_scaler, x_min=x_min, x_max=x_max
        )

        ci_factor = 1.96
        y_lower = y_mean - ci_factor * y_std
        y_upper = y_mean + ci_factor * y_std

        plt.subplot(len(length_scales), 1, i)
        plt.scatter(X_train, y_train, c="k", s=20, label="Train data")
        plt.plot(X_test, y_mean, label="Posterior mean")
        plt.fill_between(X_test.ravel(), y_lower, y_upper, alpha=0.3,
                         label="95% CI")
        plt.title(f"Effect of RBF length_scale = {l:.3f}")
        if i == len(length_scales):
            plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def study_noise_level_effect(noise_levels,
                             n_train=30,
                             noise_std=0.1,
                             length_scale=1.0):
    """
    Visual study of the effect of the WhiteKernel noise_level on the posterior.

    Here we DISABLE hyperparameter optimisation so that each noise_level
    is really used as specified.
    """
    X_train, y_train = generate_sine_data(n_train=n_train, noise_std=noise_std)
    x_min, x_max = X_train.min(), X_train.max()

    plt.figure(figsize=(9, 3 * len(noise_levels)))

    for i, sigma2 in enumerate(noise_levels, start=1):
        gpr, x_scaler = fit_gpr_model(
            X_train,
            y_train,
            length_scale=length_scale,
            noise_level=sigma2,
            normalise_inputs=True,
            optimize_hyperparams=False   # <--- FIXED noise_level here
        )
        X_test, y_true, y_mean, y_std = gpr_posterior_predictions(
            gpr, x_scaler, x_min=x_min, x_max=x_max
        )

        ci_factor = 1.96
        y_lower = y_mean - ci_factor * y_std
        y_upper = y_mean + ci_factor * y_std

        plt.subplot(len(noise_levels), 1, i)
        plt.scatter(X_train, y_train, c="k", s=20, label="Train data")
        plt.plot(X_test, y_mean, label="Posterior mean")
        plt.fill_between(X_test.ravel(), y_lower, y_upper, alpha=0.3,
                         label="95% CI")
        plt.title(f"Effect of noise_level = {sigma2:.3f}")
        if i == len(noise_levels):
            plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# 6. Main routine
# --------------------------------------------------------------------
def main():
    # 6.1 PRIOR samples (no data yet)
    plot_gp_prior_samples(length_scale=1.0,
                          x_min=0.0,
                          x_max=10.0,
                          n_test=200,
                          n_samples=5)

    # 6.2 Generate training data and fit baseline GPR (with optimisation ON)
    X_train, y_train = generate_sine_data(n_train=30, noise_std=0.15)
    print(f"Training data: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")

    gpr, x_scaler = fit_gpr_model(
        X_train,
        y_train,
        length_scale=1.0,
        noise_level=0.1,
        normalise_inputs=True,
        optimize_hyperparams=True    # <--- optimisation ON here
    )

    print("Optimised kernel hyperparameters after training:")
    print(gpr.kernel_)
    lml = gpr.log_marginal_likelihood()
    print(f"Log-marginal likelihood: {lml:.3f}")

    # 6.3 Posterior mean, std, CI and evaluation
    X_test, y_true, y_mean, y_std = gpr_posterior_predictions(gpr, x_scaler)
    plot_posterior_with_ci(X_train, y_train, X_test, y_mean, y_std,
                           title="GPR on y = sin(x) + noise (posterior mean and 95% CI)")

    rmse_val, coverage = evaluate_predictions(y_true, y_mean, y_std)
    print(f"RMSE on dense test grid: {rmse_val:.4f}")
    print(f"Empirical coverage of 95% CI: {coverage * 100:.2f}%")

    # 6.4 POSTERIOR samples (functions after conditioning on data)
    plot_gp_posterior_samples(gpr,
                              x_scaler,
                              X_train,
                              y_train,
                              x_min=0.0,
                              x_max=10.0,
                              n_test=200,
                              n_samples=5)

    # 6.5 Hyperparameter studies (optimisation OFF)
    length_scales = [0.2, 1.0, 3.0]
    study_length_scale_effect(length_scales, n_train=30, noise_std=0.15)

    noise_levels = [0.01, 0.1, 0.5]
    study_noise_level_effect(noise_levels,
                             n_train=30,
                             noise_std=0.15,
                             length_scale=1.0)


if __name__ == "__main__":
    main()
