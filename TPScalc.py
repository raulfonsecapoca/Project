import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, WhiteKernel
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# ------------------------------
# Données synthétiques
# ------------------------------
def generate_sine_data(n_train=30, noise_std=0.1, x_min=0.0, x_max=10.0):
    X = np.linspace(x_min, x_max, n_train)
    y = np.sin(X) + np.random.normal(0.0, noise_std, size=n_train)
    return X.reshape(-1, 1), y


# ------------------------------
# Choix du noyau
# ------------------------------
def make_kernel(kernel_choice, length_scale=1.0, noise_level=0.1):

    if kernel_choice == "RBF":
        base = RBF(length_scale=length_scale)
    elif kernel_choice == "Matern":
        base = Matern(length_scale=length_scale, nu=1.5)
    elif kernel_choice == "RationalQuadratic":
        base = RationalQuadratic(length_scale=length_scale, alpha=1.0)
    else:
        raise ValueError("kernel_choice must be: RBF, Matern, RationalQuadratic")

    return base + WhiteKernel(noise_level=noise_level)


# ------------------------------
# Fonction principale demandée
# ------------------------------
def run_gpr_experiment(
        kernel_choice="RBF",
        n_train=30,
        noise_std=0.1,
        n_samples_posterior=5,
        optimize_hyperparams=True,
        length_scale=1.0,
        noise_level=0.1
):
    """
    Exécute un GPR complet :
        - génération des données
        - entraînement GPR
        - prédiction + statistiques
        - tracé du graphe
    """

    t0 = time.time()

    # --- 1. Génération des données ---
    X_train, y_train = generate_sine_data(n_train=n_train, noise_std=noise_std)
    X_test = np.linspace(0, 10, 500).reshape(-1, 1)
    y_true = np.sin(X_test).ravel()

    # --- 2. Normalisation des entrées ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- 3. Noyau ---
    kernel = make_kernel(kernel_choice, length_scale, noise_level)

    # --- 4. Modèle GPR ---
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        optimizer=None if not optimize_hyperparams else "fmin_l_bfgs_b"
    )

    gpr.fit(X_train_s, y_train)

    # --- 5. Prédictions ---
    y_mean, y_std = gpr.predict(X_test_s, return_std=True)
    ci_factor = 1.96
    y_lower = y_mean - ci_factor * y_std
    y_upper = y_mean + ci_factor * y_std

    # --- 6. Calculs d'erreurs ---
    rmse = np.sqrt(mean_squared_error(y_true, y_mean))
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))

    # --- 7. Posterior samples ---
    _, cov = gpr.predict(X_test_s, return_cov=True)
    cov = (cov + cov.T) / 2 + 1e-8 * np.eye(len(cov))  # stabilité
    L = np.linalg.cholesky(cov)

    samples = y_mean.reshape(-1, 1) + L @ np.random.randn(len(X_test), n_samples_posterior)

    # --- 8. Graphique ---
    plt.figure(figsize=(10, 5))

    plt.scatter(X_train, y_train, c='k', s=30, label="Données d'entraînement")

    plt.plot(X_test, y_mean, 'C0', label="Moyenne postérieure")
    plt.fill_between(X_test.ravel(), y_lower, y_upper, alpha=0.25, color='C0',
                     label="IC 95%")

    for i in range(n_samples_posterior):
        plt.plot(X_test, samples[:, i], '--', alpha=0.6)

    plt.title(f"GPR – Kernel : {kernel_choice}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    elapsed = time.time() - t0

    # --- 9. Résultats ---
    return {
        "kernel_used": str(gpr.kernel_),
        "rmse": rmse,
        "coverage_95%": coverage,
        "log_marginal_likelihood": gpr.log_marginal_likelihood(),
        "calculation_time_sec": elapsed
    }


result = run_gpr_experiment(
    kernel_choice="Matern",
    n_train=40,
    noise_std=0.15,
    n_samples_posterior=5,
    optimize_hyperparams=True
)

print(result)
