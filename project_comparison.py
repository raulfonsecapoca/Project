"""
Comparisons of regression methods in two scenarios:

1) California Housing dataset:
   - Linear Regression
   - Polynomial Regression (fixed degrees)
   - Polynomial Ridge Regression (degree + alpha via CV)
   - Kernel Ridge Regression (RBF, alpha & gamma via CV on subset)
   - Gaussian Process Regression (RBF + WhiteKernel, on subset)

2) 1D synthetic data: y = sin(x) + Gaussian noise
   - Linear Regression
   - Polynomial Regression (fixed degrees)
   - Polynomial Ridge Regression (CV)
   - Kernel Ridge Regression (RBF, CV)
   - Gaussian Process Regression (RBF + WhiteKernel)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_SEED = 42
rng = np.random.RandomState(RANDOM_SEED)


# ====================================================================
# Utilities
# ====================================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def generate_sine_1d_data(n_train=30, noise_std=0.15, x_min=0.0, x_max=10.0):
    """
    Generate 1D training data from y = sin(x) + Gaussian noise.
    """
    X = np.linspace(x_min, x_max, n_train)
    y = np.sin(X) + rng.normal(0.0, noise_std, size=n_train)
    return X.reshape(-1, 1), y


def build_linear_regression_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lin", LinearRegression())
    ])


def build_polynomial_linear_model(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lin", LinearRegression())
    ])


def build_polynomial_ridge_cv_model(degree_candidates, alpha_candidates):
    """
    PolynomialFeatures + StandardScaler + Ridge, with degree and alpha
    chosen by cross-validation.
    """
    pipe = Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])
    param_grid = {
        "poly__degree": list(degree_candidates),
        "ridge__alpha": list(alpha_candidates),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1
    )
    return grid


def build_kernel_ridge_cv_model(alpha_candidates, gamma_candidates):
    """
    StandardScaler + KernelRidge(RBF) with alpha and gamma chosen by CV.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("krr", KernelRidge(kernel="rbf"))
    ])
    param_grid = {
        "krr__alpha": list(alpha_candidates),
        "krr__gamma": list(gamma_candidates),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1
    )
    return grid


# ====================================================================
# PART 1 – California Housing
# ====================================================================

def load_california_from_train_test(train_path="california_housing_train.csv",
                                    test_path="california_housing_test.csv"):
    """
    Load California Housing data from two CSVs with headers.

    Assumptions:
        - last column is the target (median house value)
        - all previous columns are input features
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values

    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    return X_train, X_test, y_train, y_test


def compare_models_california(train_path="california_housing_train.csv",
                              test_path="california_housing_test.csv",
                              poly_fixed_degrees=(2, 3),
                              degree_candidates=(1, 2, 3),
                              alpha_candidates_poly=(1e-2, 1e-1, 1.0, 10.0),
                              alpha_candidates_krr=(1e-2, 1e-1, 1.0, 10.0),
                              gamma_candidates_krr=(0.01, 0.1, 1.0),
                              max_gpr_train_samples=3000):
    """
    Compare Linear, Polynomial (fixed degrees), Polynomial Ridge (CV),
    Kernel Ridge (RBF, CV), and GPR on the California Housing dataset.
    """
    print("=== COMPARISON 1: California Housing ===")

    X_train, X_test, y_train, y_test = load_california_from_train_test(
        train_path=train_path, test_path=test_path
    )

    print(f"X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
    print(f"X_test  shape = {X_test.shape},  y_test  shape = {y_test.shape}")

    results = []
    preds_dict = {}  # store predictions for later plots

    # ---------------- Linear Regression ----------------
    lin_model = build_linear_regression_model()
    lin_model.fit(X_train, y_train)
    y_pred_lin = lin_model.predict(X_test)
    lin_rmse = rmse(y_test, y_pred_lin)
    lin_r2 = r2_score(y_test, y_pred_lin)
    results.append({"model": "Linear Regression", "RMSE": lin_rmse, "R2": lin_r2})
    preds_dict["Linear Regression"] = y_pred_lin
    print(f"Linear Regression: RMSE = {lin_rmse:.4f}, R2 = {lin_r2:.4f}")

    # ---------------- Polynomial Regression (fixed degrees) ----------------
    for d in poly_fixed_degrees:
        poly_model = build_polynomial_linear_model(degree=d)
        poly_model.fit(X_train, y_train)
        y_pred_poly = poly_model.predict(X_test)
        poly_rmse = rmse(y_test, y_pred_poly)
        poly_r2 = r2_score(y_test, y_pred_poly)
        name = f"Poly deg {d}"
        results.append({"model": name, "RMSE": poly_rmse, "R2": poly_r2})
        preds_dict[name] = y_pred_poly
        print(f"Polynomial degree {d}: RMSE = {poly_rmse:.4f}, R2 = {poly_r2:.4f}")

    # ---------------- Polynomial Ridge (CV) ----------------
    poly_ridge_cv = build_polynomial_ridge_cv_model(
        degree_candidates=degree_candidates,
        alpha_candidates=alpha_candidates_poly
    )
    poly_ridge_cv.fit(X_train, y_train)
    y_pred_poly_cv = poly_ridge_cv.predict(X_test)
    poly_cv_rmse = rmse(y_test, y_pred_poly_cv)
    poly_cv_r2 = r2_score(y_test, y_pred_poly_cv)

    best_params_poly = poly_ridge_cv.best_params_
    name_cv_poly = (
        f"Poly Ridge CV (deg={best_params_poly['poly__degree']}, "
        f"alpha={best_params_poly['ridge__alpha']:.3g})"
    )
    print("\nBest Polynomial Ridge (CV) on California:")
    print(f"  degree = {best_params_poly['poly__degree']}")
    print(f"  alpha  = {best_params_poly['ridge__alpha']:.3g}")
    print(f"  CV MSE = {-poly_ridge_cv.best_score_:.4f}")
    print(f"  Test RMSE = {poly_cv_rmse:.4f}, R2 = {poly_cv_r2:.4f}")

    results.append({
        "model": name_cv_poly,
        "RMSE": poly_cv_rmse,
        "R2": poly_cv_r2
    })
    preds_dict[name_cv_poly] = y_pred_poly_cv

    # ---------------- Subconjunto para KRR e GPR ----------------
    n_train = X_train.shape[0]
    max_n = min(n_train, max_gpr_train_samples)
    idx = np.arange(n_train)
    rng.shuffle(idx)
    idx_sub = idx[:max_n]

    X_train_sub = X_train[idx_sub]
    y_train_sub = y_train[idx_sub]

    # ---------------- Kernel Ridge Regression (RBF, CV) ----------------
    krr_cv = build_kernel_ridge_cv_model(
        alpha_candidates=alpha_candidates_krr,
        gamma_candidates=gamma_candidates_krr
    )
    krr_cv.fit(X_train_sub, y_train_sub)
    y_pred_krr = krr_cv.predict(X_test)

    best_params_krr = krr_cv.best_params_
    krr_rmse = rmse(y_test, y_pred_krr)
    krr_r2 = r2_score(y_test, y_pred_krr)
    krr_name = "Kernel Ridge (RBF, CV)"

    print("\nBest Kernel Ridge (RBF, CV) on California (subset training):")
    print(f"  alpha = {best_params_krr['krr__alpha']:.3g}")
    print(f"  gamma = {best_params_krr['krr__gamma']:.3g}")
    print(f"  CV MSE = {-krr_cv.best_score_:.4f}")
    print(f"  Test RMSE = {krr_rmse:.4f}, R2 = {krr_r2:.4f}")

    results.append({"model": krr_name, "RMSE": krr_rmse, "R2": krr_r2})
    preds_dict[krr_name] = y_pred_krr

    # ---------------- Gaussian Process Regression ----------------
    x_scaler = StandardScaler()
    X_train_gpr = x_scaler.fit_transform(X_train_sub)
    X_test_gpr = x_scaler.transform(X_test)

    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        random_state=RANDOM_SEED
    )
    gpr.fit(X_train_gpr, y_train_sub)
    y_pred_gpr, y_std_gpr = gpr.predict(X_test_gpr, return_std=True)

    gpr_rmse = rmse(y_test, y_pred_gpr)
    gpr_r2 = r2_score(y_test, y_pred_gpr)
    gpr_name = "GPR (RBF + White)"

    print("\nGaussian Process Regression (subset training):")
    print("  Optimised kernel:", gpr.kernel_)
    print(f"  RMSE = {gpr_rmse:.4f}, R2 = {gpr_r2:.4f}")

    results.append({"model": gpr_name, "RMSE": gpr_rmse, "R2": gpr_r2})
    preds_dict[gpr_name] = y_pred_gpr

    # ---------------- Summary and metric plots ----------------
    results_df = pd.DataFrame(results)
    print("\nSummary – California Housing:")
    print(results_df)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(results_df["model"], results_df["RMSE"])
    axes[0].set_title("RMSE by model (California)")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(results_df["model"], results_df["R2"])
    axes[1].set_title("R^2 by model (California)")
    axes[1].set_ylabel("R^2")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.show()

    # ==========================================================
    # Plots de dispersão com escala robusta
    # ==========================================================

    # juntar todos os valores para calcular uma faixa robusta (1%–99%)
    all_true = y_test
    all_pred = np.concatenate(list(preds_dict.values()))
    all_vals = np.concatenate([all_true, all_pred])

    low = np.percentile(all_vals, 1)
    high = np.percentile(all_vals, 99)

    # cores fixas por modelo
    model_colors = {
        "Linear Regression": "C0",
        "Poly deg 2": "C1",
        "Poly deg 3": "C2",
        name_cv_poly: "C3",
        krr_name: "C4",
        gpr_name: "C5",
    }

    # ---------------- Plot 1: todos os modelos ----------------
    plt.figure(figsize=(7, 7))

    plt.plot([low, high], [low, high], "k--", label="Ideal")

    for name, y_pred in preds_dict.items():
        plt.scatter(
            y_test,
            y_pred,
            alpha=0.4,
            s=15,
            label=name,
            color=model_colors.get(name, None)
        )

    plt.xlim(low, high)
    plt.ylim(low, high)
    plt.xlabel("True target (median house value)")
    plt.ylabel("Predicted target")
    plt.title("California Housing – True vs predicted (all models)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # ---------------- Plot 2: só KRR vs GPR ----------------
    plt.figure(figsize=(7, 7))

    plt.plot([low, high], [low, high], "k--", label="Ideal")

    for name in [krr_name, gpr_name]:
        y_pred = preds_dict[name]
        plt.scatter(
            y_test,
            y_pred,
            alpha=0.4,
            s=15,
            label=name,
            color=model_colors[name]
        )

    plt.xlim(low, high)
    plt.ylim(low, high)
    plt.xlabel("True target (median house value)")
    plt.ylabel("Predicted target")
    plt.title("California Housing - True vs predicted (KRR vs GPR)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    return results_df


# ====================================================================
# PART 2 – 1D sine + Gaussian noise
# ====================================================================

def compare_models_sine_1d(n_train=30,
                           noise_std=0.15,
                           x_min=0.0,
                           x_max=10.0,
                           poly_fixed_degrees=(3, 9),
                           degree_candidates_poly=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                           alpha_candidates_poly=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
                           alpha_candidates_krr=(1e-3, 1e-2, 1e-1, 1.0, 10.0),
                           gamma_candidates_krr=(0.01, 0.1, 1.0)):
    """
    Compare all models on ONE 1D synthetic dataset y = sin(x) + noise.
    Evaluation is done vs the true sin(x) on a dense grid.

    Models:
        - Linear Regression
        - Polynomial Regression (fixed degrees)
        - Polynomial Ridge Regression (CV)
        - Kernel Ridge Regression (RBF, CV)
        - GPR (RBF + WhiteKernel)
    """
    print("\n=== COMPARISON 2: 1D sine + Gaussian noise ===")

    X_train, y_train = generate_sine_1d_data(
        n_train=n_train,
        noise_std=noise_std,
        x_min=x_min,
        x_max=x_max
    )
    X_test = np.linspace(x_min, x_max, 500).reshape(-1, 1)
    y_true = np.sin(X_test).ravel()

    results = []
    preds_dict = {}

    # ------------- Linear Regression -------------
    lin_model = build_linear_regression_model()
    lin_model.fit(X_train, y_train)
    y_pred_lin = lin_model.predict(X_test)
    lin_rmse = rmse(y_true, y_pred_lin)
    results.append({"model": "Linear Regression", "rmse_true": lin_rmse})
    preds_dict["Linear Regression"] = y_pred_lin
    print(f"Linear Regression RMSE vs sin(x): {lin_rmse:.4f}")

    # ------------- Kernel Ridge (RBF, CV) -------------
    krr_cv = build_kernel_ridge_cv_model(
        alpha_candidates=alpha_candidates_krr,
        gamma_candidates=gamma_candidates_krr
    )
    krr_cv.fit(X_train, y_train)
    y_pred_krr = krr_cv.predict(X_test)
    best_params_krr = krr_cv.best_params_
    krr_rmse = rmse(y_true, y_pred_krr)
    krr_name = "Kernel Ridge (RBF, CV)"
    results.append({"model": krr_name, "rmse_true": krr_rmse})
    preds_dict[krr_name] = y_pred_krr

    print("\nBest Kernel Ridge (RBF, CV) on 1D sine (noisy y):")
    print(f"  alpha = {best_params_krr['krr__alpha']:.3g}")
    print(f"  gamma = {best_params_krr['krr__gamma']:.3g}")
    print(f"  CV MSE = {-krr_cv.best_score_:.4f}")
    print(f"  RMSE vs sin(x) = {krr_rmse:.4f}")

    # ------------- GPR (RBF + WhiteKernel) -------------
    x_scaler_gpr = StandardScaler()
    X_train_gpr = x_scaler_gpr.fit_transform(X_train)
    X_test_gpr = x_scaler_gpr.transform(X_test)

    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=noise_std)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        random_state=RANDOM_SEED
    )
    gpr.fit(X_train_gpr, y_train)
    y_mean_gpr, y_std_gpr = gpr.predict(X_test_gpr, return_std=True)

    gpr_rmse = rmse(y_true, y_mean_gpr)
    gpr_name = "GPR (RBF + White)"
    results.append({"model": gpr_name, "rmse_true": gpr_rmse})
    preds_dict[gpr_name] = y_mean_gpr
    print(f"\nGPR RMSE vs sin(x): {gpr_rmse:.4f}")
    print("Optimised GPR kernel (1D sine):", gpr.kernel_)

    # Plot GPR vs true sin(x) com IC
    ci_factor = 1.96
    y_lower_gpr = y_mean_gpr - ci_factor * y_std_gpr
    y_upper_gpr = y_mean_gpr + ci_factor * y_std_gpr

    plt.figure(figsize=(8, 4))
    plt.scatter(X_train, y_train, c="k", s=30, label="Training data")
    plt.plot(X_test, y_true, "r--", label="True sin(x)")
    plt.plot(X_test, y_mean_gpr, label="GPR mean")
    plt.fill_between(X_test.ravel(), y_lower_gpr, y_upper_gpr, alpha=0.3,
                     label="GPR 95% CI")
    plt.title("GPR vs true sin(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------- Polynomial Regression (fixed degrees) -------------
    for d in poly_fixed_degrees:
        poly_model = build_polynomial_linear_model(degree=d)
        poly_model.fit(X_train, y_train)
        y_pred_poly = poly_model.predict(X_test)
        poly_rmse = rmse(y_true, y_pred_poly)
        name = f"Poly deg {d}"
        results.append({"model": name, "rmse_true": poly_rmse})
        preds_dict[name] = y_pred_poly
        print(f"Polynomial degree {d} RMSE vs sin(x): {poly_rmse:.4f}")

    # ------------- Polynomial Ridge (CV) -------------
    poly_ridge_cv = build_polynomial_ridge_cv_model(
        degree_candidates=degree_candidates_poly,
        alpha_candidates=alpha_candidates_poly
    )
    poly_ridge_cv.fit(X_train, y_train)
    y_pred_poly_cv = poly_ridge_cv.predict(X_test)
    poly_cv_rmse = rmse(y_true, y_pred_poly_cv)
    best_params_poly = poly_ridge_cv.best_params_
    name_cv_poly = (
        f"Poly Ridge CV (deg={best_params_poly['poly__degree']}, "
        f"alpha={best_params_poly['ridge__alpha']:.3g})"
    )
    results.append({"model": name_cv_poly, "rmse_true": poly_cv_rmse})
    preds_dict[name_cv_poly] = y_pred_poly_cv

    print("\nBest Polynomial Ridge (CV) on 1D sine (noisy y):")
    print(f"  degree = {best_params_poly['poly__degree']}")
    print(f"  alpha  = {best_params_poly['ridge__alpha']:.3g}")
    print(f"  CV MSE = {-poly_ridge_cv.best_score_:.4f}")
    print(f"  RMSE vs sin(x) = {poly_cv_rmse:.4f}")

    # ------------- Single plot: ALL models em 1D -------------
    plot_colors = {
        "Linear Regression": "C0",
        krr_name: "C1",
        gpr_name: "C2",
        "Poly deg 3": "C3",
        "Poly deg 9": "C4",
        name_cv_poly: "C5",
    }

    plt.figure(figsize=(9, 5))
    plt.scatter(X_train, y_train, c="k", s=30, label="Training data")
    plt.plot(X_test, y_true, "k--", linewidth=2, label="True sin(x)")

    for name, y_pred in preds_dict.items():
        plt.plot(X_test, y_pred, label=name, color=plot_colors.get(name, None))

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("1D sine - predictions of all models")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # ------------- Summary table -------------
    results_df = pd.DataFrame(results)
    print("\nSummary RMSE vs true sin(x) (1D):")
    print(results_df.sort_values("rmse_true"))

    return results_df


# ====================================================================
# Main
# ====================================================================

def main():
    # 1) California Housing
    cali_results = compare_models_california(
        train_path="california_housing_train.csv",
        test_path="california_housing_test.csv"
    )

    # 2) 1D sine
    sine1d_results = compare_models_sine_1d(
        n_train=30,
        noise_std=0.15,
        x_min=0.0,
        x_max=10.0,
        poly_fixed_degrees=(),
    )

    # Optional: save tables to CSV
    # cali_results.to_csv("summary_california.csv", index=False)
    # sine1d_results.to_csv("summary_sine1d.csv", index=False)


if __name__ == "__main__":
    main()
