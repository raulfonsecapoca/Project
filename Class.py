import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_curve, auc, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestClassifier

# ========================================================
#              CONFIGURATION
# ========================================================

N = 2000
K = 10

# ========================================================
#              CHARGEMENT DES DONNÉES
# ========================================================

df = pd.read_csv("spambase.data", header=None)
df = df.sample(n=N, random_state=42)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ========================================================
#              NORMALISATION DES DONNÉES
# ========================================================

global_scaler = StandardScaler().fit(X)
X_scaled = global_scaler.transform(X)

print("\n=== Données normalisées avec StandardScaler ===")

# ========================================================
#     OPTIMISATION HYPERPARAMÈTRES GAUSSIAN PROCESS
# ========================================================

param_grid = {
    "kernel": [
        C(1.0) * RBF(length_scale=l) for l in [0.1]
    ]
}

print("\n=== Optimisation des hyperparamètres du Gaussian Process ===")

grid_gp = GridSearchCV(
    GaussianProcessClassifier(optimizer='fmin_l_bfgs_b', random_state=42),
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_gp.fit(X_scaled, y)
best_gp = grid_gp.best_estimator_

print("Best kernel found:", best_gp.kernel_)



# ========================================================
#              DÉFINITION DES MODÈLES
# ========================================================

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Gaussian Process (tuned)": best_gp,
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
}

# Enregistrement des résultats
all_results = {
    name: {
        "risk": [],
        "TPR": [], "FPR": [],
        "AUC": [],
        "confusion": [],
        "time_fold": []
    }
    for name in models.keys()
}

# ========================================================
#              CROSS-VALIDATION K=10
# ========================================================

kf = KFold(n_splits=K, shuffle=True, random_state=42)

plt.figure(figsize=(10, 6))
fold_index = 1

for train_idx, test_idx in kf.split(X_scaled):

    print(f"\n=== Fold {fold_index}/{K} ===")  # Affichage du fold

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    for name, model in models.items():

        start_time = time.time()

        # Entraînement
        model.fit(X_train, y_train)

        # Prédiction hard
        y_pred = model.predict(X_test)

        # Prédiction soft
        y_prob = model.predict_proba(X_test)[:, 1]

        end_time = time.time()
        fold_time = end_time - start_time
        all_results[name]["time_fold"].append(fold_time)

        # Risk
        risk = np.mean(y_pred != y_test)
        all_results[name]["risk"].append(risk)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        all_results[name]["confusion"].append(cm)

        # ROC / AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        all_results[name]["AUC"].append(roc_auc)

        # TPR / FPR
        TP = cm[1, 1]
        FN = cm[1, 0]
        FP = cm[0, 1]
        TN = cm[0, 0]

        TPR_value = TP / (TP + FN)
        FPR_value = FP / (FP + TN)

        all_results[name]["TPR"].append(TPR_value)
        all_results[name]["FPR"].append(FPR_value)

        # Affichage console pour ce fold et ce modèle
        print(f"{name} -> Risk: {risk:.4f}, AUC: {roc_auc:.4f}, Time: {fold_time:.4f} s")

        # --- Plot ROC for this fold ---
        plt.plot(fpr, tpr, alpha=0.2,
                 label=f"{name} fold {fold_index}" if fold_index == 1 else "")

    fold_index += 1

# Diagonale
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves for the 10 folds")
plt.legend()
plt.grid()
plt.show()

# ========================================================
#              AFFICHAGE DES RÉSULTATS MOYENS + ENREGISTREMENT
# ========================================================


with open("RES.txt", "w") as f:

    for name in models.keys():

        print("\n===============================")
        print(f"  Model: {name}")
        print("===============================")
        f.write("\n===============================\n")
        f.write(f"  Model: {name}\n")
        f.write("===============================\n")

        for fold_idx in range(K):
            risk = all_results[name]["risk"][fold_idx]
            auc_fold = all_results[name]["AUC"][fold_idx]
            tpr_fold = all_results[name]["TPR"][fold_idx]
            fpr_fold = all_results[name]["FPR"][fold_idx]
            fold_time = all_results[name]["time_fold"][fold_idx]
            cm = all_results[name]["confusion"][fold_idx]

            # Recalcule des points ROC pour ce fold
            y_train_idx, y_test_idx = list(kf.split(X_scaled))[fold_idx]
            X_train_fold, X_test_fold = X_scaled[y_train_idx], X_scaled[y_test_idx]
            y_train_fold, y_test_fold = y[y_train_idx], y[y_test_idx]
            model = models[name]
            model.fit(X_train_fold, y_train_fold)
            y_prob_fold = model.predict_proba(X_test_fold)[:, 1]
            fpr_points, tpr_points, _ = roc_curve(y_test_fold, y_prob_fold)

            print(f"\nFold {fold_idx+1}/{K}")
            print(f"Risk: {risk:.4f}, AUC: {auc_fold:.4f}, TPR: {tpr_fold:.4f}, FPR: {fpr_fold:.4f}, Time: {fold_time:.4f} s")
            print("Confusion matrix:\n", cm)
            print("FPR points:", np.round(fpr_points, 4))
            print("TPR points:", np.round(tpr_points, 4))

            f.write(f"\nFold {fold_idx+1}/{K}\n")
            f.write(f"Risk: {risk:.4f}, AUC: {auc_fold:.4f}, TPR: {tpr_fold:.4f}, FPR: {fpr_fold:.4f}, Time: {fold_time:.4f} s\n")
            f.write("Confusion matrix:\n")
            f.write(np.array2string(cm, precision=2) + "\n")
            f.write("FPR points: " + np.array2string(np.round(fpr_points, 4)) + "\n")
            f.write("TPR points: " + np.array2string(np.round(tpr_points, 4)) + "\n")

        # Moyennes globales
        mean_risk = np.mean(all_results[name]["risk"])
        mean_auc = np.mean(all_results[name]["AUC"])
        mean_tpr = np.mean(all_results[name]["TPR"])
        mean_fpr = np.mean(all_results[name]["FPR"])
        mean_time = np.mean(all_results[name]["time_fold"])
        mean_cm = np.mean(all_results[name]["confusion"], axis=0)

        print("\n=== Mean Results ===")
        print(f"Mean Risk: {mean_risk:.4f}, Mean AUC: {mean_auc:.4f}, Mean TPR: {mean_tpr:.4f}, Mean FPR: {mean_fpr:.4f}, Mean Time: {mean_time:.4f} s")
        print("Mean confusion matrix:\n", mean_cm)

        f.write("\n=== Mean Results ===\n")
        f.write(f"Mean Risk: {mean_risk:.4f}, Mean AUC: {mean_auc:.4f}, Mean TPR: {mean_tpr:.4f}, Mean FPR: {mean_fpr:.4f}, Mean Time: {mean_time:.4f} s\n")
        f.write("Mean confusion matrix:\n")
        f.write(np.array2string(mean_cm, precision=2) + "\n")
