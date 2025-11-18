import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 1. DATA LOADING & PREPARATION
# ==========================================
# Load the spam dataset
df = pd.read_csv('spambase.data', header=None)

# CRITICAL STEP: Subsampling
# GPC scales cubically O(N^3). We sample 1,000 instances to keep training fast.
df_sample = df.sample(n=1000, random_state=42)

X = df_sample.iloc[:, :-1].values
y = df_sample.iloc[:, -1].values

# Split Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Data (Standardization is vital for GPC and LogReg)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. MODEL TRAINING
# ==========================================

# A. Logistic Regression (Baseline - Lecture 3)
# Parametric, linear decision boundary.
log_reg = LogisticRegression(random_state=42)

# B. Random Forest (Competitor - Lecture 4)
# Non-linear, ensemble of trees.
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# C. Gaussian Process Classifier (Subject - Paper)
# Non-parametric, probabilistic. Uses Laplace approximation for the link function.
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=42)

models = {'Logistic Regression': log_reg, 'Random Forest': rf, 'Gaussian Process': gpc}
probas = {}

# ==========================================
# 3. EVALUATION METRICS
# ==========================================
print("--- Classification Performance (Test Set: 200 samples) ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] # For ROC
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    probas[name] = y_prob
    
    print(f"{name}: Accuracy = {acc:.3f}, F1-Score = {f1:.3f}")

# ==========================================
# 4. VISUALIZATION: ROC CURVES
# ==========================================
plt.figure(figsize=(10, 6))
for name, y_prob in probas.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparative Analysis: ROC Curves (Spam Classification)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# ==========================================
# 5. VISUALIZATION: CONFUSION MATRICES
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, model) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, model.predict(X_test_scaled))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Spam', 'Spam'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f"{name}")
plt.show()