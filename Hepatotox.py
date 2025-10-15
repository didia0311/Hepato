# Instalare librării necesare (rulează o singură dată)
# pip install rdkit-pypi scikit-learn xgboost pandas numpy shap tensorflow matplotlib seaborn
# ABC
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
import shap
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load dataset
# -----------------------------
data = pd.read_csv("tox21.csv")
print("Coloane CSV:", data.columns, flush=True)

# -----------------------------
# 2. Generate molecular descriptors
# -----------------------------
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_array = np.array(fp)
    # Descriptors fizico-chimici
    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol)
    ]
    return np.concatenate([fp_array, desc])

descriptor_list = []
valid_idx = []
for i, smi in enumerate(data['smiles']):
    desc = smiles_to_descriptors(smi)
    if desc is not None and not pd.isna(data.loc[i, 'SR-ARE']):
        descriptor_list.append(desc)
        valid_idx.append(i)

X = np.array(descriptor_list)
y = data.iloc[valid_idx]['SR-ARE'].values

print("X shape:", X.shape, flush=True)
print("y shape:", y.shape, flush=True)
print("Clase unice y:", np.unique(y, return_counts=True), flush=True)

if X.shape[0] == 0:
    raise ValueError("Nu există date valide pentru antrenament. Verifică CSV-ul și numele coloanelor.")

# -----------------------------
# 3. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4. Scale data (SVM / MLP)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Train classical ML models
# -----------------------------
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, kernel='rbf', random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

for name, model in models.items():
    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} ---", flush=True)
    print("ROC-AUC:", roc_auc_score(y_test, y_prob), flush=True)
    print("F1-score:", f1_score(y_test, y_pred), flush=True)
    print("Precision:", precision_score(y_test, y_pred), flush=True)
    print("Recall:", recall_score(y_test, y_pred), flush=True)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred), flush=True)

# -----------------------------
# 6. Train simple deep learning model (MLP)
# -----------------------------
input_dim = X_train_scaled.shape[1]
mlp = Sequential([
    Dense(512, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = mlp.fit(
    X_train_scaled, y_train, epochs=30, batch_size=32,
    validation_split=0.1, verbose=1
)

y_prob_dl = mlp.predict(X_test_scaled).ravel()
y_pred_dl = (y_prob_dl > 0.5).astype(int)

print("\n--- Deep Learning MLP ---", flush=True)
print("ROC-AUC:", roc_auc_score(y_test, y_prob_dl), flush=True)
print("F1-score:", f1_score(y_test, y_pred_dl), flush=True)
print("Precision:", precision_score(y_test, y_pred_dl), flush=True)
print("Recall:", recall_score(y_test, y_pred_dl), flush=True)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dl), flush=True)

# -----------------------------
# 7. Feature importance (Random Forest) using SHAP
# -----------------------------
rf_model = models['RandomForest']
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Top 20 features
shap.summary_plot(shap_values[1], X_test, plot_type="bar")
