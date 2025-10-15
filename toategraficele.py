# %% 
# ---------------------------------------------
# Instalare librării necesare (rulează o singură dată în terminal):
# pip install rdkit-pypi scikit-learn xgboost pandas numpy shap tensorflow matplotlib seaborn imbalanced-learn
# ---------------------------------------------

import pandas as pd
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import shap
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# Pentru grafice inline în VSCode / Jupyter
# %matplotlib inline   # <- doar dacă rulezi ca notebook

# Seed pentru reproducibilitate
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# %%
# 1. Load dataset
data = pd.read_csv("tox21.csv")

# 2. Molecular descriptors
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_array = np.array(fp)
    desc = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)]
    return np.concatenate([fp_array, desc])

descriptor_list, valid_idx = [], []
for i, smi in enumerate(data['smiles']):
    desc = smiles_to_descriptors(smi)
    if desc is not None and not pd.isna(data.loc[i, 'SR-ARE']):
        descriptor_list.append(desc)
        valid_idx.append(i)

X = np.array(descriptor_list)
y = data.iloc[valid_idx]['SR-ARE'].values

# %%
# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)

# 4. BALANCING - SMOTE pe setul de antrenament
smote = SMOTE(random_state=seed)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 5. Scaling
scaler = StandardScaler()
X_train_bal_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# %%
# 6. Classical ML models
neg, pos = np.bincount(y_train_bal.astype(int))
scale_pos_weight = neg / pos

models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed),
    "SVM": SVC(probability=True, kernel='rbf', random_state=seed),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, eval_metric='logloss',
        scale_pos_weight=scale_pos_weight, random_state=seed
    )
}

for name, model in models.items():
    if name == "SVM":
        model.fit(X_train_bal_scaled, y_train_bal)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    print(f"--- {name} ---")
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("F1-score:", f1_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print()

# %%
# 7. Deep Learning MLP
input_dim = X_train_bal_scaled.shape[1]
mlp = Sequential([
    Dense(512, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = mlp.fit(
    X_train_bal_scaled, y_train_bal,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

y_prob_dl = mlp.predict(X_test_scaled).ravel()
y_pred_dl = (y_prob_dl > 0.5).astype(int)

print("--- Deep Learning MLP ---")
print("ROC-AUC:", roc_auc_score(y_test, y_prob_dl))
print("F1-score:", f1_score(y_test, y_pred_dl))
print("Precision:", precision_score(y_test, y_pred_dl))
print("Recall:", recall_score(y_test, y_pred_dl))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dl))

# %% Celula 9 - Vizualizare substructuri relevante pentru toate modelele
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np

# %% 
# Subset fix de 3 molecule pentru toate modelele
subset_size = 3
np.random.seed(seed)
subset_idx = np.random.choice(len(X_test), size=subset_size, replace=False)
X_subset = X_test[subset_idx]
X_subset_scaled = X_test_scaled[subset_idx]
smiles_subset = data.iloc[valid_idx].iloc[subset_idx]['smiles'].values

def show_substructures(model, explainer, X_input, model_name):
    shap_values = explainer.shap_values(X_input)
    shap_matrix = get_shap_matrix(shap_values)
    top_features_idx = np.argsort(np.abs(shap_matrix), axis=1)[:, -5:]

    for mol_idx, mol_array in enumerate(top_features_idx):
        smiles = smiles_subset[mol_idx]
        mol = Chem.MolFromSmiles(smiles)
        bitInfo = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, bitInfo=bitInfo)

        active_bits = [bit for bit in mol_array if bit in bitInfo]
        print(f"\n{model_name} - Molecule {mol_idx+1}: {smiles}")
        if not active_bits:
            print(" Nicio top feature SHAP activă în această moleculă.")
            continue

        for bit in active_bits:
            print(f" Feature/Bit {bit} contribuie la predicție")
            for atom_idx, radius in bitInfo[bit]:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                img = Draw.MolToImage(submol, size=(300,300))
                plt.figure()
                plt.title(f"{model_name} - Bit {bit} (atom {atom_idx}, radius {radius})")
                plt.imshow(img)
                plt.axis('off')
                plt.show()

# ---------------- Random Forest ----------------
rf_explainer = shap.TreeExplainer(models['RandomForest'])
show_substructures(models['RandomForest'], rf_explainer, X_subset, "Random Forest")

# ---------------- XGBoost ----------------
xgb_explainer = shap.TreeExplainer(models['XGBoost'])
show_substructures(models['XGBoost'], xgb_explainer, X_subset, "XGBoost")

# ---------------- SVM ----------------
svm_explainer = shap.KernelExplainer(models['SVM'].predict_proba, X_train_bal_scaled[:subset_size])
show_substructures(models['SVM'], svm_explainer, X_subset_scaled, "SVM")

# ---------------- Deep Learning MLP ----------------
dl_explainer = shap.DeepExplainer(mlp, X_train_bal_scaled[:subset_size])
show_substructures(mlp, dl_explainer, X_subset_scaled, "MLP")

# %%
