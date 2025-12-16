import os
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

import plotly.express as px

# =====================================================
# CONFIGURATION STREAMLIT
# =====================================================
st.set_page_config(page_title="ML Streamlit App", layout="wide")
st.title("Interactive Machine Learning App")
st.sidebar.header("Configuration")

# Source des données
data_source = st.sidebar.radio(
    "Source des données",
    ["Projet existant", "Uploader un CSV"]
)

# Modèle ML
model_name = st.sidebar.selectbox(
    "Choisir le modèle",
    ["Logistic Regression", "Random Forest", "SVM"]
)

# Hyperparamètres
st.sidebar.subheader("Hyperparamètres")
if model_name == "Logistic Regression":
    C_value = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
elif model_name == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 50, 300, 100, step=50)
    max_depth = st.sidebar.slider("max_depth", 2, 20, 10)
else:  
    C_value = st.sidebar.slider("C", 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])

# =====================================================
# Fonction pour charger les datasets
# =====================================================
def load_project_dataset(name):
    BASE_DIR = os.path.dirname(__file__)
    if name == "Bank Marketing":
        csv_file = os.path.join(BASE_DIR, "data", "bank.csv")
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        target = "deposit"
    elif name == "Customer Churn":
        csv_file = os.path.join(BASE_DIR, "data", "customer_churn_dataset-testing-master.csv")
        df = pd.read_csv(csv_file)
        df = df.drop(columns=["CustomerID"])
        target = "Churn"
    elif name == "Income Prediction":
        csv_file = os.path.join(BASE_DIR, "data", "income_evaluation.csv")
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        df["income"] = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)
        target = "income"
    else:
        csv_file = os.path.join(BASE_DIR, "data", "loan_status.csv")
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
        df = df.drop(columns=["Loan_ID"])
        target = "Loan_Status"
    return df, target

# =====================================================
# CHARGEMENT DES DONNEES
# =====================================================
df = None
target_column = None

if data_source == "Projet existant":
    project = st.sidebar.selectbox(
        "Choisir le projet",
        ["Bank Marketing", "Customer Churn", "Income Prediction", "Loan Status"]
    )
    df, target_column = load_project_dataset(project)
else:
    file = st.sidebar.file_uploader("Uploader un fichier CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        target_column = st.sidebar.selectbox("Variable cible", df.columns)

if df is None:
    st.warning("Veuillez charger un dataset")
    st.stop()

# Séparation X / y
X = df.drop(columns=[target_column])
y = df[target_column]
if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)

# =====================================================
# PREPROCESSING
# =====================================================
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_features),
    ("cat", cat_pipeline, categorical_features)
])

# =====================================================
# CHOIX DU MODELE
# =====================================================
if model_name == "Logistic Regression":
    classifier = LogisticRegression(C=C_value, max_iter=1000)
elif model_name == "Random Forest":
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
else:
    classifier = SVC(C=C_value, kernel=kernel)

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", classifier)
])

# =====================================================
# TRAIN TEST SPLIT ET ENTRAINEMENT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =====================================================
# INTERFACE STREAMLIT : TABS
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs(["Données", "Modèle", "Résultats", "Prédiction"])

with tab1:
    st.subheader("Aperçu des données")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write("Valeurs manquantes")
    st.dataframe(df.isnull().sum())

with tab2:
    st.subheader("Modèle utilisé")
    st.write(model_name)
    st.json(classifier.get_params())

with tab3:
    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc:.3f}")
    st.text(classification_report(y_test, y_pred))
    st.plotly_chart(px.imshow(confusion_matrix(y_test, y_pred), text_auto=True))

with tab4:
    st.subheader("Prédiction utilisateur")
    user_input = {}
    for col in X.columns:
        if col in numerical_features:
            user_input[col] = st.number_input(col)
        else:
            user_input[col] = st.text_input(col)
    if st.button("Prédire"):
        pred = model.predict(pd.DataFrame([user_input]))[0]
        st.success(f"Résultat : {pred}")
