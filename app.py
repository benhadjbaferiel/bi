import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, r2_score
)

import plotly.express as px

st.set_page_config(page_title="ML Streamlit App", layout="wide")
st.title(" Application Machine Learning Interactive")

st.sidebar.header(" Configuration")

data_source = st.sidebar.radio(
    "Source des données",
    ["Dataset préchargé", "Uploader CSV"]
)

def load_dataset(name):
    if name == "Iris":
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        df = data.frame
        df["target"] = data.target
        return df
    elif name == "Wine":
        from sklearn.datasets import load_wine
        data = load_wine(as_frame=True)
        df = data.frame
        df["target"] = data.target
        return df
    elif name == "Breast Cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        df["target"] = data.target
        return df

df = None

if data_source == "Dataset préchargé":
    dataset_name = st.sidebar.selectbox(
        "Choisir un dataset",
        ["Iris", "Wine", "Breast Cancer"]
    )
    df = load_dataset(dataset_name)
else:
    file = st.sidebar.file_uploader("Uploader un fichier CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

if df is None:
    st.warning("Veuillez charger un dataset")
    st.stop()

st.sidebar.subheader(" Colonne cible")
target_column = st.sidebar.selectbox("Choisir la variable à prédire", df.columns)

y_raw = df[target_column]
is_classification = (
    y_raw.dtype == "object" or y_raw.nunique() < 20
)

task_type = "Classification" if is_classification else "Régression"
st.sidebar.success(f"Type détecté : {task_type}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [" Données", " Modèle", " Résultats", " Prédictions", "⬇ Export"]
)

with tab1:
    st.dataframe(df.head())
    st.write(df.describe(include="all"))
    st.write("Valeurs manquantes :", df.isnull().sum())

X = df.drop(target_column, axis=1)
y = y_raw

X = pd.get_dummies(X, drop_first=True)

if is_classification and y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

with tab2:
    if is_classification:
        model_name = st.selectbox(
            "Modèle",
            ["Logistic Regression", "Random Forest", "SVM"]
        )

        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100)
        else:
            model = SVC()

    else:
        model_name = st.selectbox(
            "Modèle",
            ["Linear Regression", "Random Forest Regressor", "SVR"]
        )

        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=100)
        else:
            model = SVR()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

with tab3:
    if is_classification:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        st.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.2f}")
        st.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.2f}")
        st.metric("F1-score", f"{f1_score(y_test, y_pred, average='weighted'):.2f}")

        cm = confusion_matrix(y_test, y_pred)
        st.plotly_chart(px.imshow(cm, text_auto=True, title="Confusion Matrix"))
    else:
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        st.metric("R²", f"{r2_score(y_test, y_pred):.2f}")

with tab4:
    inputs = []
    for col in X.columns:
        inputs.append(st.number_input(col))

    if st.button("🔮 Prédire"):
        scaled = scaler.transform([inputs])
        pred = model.predict(scaled)
        st.success(f"Résultat : {pred[0]}")

with tab5:
    result_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    st.download_button("⬇️ CSV", result_df.to_csv(index=False), "results.csv")
    st.download_button("⬇️ JSON", result_df.to_json(orient="records"), "results.json")
