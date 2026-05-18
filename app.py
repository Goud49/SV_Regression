import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SV Regression App",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("📈 Sales Prediction using SV Regression")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # ---------------- LOAD DATA ----------------
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📏 Dataset Shape")
    st.write(df.shape)

    st.subheader("❌ Missing Values")
    st.write(df.isnull().sum())

    # ---------------- HANDLE MISSING VALUES ----------------
    for col in df.columns:

        # Numeric Columns
        if pd.api.types.is_numeric_dtype(df[col]):

            df[col] = df[col].fillna(df[col].mean())

        # Categorical/String Columns
        else:

            df[col] = df[col].fillna(df[col].mode()[0])

    # ---------------- TARGET COLUMN ----------------
    target_column = st.selectbox(
        "🎯 Select Target Column",
        df.columns
    )

    # ---------------- MODEL FUNCTION ----------------
    @st.cache_resource
    def load_and_train_model(dataframe, target):

        # Features and Target
        X = dataframe.drop(columns=[target])
        y = dataframe[target]

        # Store Encoders
        le_dict = {}

        # ---------------- ENCODE FEATURES ----------------
        for col in X.columns:

            if not pd.api.types.is_numeric_dtype(X[col]):

                le = LabelEncoder()

                X[col] = le.fit_transform(
                    X[col].astype(str)
                )

                le_dict[col] = le

        # ---------------- ENCODE TARGET ----------------
        if not pd.api.types.is_numeric_dtype(y):

            y_encoder = LabelEncoder()

            y = y_encoder.fit_transform(
                y.astype(str)
            )

        # ---------------- FEATURE SCALING ----------------
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        # ---------------- TRAIN TEST SPLIT ----------------
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.2,
            random_state=42
        )

        # ---------------- MODEL ----------------
        svr_model = SVR(kernel='rbf')

        # ---------------- TRAIN ----------------
        svr_model.fit(X_train, y_train)

        # ---------------- PREDICTION ----------------
        y_pred = svr_model.predict(X_test)

        # ---------------- METRICS ----------------
        svr_r2 = r2_score(y_test, y_pred)

        svr_mae = mean_absolute_error(
            y_test,
            y_pred
        )

        svr_rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )

        return (
            svr_model,
            scaler,
            le_dict,
            X.columns,
            svr_r2,
            svr_mae,
            svr_rmse
        )

    # ---------------- TRAIN MODEL ----------------
    (
        svr_model,
        scaler,
        le_dict,
        feature_cols,
        svr_r2,
        svr_mae,
        svr_rmse
    ) = load_and_train_model(df, target_column)

    # ---------------- PERFORMANCE ----------------
    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "R² Score",
        round(svr_r2, 4)
    )

    col2.metric(
        "MAE",
        round(svr_mae, 4)
    )

    col3.metric(
        "RMSE",
        round(svr_rmse, 4)
    )

    # ---------------- USER INPUT ----------------
    st.subheader("🔮 Make Prediction")

    input_data = {}

    for col in feature_cols:

        # Categorical Input
        if col in le_dict:

            options = df[col].astype(str).unique().tolist()

            selected_value = st.selectbox(
                f"Select {col}",
                options
            )

            encoded_value = le_dict[col].transform(
                [selected_value]
            )[0]

            input_data[col] = encoded_value

        # Numeric Input
        else:

            value = st.number_input(
                f"Enter {col}",
                value=float(df[col].mean())
            )

            input_data[col] = value

    # ---------------- PREDICTION BUTTON ----------------
    if st.button("Predict"):

        input_df = pd.DataFrame([input_data])

        # Scale Input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = svr_model.predict(input_scaled)

        st.success(
            f"✅ Predicted Value: {prediction[0]:.2f}"
        )

    # ---------------- HEATMAP ----------------
    st.subheader("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

else:

    st.info("📂 Please upload a CSV file to continue.")
