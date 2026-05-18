import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Page Config
st.set_page_config(page_title="SV Regression App", layout="wide")

# Title
st.title("📈 Sales Prediction using SV Regression")

# Upload Dataset
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Load Dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Dataset Info
    st.subheader("Dataset Shape")
    st.write(df.shape)

    # Missing Values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Fill Missing Values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Select Target Column
    target_column = st.selectbox("Select Target Column", df.columns)

    @st.cache_resource
    def load_and_train_model(df, target_column):

        # Features and Target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode categorical columns
        le_dict = {}

        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

        # Convert target if object type
        if y.dtype == 'object':
            y_le = LabelEncoder()
            y = y_le.fit_transform(y.astype(str))

        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.2,
            random_state=42
        )

        # Model
        svr_model = SVR(kernel='rbf')

        # Train Model
        svr_model.fit(X_train, y_train)

        # Predictions
        y_pred = svr_model.predict(X_test)

        # Metrics
        svr_r2 = r2_score(y_test, y_pred)
        svr_mae = mean_absolute_error(y_test, y_pred)
        svr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return svr_model, scaler, le_dict, X.columns, svr_r2, svr_mae, svr_rmse

    # Train Model
    svr_model, scaler, le_dict, feature_cols, svr_r2, svr_mae, svr_rmse = load_and_train_model(df, target_column)

    # Display Metrics
    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("R² Score", round(svr_r2, 4))
    col2.metric("MAE", round(svr_mae, 4))
    col3.metric("RMSE", round(svr_rmse, 4))

    # User Input Section
    st.subheader("🔮 Make Prediction")

    input_data = {}

    for col in feature_cols:

        if df[col].dtype == 'object':

            options = df[col].unique().tolist()

            value = st.selectbox(f"Select {col}", options)

            # Encode input
            le = le_dict[col]
            value = le.transform([value])[0]

            input_data[col] = value

        else:

            value = st.number_input(
                f"Enter {col}",
                value=float(df[col].mean())
            )

            input_data[col] = value

    # Prediction Button
    if st.button("Predict"):

        input_df = pd.DataFrame([input_data])

        # Scale Input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = svr_model.predict(input_scaled)

        st.success(f"✅ Predicted Value: {prediction[0]:.2f}")

    # Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap='coolwarm',
        ax=ax
    )

    st.pyplot(fig)

else:
    st.info("📂 Please upload a CSV file to continue.")
