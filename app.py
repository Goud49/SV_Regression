import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Sales Prediction", layout="wide")
st.title("💰 BigMart Sales Prediction - SV Regression")

# Load and prepare data
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('Train.csv')
    
    # Store original data for visualization
    original_df = df.copy()
    
    # Handle Missing Values
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)
    
    # Store column info before encoding
    feature_cols = df.columns.tolist()
    feature_cols.remove('Item_Outlet_Sales')
    
    # Encode Categorical Columns
    le_dict = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
    
    # Split Features and Target
    X = df.drop('Item_Outlet_Sales', axis=1)
    y = df['Item_Outlet_Sales']
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train SV Regression Model
    svr_model = SVR(kernel='rbf', C=100, gamma='scale')
    svr_model.fit(X_train, y_train)
    
    svr_pred = svr_model.predict(X_test)
    svr_r2 = r2_score(y_test, svr_pred)
    svr_mae = mean_absolute_error(y_test, svr_pred)
    svr_rmse = np.sqrt(mean_squared_error(y_test, svr_pred))
    
    return svr_model, scaler, le_dict, feature_cols, X, svr_r2, svr_mae, svr_rmse, original_df

svr_model, scaler, le_dict, feature_cols, X, svr_r2, svr_mae, svr_rmse, original_df = load_and_train_model()

# Display model performance
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("R² Score", f"{svr_r2:.4f}")
with col2:
    st.metric("Mean Absolute Error", f"${svr_mae:.2f}")
with col3:
    st.metric("Root Mean Squared Error", f"${svr_rmse:.2f}")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Predict Sales", "📊 Model Performance", "📈 Dataset Analysis"])

with tab1:
    st.subheader("Enter Product & Store Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Product Information")
        item_weight = st.number_input("Item Weight (kg)", 0.0, 30.0, 10.0)
        item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
        item_type = st.selectbox("Item Type", 
                                 ["Beverages", "Breakfast", "Dairy", "Frozen Foods", "Fruits and Vegetables",
                                  "Meat", "Seafood", "Snacks", "Starchy Foods", "Breakfast", "Health and Hygiene",
                                  "Household", "Soft Drinks", "Baking Goods", "Canned", "Cereal", "Frozen",
                                  "Produce", "Meat", "Seafood", "Snacks", "Spices"])
        item_mrp = st.number_input("Item MRP ($)", 0.0, 300.0, 100.0)
    
    with col2:
        st.write("### Store Information")
        outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
        outlet_type = st.selectbox("Outlet Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])
        outlet_location_type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
        outlet_years = st.slider("Years Since Outlet Opened", 0, 30, 10)
    
    # Prepare input data
    if st.button("🔮 Predict Sales", key="predict"):
        try:
            input_data = {
                'Item_Weight': item_weight,
                'Item_Fat_Content': le_dict['Item_Fat_Content'].transform([item_fat_content])[0] if 'Item_Fat_Content' in le_dict else 0,
                'Item_Type': le_dict['Item_Type'].transform([item_type])[0] if 'Item_Type' in le_dict else 0,
                'Item_MRP': item_mrp,
                'Outlet_Size': le_dict['Outlet_Size'].transform([outlet_size])[0] if 'Outlet_Size' in le_dict else 0,
                'Outlet_Type': le_dict['Outlet_Type'].transform([outlet_type])[0] if 'Outlet_Type' in le_dict else 0,
                'Outlet_Location_Type': le_dict['Outlet_Location_Type'].transform([outlet_location_type])[0] if 'Outlet_Location_Type' in le_dict else 0,
                'Outlet_Years': outlet_years
            }
            
            # Create dataframe
            input_df = pd.DataFrame([input_data])
            
            # Ensure all required columns are present
            for col in X.columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[X.columns]
            
            # Scale input
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = svr_model.predict(input_scaled)[0]
            
            st.markdown("---")
            st.subheader("📋 Prediction Result")
            
            st.success(f"💰 Predicted Sales: **${prediction:,.2f}**")
            
            # Show additional insights
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"Item MRP: ${item_mrp}")
            with col2:
                profit_margin = ((prediction - item_mrp) / item_mrp * 100) if item_mrp > 0 else 0
                st.info(f"Estimated Margin: {profit_margin:.2f}%")
            with col3:
                st.info(f"Store Age: {outlet_years} years")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

with tab2:
    st.subheader("Model Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Model Metrics")
        metrics_data = {
            'Metric': ['R² Score', 'MAE', 'RMSE'],
            'Value': [f'{svr_r2:.4f}', f'${svr_mae:.2f}', f'${svr_rmse:.2f}']
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, hide_index=True)
    
    with col2:
        st.write("### Model Details")
        st.info(f"""
        **Algorithm**: Support Vector Regression (SVR)
        - **Kernel**: RBF (Radial Basis Function)
        - **C**: 100 (Regularization parameter)
        - **Training Samples**: {len(pd.read_csv('./archive (5)/Train.csv'))}
        - **Test Samples**: {int(len(pd.read_csv('./archive (5)/Train.csv')) * 0.2)}
        """)

with tab3:
    st.subheader("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Products", len(original_df))
    with col2:
        avg_sales = original_df['Item_Outlet_Sales'].mean()
        st.metric("Avg Sales", f"${avg_sales:,.2f}")
    with col3:
        max_sales = original_df['Item_Outlet_Sales'].max()
        st.metric("Max Sales", f"${max_sales:,.2f}")
    
    st.write("### Sales Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(original_df['Item_Outlet_Sales'], bins=50, color='#3498db', edgecolor='black')
    ax.set_title('Distribution of Item Sales')
    ax.set_xlabel('Sales ($)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    st.write("### Dataset Statistics")
    st.dataframe(original_df.describe())
    
    st.write("### Dataset Sample")
    st.dataframe(original_df.head(10))
