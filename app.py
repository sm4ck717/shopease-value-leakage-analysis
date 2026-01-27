import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn & Imbalanced-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Google GenAI
import google.generativeai as genai

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ShopEase Return Predictor",
    page_icon="🛍️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. FUNCTIONS (LOAD, TRAIN, AI)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    file_path = "shop_ease_fashion_dataset.csv"
    try:
        df = pd.read_csv(file_path)
        # Create target variable
        df['return_flag'] = (
            df['return_status']
            .str.strip()
            .str.lower()
            .eq('returned')
            .astype(int)
        )
        return df
    except FileNotFoundError:
        st.error(f"❌ Error: The file '{file_path}' was not found. Please place it in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"❌ An error occurred loading data: {e}")
        return None

@st.cache_resource
def train_model(df):
    # Define features
    features = [
        'age_group', 'city_tier', 'customer_tenure_months', 'product_category',
        'sub_category', 'brand', 'discount_percent', 'acquisition_channel',
        'channel_cost', 'delivery_time_days', 'payment_mode'
    ]
    
    # Separate Inputs and Target
    X = df[features]
    y = df['return_flag']
    
    # Identify Column Types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )
    
    # Pipeline with SMOTE
    model_pipeline = ImbPipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(sampling_strategy=0.4, random_state=42)),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]
    )
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    model_pipeline.fit(X_train, y_train)
    
    return model_pipeline, X_test, y_test

def get_ai_insights(api_key, input_data, risk_prob, risk_label):
    if not api_key:
        return "⚠️ Please enter a Google Gemini API Key in the sidebar to unlock AI insights."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        order_details = input_data.to_dict(orient='records')[0]
        
        prompt = f"""
        You are an expert E-commerce Risk Analyst.
        Analyze the following order which has been flagged as **{risk_label}** with a Return Probability of **{risk_prob:.1%}**.
        
        **Order Details:**
        {order_details}
        
        **Your Task:**
        1. **Analyze**: Why might this specific combination of factors (Age, Category, Payment, etc.) lead to a return?
        2. **Recommend**: Provide 3 specific, actionable steps to prevent this return.
        
        Keep it brief and professional.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to AI: {str(e)}"

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------------------------------------------

# Title stays outside the logic block so you always see it
st.title("🛍️ ShopEase Return Probability Engine")

# Load Data
df = load_data()

# Only proceed if Data is loaded successfully
if df is not None:
    
    # Train Model (with spinner so you know it's working)
    with st.spinner("Training predictive model on dataset..."):
        model, X_test, y_test = train_model(df)
        
    # Global threshold from notebook
    FINAL_THRESHOLD = 0.30

    # ------------------
    # SIDEBAR
    # ------------------
    st.sidebar.header("🤖 AI Configuration")
    api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter key to enable GenAI insights")
    
    st.sidebar.divider()
    
    st.sidebar.header("📝 Order Details")
    
    with st.sidebar.form("prediction_form"):
        st.subheader("Customer Profile")
        age_group = st.selectbox("Age Group", sorted(df['age_group'].unique()))
        city_tier = st.selectbox("City Tier", sorted(df['city_tier'].unique()))
        tenure = st.slider("Tenure (Months)", 
                           int(df['customer_tenure_months'].min()), 
                           int(df['customer_tenure_months'].max()), 
                           12)

        st.subheader("Product Details")
        prod_cat = st.selectbox("Category", sorted(df['product_category'].unique()))
        sub_cat = st.selectbox("Sub-Category", sorted(df['sub_category'].unique()))
        brand = st.selectbox("Brand", sorted(df['brand'].unique()))
        
        st.subheader("Transaction Info")
        payment = st.selectbox("Payment Mode", sorted(df['payment_mode'].unique()))
        channel = st.selectbox("Acquisition Channel", sorted(df['acquisition_channel'].unique()))
        
        col1, col2 = st.columns(2)
        with col1:
            discount = st.number_input("Discount %", 0.0, 100.0, 10.0)
            delivery = st.number_input("Delivery Days", 1, 30, 3)
        with col2:
            cost = st.number_input("Channel Cost", 0.0, 500.0, 50.0)

        submit_btn = st.form_submit_button("Predict Return Risk")

    # ------------------
    # DASHBOARD TABS
    # ------------------
    tab1, tab2, tab3 = st.tabs(["🚀 Prediction", "📊 Model Insights", "💾 Dataset"])

    # TAB 1: Prediction
    with tab1:
        if submit_btn:
            input_data = pd.DataFrame({
                'age_group': [age_group],
                'city_tier': [city_tier],
                'customer_tenure_months': [tenure],
                'product_category': [prod_cat],
                'sub_category': [sub_cat],
                'brand': [brand],
                'discount_percent': [discount],
                'acquisition_channel': [channel],
                'channel_cost': [cost],
                'delivery_time_days': [delivery],
                'payment_mode': [payment]
            })

            # Prediction
            probability = model.predict_proba(input_data)[0][1]
            is_high_risk = probability >= FINAL_THRESHOLD
            risk_label = "High Risk" if is_high_risk else "Low Risk"
            risk_color = "red" if is_high_risk else "green"

            st.divider()
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.metric(label="Return Probability", value=f"{probability:.2%}")
                st.markdown(f"### Status: :{risk_color}[{risk_label}]")
                st.caption(f"Threshold: {FINAL_THRESHOLD}")

            with c2:
                st.write("Risk Meter")
                st.progress(float(probability))

            # AI Insights
            st.divider()
            st.subheader("🤖 GenAI Analyst")
            if api_key:
                with st.spinner("Analyzing risk factors..."):
                    insights = get_ai_insights(api_key, input_data, probability, risk_label)
                    st.info(insights)
            else:
                st.warning("Enter API Key in sidebar to see insights.")
        else:
            st.info("👈 Please fill out the Order Details in the sidebar and click Predict.")

    # TAB 2: Insights
    with tab2:
        st.subheader("Feature Importance")
        
        # Get feature names and coeffs
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        coeffs = model.named_steps['classifier'].coef_[0]
        
        feat_df = pd.DataFrame({'feature': feature_names, 'coef': coeffs})
        # Clean names
        feat_df['feature'] = feat_df['feature'].str.replace('cat__', '').str.replace('num__', '')
        
        # Sort by magnitude
        feat_df['abs_coef'] = feat_df['coef'].abs()
        feat_df = feat_df.sort_values('abs_coef', ascending=False).head(15)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if c > 0 else 'green' for c in feat_df['coef']]
        sns.barplot(data=feat_df, x='coef', y='feature', palette=colors, ax=ax)
        plt.title("Top Drivers of Returns (Red = Increases Risk)")
        st.pyplot(fig)

    # TAB 3: Data
    with tab3:
        st.dataframe(df.head(100))

else:
    # This else block runs if df is None (Error loading data)
    st.warning("⚠️ Application halted because data could not be loaded. Please check the error message above.")