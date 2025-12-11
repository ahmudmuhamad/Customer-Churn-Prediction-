import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from src
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_pipeline.inference import load_artifacts, preprocess_new_data

# Page Config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔮",
    layout="wide"
)

# 1. Load Artifacts (Cached)
@st.cache_resource
def get_resources():
    return load_artifacts()

try:
    model, imputer, preprocessor = get_resources()
except FileNotFoundError as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# 2. UI Layout
st.title("🔮 Customer Churn Prediction")
st.markdown("Enter customer details below to predict the likelihood of churn.")

with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        
    with col2:
        st.subheader("Services")
        phone = st.selectbox("Phone Service", ["No", "Yes"])
        multiple = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
        backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
        protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
        support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
        tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
        
    with col3:
        st.subheader("Account")
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
        total = st.number_input("Total Charges", min_value=0.0, value=500.0)

    submit = st.form_submit_button("Predict Churn")

if submit:
    # 3. Construct DataFrame
    # Note: Mapping Yes/No to what model expects if necessary, 
    # but our pipeline handles string cats via OneHotEncoder, so raw strings are fine 
    # AS LONG AS they match the training encoding.
    # We map "Senior Citizen" Yes/No back to 0/1 because schema expects int
    senior_int = 1 if senior == "Yes" else 0
    
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior_int,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }
    
    raw_df = pd.DataFrame([input_data])
    
    # 4. Predict
    with st.spinner("Analyzing..."):
        try:
            X_processed = preprocess_new_data(raw_df, imputer, preprocessor)
            pred = model.predict(X_processed)[0]
            prob = model.predict_proba(X_processed)[0][1]
            
            # Display
            if pred == 1:
                st.error(f"⚠️ High Churn Risk (Probability: {prob:.2%})")
                st.markdown("**Recommendation**: Offer a discount or check for service issues immediately.")
            else:
                st.success(f"✅ Low Churn Risk (Probability: {prob:.2%})")
                st.markdown("**Recommendation**: Keep engaging with loyalty programs.")
                
            with st.expander("Feature Importance (Preview)"):
                st.info("Feature importance visualization can be added here based on the model.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
