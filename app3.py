import streamlit as st
import pickle
import numpy as np
import pandas as pd


# -------------------------------------------------
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📞",
    layout="centered"
)

# -------------------------------------------------
# Load model & feature names
# -------------------------------------------------
with open(r"C:\Users\param\OneDrive\Desktop\ML\Telecome_Churn_Predication_Project\model_1.pkl", "rb") as f:
    model = pickle.load(f)

data = pd.read_csv(r"C:\Users\param\OneDrive\Documents\NareshIT\ML_python\preprocessed_data_churn.csv")
features = data.drop("churn", axis=1).columns.tolist()

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>📞 Telecom Customer Churn</h1>
    <h4 style='text-align:center;color:#ff6b6b;'>
    Streamlit Telecom Churn ML App
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------------------------------------------------
# Input Section
# -------------------------------------------------
st.subheader("🔹 Enter Customer Details")

col1, col2 = st.columns(2)

user_input = []

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 80, 30)
    no_of_days_subscribed = st.slider("No. of Days Subscribed", 1, 500, 120)
    weekly_mins_watched = st.slider("Weekly Minutes Watched", 0, 5000, 500)
    videos_watched = st.slider("Videos Watched", 0, 500, 50)
    customer_support_calls = st.slider("Customer Support Calls", 0, 20, 1)

with col2:
    multi_screen = st.selectbox("Multi Screen Enabled", ["Yes", "No"])
    mail_subscribed = st.selectbox("Mail Subscription", ["Yes", "No"])
    minimum_daily_mins = st.slider("Minimum Daily Minutes", 0, 500, 30)
    maximum_daily_mins = st.slider("Maximum Daily Minutes", 0, 1000, 120)
    weekly_max_night_mins = st.slider("Weekly Max Night Minutes", 0, 3000, 300)
    maximum_days_inactive = st.slider("Maximum Days Inactive", 0, 30, 5)

# -------------------------------------------------
# Encoding (must match training)
# -------------------------------------------------
gender = 1 if gender == "Male" else 0
multi_screen = 1 if multi_screen == "Yes" else 0
mail_subscribed = 1 if mail_subscribed == "Yes" else 0

input_data = np.array([[
    gender,
    age,
    no_of_days_subscribed,
    multi_screen,
    mail_subscribed,
    weekly_mins_watched,
    minimum_daily_mins,
    maximum_daily_mins,
    weekly_max_night_mins,
    videos_watched,
    maximum_days_inactive,
    customer_support_calls
]])

# -------------------------------------------------
# Prediction
# -------------------------------------------------
st.markdown("---")

if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(
            f"⚠️ **Customer is likely to CHURN**\n\n"
            f"📊 Churn Probability: **{probability:.2f}**"
        )
    else:
        st.success(
            f"✅ **Customer is NOT likely to churn**\n\n"
            f"📊 Churn Probability: **{probability:.2f}**"
        )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Machine Learning Project | Logistic Regression | Streamlit UI")
