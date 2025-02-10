import streamlit as st 
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time

# Load the trained models
logreg_model = joblib.load('logistic_regression_model.joblib')
rf_model = joblib.load('random_forest_model.joblib')

# Set the title and description of the app
st.set_page_config(page_title="Customer Churn Prediction", page_icon="🔮", layout="wide")
st.title("Telecom Customer Churn Prediction")
st.markdown("""
    <h3 style='color:#1e90ff;'>Welcome to the Customer Retention Predictor!</h3>
    <p>This app predicts whether a telecom customer is likely to churn (leave) based on their attributes.</p>
    <p>Using machine learning models such as Logistic Regression and Random Forest, we can predict churn 
    and suggest strategies for retaining high-risk customers.</p>
""", unsafe_allow_html=True)

# Add an attractive image and brief about the application
st.image("logo.png", caption="Telecom Industry", use_container_width=True)

# Section: User Input for Prediction
st.sidebar.header('Please enter User Inputs')
st.sidebar.markdown("""
    Please provide the necessary information below to predict customer churn:
""")

def user_input_features():
    # Collecting user inputs through the sidebar
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.sidebar.selectbox('Senior Citizen', ['Yes', 'No'])
    partner = st.sidebar.selectbox('Partner', ['Yes', 'No'])
    dependents = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.sidebar.slider('Tenure (Months)', 0, 72, 12)
    phone_service = st.sidebar.selectbox('Phone Service', ['Yes', 'No'])
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ['Yes', 'No'])
    internet_service = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    online_security = st.sidebar.selectbox('Online Security', ['Yes', 'No'])
    online_backup = st.sidebar.selectbox('Online Backup', ['Yes', 'No'])
    device_protection = st.sidebar.selectbox('Device Protection', ['Yes', 'No'])
    tech_support = st.sidebar.selectbox('Tech Support', ['Yes', 'No'])
    streaming_tv = st.sidebar.selectbox('Streaming TV', ['Yes', 'No'])
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ['Yes', 'No'])
    contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
    monthly_charges = st.sidebar.slider('Monthly Charges', 18.25, 118.75, 60.0)
    total_charges = st.sidebar.slider('Total Charges', 18.6, 8684.8, 1500.0)

    # Map categorical inputs to numerical values
    gender_map = {'Male': 0, 'Female': 1}
    senior_citizen_map = {'Yes': 1, 'No': 0}
    partner_map = {'Yes': 1, 'No': 0}
    dependents_map = {'Yes': 1, 'No': 0}
    phone_service_map = {'Yes': 1, 'No': 0}
    multiple_lines_map = {'Yes': 1, 'No': 0}
    internet_service_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    online_security_map = {'Yes': 1, 'No': 0}
    online_backup_map = {'Yes': 1, 'No': 0}
    device_protection_map = {'Yes': 1, 'No': 0}
    tech_support_map = {'Yes': 1, 'No': 0}
    streaming_tv_map = {'Yes': 1, 'No': 0}
    streaming_movies_map = {'Yes': 1, 'No': 0}
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    paperless_billing_map = {'Yes': 1, 'No': 0}
    payment_method_map = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer': 2, 'Credit card': 3}

    # Prepare the data for prediction
    data = {
        'Gender': gender_map[gender],
        'SeniorCitizen': senior_citizen_map[senior_citizen],
        'Partner': partner_map[partner],
        'Dependents': dependents_map[dependents],
        'Tenure': tenure,
        'PhoneService': phone_service_map[phone_service],
        'MultipleLines': multiple_lines_map[multiple_lines],
        'InternetService': internet_service_map[internet_service],
        'OnlineSecurity': online_security_map[online_security],
        'OnlineBackup': online_backup_map[online_backup],
        'DeviceProtection': device_protection_map[device_protection],
        'TechSupport': tech_support_map[tech_support],
        'StreamingTV': streaming_tv_map[streaming_tv],
        'StreamingMovies': streaming_movies_map[streaming_movies],
        'Contract': contract_map[contract],
        'PaperlessBilling': paperless_billing_map[paperless_billing],
        'PaymentMethod': payment_method_map[payment_method],
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_data = user_input_features()

# Display the input data in a nice format
st.write("User Input Data:")
st.write(input_data)

# Preprocess the input features: scaling the numerical features
scaler = StandardScaler()
numerical_features = ['Tenure', 'MonthlyCharges', 'TotalCharges']
input_data[numerical_features] = scaler.fit_transform(input_data[numerical_features])

# Display the submit button
submit_button = st.button("Submit")

# Handle the submit button action
if submit_button:
    # Show "wait" message
    with st.spinner("We are predicting..."):
        # Simulate 2-second delay
        time.sleep(2)
        
        # Prediction based on the selected model
        prediction_logreg = logreg_model.predict(input_data)
        prediction_prob_logreg = logreg_model.predict_proba(input_data)[:, 1]

        prediction_rf = rf_model.predict(input_data)
        prediction_prob_rf = rf_model.predict_proba(input_data)[:, 1]

        # Display model prediction results with emojis
        if prediction_logreg == 0:
            result_logreg = "Not Churned"
            emoji_logreg = "😃"  # Happy face emoji for "Not Churned"
        else:
            result_logreg = "Churned"
            emoji_logreg = "😞"  # Sad face emoji for "Churned"

        if prediction_rf == 0:
            result_rf = "Not Churned"
            emoji_rf = "😃"  # Happy face emoji for "Not Churned"
        else:
            result_rf = "Churned"
            emoji_rf = "😞"  # sad face emoji for "Churned"

        # Display predictions with highlighting
        st.subheader(f'Prediction Results:')
        st.markdown(f"**Logistic Regression Prediction**: <span style='color:green; font-size:20px; font-weight:bold;'>{result_logreg} {emoji_logreg}</span>", unsafe_allow_html=True)
        st.markdown(f"**Random Forest Prediction**: <span style='color:red; font-size:20px; font-weight:bold;'>{result_rf} {emoji_rf}</span>", unsafe_allow_html=True)

        # Provide actionable insights based on predictions
        if result_logreg == "Churned" or result_rf == "Churned":
            st.markdown("""
                ### Retention Strategy:
            - **Offer personalized discounts**: Provide special pricing or offers to high-risk customers.
            - **Reach out personally**: Have customer service reach out directly to understand their concerns.
            - **Bundle services**: Offer packages that include additional features like streaming or tech support at a discounted price.
            - **Upgrade services**: Offer free trials for premium features such as higher internet speeds or device protection.
            - **Contract renewal incentives**: Provide incentives for renewing contracts early to increase retention.
            - **Provide extra benefits**: Offer a month of free service or complimentary services for loyal customers.
            """)
        else:
            st.write("The customer is not predicted to churn. Here are strategies to further strengthen customer loyalty:")

            st.markdown("""
            ### Loyalty and Engagement Strategies:
            - **Reward long-term customers**: Provide exclusive rewards or bonuses to customers who have been with the company for a long time.
            - **Offer loyalty discounts**: Consider offering discounts to customers based on their tenure or usage. For example, a 10% discount for customers who have stayed over 2 years.
            - **Introduce referral programs**: Encourage customers to refer friends or family by offering both the referrer and referee discounts or other perks.
            - **Exclusive member events**: Create events such as webinars, loyalty programs, or VIP events where loyal customers are given special treatment or first access to new services.
            - **Early access to new features**: Give loyal customers early access to new product offerings, services, or features before they are available to the general public.
            - **Gamification**: Implement a gamified experience, such as loyalty points for every year or month they stay, or badges for using various services (e.g., streaming, tech support). Points could be redeemed for discounts or rewards.
            """)

# Footer
st.sidebar.markdown("""
    ## About
    This is a **Customer Churn Prediction** app developed to predict whether a customer will leave (churn) or stay based on their service data.
    The app uses machine learning models like **Random Forest** and **Logistic Regression** to make predictions.
    
    ### Key Features:
    - Predict churn likelihood using customer data.
    - Provide actionable retention strategies for high-risk customers.
    - Show detailed model performance metrics.
    
    ### Developed by: Rutuja Sawant
""")
