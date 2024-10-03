import streamlit as st
import yaml
import streamlit_authenticator as stauth
import pandas as pd
import pickle
import datetime

# Load the YAML configuration file
with open("allowed_users.yaml") as file:
    config = yaml.safe_load(file)

# Load the trained Voting Classifier model
with open("voting_classifier_ex_xgb.pkl", "rb") as file:
    voting_classifier_ex_xgb = pickle.load(file)

# Load the preprocessor used during training (if applicable)
with open("preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)

# Set cookie expiry to 5 seconds
authenticator = stauth.Authenticate(
    config['credentials'],
    'news_app_cookie_test',  # Replace with your own cookie name
    'abc123',  # Replace with your own signature key
    cookie_expiry_days=300 / 86400  # Cookie expires after 5 seconds. a day has 86400 seconds
)

# Add Login Form
login_result = authenticator.login()

# Check if login_result is None before unpacking
if login_result is not None:
    name, authentication_status, username = login_result
else:
    authentication_status = None

# Authentication Status Logic

if authentication_status:
    st.write(f'Welcome *{name}*')
    st.write('You are logged in as:', username)

    # Retrieve blocked emails from the config
    blocked_emails = config.get('blocked_users', [])

    # Check if the logged-in user is in the blocked list
    if username in blocked_emails:
        st.error("Your account has been blocked.")
        st.stop()  # Stops the script from running further

    # ---- PLACE YOUR MACHINE LEARNING PREDICTION CODE BELOW ----

    # Define the options for dropdown columns
    genre_options = sorted([
        "WAR", "ASTROLOGY", "RELIGIOUS / FAITH", "HEALTH", "NATIONAL THREAT/DEFENCE NEWS",
        "INDIA-PAK", "CRIME/LAW & ORDER", "POLITICAL NEWS/GOVERNMENT NEWS", "FINANCIAL NEWS",
        "SCIENCE/SPACE", "CAREER/EDUCATION", "EVENT/CELEBRATIONS", "WEATHER/ENVIRONMENT",
        "ENTERTAINMENT NEWS", "OTHER", "MISHAPS/FAILURE OF MACHINERY", "SPORTS NEWS"
    ])

    geography_options = sorted([
        "INTERNATIONAL", "MANIPUR", "JHARKHAND", "GUJARAT", "INDIAN", "HARYANA", "RAJASTHAN",
        "BIHAR", "UTTAR PRADESH", "OTHER", "DELHI", "MAHARASHTRA", "UTTARAKHAND", "KARNATAKA",
        "JAMMU AND KASHMIR", "CHANDIGARH", "WEST BENGAL", "HIMACHAL PRADESH", "MADHYA PRADESH",
        "TELANGANA", "CHHATTISGARH"
    ])

    popularity_options = ["H", "M", "L"]

    personality_genre_options = sorted([
        "JMM", "Astrologer", "International", "JDU", "Bajrang Dal", "RJD", "Religious",
        "DMK", "BSP", "INC", "AIMIM", "OTHER", "NCP", "Defense", "SP", "BJP", "TMC", "RSS-VHP",
        "AAP", "SS", "Entertainer", "NC", "SBSP", "Cricketer"
    ])

    substories_layers_options = ["H", "M", "L"]

    logistics_options = ["ON LOCATION", "IN STUDIO", "BOTH"]

    story_format_options = sorted(["INTERVIEW", "DEBATE OR DISCUSSION", "NEWS REPORT"])

    # Streamlit app interface
    st.title("Hindi News Story Rating Prediction based on Machine Learning model")

    # Collect user inputs via Streamlit input elements
    genre = st.selectbox("Select Genre", genre_options)
    geography = st.selectbox("Select Geography (For national stories select INDIAN)", geography_options)
    personality_popularity = st.selectbox("Select Personality Popularity", popularity_options)
    personality_genre = st.selectbox("Select Personality-Genre", personality_genre_options)
    dur_hour = st.number_input("Enter Duration in Hours (Type a number between 1 to 10000. "
                               "Consider the story coverage of 6-12 months. "
                               "Typically, low durations could be less than 10 hours, "
                               "medium between 10-25 hours and high above 25 hours in 6 months to 1 year period)", min_value=1, max_value=10000, value=1)
    substories_layers = st.selectbox("Select Substories layers (HML)", substories_layers_options)
    logistics = st.selectbox("Select Logistics", logistics_options)
    story_format = st.selectbox("Select Story Format", story_format_options)

    # Create the DataFrame with the collected inputs
    new_data_show_case = pd.DataFrame({
        'Genre': [genre],
        'Geography': [geography],
        'Personality Popularity': [personality_popularity],
        'Personality-Genre': [personality_genre],
        'Dur Hour': [dur_hour],
        'Substoires layers (HML)': [substories_layers],
        'Logistics': [logistics],
        'Story_Format': [story_format]
    })

    # Display the DataFrame in Streamlit app
    st.write("User Input Data:")
    st.dataframe(new_data_show_case)

    # Button to trigger prediction
    if st.button("Predict Rating Tier"):
        # Preprocessing: Transform the new data using the preprocessor fitted on the training data
        new_data_transformed_show_case = preprocessor.transform(new_data_show_case)

        # Convert the sparse matrix to a dense matrix (if applicable)
        if hasattr(new_data_transformed_show_case, "toarray"):
            new_data_transformed_dense_show_case = new_data_transformed_show_case.toarray()
        else:
            new_data_transformed_dense_show_case = new_data_transformed_show_case

        # Make a prediction using the trained voting classifier model
        new_predictions_show_case = voting_classifier_ex_xgb.predict(new_data_transformed_dense_show_case)

        # Define the function to categorize Predicted_TVTs
        def categorize_tier(tier):
            if tier == 0:
                return 'Low viewership less than 164 TVTs avg'
            elif tier == 1:
                return 'Medium viewership between 165 and 349 TVTs avg'
            else:
                return 'High viewership more than 350 TVTs'

        # Convert the numerical prediction to the categorized tier
        predicted_value_tier = categorize_tier(new_predictions_show_case[0])

        # Display the result
        st.write(f"Predicted Rating Category: {predicted_value_tier}")

elif authentication_status is False:
    st.error('Username/password is incorrect. Please try again')

elif authentication_status is None:
    st.warning('Please enter your username and password') # Provide the login form again if authentication status is None

# Log user activity (optional)
def log_user_activity(username):
    with open("user_logs.txt", "a") as log_file:
        log_file.write(f"{datetime.datetime.now()} - {username} logged in\n")

if authentication_status:
    log_user_activity(username)

# Add the professional note at the end of the app
st.write("""
---
### Note:
This app leverages machine learning to predict news ratings, offering insights based on historical data. 
Predictions should be combined with domain expertise. The developer is not responsible for outcomes based solely on the app's predictions. 
For technical details on ML models employed and error metrics, contact:  
**Puneet Sah**  
Email: puneet2k21@gmail.com
""")