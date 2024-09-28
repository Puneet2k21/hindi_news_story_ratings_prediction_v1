# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:07:35 2024

@author: Puneet.Sah
"""

import streamlit as st
import pandas as pd
import pickle

# Load the trained Voting Classifier model
with open("voting_classifier.pkl", "rb") as file:
    voting_classifier = pickle.load(file)

# Load the preprocessor used during training (if applicable)
with open("preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)

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
    new_predictions_show_case = voting_classifier.predict(new_data_transformed_dense_show_case)

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
