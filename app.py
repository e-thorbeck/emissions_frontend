import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import requests
import plotly.graph_objects as go


#API infrastructure here
PROD = 0
HOST = 'http://127.0.0.1:8000/' if PROD==0 else ''
analytics_endpoint = HOST + 'campuses_info'
predictions_endpoint = HOST + 'predictions'

response_analytics = requests.get(analytics_endpoint).json()
#predict_output = requests.get(predictions_endpoint) - do this later

# response_analytics = {
#     'CAMPUS 1 Bundoora': {'n_buldi': 5, 'consump': 62, 'sq-ft': 10},
#     'campus 2': {'n_buldi': 8, 'consump': 67, 'sq-ft': 5}
# }


#FRONT END INFRASTRUCTURE HERE

st.markdown("""# Welcome to The UNICON Campus Emissions Predictor
## Predict carbon emissions at one of Australia's largest universities.
This project uses the open-source UNICON dataset from La Trobe University, which has collected emissions data for the past 5 years on a unified platform for the purposes of research and prediction.""")
st.markdown("""---""")


st.markdown("""
### Secion 1 - Analytics
            """)

selected_campus = st.selectbox(
    'View Emissions Statistics by Campus',
    (response_analytics.keys()))

st.write('Now viewing metrics for:', selected_campus)


col1, col2, col3 = st.columns(3)
col1.metric("Number of Buildings", response_analytics[selected_campus]['n_build'], "-$1.25")
col2.metric("Total Square Feet", response_analytics[selected_campus]['floor_area'], "0.46%")
col3.metric("Total Emissions", response_analytics[selected_campus]['consumption'], "kg of Co2")

image_url = 'https://images.unsplash.com/photo-1548407260-da850faa41e3?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1487&q=80'

col1, col2 = st.columns(2)

col1.markdown(f"Emissions per square foot at {selected_campus}")
col1.image(image_url)
col2.markdown(f"What Influences Emissions?  Correlation Matrix for Emissions at {selected_campus}")
col2.image(image_url)

chart_url = 'https://user-images.githubusercontent.com/35371660/105487165-01af3500-5cf3-11eb-9243-c66de968798c.png'

# SECTION 2 - PREDICTIONS

st.markdown("""---""")

st.markdown("""
### Secion 2 - Predictions
            """)
st.markdown('Run a few scenarios here to predict emissions for next year from electricity')
#PREDICTIONS DATA HERE
def get_predictions(n):
    predictions = {
        "building_id": "all",
        "campus_id": "all",
        "predictions": {
            "2022-04-30 00:00:00": {
                "full_preds": 0.6630154829815373,
                "lower_conf": -0.021841835066883794,
                "upper_conf": 1.3478728010299585
            },
            "2022-05-31 00:00:00": {
                "full_preds": 0.7163111307983623,
                "lower_conf": -0.03391247273685817,
                "upper_conf": 1.4665347343335828
            },
            "2022-06-30 00:00:00": {
                "full_preds": 0.6741449609941533,
                "lower_conf": -0.13618914572866436,
                "upper_conf": 1.484479067716971
            },
            "2022-07-31 00:00:00": {
                "full_preds": 0.6744950116086325,
                "lower_conf": -0.19178858729830867,
                "upper_conf": 1.5407786105155736
            }
        }
    }

    # Extract the keys and values from the predictions dictionary
    dates = list(predictions["predictions"].keys())

    # Check if n is within the valid range
    if 1 <= n <= len(dates):
        # Create and return a list of dictionaries with the date, full_preds, lower_conf, and upper_conf for n,
        # as well as all the date & prediction values before it
        selected_predictions = [
            {
                "date": date,
                "full_preds": entry["full_preds"],
                "lower_conf": entry["lower_conf"],
                "upper_conf": entry["upper_conf"]
            }
            for date, entry in predictions["predictions"].items() if dates.index(date) < n
        ]
        return selected_predictions
    else:
        return {"error": "Invalid input for n. Please choose a value between 1 and {}".format(len(dates))}

# Example usage:
user_input = st.slider('Choose a timeframe for prediction in years', 1,4,1)

#user_input = int(input("Enter a number between 1 and 4: "))
result = get_predictions(user_input)


# Extract the information from the result list
dates = [entry['date'] for entry in result]
full_preds_values = [entry['full_preds'] for entry in result]
upper_conf_values = [entry['upper_conf'] for entry in result]
lower_conf_values = [entry['lower_conf'] for entry in result]

# Create a Plotly line plot
fig = go.Figure()

# Add traces for full_preds, upper_conf, and lower_conf
fig.add_trace(go.Scatter(
    x=dates,
    y=full_preds_values,
    mode='lines+markers',
    name='Full Predictions',
    hovertemplate='%{x}: Predicted Emissions: %{y:.2f}<extra></extra>',
))

fig.add_trace(go.Scatter(
    x=dates,
    y=upper_conf_values,
    mode='lines',
    line=dict(dash='dash'),
    name='Upper Confidence',
    hovertemplate='%{x}: Upper Confidence: %{y:.2f}<extra></extra>',
))

fig.add_trace(go.Scatter(
    x=dates,
    y=lower_conf_values,
    mode='lines',
    line=dict(dash='dash'),
    name='Lower Confidence',
    hovertemplate='%{x}: Lower Confidence: %{y:.2f}<extra></extra>',
))

# Set plot title and labels
fig.update_layout(
    title='Predictions Over Time',
    xaxis_title='Date',
    yaxis_title='Prediction Values',
    # margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins as needed
    height=500,  # Adjust height as needed
    # width=800,  # Adjust width as needed
    yaxis=dict(scaleanchor="x", scaleratio=1),  # Adjust scaleratio to control the distance between lines
)

# Update x-axis tick labels for better readability
fig.update_xaxes(tickangle=45)

# Update x-axis tick labels for better readability
fig.update_xaxes(tickangle=45)
###

#prediction = st.slider('Choose a timeframe for prediction in years', 1,4,1)
st.write(user_input, ' months out')

st.markdown("ARIMAX Time Series Model Here")
st.plotly_chart(fig)
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Emissions Per Square Foot", "100", "-12% vs last year")
col2.metric("Estimated Cost - Total", "$100", "AUD")
col3.metric("Estimated Cost Per Square Foot", "$10", "AUD")
