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
emissions = {
    0: {
    'year': 2019,
    'campus_id': "Albury-Wodonga",
    'n_build': 1,
    'consumption': 11791.4872,
    'co2_from_electric': 12498.976432,
    'gross_floor_area': 1404,
    'consumption_m2': 8.398495156695157
    },
    1: {
    'year': 2019,
    'campus_id': "Bundoora",
    'n_build': 1,
    'consumption': 137471.8669,
    'co2_from_electric': 145720.17891400002,
    'gross_floor_area': 1356939.4800000023,
    'consumption_m2': 0.1013102418539696
    },
    2: {
    'year': 2020,
    'campus_id': "Albury-Wodonga",
    'n_build': 2,
    'consumption': 196738.6892,
    'co2_from_electric': 208543.010552,
    'gross_floor_area': 2649,
    'consumption_m2': 74.26904084560212
    },
    3: {
    'year': 2020,
    'campus_id': "Bundoora",
    'n_build': 3,
    'consumption': 295182.54150000005,
    'co2_from_electric': 312893.49399,
    'gross_floor_area': 4098144.070000001,
    'consumption_m2': 0.0720283466022706
    },
    4: {
    'year': 2021,
    'campus_id': "Albury-Wodonga",
    'n_build': 2,
    'consumption': 252480.6956,
    'co2_from_electric': 267629.537336,
    'gross_floor_area': 2649,
    'consumption_m2': 95.31170086825216
    },
    5: {
    'year': 2021,
    'campus_id': "Bundoora",
    'n_build': 3,
    'consumption': 371539.5707,
    'co2_from_electric': 393831.9449420001,
    'gross_floor_area': 4098144.070000001,
    'consumption_m2': 0.0906604463761567
    },
    6: {
    'year': 2022,
    'campus_id': "Albury-Wodonga",
    'n_build': 2,
    'consumption': 133575.21360000002,
    'co2_from_electric': 141589.726416,
    'gross_floor_area': 2649,
    'consumption_m2': 50.424769195923
    },
    7: {
    'year': 2022,
    'campus_id': "Bundoora",
    'n_build': 2,
    'consumption': 209331.4921,
    'co2_from_electric': 221891.381626,
    'gross_floor_area': 2631655.6500000004,
    'consumption_m2': 0.079543648539276
    }
}


emissions_df = pd.DataFrame.from_dict(emissions, orient='index')

# Filter data for the specified campus
campus_data = emissions_df[emissions_df['campus_id'] == selected_campus]

campus_data['year'] = pd.to_datetime(campus_data['year'], format='%Y')


# Create a line plot using Plotly Express
emissions_fig = px.line(
    campus_data,
    x='year',
    y='consumption_m2',
    title=f'Emissions By Year - {selected_campus}',
    labels={'year': 'Year', 'co2_from_electric': 'Co2 Emissions Per Square Foot (KG)'}
)




col1, col2, col3 = st.columns(3)
col1.metric("Number of Buildings", response_analytics[selected_campus]['n_build'], "-$1.25")
col2.metric("Total Square Feet", response_analytics[selected_campus]['floor_area'], "0.46%")
col3.metric("Total Emissions", response_analytics[selected_campus]['consumption'], "kg of Co2")

image_url = 'https://images.unsplash.com/photo-1548407260-da850faa41e3?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1487&q=80'

st.plotly_chart(emissions_fig)
# col2.markdown(f"What Influences Emissions?  Correlation Matrix for Emissions at {selected_campus}")
# col2.image(image_url)

# Sample Shapley values data (replace this with your actual data)
shapley_values_data = pd.DataFrame({
    'feature': ['Feature A', 'Feature B', 'Feature C', 'Feature D'],
    'shapley_value': [0.2, 0.3, -0.1, 0.4]
})

# Sample campus data (replace this with your actual data)
campus_data = pd.DataFrame({
    'campus_id': [1, 2, 3],
    'campus_name': ['Campus A', 'Campus B', 'Campus C']
})

# Streamlit app
st.title('Shapley Values for Selected Campus')

# Filter Shapley values data based on the selected campus
filtered_shapley_values = shapley_values_data  # Replace this line with actual filtering logic

# Plot Shapley values
fig = px.bar(filtered_shapley_values, x='feature', y='shapley_value', title=f'Shapley Values for Campus {selected_campus}')
st.plotly_chart(fig)

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
    yaxis_range=[-5,5],  # Adjust scaleratio to control the distance between lines
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
