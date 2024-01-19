import streamlit as st
import os
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import requests
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
            page_title="UNICON Campus Emissions", # => Quick reference - Streamlit
            page_icon="🐍",
            layout="centered", # wide
            initial_sidebar_state="auto"
            ) #

# page_bg_img = f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
# background-color: white;
# color: black;
# h2: black;
# h3: black;
# }}
# </style>
# """

# st.markdown(page_bg_img, unsafe_allow_html=True)

#API infrastructure here
PROD = os.getenv('PROD')
HOST = 'http://127.0.0.1:8000/' if PROD=='0' else os.getenv('API_HOST')
analytics_endpoint = HOST + 'campuses_info'
predictions_endpoint = HOST + 'predictions'
emissions_endpoint = HOST + 'campuses_year_info'
shap_endpoint = HOST + 'shap'
consumption_endpoint = HOST + 'av_consump'

response_emissions = requests.get(emissions_endpoint).json()
response_analytics = requests.get(analytics_endpoint).json()
response_shap = requests.get(shap_endpoint).json()
response_consumption = requests.get(consumption_endpoint).json()
#predict_output = requests.get(predictions_endpoint) - do this later

# response_analytics = {
#     'CAMPUS 1 Bundoora': {'n_buldi': 5, 'consump': 62, 'sq-ft': 10},
#     'campus 2': {'n_buldi': 8, 'consump': 67, 'sq-ft': 5}
# }
def format_value(value):
    return "{:,.0f}".format(value)

#FRONT END INFRASTRUCTURE HERE

st.markdown("""# Welcome to The UNICON Campus Emissions Predictor
### Predict carbon emissions at one of Australia's largest universities, and discover what factors influence emissions.

""")
st.markdown("""By Erik Thorbeck, Aime Rangel, and Ennia Castel""")
st.markdown("""---""")
st.sidebar.markdown(f"""
    # Select Your Campus Here
    """)

selected_campus = st.sidebar.selectbox(
    'View Emissions Statistics by Campus',
    (response_analytics.keys()))


st.sidebar.write('Now viewing metrics for:', selected_campus)

st.sidebar.markdown("""
                    #### Glossary
                    """)
st.sidebar.markdown("""
           This project uses the open-source UNICON dataset from La Trobe University, which has collected emissions data for the past 5 years on a unified platform for the purposes of research and prediction.
Source: https://www.kaggle.com/datasets/cdaclab/unicon/data
           """)

st.sidebar.write("""
#### Financial Data:
All financial data is displayed in Australian Dollars (AUD), unless otherwise noted.
                 """)

st.sidebar.write("""
#### Energy Data:
All energy consumption data is displayed in units of Kilowatt Hours, unless otherwise noted.
                 """)

st.sidebar.write("""
#### Emissions Data:
All Carbon Emissions are reported in the form of kilograms of Carbon Dioxide (Co2).  The Government of Victoria advises a multiple of 1.06 when converting KWH of electricity into Kilograms of Co2.
This multiple is a measure of carbon intensity, which is based on all combined sources of power generation for the State of Victoria, and is subject to change.
Source: https://www.climatechange.vic.gov.au/greenhouse-gas-emissions
                 """)

st.markdown("""
### Section 1 - Analytics
            """)


emissions_df = pd.DataFrame.from_dict(response_emissions, orient='index')
# st.write(emissions_df)

# Filter data for the specified campus
campus_data = emissions_df[emissions_df['campus_id'] == selected_campus]

campus_data['year'] = pd.to_datetime(campus_data['year'], format='%Y')


# Create a line plot using Plotly Express
emissions_fig = px.line(
    campus_data,
    x='year',
    y='co2_from_electric',
    title=f'Emissions By Year - {selected_campus}',
    labels={'year': 'Year', 'co2_from_electric': 'Total Co2 Emissions from Electricity (KG)'}
)


# format_value(round(response_analytics[selected_campus]['total_sq_ft']))

col1, col2, col3 = st.columns(3)
col1.metric("Number of Buildings", round(response_analytics[selected_campus]['n_build']), "Buildings")
col2.metric("Total Square Feet", format_value(round(response_analytics[selected_campus]['total_sq_ft'])), "Campus Footprint")
col3.metric("Total Emissions", format_value(round(response_analytics[selected_campus]['total_missions'],0)), "kg of Co2 emitted from electricity")

image_url = 'https://images.unsplash.com/photo-1548407260-da850faa41e3?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1487&q=80'

st.plotly_chart(emissions_fig)

# Map campus name to campus_id
campus_name_to_id = {'Albury-Wodonga': 2, 'Bundoora': 1}
selected_campus_id = campus_name_to_id.get(selected_campus)

#read in data from api
avg_per_hour = pd.DataFrame(response_consumption).T


# avg_per_hour
# Filter DataFrame based on selected campus
filtered_df = avg_per_hour[(avg_per_hour['campus_id'] == selected_campus_id)]
# agg_bldg_data[(agg_bldg_data['day'] > day_filter) & (agg_bldg_data['campus_id'] == selected_campus_id)]

# Group by building_id and timestamp, calculate average consumption per hour
avg_consumption_per_hour = filtered_df.groupby(['n_build', 'campus_id', 'nearest_hour'])['consumption'].mean().reset_index()

# Create a line graph using Plotly Express
consumption_fig = px.line(avg_consumption_per_hour, x='nearest_hour', y='consumption', color='n_build',
              labels={'consumption': 'Average Consumption per hour', 'nearest_hour': 'Hour', 'n_build': 'Building'})

# Customize layout
consumption_fig.update_layout(title=f'Average Consumption per Hour in Kilowatt Hours - {selected_campus}',
                  xaxis_title='Hour',
                  yaxis_title='Average Consumption per Hour - KWH',
                  legend_title='Building ID')

# Show the plot
st.plotly_chart(consumption_fig)
# col2.markdown(f"What Influences Emissions?  Correlation Matrix for Emissions at {selected_campus}")
# col2.image(image_url)

#SHAPLEY DATA HERE

# Sample Shapley values data (replace this with your actual data)
# shapley_values_data = response_shap

# # Sample campus data (replace this with your actual data)
# campus_data = pd.DataFrame({
#     'campus_id': [1, 2, 3],
#     'campus_name': ['Campus A', 'Campus B', 'Campus C']
# })

# # Streamlit app
# st.title('Shapley Values for Selected Campus')

# # Filter Shapley values data based on the selected campus
# filtered_shapley_values = shapley_values_data  # Replace this line with actual filtering logic

# # Plot Shapley values
# fig = px.bar(filtered_shapley_values, x='feature', y='shapley_value', title=f'Shapley Values for Campus {selected_campus}')

# st.plotly_chart(fig)

#coefficients here
#streamlit ready version

# campus_name_to_id = {'Albury-Wodonga': 2, 'Bundoora': 1}

# # Example JSON data
# emissions_output_correlation_json = '[{"building_id":13,"gross_floor_area":-1.0,"campus_id":2.0,"building_age":-0.8709677419,"emissions_per_sqft":61.8328449241,"event_type_HVAC_Tuning":0.0},{"building_id":14,"gross_floor_area":-1.0001248712,"campus_id":2.0,"building_age":-1.0322580645,"emissions_per_sqft":64.6036272735,"event_type_HVAC_Tuning":0.0},{"building_id":30,"gross_floor_area":0.150608806,"campus_id":1.0,"building_age":-0.5161290323,"emissions_per_sqft":0.0039008497,"event_type_HVAC_Tuning":0.0},{"building_id":39,"gross_floor_area":0.0645741185,"campus_id":1.0,"building_age":0.0,"emissions_per_sqft":0.0945262561,"event_type_HVAC_Tuning":1.0},{"building_id":62,"gross_floor_area":0.0,"campus_id":1.0,"building_age":0.1290322581,"emissions_per_sqft":-0.0158739566,"event_type_HVAC_Tuning":0.0}]'

# # Convert JSON to DataFrame
# emissions_output_df = pd.read_json(emissions_output_correlation_json)

# # Map campus name to campus_id
# selected_campus_id = campus_name_to_id.get(selected_campus)


# # Filter DataFrame based on selected campus_id
# selected_emissions_output_df = emissions_output_df[emissions_output_df['campus_id'] == selected_campus_id]

# # Selected variable
# selected_variable = 'emissions_per_sqft'

# # Calculate Pearson correlation coefficients
# correlations = selected_emissions_output_df.corrwith(selected_emissions_output_df[selected_variable])

# # Create a horizontal heatmap with Plotly
# fig = px.imshow(correlations.to_frame().T,
#                 labels=dict(color=f'Correlation with {selected_variable}'),
#                 x=list(correlations.index),
#                 y=[selected_variable],
#                 color_continuous_scale='viridis',
#                 width=800,
#                 height=400)

# # Customize layout
# fig.update_layout(title_text=f'Pearson Correlation Coefficients with {selected_variable} - {selected_campus}',
#                     xaxis=dict(side='bottom'),
#                     yaxis=dict(tickmode='array', tickvals=[selected_variable], ticktext=['']))

# st.plotly_chart(fig)



def plot_shapley_values(shapley_json, selected_campus):
    # Mapping of campus names to IDs
    campus_name_to_id = {'Albury-Wodonga': 2, 'Bundoora': 1}

    # Convert the selected campus name to campus ID
    selected_campus_id = campus_name_to_id.get(selected_campus)

    if selected_campus_id is None:
        st.error(f"Invalid campus name: {selected_campus}")
        return None

    # Parse the JSON response
    shapley_data = []
    for key, values in shapley_json.items():
        if values['campus_id'] == selected_campus_id:
            row = {'campus_id': values['campus_id']}
            row.update(values)
            shapley_data.append(row)

    # Check if there is data for the selected campus
    if not shapley_data:
        st.warning(f"No data found for campus {selected_campus}")
        return None

    # Create a DataFrame
    shapley_df = pd.DataFrame(shapley_data)

    # Plotly bar chart
    fig = px.bar(shapley_df, x='campus_id', y=shapley_df.columns[1:],
                 title=f'Shapley Values for Features - Campus {selected_campus}',
                 labels={'value': 'Shapley Value', 'variable': 'Feature'},
                 barmode='group')

    return fig

# Streamlit app
def main():
    st.markdown("### What Affects Consumption?")
    st.markdown("Shapley values are a concept from game theory, attributing average marginal contributions to an outcome.  In this case, we look at the average marginal contribution of each factor to our target variable 'consumption'")

    # Plot Shapley values
    shap_fig = plot_shapley_values(response_shap, selected_campus)

    # Display the figure
    if shap_fig:
        st.plotly_chart(shap_fig)


if __name__ == "__main__":
    main()
st.markdown("""
#### Interpretation of Shapley Values:
**Positive Shapley Value**: If a feature has a positive Shapley value, it means that this feature contributed positively to the prediction, increasing the energy consumption. In simpler words, when this feature is present or takes a higher value, it tends to push the energy consumption higher.

**Negative Shapley Value**: Conversely, if a feature has a negative Shapley value, it means that this feature contributed negatively to the prediction, decreasing the energy consumption. In simpler terms, when this feature is present or takes a higher value, it tends to reduce the energy consumption.

**Magnitude of Shapley Value**: The magnitude (absolute value) of the Shapley value indicates the strength of the contribution. The larger the magnitude, the more influential the feature is in affecting the outcome.
""")
chart_url = 'https://user-images.githubusercontent.com/35371660/105487165-01af3500-5cf3-11eb-9243-c66de968798c.png'

# SECTION 2 - PREDICTIONS

st.markdown("""---""")

st.markdown("""
### Section 2 - Predictions
            """)
st.markdown('Run a few scenarios here to predict emissions for next year from electricity')
#PREDICTIONS DATA HERE
campus_name_to_id = {'Bundoora': 1, 'Albury-Wodonga': 2}

def get_predictions(selected_campus_name, n):
    # Convert the selected campus name to campus ID
    selected_campus_id = campus_name_to_id.get(selected_campus_name)

    if selected_campus_id is None:
        return {"error": f"Invalid campus name: {selected_campus_name}"}

    # predictions = response_predictions

    params = {'campus_id': selected_campus_id}
    predictions = requests.get(predictions_endpoint, params=params).json()

    # Extract the keys and values from the predictions dictionary for the selected campus
    dates = list(predictions["predictions"].keys())

    # Check if n is within the valid range
    if 1 <= n <= len(dates):
        # Create and return a list of dictionaries with the date, full_preds, lower_conf, and upper_conf for n,
        # as well as all the date & prediction values before it for the selected campus
        selected_predictions = [
            {
                "date": date,
                "full_preds": entry["full_preds"],
                "lower_conf": entry["lower_conf"],
                "upper_conf": entry["upper_conf"],
                "cost": entry['cost']
            }
            for date, entry in predictions["predictions"].items() if dates.index(date) < n
        ]
        return selected_predictions
    else:
        return {"error": f"Invalid input for n. Please choose a value between 1 and {len(dates)}"}

# Example usage:
# selected_campus_name = input("Enter campus name (Bundoora or Albury-Wodonga): ")
# user_input = int(input("Enter a number between 1 and 4: "))
time_span = st.slider('Choose a timeframe for prediction in days', 1,100,10)

#user_input = int(input("Enter a number between 1 and 4: "))
result = get_predictions(selected_campus, time_span)

# Extract the information from the result list
dates = [entry['date'] for entry in result]
full_preds_values = [entry['full_preds'] for entry in result]
upper_conf_values = [entry['upper_conf'] for entry in result]
lower_conf_values = [entry['lower_conf'] for entry in result]

# Create a Plotly line plot
preds_fig = go.Figure()

# Add traces for full_preds, upper_conf, and lower_conf
preds_fig.add_trace(go.Scatter(
    x=dates,
    y=full_preds_values,
    mode='lines+markers',
    name='Full Predictions',
    hovertemplate='%{x}: Predicted Cumulative KW Consumption: %{y:.2f}<extra></extra>',
))

preds_fig.add_trace(go.Scatter(
    x=dates,
    y=upper_conf_values,
    mode='lines',
    line=dict(dash='dash'),
    name='Upper Confidence',
    hovertemplate='%{x}: Upper Confidence: %{y:.2f}<extra></extra>',
))

preds_fig.add_trace(go.Scatter(
    x=dates,
    y=lower_conf_values,
    mode='lines',
    line=dict(dash='dash'),
    name='Lower Confidence',
    hovertemplate='%{x}: Lower Confidence: %{y:.2f}<extra></extra>',
))

def final_date(result, n):
    # Check if n is within the valid range
    if 1 <= n <= len(result):
        # Extract the 'full_pred' and 'cost' values for the nth date
        end_date = result[n - 1]["date"]
        date_object = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

        # Format the datetime object to '5 Jan 2022' format
        formatted_date = date_object.strftime('%d %b %Y')

        return formatted_date
    else:
        print(f"Invalid input for n. Please choose a value between 1 and {len(result)}")
        return None
finaldate = final_date(result, time_span)

# Set plot title and labels
preds_fig.update_layout(
    title="",
    xaxis_title='Date',
    yaxis_title='Predicted Consumption KWH',
    height=500,
    yaxis=dict(scaleanchor="x", scaleratio=0.1)
    # yaxis_range=[-5, 10],
)

# Update x-axis tick labels for better readability
preds_fig.update_xaxes(tickangle=45)

st.write(time_span, ' days out')

st.markdown(f"""
            ##### ARIMAX Time Series Model
            Predicting Until: {finaldate}
            """)
st.plotly_chart(preds_fig)
def calculate_cost(result, n):
    # Check if n is within the valid range
    if 1 <= n <= len(result):
        # Extract the 'full_pred' and 'cost' values for the nth date
        full_pred = result[n - 1]["full_preds"]
        cost = result[n - 1]["cost"]

        # Calculate the cost by multiplying 'full_pred' with 'cost'
        total_cost = full_pred * cost

        return total_cost
    else:
        print(f"Invalid input for n. Please choose a value between 1 and {len(result)}")
        return None

# Example usage:
# Assuming you have obtained the selected_predictions list from get_predictions function

def calculate_emissions(result, n):
    # Check if n is within the valid range
    if 1 <= n <= len(result):
        # Extract the 'full_pred' and 'cost' values for the nth date
        full_pred = result[n - 1]["full_preds"]

        return full_pred
    else:
        print(f"Invalid input for n. Please choose a value between 1 and {len(result)}")
        return None



total_cost = calculate_cost(result, time_span)
cost_per_sqft = total_cost / response_analytics[selected_campus]['total_sq_ft']
emissions_sqft = (calculate_emissions(result, time_span)*1.06) / response_analytics[selected_campus]['total_sq_ft']

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Emissions", f"{emissions_sqft:.7f}", "KG Co2 Per Square Foot")
col2.metric("Projected Electricity Costs - Total", f"${format_value(round(total_cost,2))}", "AUD")
col3.metric("Estimated Cost Per Square Foot", f"${(round(cost_per_sqft,4))}", "AUD")

st.write(f"Please note, forecasted costs are due to historical costs per KwH of electricity, as of {finaldate}")
