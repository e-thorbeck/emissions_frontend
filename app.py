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
emissions_endpoint = HOST + 'campuses_year_info'
shap_endpoint = HOST + 'shap'

response_emissions = requests.get(emissions_endpoint).json()
response_analytics = requests.get(analytics_endpoint).json()
response_shap = requests.get(shap_endpoint).json()
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
### Section 1 - Analytics
            """)

selected_campus = st.selectbox(
    'View Emissions Statistics by Campus',
    (response_analytics.keys()))

st.write('Now viewing metrics for:', selected_campus)

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




col1, col2, col3 = st.columns(3)
col1.metric("Number of Buildings", round(response_analytics[selected_campus]['n_build']), "Buildings")
col2.metric("Total Square Feet", round(response_analytics[selected_campus]['total_sq_ft']), "Campus Footprint")
col3.metric("Total Emissions", round(response_analytics[selected_campus]['total_missions']), "kg of Co2 emitted from electricity")

image_url = 'https://images.unsplash.com/photo-1548407260-da850faa41e3?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1487&q=80'

st.plotly_chart(emissions_fig)
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
    st.title("Shapley Values Visualization")

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
user_input = st.slider('Choose a timeframe for prediction in days', 1,50,1)

#user_input = int(input("Enter a number between 1 and 4: "))
result = get_predictions(selected_campus, user_input)

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

# Set plot title and labels
preds_fig.update_layout(
    title='Predictions Over Time',
    xaxis_title='Date',
    yaxis_title='Predicted Consumption KWH',
    height=500,
    yaxis=dict(scaleanchor="x", scaleratio=0.1)
    # yaxis_range=[-5, 10],
)

# Update x-axis tick labels for better readability
preds_fig.update_xaxes(tickangle=45)

# Show the plot
# preds_fig.show()
# user_input = st.slider('Choose a timeframe for prediction in years', 1,4,1)

# #user_input = int(input("Enter a number between 1 and 4: "))
# result = get_predictions(selected_campus, user_input)


# # Extract the information from the result list
# dates = [entry['date'] for entry in result]
# full_preds_values = [entry['full_preds'] for entry in result]
# upper_conf_values = [entry['upper_conf'] for entry in result]
# lower_conf_values = [entry['lower_conf'] for entry in result]

# # Create a Plotly line plot
# fig = go.Figure()

# # Add traces for full_preds, upper_conf, and lower_conf
# fig.add_trace(go.Scatter(
#     x=dates,
#     y=full_preds_values,
#     mode='lines+markers',
#     name='Full Predictions',
#     hovertemplate='%{x}: Predicted Emissions: %{y:.2f}<extra></extra>',
# ))

# fig.add_trace(go.Scatter(
#     x=dates,
#     y=upper_conf_values,
#     mode='lines',
#     line=dict(dash='dash'),
#     name='Upper Confidence',
#     hovertemplate='%{x}: Upper Confidence: %{y:.2f}<extra></extra>',
# ))

# fig.add_trace(go.Scatter(
#     x=dates,
#     y=lower_conf_values,
#     mode='lines',
#     line=dict(dash='dash'),
#     name='Lower Confidence',
#     hovertemplate='%{x}: Lower Confidence: %{y:.2f}<extra></extra>',
# ))

# # Set plot title and labels
# fig.update_layout(
#     title='Predictions Over Time',
#     xaxis_title='Date',
#     yaxis_title='Prediction Values',
#     # margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins as needed
#     height=500,  # Adjust height as needed
#     # width=800,  # Adjust width as needed
#     yaxis_range=[-5,5],  # Adjust scaleratio to control the distance between lines
# )

# # Update x-axis tick labels for better readability
# fig.update_xaxes(tickangle=45)

# # Update x-axis tick labels for better readability
# fig.update_xaxes(tickangle=45)
# ###

# #prediction = st.slider('Choose a timeframe for prediction in years', 1,4,1)
st.write(user_input, ' days out')

st.markdown("ARIMAX Time Series Model Here")
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

total_cost = calculate_cost(result, user_input)
cost_per_sqft = total_cost / response_analytics[selected_campus]['total_sq_ft']
emissions_sqft = calculate_emissions(result, user_input) / response_analytics[selected_campus]['total_sq_ft']

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Emissions", round(emissions_sqft,5), "KG Co2 Per Square Foot")
col2.metric("Estimated Electricity Costs - Total", f"${round(total_cost)}", "AUD")
col3.metric("Estimated Cost Per Square Foot", f"${round(cost_per_sqft,4)}", "AUD")
