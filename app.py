import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.markdown("""# Welcome to The UNICON Campus Emissions Predictor
## Predict carbon emissions at one of Australia's largest universities.
This project uses the open-source UNICON dataset from La Trobe University, which has collected emissions data for the past 5 years on a unified platform for the purposes of research and prediction.""")
st.markdown("""---""")


st.markdown("""
### Secion 1 - Analytics
            """)

selected_campus = st.selectbox(
    'View Emissions Statistics by Campus',
    ('Campus 1 - Bundoora', 'Campus 2 - Albury-Wodonga'))

st.write('Now viewing metrics for:', selected_campus)

col1, col2, col3 = st.columns(3)
col1.metric("Number of Buildings", "5", "-$1.25")
col2.metric("Total Square Feet", "XXXX", "0.46%")
col3.metric("Years of Emissions", "XXXX", "+4.87%")

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

prediction = st.slider('Choose a timeframe for prediction in years', 1,5,1)
st.write(prediction, ' years out')

st.markdown("ARIMAX Time Series Model Here")
st.image(chart_url)
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Emissions Per Square Foot", "100", "-12% vs last year")
col2.metric("Estimated Cost - Total", "$100", "AUD")
col3.metric("Estimated Cost Per Square Foot", "$10", "AUD")

# scaled_5min_bldg_data = pd.read_csv('scaled_data/agg_building_data_filtered_encoded_scaled_12JAN_final.csv')

# scaled_5min_bldg_data = pd.DataFrame(scaled_5min_bldg_data)

# emissions_output_scaled = scaled_5min_bldg_data.groupby('building_id').agg({
#     'gross_floor_area': 'first',
#     'building_age': 'first',
#     'consumption' : 'sum',
#     'emissions_per_sqft_yearly': 'mean',
#     'co2_from_electric': 'sum'
# }).reset_index()


# correlation_matrix_scaled = emissions_output_scaled.corr()

# # Create an interactive correlation heatmap with Plotly Express
# fig = px.imshow(
#     correlation_matrix_scaled,
#     x=correlation_matrix_scaled.index,
#     y=correlation_matrix_scaled.columns,
#     color_continuous_scale='Viridis',
#     title='Correlation Heatmap',
#     height=600,
#     width=800
# )

# # Show the plot
# fig.show()

# st.plotly_chart(fig)
