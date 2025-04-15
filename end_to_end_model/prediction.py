import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import joblib

# Initialize the Dash app
app = dash.Dash(__name__)

# Load the saved model
model = joblib.load('/home/hariprasath/python/machine_learning_python/end_to_end_model/house_price_model.pkl')

# Print expected features if available
try:
    expected_features = model.feature_names_in_
    print("Model expects features:", expected_features)
except AttributeError:
    expected_features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
    print("Defaulting to feature names:", expected_features)

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.H1("Real Estate Price Prediction", style={'text-align': 'center'}),

        html.Div([
            dcc.Input(id='distance_to_mrt', type='number', placeholder='Distance to MRT Station (meters)',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='num_convenience_stores', type='number', placeholder='Number of Convenience Stores',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='latitude', type='number', placeholder='Latitude',
                      style={'margin': '10px', 'padding': '10px'}),
            dcc.Input(id='longitude', type='number', placeholder='Longitude',
                      style={'margin': '10px', 'padding': '10px'}),
            html.Button('Predict Price', id='predict_button', n_clicks=0,
                        style={'margin': '10px', 'padding': '10px', 'background-color': '#007BFF', 'color': 'white'}),
        ], style={'text-align': 'center'}),

        html.Div(id='prediction_output', style={'text-align': 'center', 'font-size': '20px', 'margin-top': '20px'})
    ], style={'width': '50%', 'margin': '0 auto', 'border': '2px solid #007BFF', 'padding': '20px', 'border-radius': '10px'})
])

# Define callback to update output
@app.callback(
    Output('prediction_output', 'children'),
    Input('predict_button', 'n_clicks'),
    State('distance_to_mrt', 'value'),
    State('num_convenience_stores', 'value'),
    State('latitude', 'value'),
    State('longitude', 'value')
)
def update_output(n_clicks, distance_to_mrt, num_convenience_stores, latitude, longitude):
    try:
        if n_clicks > 0:
            if all(v is not None for v in [distance_to_mrt, num_convenience_stores, latitude, longitude]):
                # Prepare features based on model's expectation
                input_df = pd.DataFrame([[distance_to_mrt, num_convenience_stores, latitude, longitude]],
                                        columns=expected_features)

                print("Input features:\n", input_df)

                prediction = model.predict(input_df)[0]
                return f'🏠 Predicted House Price of Unit Area: {prediction:.2f}'

            else:
                return '⚠️ Please enter all values to get a prediction.'
        return ''
    except Exception as e:
        print("Error occurred during prediction:", str(e))
        return f'❌ Error occurred: {str(e)}'

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8051)

