# app.py
from flask import Flask, request, render_template, redirect, url_for
import joblib  # Using joblib as per the Python training script
import numpy as np
import pandas as pd # Needed for DataFrame creation for consistent preprocessing

app = Flask(__name__)

# Define paths for the saved model, scaler, and encoders
MODEL_PATH = 'traffic_model.joblib'
SCALER_PATH = 'scaler.joblib'
ENCODERS_PATH = 'encoders.joblib'

# Global variables for loaded components
model = None
scaler = None
encoders = None
training_features_order = None # To store the order of features during training

def load_ml_components():
    """
    Loads the machine learning model, scaler, and encoders.
    This function should be called once when the Flask app starts.
    """
    global model, scaler, encoders, training_features_order
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoders = joblib.load(ENCODERS_PATH)

        # Get feature order from the model or from the training script's X.columns
        # The training script uses X.columns.tolist() for training_features_order,
        # so we need to ensure this is replicated here.
        # This assumes the original X columns before scaling are consistent.
        # A more robust way would be to save X.columns.tolist() in training script
        # and load it here. For now, we manually define based on the training script's X.columns
        # order: ['temp', 'rain', 'snow', 'day', 'month', 'year', 'hours', 'minutes', 'seconds', 'holiday_encoded', 'weather_encoded']
        # Note: In the training script, X is formed by dropping 'traffic_volume' and then `holiday` and `weather` are replaced by `holiday_encoded` and `weather_encoded`.
        # The order in the `predict_traffic_volume` function's `processed_input_df` is:
        # ['temp', 'rain', 'snow', 'day', 'month', 'year', 'hours', 'minutes', 'seconds', 'holiday_encoded', 'weather_encoded']
        training_features_order = ['temp', 'rain', 'snow', 'day', 'month', 'year',
                                   'hours', 'minutes', 'seconds',
                                   'holiday_encoded', 'weather_encoded']


        print("✅ ML components (model, scaler, encoders) loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ Error: Required ML component file not found: {e}. Please ensure '{MODEL_PATH}', '{SCALER_PATH}', and '{ENCODERS_PATH}' are in the same directory as app.py")
        model = None
        scaler = None
        encoders = None
    except Exception as e:
        print(f"❌ An unexpected error occurred while loading ML components: {e}")
        model = None
        scaler = None
        encoders = None

# Load components when the app starts
load_ml_components()

# Home page with navigation
@app.route('/')
def home():
    return render_template('home.html')

# Prediction form handler
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        # Provide possible values for dropdowns (e.g., from encoder classes)
        # These should ideally be loaded from the encoders if they're always fixed.
        # For demonstration, hardcoding common values.
        holidays = encoders['holiday_encoder'].classes_.tolist() if encoders and 'holiday_encoder' in encoders else ['None', 'New Year\'s Day', 'Christmas Day']
        weathers = encoders['weather_encoder'].classes_.tolist() if encoders and 'weather_encoder' in encoders else ['Clear', 'Clouds', 'Rain', 'Snow', 'Fog', 'Mist']
        return render_template('predict.html', holidays=holidays, weathers=weathers)

    # If POST request, process the prediction
    if model is None or scaler is None or encoders is None or training_features_order is None:
        return "Prediction service not fully initialized. Please check server logs.", 500

    try:
        # Get data from form
        holiday_str = request.form['holiday']
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather_str = request.form['weather']
        date_str = request.form['date']
        time_str = request.form['Time']

        # Feature Engineering: Extract date & time components
        day, month, year = map(int, date_str.split('-'))
        hours, minutes, seconds = map(int, time_str.split(':'))

        # Preprocessing: Apply Label Encoding using loaded encoders
        # Ensure that the string values exist in the encoder's classes
        try:
            holiday_encoded = encoders['holiday_encoder'].transform([holiday_str])[0]
        except ValueError:
            return render_template('output.html', prediction_text=f"Error: Invalid 'holiday' value '{holiday_str}'. Please select from the provided options.")
        try:
            weather_encoded = encoders['weather_encoder'].transform([weather_str])[0]
        except ValueError:
            return render_template('output.html', prediction_text=f"Error: Invalid 'weather' value '{weather_str}'. Please select from the provided options.")

        # Create a DataFrame for the single input, ensuring column order matches training
        # The keys here must match the order in `training_features_order`
        input_data_dict = {
            'temp': temp,
            'rain': rain,
            'snow': snow,
            'day': day,
            'month': month,
            'year': year,
            'hours': hours,
            'minutes': minutes,
            'seconds': seconds,
            'holiday_encoded': holiday_encoded,
            'weather_encoded': weather_encoded
        }

        # Convert to DataFrame, ensuring column order
        input_df = pd.DataFrame([input_data_dict], columns=training_features_order)

        # Scale the input data using the loaded scaler
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        return render_template('output.html', prediction_text=f"Predicted Traffic Volume: {prediction:.2f} vehicles")

    except Exception as e:
        # Log the full error for debugging
        app.logger.error(f"Prediction error: {e}", exc_info=True)
        return render_template('output.html', prediction_text=f"An error occurred during prediction. Please check your inputs or contact support. Error details: {str(e)}")

# About page
@app.route('/about')
def about():
    return render_template('about.html')

# Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Team member profiles (Routes corrected for clarity and consistency)
@app.route('/contact/yaswanth')
def yaswanth():
    return render_template('yaswanth.html')

@app.route('/contact/jashwanth')
def jashwanth():
    return render_template('jashwanth.html')

@app.route('/contact/chaithu')
def chaithu(): # Changed function name to match route
    return render_template('chaithu.html')

# Run the server
if __name__ == '__main__':
    # It's generally not recommended to run debug=True in production
    # but it's useful for development.
    app.run(debug=True, port=3000)
