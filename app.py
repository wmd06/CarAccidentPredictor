import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.pipeline import Pipeline
import requests
from datetime import datetime

# Load the saved model once during app initialization
try:
    model = joblib.load('random_forest_model.pkl')
except FileNotFoundError:
    raise ValueError("Model not trained. Please train the model first.")
    
def get_weather_data(lat, lon):
    url = f"https://open-weather13.p.rapidapi.com/city/latlon/{lat}/{lon}"
    headers = {
        "X-RapidAPI-Host": "open-weather13.p.rapidapi.com",
        "x-rapidapi-key": "62a01963aemsh4d8cd8836950d56p11efa6jsndb1fa013d449"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Weather API Error: {response.status_code} - {response.text}")

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live')
def livepage():
    return render_template('live.html')

@app.route('/live/predict', methods=['POST'])
def live():
    try:
        current_datetime = datetime.now()

        start_lng = float(request.form['Start_Lng'])
        start_lat = float(request.form['Start_Lat'])

        weather_data = get_weather_data(start_lat, start_lng)

        temp = float((weather_data['main']['temp'] - 273.15) * 9/5 + 32)
        humidity = float(weather_data['main']['humidity'])
        visibility = float(weather_data['visibility']/10000)
        pressure = float(weather_data['main']['pressure'] * 0.02953)
        wind_speed = float(weather_data['wind']['speed'] * 2.23694)
        wind_chill = float(35.74 + (0.6215*temp) - (35.75* wind_speed**0.16) + (0.4275*temp*wind_speed**0.16))

        # Get input data from form
        input_data = {
                'Start_Lng': start_lng,
                'Start_Lat': start_lat,
                'time': current_datetime.hour,  # Current hour (0–23)
                'month': current_datetime.month,  # Current month (1–12)
                'day_of_week': current_datetime.weekday(),
                # Weather condition is encoded, so we use the numeric value for weather
                'weather_encoded': int(request.form['weather_encoded']),  # Weather condition encoded
                'Temperature(F)': temp,  # Temperature input
                'Wind_Chill(F)': wind_chill,  # Wind chill input
                'Humidity(%)': humidity,  # Humidity input
                'Wind_Speed(mph)': wind_speed, # Wind speed input
                'Precipitation(in)': float(request.form['precipitation']),
                'Visibility(mi)': visibility,  # Visibility input
                'Pressure(in)': pressure,  # Pressure input
                
                # Road conditions are binary (0 or 1)
                'traffic_signal': int(request.form.get('traffic_signal', 0)),
                'bump': int(request.form.get('bump', 0)),
                'crossing': int(request.form.get('crossing', 0)),
                'give_way': int(request.form.get('give_way', 0)),
                'junction': int(request.form.get('junction', 0)),
                'no_exit': int(request.form.get('no_exit', 0)),
                'railway': int(request.form.get('railway', 0)),
                'roundabout': int(request.form.get('roundabout', 0)),
                'stop': int(request.form.get('stop', 0)),
                'traffic_calming': int(request.form.get('traffic_calming', 0)),
                'turning_loop': int(request.form.get('turning_loop', 0))
        }
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict the probabilities for both classes (accident and no accident)
        probabilities = model.predict_proba(input_df)

        # The probability for class 1 (accident)
        accident_probability = probabilities[0][1]  # Class 1 corresponds to an accident

        # Define the risk levels based on the model's predicted probability
        if accident_probability < 0.25:
            predicted_risk = 'Low'
        elif 0.25 <= accident_probability < 0.5:
            predicted_risk = 'Medium-Low'
        elif 0.5 <= accident_probability < 0.75:
            predicted_risk = 'Medium-High'
        else:
            predicted_risk = 'High'

        # Return the result along with the predicted risk and the accident probability
        return render_template('result.html', 
                            risk=predicted_risk, 
                            probability=accident_probability.round(3))

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:

        # Get input data from form
        input_data = {
                'Start_Lng': float(request.form['Start_Lng']),  # Longitude input
                'Start_Lat': float(request.form['Start_Lat']),  # Latitude input
                'time': int(request.form['time']),  # Time of day (integer)
                'month': int(request.form['month']),  # Month input
                'day_of_week': int(request.form['day_of_week']),  # Day of week (0=Monday, 6=Sunday)
                
                # Weather condition is encoded, so we use the numeric value for weather
                'weather_encoded': int(request.form['weather_encoded']),  # Weather condition encoded

                'Temperature(F)': float(request.form['temperature']),
                'Wind_Chill(F)': float(request.form['wind_chill']),  # Wind chill input
                'Humidity(%)': float(request.form['humidity']),
                'Wind_Speed(mph)': float(request.form['wind_speed']),
                'Precipitation(in)': float(request.form['precipitation']),
                'Visibility(mi)': float(request.form['visibility']),
                'Pressure(in)': float(request.form['pressure']),

                # Road conditions are binary (0 or 1)
                'traffic_signal': int(request.form.get('traffic_signal', 0)),
                'bump': int(request.form.get('bump', 0)),
                'crossing': int(request.form.get('crossing', 0)),
                'give_way': int(request.form.get('give_way', 0)),
                'junction': int(request.form.get('junction', 0)),
                'no_exit': int(request.form.get('no_exit', 0)),
                'railway': int(request.form.get('railway', 0)),
                'roundabout': int(request.form.get('roundabout', 0)),
                'stop': int(request.form.get('stop', 0)),
                'traffic_calming': int(request.form.get('traffic_calming', 0)),
                'turning_loop': int(request.form.get('turning_loop', 0))
        }


        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict the probabilities for both classes (accident and no accident)
        probabilities = model.predict_proba(input_df)

        # The probability for class 1 (accident)
        accident_probability = probabilities[0][1]  # Class 1 corresponds to an accident

        # Define the risk levels based on the model's predicted probability
        if accident_probability < 0.25:
            predicted_risk = 'Low'
        elif 0.25 <= accident_probability < 0.5:
            predicted_risk = 'Medium-Low'
        elif 0.5 <= accident_probability < 0.75:
            predicted_risk = 'Medium-High'
        else:
            predicted_risk = 'High'

        # Return the result along with the predicted risk and the accident probability
        return render_template('result.html', 
                            risk=predicted_risk, 
                            probability=accident_probability.round(3))

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Uncomment and run this once to save the model
    # from your_training_script import predictor, model
    # save_model(model, predictor)
    app.run(debug=True)