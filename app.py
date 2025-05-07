from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Load model and scalers
model = pickle.load(open('model.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/smart-crop')
def smart_crop():

    
    return render_template('smart_crop.html')

@app.route('/soil-analysis')
def soil_analysis():
    return render_template('soil_analysis.html')

@app.route('/weather-insights')
def weather_insights():
    return render_template('weather_insights.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form values
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorus'])
        K = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare and scale input
        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaled_input = ms.transform(input_features)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        # Get probabilities for all crops
        probabilities = model.predict_proba(scaled_input)[0]
        crop_classes = model.classes_

        # Pair crops with their probabilities
        crop_confidence = sorted(
            zip(crop_classes, probabilities),
            key=lambda x: x[1],
            reverse=True
        )

        # Send crop names and probabilities to frontend for graph
        chart_labels = [crop for crop, _ in crop_confidence]
        chart_data = [round(prob * 100, 2) for _, prob in crop_confidence]

        return render_template(
            'index.html',
            prediction=prediction,
            crop_confidence=crop_confidence,
            chart_labels=[crop for crop , _ in crop_confidence],
            chart_data=[round(prob * 100, 2) for _, prob in crop_confidence]
        )

    except Exception as e:
        return f"Error: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)