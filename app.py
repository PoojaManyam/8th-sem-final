import os
import joblib
import pandas as pd
import requests
import random
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from api_handler import get_pollution_data
from state_crops import get_recommended_crops

app = Flask(__name__)

# Ensure the static folder exists for saving images
if not os.path.exists("static"):
    os.makedirs("static")

# OpenWeatherMap API Key (Replace with your actual key)
OPENWEATHER_API_KEY = "68b32df855b0fc961ff05362aa540650"

# Load the trained pollution prediction model and label encoder
# Ensure this file exists
model = joblib.load("air_pollution_model.pkl")
# Label encoder for city names
label_encoder = joblib.load("label_encoder.pkl")

# List of cities from the trained dataset
cities = list(label_encoder.classes_)

# Pollution categories
pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
              'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

# Air Quality Index Ranges
AQI_RANGES = {
    "Good": {"SO2": (0, 20), "NO2": (0, 40), "PM10": (0, 20), "PM2.5": (0, 10), "O3": (0, 60), "CO": (0, 4400)},
    "Fair": {"SO2": (20, 80), "NO2": (40, 70), "PM10": (20, 50), "PM2.5": (10, 25), "O3": (60, 100), "CO": (4400, 9400)},
    "Moderate": {"SO2": (80, 250), "NO2": (70, 150), "PM10": (50, 100), "PM2.5": (25, 50), "O3": (100, 140), "CO": (9400, 12400)},
    "Poor": {"SO2": (250, 350), "NO2": (150, 200), "PM10": (100, 200), "PM2.5": (50, 75), "O3": (140, 180), "CO": (12400, 15400)},
    "Very Poor": {"SO2": (350, float('inf')), "NO2": (200, float('inf')), "PM10": (200, float('inf')), "PM2.5": (75, float('inf')), "O3": (180, float('inf')), "CO": (15400, float('inf'))}
}

# Function to determine AQI category


def determine_aqi_category(pollution_data):
    category_scores = {category: 0 for category in AQI_RANGES.keys()}

    for pollutant, value in pollution_data.items():
        for category, ranges in AQI_RANGES.items():
            if pollutant in ranges:
                min_val, max_val = ranges[pollutant]
                if min_val <= value < max_val:
                    category_scores[category] += 1

    return max(category_scores, key=category_scores.get)

# Function to get latitude & longitude of a city


def get_city_coordinates(city_name):
    geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={OPENWEATHER_API_KEY}"

    response = requests.get(geocode_url)
    data = response.json()

    if data:
        return data[0]['lat'], data[0]['lon']
    return None, None


@app.route('/')
def home():
    return render_template('index.html', cities=cities)


@app.route('/pollution', methods=['GET', 'POST'])
def pollution():
    import csv
    import datetime

    city = None
    city_lat, city_lng = None, None
    data = None
    aqi_category = None

    if request.method == "POST":
        city = request.form.get("city")
        city_lat, city_lng = get_city_coordinates(city)

        if city_lat and city_lng:
            data = get_pollution_data(city)
            aqi_category = determine_aqi_category(data)

            # Append the data to air.csv
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv_filename = "air.csv"

            with open(csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                pollution_values = list(data.values()) if data else []
                row = ["Null", city, timestamp,
                       aqi_category] + pollution_values
                writer.writerow(row)

    return render_template(
        "pollution.html",
        city=city,
        city_lat=city_lat,
        city_lng=city_lng,
        data=data,
        aqi_category=aqi_category
    )


@app.route('/comparison', methods=['GET', 'POST'])
def comparison():
    state1 = request.form.get('state1')
    state2 = request.form.get('state2')

    if state1 and state2:
        data1 = get_pollution_data(state1)
        data2 = get_pollution_data(state2)

        if data1 and data2:
            pollutants = list(data1.keys())
            state1_values = [data1[p] for p in pollutants]
            state2_values = [data2[p] for p in pollutants]

            # Create a static line graph using Matplotlib
            plt.figure(figsize=(10, 5))
            plt.plot(pollutants, state1_values, marker='o',
                     linestyle='-', label=state1, color='blue')
            plt.plot(pollutants, state2_values, marker='s',
                     linestyle='--', label=state2, color='red')

            plt.xlabel("Pollutants")
            plt.ylabel("Pollution Levels (µg/m³)")
            plt.title(f"Pollution Comparison: {state1} vs {state2}")
            plt.legend()
            plt.grid(True)

            # Save the graph as an image
            graph_path = "static/comparison_graph.png"
            plt.savefig(graph_path)
            plt.close()

            return render_template('comparison.html', data1=data1, data2=data2, state1=state1, state2=state2)

    return render_template('comparison.html')


@app.route('/remedies', methods=['POST'])
def remedies():
    city = request.form.get('state')  # this is actually the city

    # City to State Mapping
    city_to_state = {
        "bangalore": "Karnataka",
        "bengaluru": "Karnataka",
        "harohalli": "Karnataka",
        "mumbai": "Maharashtra",
        "pune": "Maharashtra",
        "chennai": "Tamil Nadu",
        "coimbatore": "Tamil Nadu",
        "lucknow": "Uttar Pradesh",
        "kanpur": "Uttar Pradesh",
        "ludhiana": "Punjab",
        "amritsar": "Punjab"
    }

    state = city_to_state.get(city.lower(), "Unknown")

    if state == "Unknown":
        remedies = {"pollutants": [], "crops": [
            "General Greenery: Neem, Tulsi, Peepal"]}
    else:
        remedies = get_recommended_crops(state)

    pollution_data = get_pollution_data(city)

    if pollution_data:
        dominant_pollutant = max(pollution_data, key=pollution_data.get)
    else:
        dominant_pollutant = "Unknown"

    # Check for pollutant match (case-insensitive)
    if dominant_pollutant and dominant_pollutant.upper() in [p.upper() for p in remedies["pollutants"]]:
        crop_suggestion = remedies["crops"]
    else:
        # Show both state crops + general greenery
        crop_suggestion = remedies["crops"] + \
            ["General Greenery: Neem, Tulsi, Peepal"]

    return render_template(
        'remedies.html',
        city=city,
        state=state,
        remedies=remedies,
        pollutant=dominant_pollutant,
        crops=crop_suggestion
    )


@app.route('/predict_page', methods=['GET'])
def predict_page():
    # Ensure predict.html exists
    return render_template('predict.html', cities=cities)


@app.route('/predict', methods=['POST'])
def predict():
    import csv
    import datetime

    city = request.form.get("city")

    if city not in cities:
        return jsonify({"error": "City not found in the dataset"}), 400

    # Encode city
    city_encoded = label_encoder.transform([city])[0]

    # Predict pollution levels
    input_data = pd.DataFrame([[city_encoded]], columns=["City"])
    predictions = model.predict(input_data)
    predictions_df = pd.DataFrame(predictions, columns=pollutants)

    # Extract pollution parameters from predictions
    prediction_data = predictions_df.iloc[0].to_dict()
    pm25 = prediction_data.get("PM2.5", 0.0)
    pm10 = prediction_data.get("PM10", 0.0)
    no2 = prediction_data.get("NO2", 0.0)
    o3 = prediction_data.get("O3", 0.0)
    co = prediction_data.get("CO", 0.0)
    so2 = prediction_data.get("SO2", 0.0)
    nh3 = prediction_data.get("NH3", 0.0)
    aqi = prediction_data.get("AQI", 0.0)
    aqi_category = determine_aqi_category({"AQI": aqi})

    # Timestamp for the data
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare row for appending with "null" for city_code
    row = [
        "null", city, timestamp, pm25, pm10, no2, o3, co, so2, nh3, aqi,
        aqi_category
    ]

    # Append to air.csv
    csv_filename = "Air_dataset.csv"
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Add header if the file is new
        if not file_exists:
            writer.writerow([
                "City Code", "City", "Timestamp", "PM2.5", "PM10", "NO2", "O3", "CO",
                "SO2", "NH3", "AQI", "AQI Category"
            ])

        # Write the data row
        writer.writerow(row)

    return jsonify(predictions_df.to_dict(orient="records")[0])


if __name__ == '__main__':
    app.run(debug=True)
