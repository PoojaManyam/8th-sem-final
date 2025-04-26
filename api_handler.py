import requests

# add api from OpenWeather C:\Users\DELL\Downloads\pollution tracker_ML (2)\pollutiontracker\api_handler.py
API_KEY = "68b32df855b0fc961ff05362aa540650"
BASE_URL = "http://api.openweathermap.org/data/2.5/air_pollution"


def get_pollution_data(city):
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
    geo_response = requests.get(geo_url).json()

    if not geo_response:
        return None

    lat, lon = geo_response[0]['lat'], geo_response[0]['lon']
    response = requests.get(f"{BASE_URL}?lat={lat}&lon={lon}&appid={API_KEY}")
    data = response.json()

    if 'list' in data:
        pollutants = data['list'][0]['components']
        return pollutants
    return None
