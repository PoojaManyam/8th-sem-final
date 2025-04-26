def get_recommended_crops(state):
    crop_recommendations = {
        "Maharashtra": {
            "pollutants": ["PM2.5", "NO2", "CO", "O3"],
            "crops": ["Jowar", "Sugarcane", "Soybean"]
        },
        "Karnataka": {
            "pollutants": ["CO", "SO2", "PM2.5"],
            "crops": ["Ragi", "Coffee", "Coconut"]
        },
        "Uttar Pradesh": {
            "pollutants": ["PM10", "O3", "CO"],
            "crops": ["Wheat", "Rice", "Sugarcane"]
        },
        "Punjab": {
            "pollutants": ["PM2.5", "NH3", "CO"],
            "crops": ["Wheat", "Rice", "Maize"]
        },
        "Tamil Nadu": {
            "pollutants": ["NO2", "SO2", "CO"],
            "crops": ["Millets", "Pulses", "Groundnut"]
        },
    }

    return crop_recommendations.get(state, {
        "pollutants": [],
        "crops": ["No specific crops available"]
    })
