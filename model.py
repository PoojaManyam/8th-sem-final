import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
file_path = 'Air_dataset.csv'
dataset = pd.read_csv(file_path)

# Drop unnecessary columns
data = dataset.drop(columns=['Unnamed: 0', 'AQI', 'AQI_Bucket', 'Date'])
data = data.dropna()

# Encode city names
label_encoder = LabelEncoder()
data['City'] = label_encoder.fit_transform(data['City'])

# Define features and targets
features = ['City']
target_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
                  'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
X = data[features]
y = data[target_columns]

# Normalize targets (LSTM works better with normalized data)
scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler.fit_transform(y)

# Reshape features and targets for LSTM
# LSTM expects 3D input (samples, timesteps, features)
X_reshaped = np.array(X).reshape(X.shape[0], 1, X.shape[1])
y_reshaped = np.array(y_scaled)  # Output remains 2D (samples, target_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_reshaped, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(
    X_train.shape[1], X_train.shape[2])))
model.add(Dense(y_train.shape[1]))  # Output layer for all pollutants

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
y_pred_scaled = model.predict(X_test)
# Convert predictions back to original scale
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_original = scaler.inverse_transform(y_test)

# Calculate RMSE for each pollutant
rmse = np.sqrt(mean_squared_error(
    y_test_original, y_pred, multioutput='raw_values'))
print("RMSE for each pollutant:", rmse)

# Save the model and label encoder
model.save('air_pollution_model.h5')  # Save the LSTM model
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model, Label Encoder, and Scaler saved successfully!")
