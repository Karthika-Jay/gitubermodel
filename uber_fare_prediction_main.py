import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import streamlit as st
import joblib

# Load dataset
@st.cache
def load_data():
    data = pd.read_csv('uber_ride_data.csv')
    return data

# Preprocess data
def preprocess_data(data):
    # Handle missing values
    data = data.dropna()
    
    # Extract features from pickup_datetime
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['hour'] = data['pickup_datetime'].dt.hour
    data['day'] = data['pickup_datetime'].dt.day
    data['month'] = data['pickup_datetime'].dt.month
    data['year'] = data['pickup_datetime'].dt.year
    data['day_of_week'] = data['pickup_datetime'].dt.dayofweek
    
    # Calculate trip distances
    data['trip_distance'] = np.sqrt(
        (data['dropoff_longitude'] - data['pickup_longitude']) ** 2 +
        (data['dropoff_latitude'] - data['pickup_latitude']) ** 2
    )
    
    # Segment data based on time of day
    data['time_of_day'] = pd.cut(data['hour'], bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'])
    
    return data

# Train model
def train_model(data):
    X = data[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'hour', 'day', 'month', 'year', 'day_of_week', 'trip_distance']]
    y = data['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return model, rmse

# Save model
def save_model(model, filename='uber_fare_model.pkl'):
    joblib.dump(model, filename)

# Load model
def load_model(filename='uber_fare_model.pkl'):
    model = joblib.load(filename)
    return model

# Streamlit app
def main():
    st.title("Uber Fare Prediction")
    
    model = load_model()
    
    pickup_longitude = st.number_input("Pickup Longitude")
    pickup_latitude = st.number_input("Pickup Latitude")
    dropoff_longitude = st.number_input("Dropoff Longitude")
    dropoff_latitude = st.number_input("Dropoff Latitude")
    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, step=1)
    hour = st.number_input("Hour of Pickup", min_value=0, max_value=23, step=1)
    day = st.number_input("Day of Pickup", min_value=1, max_value=31, step=1)
    month = st.number_input("Month of Pickup", min_value=1, max_value=12, step=1)
    year = st.number_input("Year of Pickup", min_value=2009, max_value=2024, step=1)
    day_of_week = st.number_input("Day of Week", min_value=0, max_value=6, step=1)
    
    if st.button("Predict Fare"):
        trip_distance = np.sqrt((dropoff_longitude - pickup_longitude) ** 2 + (dropoff_latitude - pickup_latitude) ** 2)
        features = np.array([[pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count, hour, day, month, year, day_of_week, trip_distance]])
        fare = model.predict(features)
        st.write(f"Estimated Fare: ${fare[0]:.2f}")
        
if __name__ == "__main__":
    data = load_data()
    data = preprocess_data(data)
    model, rmse = train_model(data)
    save_model(model)
    main()
