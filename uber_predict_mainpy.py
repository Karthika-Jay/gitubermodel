import streamlit as st
import pickle
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import base64
from streamlit_option_menu import option_menu

# Load the trained model
with open('uber_model_pickle.pkl', 'rb') as model_file:
    model_r = pickle.load(model_file)

# Load the dataset
df = pd.read_csv('streamlit.csv')

# Title of the app
st.markdown("<div style='background-color:rgb(0, 0, 0);padding:10px;text-align:center;'><h1 style='color:white;'>Uber Fare Amount Prediction</h1></div>", unsafe_allow_html=True)
tabs = ["👋😊Home", "📊Fare Predictions"]
selected_tab = st.selectbox(":orange[**SELECT AN OPTION👇**]", tabs)
if selected_tab == "👋😊Home":
        # Load the image and encode it in base64
    with open("C:/Users/lenovo/Desktop/try/uber_fare_prediction_git/Picture1.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # CSS to set the background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
    """
    <style>
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0; }
        100% { opacity: 1; }
    }
    .blink {
        animation: blink 1s infinite;
        font-size: 3em; 
        color: rainbow(10); 
        text-align: center; 
        display: block;
    }
    .container {
        text-align: center; 
    }
    </style>
    <div class='container'>
        <div class='blink'>🚗🆆🅴🅻🅲🅾🅼🅴🛺</div>
    </div>
    """,
    unsafe_allow_html=True
)

    st.markdown('''<div style="background-color:white;padding:6px;text-align:left;"><p style="color:black;font-size:20px;">

The project "Uber Fare Prediction and Streamlit Web Application" aims to develop a machine learning model to predict Uber ride fares based on various features extracted from ride data. The project includes the following key components:
                              
🔹Data Cleaning and Preprocessing: Handling missing values, converting data types, and performing exploratory data analysis.

🔹Feature Engineering: Extracting features from pickup_datetime, calculating trip distances, and segmenting data based on time of day and passenger count.
                
🔹Regression Modeling: Training a regression model to predict fare amounts.

🔹Model Evaluation: Assessing model performance using various metrics.
        
🔹Geospatial Analysis: Utilizing spatial data to enhance predictions.
        
🔹Time Series Analysis: Analyzing temporal patterns in ride data.
                
🔹 Time Series Analysis: Analyzing temporal patterns in ride data.
        
🔹Web Application Development: Creating a Streamlit web application to allow users to input ride details and receive fare estimates.
        
🔹Deployment: Deploying the web application on cloud platforms like AWS.
                
The project delivers a trained regression model, a functional Streamlit app, and detailed performance metrics. The dataset used is provided in CSV format and includes variables such as fare amount, pickup and dropoff locations, datetime, and passenger count. The project follows best practices for coding standards, version control, documentation, and data privacy.
                </p></div>''', unsafe_allow_html=True)
    
    st.image("C:/Users/lenovo/Desktop/try/uber_fare_prediction_git/ug1.gif")

elif selected_tab == "📊Fare Predictions":
    # Load the image and encode it in base64
    with open("C:/Users/lenovo/Desktop/try/uber_fare_prediction_git/ui5.jpeg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # CSS to set the background image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    # Function to filter DataFrame based on selections
    def filter_dataframe(df, passenger_count, year, month, day):
        filtered_df = df[(df['passenger_count'] == passenger_count) &
                        (df['Year'] == year) &
                        (df['Month'] == month) &
                        (df['Day'] == day)]
        return filtered_df

    # Selection box for passenger count
    passenger_count = st.selectbox(":rainbow[Passenger Count]", options=list(range(1, 7)), index=0)

    # Year, Month, Day selection boxes
    year_options = df['Year'].unique()
    month_options = df['Month'].unique()
    day_options = df['Day'].unique()

    Year = st.selectbox(":rainbow[Year]", options=year_options, index=0)
    Month = st.selectbox(":rainbow[Month]", options=month_options, index=0)
    Day = st.selectbox(":rainbow[Day]", options=day_options, index=0)

    # Filter DataFrame based on selected values
    filtered_df = filter_dataframe(df, passenger_count, Year, Month, Day)

    # Create selection boxes for pickup and dropoff coordinates based on filtered DataFrame
    pickup_longitude = st.selectbox(":rainbow[Pickup Longitude]", filtered_df['pickup_longitude'].unique())
    pickup_latitude = st.selectbox(":rainbow[Pickup Latitude]", filtered_df['pickup_latitude'].unique())
    dropoff_longitude = st.selectbox(":rainbow[Dropoff Longitude]", filtered_df['dropoff_longitude'].unique())
    dropoff_latitude = st.selectbox(":rainbow[Dropoff Latitude]", filtered_df['dropoff_latitude'].unique())

    # Filter distance travelled options based on the filtered DataFrame
    distance_travelled_km = st.selectbox(":rainbow[Distance Travelled_in_KM]", filtered_df['distance_travelled_km'].unique())

    # Select box for day of the week
    if 'Day_of_Week_num' in filtered_df.columns:
        day_of_week_num_options = filtered_df['Day_of_Week_num'].unique()
        Day_of_Week_num = st.selectbox(":rainbow[Day_of_Week_num]", options=day_of_week_num_options, index=0)
    else:
        st.error("Column 'Day_of_Week_num' not found in the dataset. Please check the column names.")
        Day_of_Week_num = 0  # Default value if column is missing

    # Hour selection
    hour_options = filtered_df['Hour'].unique() if 'Hour' in filtered_df.columns else list(range(0, 24))
    Hour = st.selectbox(":rainbow[Hour of the Day]", options=hour_options, index=0)

    # Mapping for seasons
    season_options = {
        1: "Summer",
        2: "Winter",
        3: "Fall",
        4: "Spring"
    }

    # Create a select box for season with user-friendly labels
    if 'season' in filtered_df.columns:
        season_options_keys = filtered_df['season'].unique()
        season_label = st.selectbox(":rainbow[☀️⛄Season🍂🌼]", options=season_options_keys, format_func=lambda x: season_options.get(x, x), index=0)
        season = next((key for key, value in season_options.items() if value == season_label), None)
    else:
        st.error("Column 'season' not found in the dataset. Please check the column names.")
        season = 1  # Default value if column is missing

    # Create the user_data array with all required features
    user_data = np.array([[pickup_longitude, pickup_latitude,
                        dropoff_longitude, dropoff_latitude, passenger_count,
                        distance_travelled_km, Year, Month, Day, Day_of_Week_num,
                        Hour, 1, season]])

    # Initialize Nominatim geocoder with a custom user agent
    geolocator = Nominatim(user_agent="your-unique-user-agent")

    # Button to predict the fare amount
    if st.button(":rainbow[Predict Fare Amount]"):
        
        # Make prediction
        y_pred = model_r.predict(user_data)
        
        # Attempt to retrieve pickup and dropoff locations
        try:
            pickup_location = df[(df['pickup_longitude'] == pickup_longitude) & 
                                (df['pickup_latitude'] == pickup_latitude)]['pickup'].iloc[0]
            dropoff_location = df[(df['dropoff_longitude'] == dropoff_longitude) & 
                                (df['dropoff_latitude'] == dropoff_latitude)]['drop off'].iloc[0]

            # Reverse geocode pickup location
            pickup_address = geolocator.reverse((pickup_latitude, pickup_longitude), exactly_one=True)
            pickup_full_address = pickup_address.address if pickup_address else "Address not found"

            # Reverse geocode dropoff location
            dropoff_address = geolocator.reverse((dropoff_latitude, dropoff_longitude), exactly_one=True)
            dropoff_full_address = dropoff_address.address if dropoff_address else "Address not found"

            # Display the prediction and location information
            st.write(f"<span style='color:#FF00FF;font-weight:bold;'>Predicted Fare Amount: 💸$</span><b>{round(y_pred[0], 2)}</b>", unsafe_allow_html=True)
            st.write(f"<span style='color:#FF00FF;font-weight:bold;'>Pickup Location:</span><b>{pickup_full_address}</b>", unsafe_allow_html=True)
            st.write(f"<span style='color:#FF00FF;font-weight:bold;'>Dropoff Location:</span><b>{dropoff_location}</b>, Address:</span><b>{dropoff_full_address}</b>", unsafe_allow_html=True)

            # Add a delay between requests
            time.sleep(1)

        except (GeocoderTimedOut, GeocoderServiceError) as e:
            st.error(f"Geocoding error: {e}")
        except IndexError:
            st.error("No matching location found for the selected coordinates.")
        st.write(" ")
        st.write(" ")
        st.markdown(
            """
            <style>
            @keyframes blink {
                0% { opacity: 1; }
                50% { opacity: 0; }
                100% { opacity: 1; }
            }
            .blink {
                animation: blink 1s infinite;
                font-size: 2em; 
                color: blue; 
                text-align: center; 
                display: block;
            }
            .container {
                text-align: center; 
            }
            </style>
            <div class='container'>
                <div class='blink'> GOT YOUR FARE AMOUNT💸!! 👋😊BYE!</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        # Create a centered column
        col1, col2, col3 = st.columns([1, 3, 1])

        # Display the GIF in the middle column
        with col2:
            st.image("C:/Users/lenovo/Desktop/try/uber_fare_prediction_git/ug2.gif")
        st.balloons()
