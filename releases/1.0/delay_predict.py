import requests
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

progress_val = 0

def getdata(airline):
    apiKey = "gQvOAvCu725jH4VcgCA300hbQJNwFfz2"
    apiUrl = "https://aeroapi.flightaware.com/aeroapi/"

    operators = airline
    payload = {'max_pages': 5}
    auth_header = {'x-apikey':apiKey}

    response = requests.get(apiUrl + f"operators/{operators}/flights",
    params=payload, headers=auth_header)

    if response.status_code == 200:
        output = response.json()
        return output
    else:
        return("Error executing request")

def getflight(callsign):
    apiKey = "gQvOAvCu725jH4VcgCA300hbQJNwFfz2"
    apiUrl = "https://aeroapi.flightaware.com/aeroapi/"

    flight = str(callsign)
    current_utc_time = datetime.datetime.utcnow()
    utc = current_utc_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    one_day_ago = current_utc_time - datetime.timedelta(days=1)  # gets all data of past 24 hrs
    utc_1 = one_day_ago.strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {'max_pages': 1,'start':utc_1,'end':utc}
    auth_header = {'x-apikey':apiKey}

    response = requests.get(apiUrl + f"flights/{flight}",
        params=payload, headers=auth_header)

    if response.status_code == 200:
        output = response.json()
        return output
    else:
        return("Error executing request")

def get_airport_arrivals(airport):
    apiKey = "gQvOAvCu725jH4VcgCA300hbQJNwFfz2"
    apiUrl = "https://aeroapi.flightaware.com/aeroapi/"

    airports = airport
    payload = {'max_pages': 2}
    auth_header = {'x-apikey':apiKey}

    response = requests.get(apiUrl + f"airports/{airports}/flights",
        params=payload, headers=auth_header)

    if response.status_code == 200:
        airport_output = response.json()
    else:
       return("Error executing request")

    airport_arrivals = pd.json_normalize(airport_output['arrivals'])
    airport_arrivals_cleaned=airport_arrivals[["ident_icao","operator","departure_delay","arrival_delay","aircraft_type","route_distance","origin.code_icao","destination.code_icao"]]
    return airport_arrivals_cleaned

def get_historical_flight(callsign):
    # API credentials and URL
    apiKey = "gQvOAvCu725jH4VcgCA300hbQJNwFfz2"
    apiUrl = "https://aeroapi.flightaware.com/aeroapi/"

    # Convert callsign to string
    flight = str(callsign)

    # Calculate UTC timestamps for historical data retrieval
    current_utc_time = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    utc = current_utc_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    T_15 = current_utc_time - datetime.timedelta(days=7)
    utc_1 = T_15.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Define payload for API request
    payload = {'max_pages': 4, 'start': utc_1, 'end': utc}
    auth_header = {'x-apikey': apiKey}

    # Make API request to get historical flight data
    response = requests.get(apiUrl + f"flights/{flight}",
                            params=payload, headers=auth_header)

    if response.status_code == 200:
        output = response.json()
        return output
    else:
        return "Error executing request"

def predict(flight_number):
    global progress_val

    # Get user inputted flight number
    """ target_flight = input('Input flight callsign. e.g."UAL1"') """
    target_flight = flight_number

    # Get current flight data
    flightdata = getflight(target_flight)
    flight = pd.json_normalize(flightdata['flights'])

    # Extract relevant information from current flight data
    airline = flight['operator_icao'][0]
    departure_delay = flight['departure_delay'][0]  # Use departure delay if the flight is en route
    aircraft = flight['aircraft_type'][0]
    dist = flight['route_distance'][0]  # Account for cycles above the arrival airport
    origin = flight['origin.code_icao'][0]
    destination = flight['destination.code_icao'][0]

    # Print the origin and update progress value
    print(origin)
    progress_val += 10

    # Get historical data on this flight
    historical_data = get_historical_flight(target_flight)
    print(historical_data)
    progress_val += 10

    # Extract relevant information from historical flight data
    historical_flight = pd.json_normalize(historical_data['flights'])
    historical_airline = historical_flight['operator_icao']
    historical_departure_delay = historical_flight['departure_delay']
    historical_arrival_delay = historical_flight['arrival_delay']
    historical_aircraft = historical_flight['aircraft_type']
    historical_dist = historical_flight['route_distance']
    historical_origin = historical_flight['origin.code_icao']
    historical_destination = historical_flight['destination.code_icao']

    # Create a DataFrame for historical flight data
    flight_requested_historical = pd.DataFrame({
        'operator': historical_airline,
        'departure_delay': historical_departure_delay,
        'aircraft_type': historical_aircraft,
        'route_distance': historical_dist,
        'origin.code_icao': historical_origin,
        'destination.code_icao': historical_destination,
        'arrival_delay': historical_arrival_delay
    })
    print(flight_requested_historical)
    progress_val += 10

    # Creating a DataFrame for the requested flight
    flight_requested = pd.DataFrame({
        'operator': [airline],
        'departure_delay': [departure_delay],
        'aircraft_type': [aircraft],
        'route_distance': [dist],
        'origin.code_icao': [origin],
        'destination.code_icao': [destination],
        'arrival_delay': [0]
    })
    print(flight_requested)
    progress_val += 10

    # Converting airline to string and retrieving flight data
    airline = str(airline)
    output = getdata(airline)
    scheduled = pd.json_normalize(output['scheduled'])
    arrivals = pd.json_normalize(output['arrivals'])

    # Combining requested flight data with arrivals and additional data
    arrivals_cleaned = pd.concat([flight_requested, arrivals[["ident_icao", "operator", "departure_delay", "arrival_delay", "aircraft_type", "route_distance", "origin.code_icao", "destination.code_icao"]]])
    destination_airport = get_airport_arrivals(destination)
    arrivals_cleaned = pd.concat([arrivals_cleaned, destination_airport])
    arrivals_cleaned = pd.concat([arrivals_cleaned, flight_requested_historical])

    # Handling missing values
    numeric_cols = arrivals_cleaned.select_dtypes(include=[int, float])
    arrivals_cleaned[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean())
    arrivals_cleaned = arrivals_cleaned.fillna('Unknown')

    # Creating features (X) and target variable (y)
    X = arrivals_cleaned[["operator", "departure_delay", "aircraft_type", "route_distance", "origin.code_icao", "destination.code_icao"]]
    y = arrivals_cleaned['arrival_delay']

    # Tokenization and Padding
    tokenizer = Tokenizer(num_words=10000)
    columns_to_tokenize = ['operator', 'aircraft_type', 'origin.code_icao', 'destination.code_icao']
    for column in columns_to_tokenize:
        tokenizer.fit_on_texts(X[column])
        sequences = tokenizer.texts_to_sequences(X[column])
        max_sequence_length = 20
        padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
        X[column] = np.array(sequences)

    # Train-test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and compile the neural network model
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=6))  # Input layer with 6 features
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='linear'))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error')  # Use an appropriate loss function for your task

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=1500, batch_size=32, validation_data=(X_test_scaled, y_test))

    # Evaluate the model on the test set
    loss = model.evaluate(X_test_scaled, y_test)
    print(f"Test Loss: {loss}")
    progress_val += 20

    # Predictions on the test set
    y_pred = model.predict(X_test_scaled)

    print(arrivals_cleaned)
    print(y_pred)
    print(y_test)
    print(X_test)

    # Create a DataFrame for the requested flight
    flight_requested = pd.DataFrame({
        'operator': [airline],
        'departure_delay': [departure_delay],
        'aircraft_type': [aircraft],
        'route_distance': [dist],
        'origin.code_icao': [origin],
        'destination.code_icao': [destination],
        'arrival_delay': [0]
    })
    print(flight_requested)
    progress_val += 10

    # Scale the input data for prediction
    flight_to_predict_scaled = scaler.fit_transform(X)
    ETA = model.predict(flight_to_predict_scaled)
    print("The estimated delay of this flight, in seconds, is:", ETA[0][0])
    progress_val += 30

    # Extract ETA values from the prediction results
    ETA_new = [eta[0] for eta in ETA]

    # Get actual arrival delays
    ATA = arrivals_cleaned['arrival_delay']

    # Visualize the relationship between actual and estimated delays
    # plt.scatter(ATA, ETA_new)
    # plt.xlabel('Actual Delay')
    # plt.ylabel('Estimated Delay')

    return ETA[0][0]