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
    one_day_ago = current_utc_time - datetime.timedelta(days=1)
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
    apiKey = "gQvOAvCu725jH4VcgCA300hbQJNwFfz2"
    apiUrl = "https://aeroapi.flightaware.com/aeroapi/"

    flight = str(callsign)
    current_utc_time = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    utc = current_utc_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    T_15 = current_utc_time - datetime.timedelta(days=7)
    utc_1 = T_15.strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {'max_pages': 4,'start':utc_1,'end':utc}
    auth_header = {'x-apikey':apiKey}

    response = requests.get(apiUrl + f"flights/{flight}",
        params=payload, headers=auth_header)

    if response.status_code == 200:
        output = response.json()
        return output
    else:
        return("Error executing request")

def predict(flight_number):
    #target_flight = input('Input flight callsign. e.g."UAL1"')

    target_flight = flight_number

    flightdata = getflight(target_flight)
    flight = pd.json_normalize(flightdata['flights'])
    airline = flight['operator_icao'][0]
    departure_delay = flight['departure_delay'][0]
    aircraft = flight['aircraft_type'][0]
    dist = flight['route_distance'][0]
    origin = flight['origin.code_icao'][0]
    destination = flight['destination.code_icao'][0]
    print(origin)
    historical_data = get_historical_flight(target_flight)
    print(historical_data)

    historical_flight = pd.json_normalize(historical_data['flights'])
    historical_airline = historical_flight['operator_icao']
    historical_departure_delay = historical_flight['departure_delay']
    historical_arrival_delay = historical_flight['arrival_delay']
    historical_aircraft = historical_flight['aircraft_type']
    historical_dist = historical_flight['route_distance']
    historical_origin = historical_flight['origin.code_icao']
    historical_destination = historical_flight['destination.code_icao']
    flight_requested_historical = pd.DataFrame({'operator':historical_airline,'departure_delay':historical_departure_delay,'aircraft_type':historical_aircraft,'route_distance':historical_dist,'origin.code_icao':historical_origin,'destination.code_icao':historical_destination,'arrival_delay':historical_arrival_delay})
    print(flight_requested_historical)

    flight_requested = pd.DataFrame({'operator': [airline], 'departure_delay': [departure_delay], 'aircraft_type': [aircraft], 'route_distance': [dist], 'origin.code_icao': [origin], 'destination.code_icao': [destination], 'arrival_delay': [0]})
    print(flight_requested)
    airline = str(airline)
    output = getdata(airline)
    scheduled = pd.json_normalize(output['scheduled'])
    arrivals = pd.json_normalize(output['arrivals'])
    arrivals_cleaned = pd.concat([flight_requested, arrivals[["ident_icao","operator","departure_delay","arrival_delay","aircraft_type","route_distance","origin.code_icao","destination.code_icao"]]])  # Fixed the syntax error here
    destination_airport = get_airport_arrivals(destination)
    arrivals_cleaned = pd.concat([arrivals_cleaned, destination_airport])
    arrivals_cleaned = pd.concat([arrivals_cleaned, flight_requested_historical])
    numeric_cols = arrivals_cleaned.select_dtypes(include=[int, float])
    arrivals_cleaned[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean())
    arrivals_cleaned = arrivals_cleaned.fillna('Unknown')

    X = arrivals_cleaned[["operator","departure_delay","aircraft_type","route_distance","origin.code_icao","destination.code_icao"]]
    y = arrivals_cleaned['arrival_delay']

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X['operator'])
    sequences = tokenizer.texts_to_sequences(X['operator'])
    max_sequence_length = 20
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    sequences = np.array(sequences)
    X['operator'] = sequences
    tokenizer.fit_on_texts(X['aircraft_type'])
    sequences = tokenizer.texts_to_sequences(X['aircraft_type'])
    max_sequence_length = 20
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    sequences = np.array(sequences)
    X['aircraft_type'] = sequences
    tokenizer.fit_on_texts(X['origin.code_icao'])
    sequences = tokenizer.texts_to_sequences(X['origin.code_icao'])
    max_sequence_length = 20
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    sequences = np.array(sequences)
    X['origin.code_icao'] = sequences
    tokenizer.fit_on_texts(X['destination.code_icao'])
    sequences = tokenizer.texts_to_sequences(X['destination.code_icao'])
    max_sequence_length = 20
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    sequences = np.array(sequences)
    X['destination.code_icao'] = sequences
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=6))  # Input layer with 6 features
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='linear'))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error')  # Use an appropriate loss function for your task
    model.fit(X_train, y_train, epochs=1500, batch_size=32, validation_data=(X_test, y_test))
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    y_pred = model.predict(X_test)

    print(arrivals_cleaned)

    print(y_pred)

    print(y_test)

    print(X_test)

    flight_requested = pd.DataFrame({'operator':[airline],'departure_delay':[departure_delay],'aircraft_type':[aircraft],'route_distance':[dist],'origin.code_icao':[origin],'destination.code_icao':[destination],'arrival_delay':[0]})
    print(flight_requested)

    flight_to_predict = scaler.fit_transform(X)
    ETA = model.predict(flight_to_predict)
    print("The estimated delay of this flight, in seconds, is:",ETA[0][0])

    ETA_new = []
    for i in range(len(ETA)):
        ETA_new.append(ETA[i][0])
    ATA = arrivals_cleaned['arrival_delay']
    plt.scatter(ATA,ETA_new)
    plt.xlabel='Actual Delay'
    plt.ylabel='Estimated Delay'

    return ETA[0][0]