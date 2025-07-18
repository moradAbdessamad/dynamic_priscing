import requests #type: ignore
import json
import os
import sys
import openmeteo_requests #type: ignore
import requests_cache #type: ignore
from retry_requests import retry #type: ignore
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim #type: ignore
import pandas as pd #type: ignore
import SRC
metaData = SRC.load_model_metadata()


def get_coordinates(city_name):
    """
    Get the latitude and longitude of a city using Nominatim geocoding service.
    """
    geolocator = Nominatim(user_agent="my_app")
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None
    

def fetch_weather_data(start_date, 
                        end_date, 
                        latitude, 
                        longitude, 
                        archive_api = metaData['weatherInformation']['archive_api'], 
                        forecast_api = metaData['weatherInformation']['forecast_api']):
    """
    Fetch weather data using the official Open-Meteo client.
    Automatically determines whether to use historical or forecast API based on dates.
    No recursion - handles split date ranges directly.

    Args:
        start_date (str): Start date in the format 'YYYY-MM-DD'
        end_date (str): End date in the format 'YYYY-MM-DD'
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location
    """
    session = requests.Session()
    openmeteo = openmeteo_requests.Client(session=session)

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)

    combined_weather_data = {}

    if end_dt.date() < today:
        print("Fetching historical data only")
        hist_data = _fetch_single_range(archive_api, start_date, end_date, latitude, longitude, openmeteo)
        if hist_data:
            combined_weather_data.update(hist_data)
            
    elif start_dt.date() > today:
        print("Fetching forecast data only")
        forecast_data = _fetch_single_range(forecast_api, start_date, end_date, latitude, longitude, openmeteo)
        if forecast_data:
            combined_weather_data.update(forecast_data)
            
    else:
        print("Date range spans past and future - making separate requests")
        
        if start_dt.date() <= yesterday:
            hist_end = min(end_dt.date(), yesterday)
            hist_end_str = hist_end.strftime('%Y-%m-%d')
            print(f"Fetching historical data: {start_date} to {hist_end_str}")
            
            hist_data = _fetch_single_range(archive_api, start_date, hist_end_str, latitude, longitude, openmeteo)
            if hist_data:
                combined_weather_data.update(hist_data)
        
        if end_dt.date() >= today:
            forecast_start = max(start_dt.date(), today)
            forecast_start_str = forecast_start.strftime('%Y-%m-%d')
            print(f"Fetching forecast data: {forecast_start_str} to {end_date}")
            
            forecast_data = _fetch_single_range(forecast_api, forecast_start_str, end_date, latitude, longitude, openmeteo)
            if forecast_data:
                combined_weather_data.update(forecast_data)

    return combined_weather_data


def _fetch_single_range(url, start_date, end_date, latitude, longitude, openmeteo):
    """
    Fetch weather data for a single date range from a specific API endpoint.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", 
                 "precipitation_sum", "windspeed_10m_max"]
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation {response.Elevation()} m asl")
        print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")

        daily = response.Daily()

        daily_data = {"date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )}

        daily_data["temperature_2m_max"] = daily.Variables(0).ValuesAsNumpy()
        daily_data["temperature_2m_min"] = daily.Variables(1).ValuesAsNumpy()
        daily_data["temperature_2m_mean"] = daily.Variables(2).ValuesAsNumpy()
        daily_data["precipitation_sum"] = daily.Variables(3).ValuesAsNumpy()
        daily_data["windspeed_10m_max"] = daily.Variables(4).ValuesAsNumpy()

        df = pd.DataFrame(data=daily_data)

        weather_data = {}
        for _, row in df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            weather_data[date_str] = {
                'temperature_max': row['temperature_2m_max'],
                'temperature_min': row['temperature_2m_min'],
                'temperature_mean': row['temperature_2m_mean'],
                'precipitation': row['precipitation_sum'],
                'windspeed_max': row['windspeed_10m_max']
            }

        print(f"Successfully fetched weather data for {len(weather_data)} dates")
        return weather_data

    except Exception as e:
        print(f"Error fetching weather data from {url}: {e}")
        return {}    


def get_weather_data_for_city(start_date: str, end_date: str, city: str):
    """
    Fetches historical weather data for a specific city between start_date and end_date.

    Args:
        start_date (str): The start date in "YYYY-MM-DD" format.
        end_date (str): The end date in "YYYY-MM-DD" format.
        city (str): The name of the city.
    """
    try:
        lat, lon = get_coordinates(city)
        weather_data = fetch_weather_data(start_date, end_date, lat, lon)
        return weather_data
    except Exception as e:
        print(f"Error fetching weather data for city {city}: {e}")
        return {}


def make_weather_data_dataframe(weather_data_path=f"../{metaData['weatherInformation']['archive_json_path']}"):
    """
    Convert weather data to DataFrame with additional calculated columns.
    
    Args:
        weather_data (dict): Dictionary with dates as keys and weather metrics as values
    """
    with open(weather_data_path, 'r') as file:
        weather_data = json.load(file)

    if not weather_data:
        print("No weather data provided in the JSON file.")
        return pd.DataFrame()
    
    df = pd.DataFrame.from_dict(weather_data, orient='index')
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'
    
    df['temperature_range'] = df['temperature_max'] - df['temperature_min']
        
    numerical_cols = ['temperature_max', 'temperature_min', 'temperature_mean', 
                     'precipitation', 'windspeed_max', 'temperature_range']
    df[numerical_cols] = df[numerical_cols].round(3)

    df.fillna(0, inplace=True)
    
    return df

def get_weather_dataframe_for_city_single_date(date, city):
    """
    Fetches weather data for a specific city for a single date and returns it as a pandas DataFrame.
    
    Args:
        date (str): The date in "YYYY-MM-DD" format.
        city (str): The name of the city.
    """
    try:
        lat, lon = get_coordinates(city)
        
        if lat is None or lon is None:
            print(f"Could not find coordinates for city: {city}")
            return pd.DataFrame()
        
        print(f"Found coordinates for {city}: {lat:.4f}, {lon:.4f}")
        
        weather_data = fetch_weather_data(date, date, lat, lon)
        
        if not weather_data:
            print(f"No weather data retrieved for {city} on {date}")
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(weather_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df.index.name = 'date'
        
        df['temperature_range'] = df['temperature_max'] - df['temperature_min']
        
        numerical_cols = ['temperature_max', 'temperature_min', 'temperature_mean', 
                         'precipitation', 'windspeed_max', 'temperature_range']
        df[numerical_cols] = df[numerical_cols].round(3)
        
        df.fillna(0, inplace=True)
        
        print(f"Successfully created DataFrame for {city} on {date}")
        return df
        
    except Exception as e:
        print(f"Error creating weather DataFrame for city {city} on {date}: {e}")
        return pd.DataFrame()
    

def get_weather_dataframe_for_city_single_date_fast(date, weather_data_dict):
    """
    Fetches weather data for a specific date from pre-loaded weather data dictionary and returns it as a pandas DataFrame.
    This is a fast version that doesn't make API calls.
    
    Args:
        date (str): The date in "YYYY-MM-DD" format.
        weather_data_dict (dict): Pre-loaded weather data dictionary with dates as keys.
    """
    try:
        if date not in weather_data_dict:
            print(f"No weather data found for date: {date}")
            return pd.DataFrame()
        
        day_weather = weather_data_dict[date]
        
        single_date_data = {date: day_weather}
        
        df = pd.DataFrame.from_dict(single_date_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df.index.name = 'date'
        
        df['temperature_range'] = df['temperature_max'] - df['temperature_min']
        
        numerical_cols = ['temperature_max', 'temperature_min', 'temperature_mean', 
                         'precipitation', 'windspeed_max', 'temperature_range']
        df[numerical_cols] = df[numerical_cols].round(3)
        
        df.fillna(0, inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Error creating weather DataFrame for date {date}: {e}")
        return pd.DataFrame()


def load_weather_data_from_json(json_file_path=f"../{metaData['weatherInformation']['archive_json_path']}"):
    """
    Helper function to load weather data from JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file containing weather data.
    """
    try:
        with open(json_file_path, 'r') as file:
            weather_data = json.load(file)
        print(f"Successfully loaded weather data for {len(weather_data)} dates from {json_file_path}")
        return weather_data
    except Exception as e:
        print(f"Error loading weather data from {json_file_path}: {e}")
        return {}