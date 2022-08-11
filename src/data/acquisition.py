from io import StringIO
import numpy as np
import pandas as pd
import requests


def get_weather_data(start_year: int):
    """
    Acquires the weather data from the Oxford station.
    Parameters
    ----------
    start_year : int
        Year to acquire weather data from.
    Returns
    -------
    pd.DataFrame data with maximum temperature, minimum temperature and rainfall
    """
    url = 'https://metoffice.gov.uk/pub/data/weather/uk/climate/stationdata/oxforddata.txt'
    # Makes a request to a web page and fetches the data from the url
    r = requests.get(url)
    buff = StringIO(r.text)
    # Reads a table of fixed-width formatted lines into a dataframe
    df_weather = pd.read_fwf(buff, skiprows=5)
    # Cleaning the data types and renaming some columns
    df_weather = df_weather.dropna()
    df_weather.mm = df_weather.mm.astype(int)
    df_weather.yyyy = df_weather.yyyy.astype(int)
    df_weather = df_weather.rename(columns={'yyyy': 'year', 'mm': 'month'}).drop(columns=['af', 'sun'])
    # Stating the start year of when we want to read the weather data from
    df_weather = df_weather[df_weather.year >= start_year]
    # Cleaning the data types of some columns
    df_weather = df_weather.astype({'tmax': float,
                                    'tmin': float,
                                    'rain': float})
    return df_weather


def get_bank_holiday_data():
    """
    Read the data containing bank holiday dates and the bank holiday description from CSV
    Returns
    -------
    pd.DataFrame for bank holidays
    """
    df = pd.read_csv('bank_holiday_data_1953_to_1954.csv')
    # Create another column for bank holiday
    # When there is a bank holiday add Yes and when not add No
    df['bank_holiday'] = np.where(df.Bank_Holiday_Description.isna(), 'No', 'Yes')
    # Cleaning column names and data types
    df = df[['Date', 'bank_holiday']].rename(columns={'Date': 'date'})
    df['date'] = df['date'].astype('datetime64[ns]')
    return df
