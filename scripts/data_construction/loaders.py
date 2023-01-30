import os
import pandas as pd

DIRNAME = os.path.dirname(__file__)
DATA_PATH = os.path.join(DIRNAME, "../../data")


def load_bay_area_dataframes(start_date, end_date):
    station_df = pd.read_csv(DATA_PATH + "/bikeshare_raw/station.csv", sep=',', encoding="ISO-8859-1", engine='python')
    station_df = station_df.set_index('id')

    trip_df = pd.read_csv(DATA_PATH + "/bikeshare_raw/trip.csv", sep=',', encoding="ISO-8859-1", engine='python',
                          parse_dates=['start_date', 'end_date'], infer_datetime_format=True)
    trip_df = trip_df.loc[(trip_df["start_date"] >= start_date) & (trip_df["end_date"] <= end_date)]

    weather_df = pd.read_csv(DATA_PATH + "/bikeshare_raw/weather.csv", sep=',', encoding="ISO-8859-1", engine='python',
                             parse_dates=['date'], infer_datetime_format=True)
    weather_df = weather_df.loc[(weather_df["date"] >= start_date) & (weather_df["date"] < end_date)]

    # filter status and station that do not exist in early trip_df
    station_df = station_df[:-2]
    trip_df = trip_df[trip_df.start_station_id.isin(station_df.index.unique())]
    trip_df = trip_df[trip_df.end_station_id.isin(station_df.index.unique())]

    return station_df, trip_df, weather_df


def load_metro_dataframes(start_date, end_date):
    station_df = pd.read_csv(DATA_PATH + "/metro_raw/station.csv", sep=",")
    station_df.columns = ["id", "lat", "long"]
    station_df = station_df.set_index('id')
    station_df = station_df.fillna(method="ffill")

    trip_df = pd.read_csv(DATA_PATH + "/metro_raw/trip.csv", sep=",", encoding="ISO-8859-1", engine='python', parse_dates=['start_time', 'end_time'], infer_datetime_format=True)
    trip_df.columns = ["id", "Duration", "start_date", "end_date", "start_station_id", "start_station_lat", "start_station_lon", "end_station_id", "end_station_lat", "end_station_lon", "bike_id",
            "plan_duration", "trip_route_category", "passholder_type"]
    trip_df = trip_df[trip_df.start_station_id.isin(station_df.index.unique())]
    trip_df = trip_df[trip_df.end_station_id.isin(station_df.index.unique())]
    trip_df = trip_df.loc[(trip_df["start_date"] >= start_date) & (trip_df["end_date"] <= end_date)]

    weather_df = pd.read_csv(DATA_PATH + "/metro_raw/weather.csv", sep=",", encoding="ISO-8859-1", engine='python', parse_dates=['date'], infer_datetime_format=True)
    weather_df = weather_df.loc[(weather_df["date"] >= start_date) & (weather_df["date"] < end_date)]

    return station_df, trip_df, weather_df


def load_bluebikes_dataframes(start_date, end_date):
    station_df = pd.read_csv(DATA_PATH + "/bluebikes_raw/station.csv", sep=",")
    station_df.columns = ["name", "id", "lat", "long", "city", "dock_count"]
    station_df = station_df.set_index('id')

    trip_df = pd.read_csv(DATA_PATH + "/bluebikes_raw/trip.csv", sep=",", encoding="ISO-8859-1", engine='python', parse_dates=['Start date', 'End date'], infer_datetime_format=True)
    trip_df.columns = ["Duration", "start_date", "end_date", "start_station_id", "start_station_name", "end_station_id",
                       "end_station_name", "Bike number", "Member type", "Zip code", "Gender"]
    trip_df = trip_df[trip_df.start_station_id.isin(station_df.index.unique())]
    trip_df = trip_df[trip_df.end_station_id.isin(station_df.index.unique())]
    trip_df = trip_df.loc[(trip_df["start_date"] >= start_date) & (trip_df["end_date"] <= end_date)]

    weather_df = pd.read_csv(DATA_PATH + "/bluebikes_raw/weather.csv", sep=",", encoding="ISO-8859-1", engine='python', parse_dates=['DATE'], infer_datetime_format=True)
    weather_df = weather_df.rename(columns={"PRCP": "precipitation_inches", "DATE": "date", "LATITUDE": "lat", "LONGITUDE": "long"})
    weather_df = weather_df.loc[(weather_df["date"] >= start_date) & (weather_df["date"] < end_date)]

    return station_df, trip_df, weather_df
