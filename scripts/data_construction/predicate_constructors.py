"""
Define class with methods for constructing and writing PSL predicates from raw bikeshare data.
"""

import numpy as np
import os
import pandas as pd
import datetime
import geopy
import pgeocode
import pickle

from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
DATASET = "Bay_Area"
DIRNAME = os.path.dirname(__file__)
MODEL_PATH = os.path.join(DIRNAME, "../../data/", DATASET, "local_models")

# Number of pools to parallelize predicate construction.
N_POOLS = 20

# Set ARIMA parameters. Justified in jupyter notebook.
p = 2
d = 0
q = 0
P = 2
D = 0
Q = 0
s = 12


class PredicateConstructor:
    time_to_constant_dict = {}
    station_id_to_constant_dict = {}
    raining_df = pd.DataFrame()

    def __init__(self, start_time, end_time, time_step_hr, station_df, out_directory):
        time_range = pd.date_range(start_time, end_time, freq=time_step_hr, closed='left')
        self.time_to_constant_dict = dict(zip(np.sort(time_range), np.arange(len(time_range), dtype=int)))

        unique_station_list = station_df.index.tolist()
        if pd.api.types.is_object_dtype(station_df.index):
            self.station_id_to_constant_dict = dict(zip(unique_station_list,
                                                        np.arange(len(unique_station_list), dtype=int)))
        else:
            self.station_id_to_constant_dict = dict(zip(unique_station_list, unique_station_list))

        pd.Series(self.station_id_to_constant_dict).to_csv(os.path.join(out_directory, "station_to_int_id.csv"))

        print("Making " + MODEL_PATH)
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)

    def station_to_int_id(self, data):
        data_copy = data.copy(deep=True)

        if 'station' in data_copy.columns:
            data_copy["station"] = data_copy["station"].apply(lambda x: self.station_id_to_constant_dict[x] if not isinstance(x, int) else x)

        if 'station_id' in data_copy.columns:
            data_copy["station_id"] = data_copy["station_id"].apply(lambda x: self.station_id_to_constant_dict[x] if not isinstance(x, int) else x)

        if 'station_1' in data_copy.columns and 'station_2' in data_copy.columns:
            data_copy["station_1"] = data_copy["station_1"].apply(lambda x: self.station_id_to_constant_dict[x] if not isinstance(x, int) else x)
            data_copy["station_2"] = data_copy["station_2"].apply(lambda x: self.station_id_to_constant_dict[x] if not isinstance(x, int) else x)

        if 'station_id_1' in data_copy.columns and 'station_id_2' in data_copy.columns:
            data_copy["station_id_1"] = data_copy["station_id_1"].apply(lambda x: self.station_id_to_constant_dict[x] if not isinstance(x, int) else x)
            data_copy["station_id_2"] = data_copy["station_id_2"].apply(lambda x: self.station_id_to_constant_dict[x] if not isinstance(x, int) else x)

        return data_copy

    def time_to_int_ids(self, data):
        data_copy = data.copy(deep=True)

        # Convert times to int ids.
        for col in data_copy.select_dtypes(include=np.datetime64).columns:
            data_copy.loc[:, col] = data_copy.loc[:, col].map(self.time_to_constant_dict)

        # Convert stations to ints if necessary
        data_copy = self.station_to_int_id(data_copy)

        return data_copy

    def write(self, data, predicate_name, path):
        data_copy = self.time_to_int_ids(data)
        data_copy = self.station_to_int_id(data_copy)

        # Path to this file relative to caller
        data_copy.to_csv(os.path.join(path, predicate_name + '.txt'), sep='\t', header=False, index=False)

    def read_local_model(self, model_type, station_id):
        return pickle.load(open(os.path.join(MODEL_PATH, str(station_id) + "_" + model_type + ".pkl"), "rb"))

    def write_local_model(self, model, model_type, station_id):
        pickle.dump(model, open(os.path.join(MODEL_PATH, str(station_id) + "_" + model_type + ".pkl"), "wb"))

    def read_local_model_params(self, model_type, station_id):
        return pickle.load(open(os.path.join(MODEL_PATH, str(station_id) + "_" + model_type + "_params.pkl"), "rb"))

    def write_local_model_params(self, params, model_type, station_id):
        pickle.dump(params, open(os.path.join(MODEL_PATH, str(station_id) + "_" + model_type + "_params.pkl"), "wb"))

    def station_to_zipcode_map(self, station_df, weather_df):
        zipcode_list = weather_df["zip_code"].unique()
        zipmodel = pgeocode.Nominatim('us')
        zip_to_station = dict({})
        for idx, row in station_df.iterrows():
            zip_distances = []

            for zipcode in zipcode_list:
                zipcode_info = zipmodel.query_postal_code(str(zipcode))
                zip_lat = zipcode_info["latitude"]
                zip_lon = zipcode_info["longitude"]

                zip_distances += [(zipcode, haversine_distances([[row["lat"], row["long"]]],
                                                                [[zip_lat, zip_lon]])[0][0])]

            nearest_zip = sorted(zip_distances, key=lambda x: x[1])[0][0]

            if nearest_zip not in zip_to_station.keys():
                zip_to_station[nearest_zip] = [idx]
            else:
                zip_to_station[nearest_zip] += [idx]

        return zip_to_station

    def get_nearest_zip_observations(self, unobserved_zip_code, weather_df, zip_codes_with_observations):
        zipmodel = pgeocode.Nominatim('us')
        unobserved_zipcode_info = zipmodel.query_postal_code(unobserved_zip_code)
        unobserved_zip_lat = unobserved_zipcode_info["latitude"]
        unobserved_zip_lon = unobserved_zipcode_info["longitude"]

        zip_distances = []

        for zip_code in zip_codes_with_observations:
            zip_info = zipmodel.query_postal_code(zip_code)
            zip_lat = zip_info["latitude"]
            zip_lon = zip_info["longitude"]

            zip_distances += [(zip_code, haversine_distances([[unobserved_zip_lat, unobserved_zip_lon]], 
                                                            [[zip_lat, zip_lon]])[0][0])]

        nearest_zip = sorted(zip_distances, key=lambda x: x[1])[0][0]
        return weather_df.loc[weather_df.zip_code == nearest_zip]["precipitation_inches"].reset_index()

    """
    Evidence and Open Predicates
    """
    def demand_predicate(self, demand_df, path, partition, write_value=True, arrival=False):
        if arrival:
            filename = 'ArrivalDemand_{}'.format(partition)
        else:
            filename = 'DepartureDemand_{}'.format(partition)

        if write_value:
            self.write(demand_df, filename, path)
        else:
            self.write(demand_df[['station_id', 'time']], filename, path)

    """
    Blocking Predicates
    """
    def station_predicate(self, station_df, path):
        self.write(pd.DataFrame(data={'station': station_df.index,
                                      'value': 1}), "Station_obs", path)

    def time_block_predicate(self, demand_df, path, partition):
        # truth
        target_dataframe = demand_df.loc[:, ['station_id', 'time']]
        target_dataframe.loc[:, 'value'] = 1
        self.write(target_dataframe, 'TimeBlock_{}'.format(partition), path)

    def commute_hours(self, time_df, path):
        unique_times = time_df.time.unique()
        commute_hours_df = pd.DataFrame(data={'time': unique_times,
                                              'value': [int((6 <= pd.Timestamp(t).hour < 10) |
                                                            (15 <= pd.Timestamp(t).hour < 19))
                                                        for t in unique_times]})
        self.write(commute_hours_df, 'CommuteHour_obs', path)

    """
    Auto-Regressive Predicates
    """
    def lag_n_predicate(self, time_df, time_step_hr, lag_hr, n, forecast_start_date, path):
        start_date = max(forecast_start_date - datetime.timedelta(hours=n), time_df.time.min())
        end_date = time_df.time.max()
        unique_times = pd.date_range(start_date + datetime.timedelta(hours=(n * lag_hr)), end_date, freq=time_step_hr)
        lag_n_df = pd.DataFrame(data={'time1': unique_times,
                                      'time2': unique_times - datetime.timedelta(hours=n * lag_hr),
                                      'value': 1})
        self.write(lag_n_df, "Lag{}_obs".format(n), path)
        return lag_n_df.set_index(['time1', 'time2'])
    
    def ishour_predicate(self, time_df, path):
        unique_times = time_df.time.unique()
        ishour_df = pd.DataFrame(data={'time': unique_times,
                                       'hour': [pd.Timestamp(t).hour for t in unique_times],
                                       'value': 1})
        self.write(ishour_df, "IsHour_obs", path)
        return ishour_df.set_index(['time', 'hour'])

    def isday_predicate(self, time_df, path):
        unique_times = time_df.time.unique()
        isday_df = pd.DataFrame(data={'time': unique_times,
                                      'day': [pd.Timestamp(t).floor('d') for t in unique_times],
                                      'value': 1})
        self.write(isday_df, "IsDay_obs", path)
        return isday_df.set_index(['time', 'day'])

    def isdayofweek_predicate(self, time_df, path):
        unique_times = time_df.time.unique()
        sameweekday_df = pd.DataFrame(data={'time': unique_times,
                                            'day': [pd.Timestamp(t).dayofweek for t in unique_times],
                                            'value': 1})
        self.write(sameweekday_df, "IsDayOfWeek_obs", path)
        return sameweekday_df.set_index(['time', 'day'])

    def isweekend_predicate(self, time_df, path):
        unique_times = time_df.time.unique()
        isweekend_df = pd.DataFrame(data={'time': unique_times,
                                          'value': [np.floor(pd.Timestamp(t).dayofweek / 5) for t in unique_times]})
        self.write(isweekend_df, "IsWeekend_obs", path)
        return isweekend_df.set_index(['time'])

    def samedaysamehour_blocked(self, time_df, raining_df, forecast_start_date, blocked_weeks, path):
        samedaysamehour_blocked_df = pd.DataFrame(columns=['time_1', 'time_2', 'value'])
        samedaysamehour_blocked_time_list = []

        times_in_range = time_df[time_df.time >= forecast_start_date - blocked_weeks * 7 * datetime.timedelta(days=1)]
        all_times = times_in_range.time.unique()
        rainy_times = raining_df.reset_index().time.unique()

        for current_time in all_times:

            current_time_raining = False
            if current_time in rainy_times:
                current_time_raining = True

            for week_count in range(blocked_weeks):
                candidate_time = pd.Timestamp(current_time) + ((week_count + 1) * 7 * datetime.timedelta(days=1))

                # Don't run past the end of the time list
                if candidate_time not in all_times:
                    break

                if ((candidate_time in rainy_times) == current_time_raining):
                    samedaysamehour_blocked_time_list += [{'time_1': current_time,
                                                           'time_2': candidate_time,
                                                           'value': 1}]

        samedaysamehour_blocked_df = samedaysamehour_blocked_df.append(samedaysamehour_blocked_time_list)

        samedaysamehour_blocked_df['time_1'] = pd.to_datetime(samedaysamehour_blocked_df['time_1'])
        samedaysamehour_blocked_df['time_2'] = pd.to_datetime(samedaysamehour_blocked_df['time_2'])
        self.write(samedaysamehour_blocked_df, "SameDaySameHourBlocked_obs", path)

    def samehour_blocked(self, time_df, raining_df, forecast_start_date, blocked_weeks, path):
        samehour_blocked_df = pd.DataFrame(columns=['time_1', 'time_2', 'value'])
        samehour_blocked_time_list = []

        times_in_range = time_df[time_df.time >= forecast_start_date - blocked_weeks * 7 * datetime.timedelta(days=1)]
        all_times = times_in_range.time.unique()
        rainy_times = raining_df.reset_index().time.unique()

        for current_time in all_times:

            current_time_raining = False
            if current_time in rainy_times:
                current_time_raining = True

            current_time_is_weekend = False
            if pd.Timestamp(current_time).dayofweek in [5, 6]:
                current_time_is_weekend = True

            for day_count in range(blocked_weeks * 7):
                candidate_time = pd.Timestamp(current_time) + ((day_count + 1) * datetime.timedelta(days=1))

                # Don't run past the end of the time list
                if candidate_time not in all_times:
                    break

                if ((candidate_time in rainy_times) == current_time_raining) and (
                        (pd.Timestamp(candidate_time).dayofweek in [5, 6]) == current_time_is_weekend):
                    samehour_blocked_time_list += [{'time_1': current_time,
                                                    'time_2': candidate_time,
                                                    'value': 1}]

        samehour_blocked_df = samehour_blocked_df.append(samehour_blocked_time_list)

        samehour_blocked_df['time_1'] = pd.to_datetime(samehour_blocked_df['time_1'])
        samehour_blocked_df['time_2'] = pd.to_datetime(samehour_blocked_df['time_2'])
        self.write(samehour_blocked_df, "SameHourBlocked_obs", path)

    """
    Collective Predicates
    """
    def destination_predicate(self, trip_df, path, top_n=3):
        # Station2 is a top destination for Station1
        trip_count_df = trip_df.loc[:, ["id", "start_station_id", "end_station_id"]].groupby(
            ["start_station_id", "end_station_id"]).count()
        trip_count_df.columns = ["count"]
        trip_count_df = trip_count_df.groupby(level=0)["count"].nlargest(top_n).droplevel(0).reset_index()
        trip_count_df["count"] = 1
        trip_count_df.columns = ["station_1", "station_2", "value"]

        self.write(trip_count_df.loc[:, ["station_1", "station_2", "value"]], "Destination_obs", path)
        trip_count_df = trip_count_df.loc[:, ["station_1", "station_2", "value"]].set_index(["station_1", "station_2"])
        return trip_count_df

    def source_predicate(self, trip_df, path, top_n=3):
        # Station2 is a top source for Station1
        trip_count_df = trip_df.loc[:, ["id", "start_station_id", "end_station_id"]].groupby(
            ["end_station_id", "start_station_id"]).count()
        trip_count_df.columns = ["count"]
        trip_count_df = trip_count_df.groupby(level=0)["count"].nlargest(top_n).droplevel(0).reset_index()
        trip_count_df["count"] = 1
        trip_count_df.columns = ["station_1", "station_2", "value"]

        self.write(trip_count_df.loc[:, ["station_1", "station_2", "value"]], "Source_obs", path)
        trip_count_df = trip_count_df.loc[:, ["station_1", "station_2", "value"]].set_index(["station_1", "station_2"])
        return trip_count_df

    def nearby_predicate(self, station_df, path, n=5):
        station_df.loc[:, 'lat_rad'] = np.radians(station_df.loc[:, 'lat'])
        station_df.loc[:, 'long_rad'] = np.radians(station_df.loc[:, 'long'])
        distances_df = pd.DataFrame(data=haversine_distances(station_df.loc[:, ['lat_rad', 'long_rad']]),
                                    index=station_df.index, columns=station_df.index)
        distances_df = distances_df * 6371000/1000  # multiply by Earth radius to get kilometers

        # Keep 0.5km radius.
        distances_df = distances_df <= 0.5
        distances_df = distances_df.astype(int)
        nearby_series = distances_df.stack()
        nearby_series.index.set_names(['station_id_1', 'station_id_2'], inplace=True)

        self.write(nearby_series.reset_index(), 'Nearby_obs', path)

    def commute_predicate(self, trip_df, station_df, path):
        # Trips.
        # Filter trips to those between 6am and 7pm on weekdays (common work hours).
        filtered_trip_df = trip_df[(trip_df.start_date.dt.hour > 6) &
                                   (trip_df.start_date.dt.hour < 19) &
                                   (trip_df.start_date.dt.dayofweek < 5) &
                                   (trip_df.start_station_id != trip_df.end_station_id) &
                                   (trip_df.subscription_type == 'Subscriber')]

        # Count the number of relevant trips between stations.
        trip_count_df = filtered_trip_df.loc[:, ["id", "start_station_id", "end_station_id"]].groupby(
            ["start_station_id", "end_station_id"]).count()
        trip_count_df.columns = ["count"]
        station_index = pd.MultiIndex.from_product([station_df.index, station_df.index],
                                                   names=['station_1', 'station_2'])
        trip_count_df = trip_count_df.reindex(station_index, fill_value=0)

        # Make commute counts df
        commute_df = pd.DataFrame(index=station_index, columns=['departure', 'return'])
        commute_df.loc[:, "departure"] = trip_count_df["count"]
        commute_df.loc[:, "return"] = trip_count_df.loc[trip_count_df.swaplevel().index, "count"].values
        commute_df.astype({'departure': np.int32, 'return': np.int32}, copy=False)
        commute_df.index = commute_df.index.set_names(['station_1', 'station_2'])
        commute_df = commute_df.reset_index()
        commute_df = commute_df[(commute_df['departure'] != 0) & (commute_df['return'] != 0)]
        commute_df = commute_df.loc[commute_df.groupby('station_1')['departure'].nlargest(5).index.get_level_values(1)]

        commute_df.loc[:, "value"] = 1.0
        self.write(commute_df.loc[:, ["station_1", "station_2", "value"]], "Commute_obs", path)
        return commute_df

    def stationcluster_predicate(self, station_df, demand_df, k, path):
        station_demand_series_map = dict({})
        not_clustered_stations = []

        for station in station_df.index.unique():
            demand = demand_df.loc[demand_df.station_id == station].demand
            if demand.shape[0] > 0:
                station_demand_series_map[station] = demand
            else:
                not_clustered_stations += [station]

        agglomerative_clustering_model = AgglomerativeClustering(n_clusters=k)
        results = agglomerative_clustering_model.fit_predict([x[1] for x in station_demand_series_map.items() if x[1].shape[0] > 0])

        station_cluster_map = dict({})
        cluster_station_map = dict({})

        for idx, station in enumerate(station_demand_series_map.keys()):
            cluster = results[idx]
            station_cluster_map[station] = cluster
            if cluster not in cluster_station_map.keys():
                cluster_station_map[cluster] = [station]
            else:
                cluster_station_map[cluster] += [station]

        station_cluster_df = pd.DataFrame(columns=['station_id', 'cluster', 'value'])

        for cluster in cluster_station_map.keys():
            for station in cluster_station_map[cluster]:
                station_cluster_df = station_cluster_df.append({'station_id': station, 'cluster': cluster, 'value': 1}, ignore_index=True)

        # Assign all stations which were not clustered because of a lack of demand observations
        # to their own cluster with id=k.
        for station in not_clustered_stations:
            station_cluster_df = station_cluster_df.append({'station_id': station, 'cluster': k, 'value': 1}, ignore_index=True)

        same_cluster_df = pd.DataFrame(columns=["station_1", "station_2", "value"])

        # Stations that were not clustered don't even need to appear in the SameCluster_obs file

        for cluster in cluster_station_map.keys():
            for station in cluster_station_map[cluster]:
                for station_2 in cluster_station_map[cluster]:
                    if station == station_2:
                        continue
                    same_cluster_df = same_cluster_df.append({"station_1": station, "station_2": station_2, "value": 1}, ignore_index=True)

        self.write(same_cluster_df.loc[:, ["station_1", "station_2", "value"]], "SameCluster_obs", path)
        self.write(station_cluster_df.loc[:, ["station_id", "cluster", "value"]], "StationCluster_obs", path)

    def similarstation_predicate(self, station_df, demand_df, count, path, arrival=False):
        station_demand_map = dict({})

        similar_station_df = pd.DataFrame(columns=["station_1", "station_2", "similarity"])
        station_similarities = []

        for station in station_df.index.tolist():
            station_demand_map[station] = demand_df.loc[demand_df.station_id == station].demand.values

        for idx, station_1 in enumerate(station_df.index.tolist()):
            station_1_demand = station_demand_map[station_1]

            if (station_1_demand.size == 0) or (station_1_demand.mean() == 0.0):
                continue

            for station_2 in station_df.index.tolist()[idx + 1:]:
                station_2_demand = station_demand_map[station_2]

                if (station_2_demand.size == 0) or (station_2_demand.mean() == 0):
                    continue

                station_similarities += [(station_1, station_2, np.corrcoef(station_1_demand, station_2_demand)[0][1])]

        sorted_similarities = sorted(station_similarities, key=lambda x: x[2])
        station_similarity_rows = []

        for station_pair in sorted_similarities[-count:]:
            station_similarity_rows += [{"station_1": station_pair[0], "station_2": station_pair[1], "similarity": 1}]

        similar_station_df = similar_station_df.append(station_similarity_rows)

        demand_type = "Departure"
        if arrival:
            demand_type = "Arrival"

        self.write(similar_station_df.loc[:, ["station_1", "station_2", "similarity"]], demand_type + "SimilarStation_obs", path)

    """
    Local Predictor Predicates
    """
    def fit_global_AR_models(self, departure_demand_df, arrival_demand_df, start_date, end_date, lags):
        AR_Arrival_model_path = os.path.join(MODEL_PATH, "AR_Arrival.pkl")
        AR_Departure_model_path = os.path.join(MODEL_PATH, "AR_Departure.pkl")

        if os.path.exists(AR_Arrival_model_path) and os.path.exists(AR_Departure_model_path):
            self.arrival_AR_model = pickle.load(open(AR_Arrival_model_path, "rb"))
            self.departure_AR_model = pickle.load(open(AR_Departure_model_path, "rb"))
            return self.arrival_AR_model.coef_

        ## Get all stations
        stations = arrival_demand_df.station_id.unique()

        # Map station IDs to their arrival/departure demand series
        arrival_station_df_map = dict()
        departure_station_df_map = dict()

        for station in stations:
            arrival_station_df_map[station] = arrival_demand_df.loc[(arrival_demand_df.station_id == station) & (arrival_demand_df.time < end_date)]["demand"].values
            departure_station_df_map[station] = departure_demand_df.loc[(departure_demand_df.station_id == station) & (departure_demand_df.time < end_date)]["demand"].values

        global_train_arrival_demands = []
        global_train_departure_demands = []
        global_train_arrival_lagged_demands = []
        global_train_departure_lagged_demands = []

        for station in stations:
            if len(arrival_station_df_map[station]) == 0 or len(departure_station_df_map[station]) == 0:
                continue

            for day_idx in range(len(arrival_station_df_map[station]) - lags[-1]):
                day = day_idx + lags[-1]
                
                global_train_arrival_demands += [arrival_station_df_map[station][day]]
                global_train_departure_demands += [departure_station_df_map[station][day]]
                
                day_lagged_arrival_demands = []
                day_lagged_departure_demands = []

                for lag in lags:
                    day_lagged_arrival_demands += [arrival_station_df_map[station][day - lag]]
                    day_lagged_departure_demands += [departure_station_df_map[station][day - lag]]

                global_train_arrival_lagged_demands += [day_lagged_arrival_demands]
                global_train_departure_lagged_demands += [day_lagged_departure_demands]


        # Fit arrival/departure AR models
        self.arrival_AR_model = LinearRegression().fit(global_train_arrival_lagged_demands, global_train_arrival_demands)
        self.departure_AR_model = LinearRegression().fit(global_train_departure_lagged_demands, global_train_departure_demands)
        
        pickle.dump(self.arrival_AR_model, open(AR_Arrival_model_path, "wb"))
        pickle.dump(self.departure_AR_model, open(AR_Departure_model_path, "wb"))

        coef_out_file = open(os.path.join(MODEL_PATH, "global_AR_coefs.txt"), "w")
        coef_out_file.write("DemandType\t" + "\t".join(["Lag" + str(lag) for lag in lags]) + "\tBias\n")
        coef_out_file.write("Arrival\t" + "\t".join([str(coef) for coef in self.arrival_AR_model.coef_]) + "\t" + str(self.arrival_AR_model.intercept_) + "\n")
        coef_out_file.write("Departure\t" + "\t".join([str(coef) for coef in self.departure_AR_model.coef_]) + "\t" + str(self.departure_AR_model.intercept_) + "\n")
        coef_out_file.close()

        return self.arrival_AR_model.coef_

    def ar_predicate(self, arrival_demand_df, departure_demand_df, start_date, num_time_steps, time_step_hr, lags, path):
        # Get all stations
        stations = arrival_demand_df.station_id.unique()

        # Map station IDs to their arrival/departure demand series
        arrival_station_df_map = dict()

        print("Creating station-demand maps")

        for station in stations:
            arrival_station_df_map[station] = arrival_demand_df.loc[arrival_demand_df.station_id == station][["time", "demand"]]
            arrival_station_df_map[station].columns = ['time', 'Demand']
            #f arrival_station_df_map[station].isna().sum() > 0:
            #   stations = [station_id for station_id in stations if station

        departure_station_df_map = dict()

        for station in stations:
            departure_station_df_map[station] = departure_demand_df.loc[arrival_demand_df.station_id == station][["time", "demand"]]
            departure_station_df_map[station].columns = ['time', 'Demand']

        arrival_pred_df = pd.DataFrame(columns=['station_id', 'time', 'Demand'])
        departure_pred_df = pd.DataFrame(columns=['station_id', 'time', 'Demand'])

        print("Making predictions")
        for time_step in range(num_time_steps):
            cur_time = start_date + datetime.timedelta(hours=time_step)
            
            for station in stations:
                station_arrival_demand = arrival_station_df_map[station]
                station_departure_demand = departure_station_df_map[station]

                prev_arrival_demands = []
                prev_departure_demands = []

                for lag in lags:
                    lag_time = cur_time - datetime.timedelta(hours=lag)
                    prev_arrival_demands += [station_arrival_demand.loc[station_arrival_demand.time == lag_time]["Demand"].values[0]]
                    prev_departure_demands += [station_departure_demand.loc[station_departure_demand.time == lag_time]["Demand"].values[0]]

                # Add bias
                prev_arrival_demands += [1]
                prev_departure_demands += [1]

                # Predict and append to station
                predicted_arrival_demand = self.arrival_AR_model.predict([prev_arrival_demands])[0]
                predicted_departure_demand = self.departure_AR_model.predict([prev_departure_demands])[0]

                predicted_arrival_demand_df_row = [{'station_id': station, 'time': cur_time, 'Demand': predicted_arrival_demand}]
                predicted_departure_demand_df_row = [{'station_id': station, 'time': cur_time, 'Demand': predicted_departure_demand}]

                arrival_station_df_map[station] = arrival_station_df_map[station].append(predicted_arrival_demand_df_row)
                arrival_pred_df = arrival_pred_df.append(predicted_arrival_demand_df_row)

                departure_station_df_map[station] = departure_station_df_map[station].append(predicted_departure_demand_df_row)
                departure_pred_df = departure_pred_df.append(predicted_departure_demand_df_row)

        self.write(arrival_pred_df.loc[:, ['station_id', 'time', 'Demand']], 'ARArrival_obs', path)
        self.write(departure_pred_df.loc[:, ['station_id', 'time', 'Demand']], 'ARDeparture_obs', path)

        return arrival_pred_df, departure_pred_df

    """
    Exogenous Variable Predicates
    """
    def raining_predicate(self, weather_df, station_df, forecast_start_date, path):
        self.raining_df = pd.DataFrame(columns=['station_id', 'time', 'raining'])
        self.raining_df = self.raining_df.astype({'station_id': int, 'time': 'datetime64[ns]', 'raining': float})

        weather_df = weather_df.reset_index()

        geolocator = geopy.Nominatim(user_agent='STPSL')

        if "zip_code" not in weather_df:
            weather_df["zip_code"] = ""
            seen_locations = dict({})
            for idx, row in weather_df.iterrows():
                if (row["lat"], row["long"]) not in seen_locations.keys():
                    weather_df.loc[idx, "zip_code"] = geolocator.geocode(geolocator.reverse((row["lat"], row["long"])), addressdetails=True).raw['address']['postcode'][:5]
                    seen_locations[(row["lat"], row["long"])] = weather_df.loc[idx, "zip_code"]
                else:
                    weather_df.loc[idx, "zip_code"] = seen_locations[(row["lat"], row["long"])]

        zip_to_station = self.station_to_zipcode_map(station_df, weather_df)
        weather_df.precipitation_inches = weather_df.precipitation_inches.apply(lambda x: 0.0 if x == 'T' else float(x))
        weather_events_df = weather_df[weather_df.precipitation_inches > 0.0]
        for idx, row in weather_events_df.iterrows():
            for station_id in zip_to_station[row["zip_code"]]:
                self.raining_df = self.raining_df.append(
                    [{"station_id": station_id, "time": row["date"] + (hour * datetime.timedelta(hours=1)), "raining": 1}
                     for hour in range(24)], ignore_index=True)

        self.write(self.raining_df[self.raining_df.time >= forecast_start_date], "Raining_obs", path)

        return self.raining_df.set_index(['station_id', 'time'])


def get_station_demand(demand_df, station_id, time_step_hr, min_date=None):
    station_demand_df = demand_df.loc[demand_df['station_id'] == station_id]
    station_demand_df = station_demand_df.set_index('time')
    if min_date is None:
        station_demand_df = station_demand_df.reindex(
            pd.date_range(min(station_demand_df.index), max(station_demand_df.index), freq=time_step_hr),
            fill_value=0)
    else:
        station_demand_df = station_demand_df.reindex(
            pd.date_range(min_date, max(station_demand_df.index), freq=time_step_hr),
            fill_value=0)

    return station_demand_df
