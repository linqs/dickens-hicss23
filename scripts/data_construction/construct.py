"""
Construct PSL formatted data from raw bikehsare data.
"""

import datetime
import numpy as np
import os
import pandas as pd

import predicate_constructors
import loaders
import client_command_constructors
from demand import construct_trip_based_demand_dfs

DATASET = "Bay_Area"
DIRNAME = os.path.dirname(__file__)
PSL_DATA_PATH = os.path.join(DIRNAME, "../../data/", DATASET)
LOAD_METHOD = {
    'Bay_Area': loaders.load_bay_area_dataframes,
    'Blue_Bikes': loaders.load_bluebikes_dataframes,
    "Metro": loaders.load_metro_dataframes
}

CLIENT_COMMAND_METHOD = {
    'Incremental': client_command_constructors.incremental_construct_client_commands,
    'Online': client_command_constructors.online_construct_client_commands
}

# Data is split using forward chaining.
# This method splits the train and tests data like:
#     learn : Observed [0, ..., 7], train[8, 9, 10, 11]
#     fold 1: Observed [0, ..., 11], test[12, 13, 14, 15]
#     fold 2: Observed [0, ..., 12], test[13, 14, 15, 16]
#     fold 3: Observed [0, ..., 13], test[14, 15, 16, 17]
# Time steps are separated into 1 hour blocks and data is added and tested on a daily basis.
# The time step size in the formats needed in the code.
TIME_STEP_HR_INT = 1
TIME_STEP_HR = '1h'
TIME_STEP_MIN = '60min'
HOURS_PER_WINDOW = 24
WINDOWS_PER_TEST_SET = 1
INITIAL_SEGMENT_SIZE = 365
# Bluebikes
# INITIAL_SEGMENT_SIZE = 304

BLOCKED_WEEKS = 4
TOP_K_SIMILAR = {
    "Bay_Area": 1000,
    "Blue_Bikes": 1000,
    "Metro": 1000
}

CLUSTER_COUNT = {
    "Bay_Area": 13,
    "Blue_Bikes": 30,
    "Metro": 10
}

# Set a start date and end date.
START_DATE = {
    "Bay_Area": datetime.datetime(2013, 8, 29),
    "Blue_Bikes": datetime.datetime(2013, 4, 3),
    "Metro": datetime.datetime(2018, 2, 1)
}
VALIDATION_END_DATE = {
    "Bay_Area": datetime.datetime(2015, 5, 1),
    "Blue_Bikes": datetime.datetime(2014, 8, 1),
    "Metro": datetime.datetime(2019, 8, 1)
}
TEST_END_DATE = {
    "Bay_Area": datetime.datetime(2015, 6, 1),
    "Blue_Bikes": datetime.datetime(2014, 9, 1),
    "Metro": datetime.datetime(2019, 9, 1)
}

DATES = [START_DATE[DATASET], VALIDATION_END_DATE[DATASET]]

# Set the learn fold boundaries
LEARN_OBSERVED_WINDOWS = (DATES[1] - DATES[0]).days - 30
LEARN_FORECAST_WINDOW = 1
LEARN_END_DATE = DATES[0] + datetime.timedelta(hours=int(HOURS_PER_WINDOW * (LEARN_OBSERVED_WINDOWS + LEARN_FORECAST_WINDOW)))

# Partition names
OBS = 'obs'
TRUTH = 'truth'
TARGET = 'target'

# AR lags
LAGS = [1, 2, 12, 24, 168]

# TODO(Charles): The first target set of incremental should not include the entire day, but only the next timestep.
def construct_predicates():
    # Create out directory
    out_directory = PSL_DATA_PATH + '/eval/'
    print("Making " + out_directory)
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    # Load the raw data.
    station_df, trip_df, weather_df = LOAD_METHOD[DATASET](DATES[0], DATES[1])

    # Get demand.
    departure_demand_df, arrival_demand_df = construct_trip_based_demand_dfs(
        trip_df, station_df, DATES[0], DATES[1], TIME_STEP_MIN, TIME_STEP_HR, out_directory)

    # Partition dates into ranges for folds and splits.
    total_day_count = (DATES[1] - DATES[0]).days
    window_dates = [[DATES[0] + datetime.timedelta(hours=int(HOURS_PER_WINDOW * i)),
                     DATES[0] + datetime.timedelta(hours=int(HOURS_PER_WINDOW * (i + 1)))]
                    for i in np.arange((LEARN_OBSERVED_WINDOWS + LEARN_FORECAST_WINDOW), int(total_day_count / (HOURS_PER_WINDOW / 24)))]

    # Add an initial year worth of data.
    initial_year = [[DATES[0], DATES[0] + datetime.timedelta(hours=int(HOURS_PER_WINDOW * INITIAL_SEGMENT_SIZE))]]
    window_dates = initial_year + [[DATES[0] + datetime.timedelta(hours=int(HOURS_PER_WINDOW * i)),
                                    DATES[0] + datetime.timedelta(hours=int(HOURS_PER_WINDOW * (i + 1)))]
                                   for i in np.arange(INITIAL_SEGMENT_SIZE, LEARN_OBSERVED_WINDOWS + LEARN_FORECAST_WINDOW)] + window_dates

    END_OF_INITIAL_SEGMENT = window_dates[0][1]

    # Instantiate predicate constructor object.
    predicate_constructor = predicate_constructors.PredicateConstructor(DATES[0], DATES[1], TIME_STEP_HR, station_df, out_directory)

    # Fit global AR models for arrival/departure demand
    predicate_constructor.fit_global_AR_models(departure_demand_df, arrival_demand_df, DATES[0], END_OF_INITIAL_SEGMENT, LAGS)

    # Construct predicates that are static.
    print("Constructing static predicates.")
    construct_static_predicates(predicate_constructor,
                                trip_df[(trip_df["end_date"] >= DATES[0]) &
                                        (trip_df["end_date"] < LEARN_END_DATE)],
                                departure_demand_df.loc[:, ["station_id", "time"]],
                                station_df, weather_df, window_dates,
                                departure_demand_df[departure_demand_df["time"] < LEARN_END_DATE],
                                arrival_demand_df[arrival_demand_df["time"] < LEARN_END_DATE],
                                out_directory)

    # Construct predicates that are dynamic for this fold.
    print("Constructing dynamic predicates.")
    construct_dynamic_predicates(predicate_constructor, departure_demand_df, arrival_demand_df, window_dates, out_directory)


def construct_dynamic_predicates(predicate_constructor, departure_demand_df,
                                 arrival_demand_df, window_dates, out_directory):
    """
    Construct the predicates change between time steps.
    """

    # Initialize aggregate dataframes.
    # These dataframes will grow with each time step by adding the previous time step's data.
    aggregated_observed_departure_demand_df = pd.DataFrame(columns=departure_demand_df.columns)
    aggregated_observed_departure_demand_df = aggregated_observed_departure_demand_df.astype(departure_demand_df.dtypes)

    aggregated_observed_arrival_demand_df = pd.DataFrame(columns=arrival_demand_df.columns)
    aggregated_observed_arrival_demand_df = aggregated_observed_arrival_demand_df.astype(arrival_demand_df.dtypes)

    for time_step, split_date_range in enumerate(window_dates[:-WINDOWS_PER_TEST_SET]):
        if time_step == 32:
            break

        print("Constructing predicates for time step: " + str(time_step).zfill(3))
        # Set the shared path between these predicates.
        path = os.path.join(out_directory, str(time_step).zfill(3))
        if not os.path.exists(path):
            os.makedirs(path)

        # Add new observations.
        observed_departure_demands_df = departure_demand_df.loc[
                                        (departure_demand_df["time"] >= window_dates[time_step][0]) &
                                        (departure_demand_df["time"] < window_dates[time_step][1]), :]
        aggregated_observed_departure_demand_df = aggregated_observed_departure_demand_df.append(observed_departure_demands_df)

        observed_arrival_demands_df = arrival_demand_df.loc[
                                      (arrival_demand_df["time"] >= window_dates[time_step][0]) &
                                      (arrival_demand_df["time"] < window_dates[time_step][1]), :]
        aggregated_observed_arrival_demand_df = aggregated_observed_arrival_demand_df.append(observed_arrival_demands_df)


        # Assumes no lags greater than a week. And no overlapping windows.
        expired_departure_demands_df = departure_demand_df.loc[
           (departure_demand_df["time"] >= window_dates[time_step][0] - datetime.timedelta(hours=int(24 * 7))) &
           (departure_demand_df["time"] < window_dates[time_step][1] - datetime.timedelta(hours=int(24 * 7))), :]

        expired_arrival_demands_df = arrival_demand_df.loc[
           (arrival_demand_df["time"] >= window_dates[time_step][0] - datetime.timedelta(hours=int(24 * 7))) &
           (arrival_demand_df["time"] < window_dates[time_step][1] - datetime.timedelta(hours=int(24 * 7))), :]

        # Initial observations are loaded from command line
        if time_step == 0:
            observed_departure_demands_df = None
            observed_arrival_demands_df = None

        # Departure Demand Targets
        if time_step == 0:
            target_departure_demands_df = departure_demand_df.loc[
                                          (departure_demand_df["time"] >= window_dates[time_step + 1][0]) &
                                          (departure_demand_df["time"] < window_dates[time_step + 1][1]), :]

            new_target_departure_demands_df = target_departure_demands_df
        elif time_step == 1:
            target_departure_demands_df = departure_demand_df.loc[
                                          (departure_demand_df["time"] >= window_dates[time_step + 1][0]) &
                                          (departure_demand_df["time"] < window_dates[time_step + WINDOWS_PER_TEST_SET][1]), :]

            new_target_departure_demands_df = target_departure_demands_df
        else:
            target_departure_demands_df = departure_demand_df.loc[
                                          (departure_demand_df["time"] >= window_dates[time_step + 1][0]) &
                                          (departure_demand_df["time"] < window_dates[time_step + WINDOWS_PER_TEST_SET][1]), :]

            new_target_departure_demands_df = departure_demand_df.loc[
                                              (departure_demand_df["time"] >= window_dates[time_step + WINDOWS_PER_TEST_SET][0]) &
                                              (departure_demand_df["time"] < window_dates[time_step + WINDOWS_PER_TEST_SET][1]), :]

        # Departure Demand
        if time_step == 0:
            predicate_constructor.demand_predicate(
                aggregated_observed_departure_demand_df[
                    aggregated_observed_departure_demand_df["time"] >= window_dates[1][0] - BLOCKED_WEEKS * 7 * datetime.timedelta(days=1)
                ],
                path, OBS)

        predicate_constructor.demand_predicate(target_departure_demands_df, path, TARGET, write_value=False)
        predicate_constructor.demand_predicate(target_departure_demands_df, path, TRUTH)

        # Arrival Demand Targets
        if time_step == 0:
            target_arrival_demands_df = arrival_demand_df.loc[
                                        (arrival_demand_df["time"] >= window_dates[time_step + 1][0]) &
                                        (arrival_demand_df["time"] < window_dates[time_step + 1][1]), :]

            new_target_arrival_demands_df = target_arrival_demands_df
        elif time_step == 1:
            target_arrival_demands_df = arrival_demand_df.loc[
                                        (arrival_demand_df["time"] >= window_dates[time_step + 1][0]) &
                                        (arrival_demand_df["time"] < window_dates[time_step + WINDOWS_PER_TEST_SET][1]), :]

            new_target_arrival_demands_df = target_arrival_demands_df
        else:
            target_arrival_demands_df = arrival_demand_df.loc[
                                        (arrival_demand_df["time"] >= window_dates[time_step + 1][0]) &
                                        (arrival_demand_df["time"] < window_dates[time_step + WINDOWS_PER_TEST_SET][1]), :]

            new_target_arrival_demands_df = arrival_demand_df.loc[
                                            (arrival_demand_df["time"] >= window_dates[time_step + WINDOWS_PER_TEST_SET][0]) &
                                            (arrival_demand_df["time"] < window_dates[time_step + WINDOWS_PER_TEST_SET][1]), :]

        # Arrival Demand
        if time_step == 0:
            predicate_constructor.demand_predicate(
                aggregated_observed_arrival_demand_df[
                    aggregated_observed_arrival_demand_df["time"] >= window_dates[1][0] - BLOCKED_WEEKS * 7 * datetime.timedelta(days=1)
                ],
                path, OBS, arrival=True)

        predicate_constructor.demand_predicate(target_arrival_demands_df, path, TARGET, write_value=False, arrival=True)
        predicate_constructor.demand_predicate(target_arrival_demands_df, path, TRUTH, arrival=True)

        #  TimeBlock predicate.
        if time_step == 0:
            # Initial TimeBlock Predicate: Include the observed.
            new_time_block_df = departure_demand_df.loc[
                                (departure_demand_df["time"] >= window_dates[1][0] - BLOCKED_WEEKS * 7 * datetime.timedelta(days=1)) &
                                (departure_demand_df["time"] < window_dates[1][1]), :]
            new_time_block_df.loc[:, 'value'] = 1.0
            predicate_constructor.time_block_predicate(new_time_block_df, path, OBS)
        else:
            # Update TimeBlock Predicate: Only the added targets
            new_time_block_df = new_target_departure_demands_df.copy()
            new_time_block_df.loc[:, 'value'] = 1.0

        # Assumes no lags greater than a week.
        expired_time_block_df = departure_demand_df.loc[
               (departure_demand_df["time"] >= window_dates[time_step][0] - datetime.timedelta(hours=int(24 * 7))) &
               (departure_demand_df["time"] < window_dates[time_step][1] - datetime.timedelta(hours=int(24 * 7))), :]
        if expired_time_block_df.shape[0] > 0:
            expired_time_block_df.loc[:, 'value'] = 1.0
        else:
            expired_time_block_df = None

        # Get client commands for this timestep.
        for experiment in ['Online', 'Incremental']:
            print("Constructing " + experiment + " client commands for time step: " + str(time_step).zfill(2))
            command_list = CLIENT_COMMAND_METHOD[experiment](predicate_constructor,
                                                             observed_departure_demands_df, observed_arrival_demands_df,
                                                             new_target_departure_demands_df, new_target_arrival_demands_df,
                                                             expired_departure_demands_df, expired_arrival_demands_df,
                                                             new_time_block_df, expired_time_block_df,
                                                             time_step)

            command_list += ["Exit"]
            client_command_constructors.command_file_write(command_list, path, experiment)


def construct_static_predicates(predicate_constructor, initial_observed_trip_df, time_df, station_df, weather_df, window_dates, departure_demand_df, arrival_demand_df, out_directory):
    """
    Construct the predicates that do not change between time steps.
    """
    
    # Set the shared path between these predicates.
    path = os.path.join(out_directory, "0".zfill(3))
    if not os.path.exists(path):
        os.makedirs(path)

    # Station similarity predicates
    predicate_constructor.similarstation_predicate(station_df, departure_demand_df, TOP_K_SIMILAR[DATASET], path, arrival=False)
    predicate_constructor.similarstation_predicate(station_df, arrival_demand_df, TOP_K_SIMILAR[DATASET], path, arrival=True)

    # Location / station related predicates.
    predicate_constructor.station_predicate(station_df, path)
    predicate_constructor.nearby_predicate(station_df, path)

    # Commute hours
    # Only write commute hours for time in forecast range.
    predicate_constructor.commute_hours(time_df[time_df.time >= window_dates[1][0]], path)

    # Exogenous variables.
    # We only need the exogenous observations for the forecasted values.
    raining_df = predicate_constructor.raining_predicate(weather_df, station_df, window_dates[1][0], path)

    # Source Destination
    predicate_constructor.destination_predicate(initial_observed_trip_df, path)
    predicate_constructor.source_predicate(initial_observed_trip_df, path)

    # Time
    predicate_constructor.lag_n_predicate(time_df, TIME_STEP_HR, TIME_STEP_HR_INT, 1, window_dates[1][0], path)
    predicate_constructor.lag_n_predicate(time_df, TIME_STEP_HR, TIME_STEP_HR_INT, 2, window_dates[1][0], path)
    predicate_constructor.lag_n_predicate(time_df, TIME_STEP_HR, TIME_STEP_HR_INT, 12, window_dates[1][0], path)
    predicate_constructor.lag_n_predicate(time_df, TIME_STEP_HR, TIME_STEP_HR_INT, 24, window_dates[1][0], path)
    predicate_constructor.lag_n_predicate(time_df, TIME_STEP_HR, TIME_STEP_HR_INT, 48, window_dates[1][0], path)
    predicate_constructor.lag_n_predicate(time_df, TIME_STEP_HR, TIME_STEP_HR_INT, 168, window_dates[1][0], path)

    predicate_constructor.isday_predicate(time_df[time_df.time >= window_dates[1][0]], path)
    predicate_constructor.isweekend_predicate(time_df[time_df.time >= window_dates[1][0]], path)
    predicate_constructor.samehour_blocked(time_df, raining_df, window_dates[1][0], BLOCKED_WEEKS, path)
    predicate_constructor.samedaysamehour_blocked(time_df, raining_df, window_dates[1][0], BLOCKED_WEEKS, path)

    # Station Clustering Predicates
    predicate_constructor.stationcluster_predicate(station_df,
                                                   departure_demand_df[departure_demand_df.time <= LEARN_END_DATE],
                                                   CLUSTER_COUNT[DATASET],
                                                   path)


def main():
    construct_predicates()


if __name__ == '__main__':
    main()
