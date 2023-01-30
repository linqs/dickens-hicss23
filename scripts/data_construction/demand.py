"""
Demand dataframe constructors.
"""

import numpy as np
import pandas as pd
import pickle
import os


def construct_trip_based_demand_dfs(trip_df, station_df, start_date, end_date, time_step_min, time_step_hr, out_path):
    trip_df.loc[:, 'id'] = trip_df.index

    # Demand is defined as the number trips starting at a stations.
    trip_df_departure_subset = trip_df.loc[:, ['start_station_id', 'start_date', 'id']]
    trip_df_arrival_subset = trip_df.loc[:, ['end_station_id', 'end_date', 'id']]

    trip_df_departure_subset.loc[:, 'start_date'] = trip_df_departure_subset.start_date.dt.floor(time_step_min)
    trip_df_arrival_subset.loc[:, 'end_date'] = trip_df_arrival_subset.end_date.dt.floor(time_step_min)

    departure_demand_df = trip_df_departure_subset.groupby(['start_station_id', 'start_date']).count().reset_index()
    arrival_demand_df = trip_df_arrival_subset.groupby(['end_station_id', 'end_date']).count().reset_index()

    departure_demand_df.columns = ['station_id', 'time', 'demand']
    departure_demand_df = departure_demand_df[(departure_demand_df.time >= start_date) & (departure_demand_df.time < end_date)]
    arrival_demand_df.columns = ['station_id', 'time', 'demand']
    arrival_demand_df = arrival_demand_df[(arrival_demand_df.time >= start_date) & (arrival_demand_df.time < end_date)]

    # Normalize by dock counts.
    # departure_demand_df.demand = np.clip(departure_demand_df.demand / station_df.loc[departure_demand_df.station_id, 'dock_count'].values, 0.0, 1.0)
    # arrival_demand_df.demand = np.clip(arrival_demand_df.demand / station_df.loc[arrival_demand_df.station_id, 'dock_count'].values, 0.0, 1.0)

    # Normalize with quantiles.
    # departure_quantiles = departure_demand_df.loc[:, ['station_id', 'demand']].groupby('station_id').quantile(0.98)
    # departure_demand_df.demand = np.clip(departure_demand_df.demand / departure_quantiles.loc[departure_demand_df.station_id, 'demand'].values, 0.0, 1.0)
    #
    # arrival_quantiles = arrival_demand_df.loc[:, ['station_id', 'demand']].groupby('station_id').quantile(0.98)
    # arrival_demand_df.demand = np.clip(arrival_demand_df.demand / arrival_quantiles.loc[arrival_demand_df.station_id, 'demand'].values, 0.0, 1.0)

    # Fill in missing data ranges.
    departure_demand_df = departure_demand_df.append(pd.DataFrame(data={
        'station_id': -1,
        'time': pd.date_range(start_date, end_date, freq=time_step_hr, closed='left'),
        'demand': 0
    }))
    departure_demand_df = departure_demand_df.set_index(['station_id', 'time']).unstack(fill_value=0).stack().reset_index()
    departure_demand_df.drop(departure_demand_df[departure_demand_df.station_id == -1].index, inplace=True)

    arrival_demand_df = arrival_demand_df.append(pd.DataFrame(data={
        'station_id': -1,
        'time': pd.date_range(start_date, end_date, freq=time_step_hr, closed='left'),
        'demand': 0
    }))
    arrival_demand_df = arrival_demand_df.set_index(['station_id', 'time']).unstack(fill_value=0).stack().reset_index()
    arrival_demand_df.drop(arrival_demand_df[arrival_demand_df.station_id == -1].index, inplace=True)

    # Normalize with std.
    departure_stds = departure_demand_df.loc[:, ['station_id', 'demand']].groupby('station_id').std()
    (departure_stds * 6).to_csv(os.path.join(out_path, "departure_demand_scaling.csv"))
    departure_demand_df.demand = np.clip(
        departure_demand_df.demand / (6 * departure_stds.loc[departure_demand_df.station_id, 'demand'].values), 0.0, 1.0)

    arrival_stds = arrival_demand_df.loc[:, ['station_id', 'demand']].groupby('station_id').std()
    (arrival_stds * 6).to_csv(os.path.join(out_path, "arrival_demand_scaling.csv"))
    arrival_demand_df.demand = np.clip(
        arrival_demand_df.demand / (6 * arrival_stds.loc[arrival_demand_df.station_id, 'demand'].values), 0.0, 1.0)

    # Pickle Re-Scaled demands for baselines.
    with open(os.path.join(out_path, "departure_demand.pickle"), "wb") as output:
        rescaled_departure_demand_df = departure_demand_df.copy()
        rescaled_departure_demand_df.demand = (
                rescaled_departure_demand_df.demand * 6 * departure_stds.loc[rescaled_departure_demand_df.station_id].demand.values
        )
        pickle.dump(rescaled_departure_demand_df.set_index(['time', 'station_id']).unstack().values, output, protocol=2)

    with open(os.path.join(out_path, "arrival_demand.pickle"), "wb") as output:
        rescaled_arrival_demand_df = arrival_demand_df.copy()
        rescaled_arrival_demand_df.demand = (
                rescaled_arrival_demand_df.demand * 6 * arrival_stds.loc[rescaled_arrival_demand_df.station_id].demand.values
        )
        pickle.dump(rescaled_arrival_demand_df.set_index(['time', 'station_id']).unstack().values, output, protocol=2)

    return departure_demand_df, arrival_demand_df


def construct_capacity_based_demand_df(trip_df, status_df, station_df, time_step_min):
    # Demand is defined as the number of bikes checked out / total capacity.
    demand_df = status_df.copy(deep=True)
    demand_df.time = demand_df.time.dt.floor(time_step_min)
    demand_df['demand'] = (demand_df['docks_available']) / (demand_df['docks_available'] + demand_df['bikes_available'])
    demand_df.demand = demand_df.groupby(['station_id', 'time'])['demand'].mean()
    demand_df.demand = np.clip(demand_df.demand.values, 0.0, 1.0)
    demand_df = demand_df.drop(['bikes_available', 'docks_available'], axis=1)
    return demand_df
