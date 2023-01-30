import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import datetime

MODELS = ["collective"]
DATASETS = ["Bay_Area"]
EXPERIMENTS = ["Online"]

DAYS = {
    "Blue_Bikes": [str(i).zfill(3) for i in range(1, 30)],
    'Metro': [str(i).zfill(3) for i in range(1, 30)],
    'Bay_Area': [str(i).zfill(3) for i in range(1, 30)]
}



def computeStationRMSE(station_truthDemand, station_predictedDemand):
    station_predictedDemand = station_predictedDemand.set_index('time')
    station_truthDemand = station_truthDemand.set_index('time')
    times = station_predictedDemand.index.sort_values()
    return np.sqrt(mean_squared_error(
        station_truthDemand.loc[times, 'demand'], 
        station_predictedDemand.loc[times, 'demand']))

def computeStationR2(station_truthDemand, station_predictedDemand):
    station_predictedDemand = station_predictedDemand.set_index('time')
    station_truthDemand = station_truthDemand.set_index('time')
    times = station_predictedDemand.index.sort_values()
    return r2_score(station_truthDemand.loc[times, 'demand'].values, station_predictedDemand.loc[times, 'demand'].values)



def loadStationToIntIdMap(dataset):
    stationToIntIdMap = pd.read_csv("../data/{}/eval/station_to_int_id.csv".format(dataset), index_col=0)
    stationToIntIdMap.columns = ["station_id"]
    return stationToIntIdMap

def loadScalingInformation(dataset, demand_type):
    scaling_information = pd.read_csv("../data/{}/eval/{}_demand_scaling.csv".format(dataset, demand_type), index_col=0)
    stationToIntIdMap = loadStationToIntIdMap(dataset)
    scaling_information.index = stationToIntIdMap.loc[scaling_information.index, "station_id"]
    return scaling_information

def loadTruth(data_dir, truth_file_name, scaling_information):
    truth_demand = pd.read_csv("{}/{}".format(data_dir, truth_file_name), sep='\t', header=None)
    truth_demand.columns = ['station_id', 'time', 'demand']
    truth_demand.demand = (
        truth_demand.demand * scaling_information.loc[truth_demand.station_id].demand.values
    )
    return truth_demand



def loadIncrementalDailyPredictions(day_results_dir, prediction_file_name, scaling_information):
    demand_df = pd.DataFrame()
    
    for time_step in INCREMENTAL_DAILY_TIME_STEPS:
        time_step_demand_df = pd.read_csv("{}/{}/{}".format(day_results_dir, time_step, prediction_file_name), sep='\t', header=None)
        time_step_demand_df.columns = ['station_id', 'time', 'demand']
        time_step_demand_df.demand = (
            time_step_demand_df.demand * scaling_information.loc[time_step_demand_df.station_id].demand.values
        )
        
        demand_df = pd.concat([demand_df, time_step_demand_df])
        
    return demand_df



def loadOnlineDailyPredictions(day_results_dir, prediction_file_name, scaling_information):    
    demand_df = pd.read_csv("{}/{}".format(day_results_dir, prediction_file_name), sep='\t', header=None)
    demand_df.columns = ['station_id', 'time', 'demand']
    demand_df.demand = (
        demand_df.demand * scaling_information.loc[demand_df.station_id].demand.values
    )        
    return demand_df

prediction_loaders = {
    "Incremental": loadIncrementalDailyPredictions,
    "Online": loadOnlineDailyPredictions
}

results_frame = pd.DataFrame()

# Forecast windows.
forecast_window = 24

for experiment in EXPERIMENTS:
    print(experiment)
    
    prediction_loader = prediction_loaders[experiment]
    
    for dataset in DATASETS:
        print(dataset)
        arrival_scaling_information = loadScalingInformation(dataset, "arrival")
        departure_scaling_information = loadScalingInformation(dataset, "departure")

        for model in MODELS:
            print(model)
            
            for day in DAYS[dataset][:]:
                day_results_dir = "../results/{}/{}/{}/inferred-predicates/{}".format(experiment, dataset, model, day)
                data_dir = "../data/{}/eval/{}".format(dataset, day)
        
                predicted_arrival_demand = prediction_loader(day_results_dir, 'ARRIVALDEMAND.txt', arrival_scaling_information)
                predicted_departure_demand = prediction_loader(day_results_dir, 'DEPARTUREDEMAND.txt', departure_scaling_information)
                                
                truth_arrival_demand = loadTruth(data_dir, "ArrivalDemand_truth.txt", arrival_scaling_information)
                truth_departure_demand = loadTruth(data_dir, "DepartureDemand_truth.txt", departure_scaling_information)

                # Aggregate Results.
                stations = predicted_departure_demand.station_id.unique()

                ArrivalMeanRMSE = np.mean([computeStationRMSE(
                        truth_arrival_demand.loc[truth_arrival_demand.station_id == station_id, ["time", "demand"]], 
                        predicted_arrival_demand.loc[predicted_arrival_demand.station_id == station_id, ["time", "demand"]]
                    ) for station_id in stations])
                DepartureMeanRMSE = np.mean([computeStationRMSE(
                        truth_departure_demand.loc[truth_departure_demand.station_id == station_id, ["time", "demand"]], 
                        predicted_departure_demand.loc[predicted_departure_demand.station_id == station_id, ["time", "demand"]]
                    ) for station_id in stations])
                ArrivalR2 = np.mean([computeStationR2(
                        truth_arrival_demand.loc[truth_arrival_demand.station_id == station_id, ["time", "demand"]], 
                        predicted_arrival_demand.loc[predicted_arrival_demand.station_id == station_id, ["time", "demand"]]
                    ) for station_id in stations])

                DepartureR2 = np.mean([computeStationR2(
                        truth_departure_demand.loc[truth_departure_demand.station_id == station_id, ["time", "demand"]], 
                        predicted_departure_demand.loc[predicted_departure_demand.station_id == station_id, ["time", "demand"]]
                    ) for station_id in stations])

                results = pd.DataFrame({
                    'Experiment': experiment,
                    'Model': model,
                    'Dataset': dataset,
                    'Day': day,
                    'ArrivalMeanRMSE': ArrivalMeanRMSE,
                    'DepartureMeanRMSE': DepartureMeanRMSE,
                    'OverallRMSE': (ArrivalMeanRMSE + DepartureMeanRMSE) / 2,
                    'ArrivalR2': ArrivalR2,
                    'DepartureR2': DepartureR2,
                    'OverallR2': (ArrivalR2 + DepartureR2) / 2
                }, index=[1])

                results_frame = pd.concat([results_frame, results])

for experiment in EXPERIMENTS:
    for dataset in DATASETS:
        for model in MODELS:
            print(experiment + ", Dataset: " + dataset + ", model: " + model + ", mean RMSE:" + str(np.mean(results_frame[(results_frame["Experiment"] == experiment) & (results_frame["Dataset"] == dataset) & (results_frame["Model"] == model)].groupby(by="Day")["OverallRMSE"].mean().values)))
            print(experiment + ", Dataset: " + dataset + ", model: " + model + ", median RMSE:" + str(np.median(results_frame[(results_frame["Experiment"] == experiment) & (results_frame["Dataset"] == dataset) & (results_frame["Model"] == model)].groupby(by="Day")["OverallRMSE"].median().values)))
            print(experiment + ", Dataset: " + dataset + ", model: " + model + ", mean R2:" + str(np.mean(results_frame[(results_frame["Experiment"] == experiment) & (results_frame["Dataset"] == dataset) & (results_frame["Model"] == model)].groupby(by="Day")["OverallR2"].mean().values)))
            print(experiment + ", Dataset: " + dataset + ", model: " + model + ", median R2:" + str(np.median(results_frame[(results_frame["Experiment"] == experiment) & (results_frame["Dataset"] == dataset) & (results_frame["Model"] == model)].groupby(by="Day")["OverallR2"].median().values)))
