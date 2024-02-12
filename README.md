# bias_correction_of_storm_surge_model_with_nn
Bias correction with Neural Networks of the operational storm surge model (ROMS) that runs at the Norwegian Meteorological Institute.


# Before using the files in this repository
Note that the code in this repository works for internal paths to the datasets stored in MET Norway’s systems (PPI). The user should update the paths. 

# How to make inference

## Run the script main_predict.py

This script uses pre-trained models to predict the residuals in one “kyststasjoner” (Nordic4-SS) file and is designed to run operationally.

Example of how to run the script:
nohup time python3 -u main_predict.py '2022-05-09 00:00' > nohup_predict.out & 

This will predict the residuals in the Nordic4-SS prediction generated at 2022-05-09 00:00.
The output will be saved in a NetCDF file named 2022050900.nc in the directory /lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast/ml_predict_storm_surge

Note that the script only works for Nordic4-SS predictions generated at 00 or 12 UTC (not 06 or 18).

# How to train the models

## Run the script main_run_models_for_all_stations.py.
This script will train the ML models at all stations and for all lead times.

The following files are needed:
prepare_df_operational.py to get a DataFrame with all the data
preprocessor_operational_train_and_test.py to get a dict with preprocessed arrays ( X_train, Y_train, X_test, Y_test, etc.)
some helpers


## prepare_df_operational.py

Make one big DataFrame with all the raw data that we need. This will later be preprocessed and put into training and test arrays.

The data come from different sources: kyststasjoner_files, Jean’s prepared dataset containing observations and tide data, and weather forecasts. Everything is saved on PPI.

Some variables are derived from others, for instance, (obs - tide) or wind speed.


First, an auxiliary DataFrame is created with all the variables needed to make the final DataFrame with the requested variables. For instance, if wind_speed is requested, the auxiliary DataFrame will contain u10 and v10. 

Then, the requested variables are computed from the data in the auxiliary DataFrame and stored in a new DataFrame.

Obs: To avoid opening all the files (which involves many slow operations), a DataFrame for each set of variables with the same source have been pre-generated and saved on PPI. The functions that prepare these datasets are:

generate_obs_tide_df
add_obs_tide_df
generate_station_forcing_df
generate_operational_df
add_roms_obs_tide

## preprocessor_operational_train_test.py
This function takes the DataFrames with all the raw data and generates training and test arrays for the ML models, by lagging some variables, filling in gaps, and splitting the dataset.

# Notes:
Some variables can be lagged to generate a simulated lead time (e.g., obs, tide), but for forecast variables (e.g., storm surge, wind) we use the actual lead time provided in the forecast files.

Directories

MEPS:
data_dir = '/lustre/storeB/immutable/archive/projects/metproduction/MEPS'

Nordic4-SS:
data_dir = '/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive' 

Column names:
varName_t_leadTime_whenForecastBegins_stationID

analysis time (lead time = 0), t0: do not use t_leadTime
forecasts generated at analysis time, t0: do not add whenforecastBegins



