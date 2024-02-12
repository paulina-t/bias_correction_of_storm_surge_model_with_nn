import numpy as np
import keras
import datetime as dt
import pickle
import xarray as xr
import sys
import pytz


from ml_storm_surge_operational.data_loader.preprocessor_operational_predict import (
    PreprocessInput 
    as PreprocessInputPredict
)
import ml_storm_surge_operational.utils.helpers as hlp

def predict_residuals(datetime_predict):
    """
     Predict the residuals at all stations and lead times for a given time. 
     
     Args:
     	 datetime_predict: datetime of predicting time in YYYYMMDD format
     
     Returns: 
     	 Y_test_pred
    """
    stations = get_list_of_stations()
    Y_test_pred = np.full([60, len(stations)], np.nan)
    count_station = 0
    # Takes a list of stations and returns the predictions for each station.
    for station in stations:
        train_dict = load_training_data(station)
        y_train_mean = train_dict['y_train_mean']
        y_train_std = train_dict['y_train_std']
        test_dict = preprocess_test_data(
            station, 
            datetime_predict
            )
        x_test_norm = test_dict['x_norm']
        # Iterate over lead times
        for t in range(1, 61):
            model = load_trained_model(station, t)
            # model.predict(x_test_norm) triggers a warning: 
            # https://stackoverflow.com/questions/66271988/warningtensorflow11-out-of-the-last-11-calls-to-triggered-tf-function-retracin
            y_test_norm_pred = model(x_test_norm, training=False) 
            y_test_pred = inverse_norm(
                x_norm=y_test_norm_pred, 
                mu=y_train_mean[t - 1], 
                sigma=y_train_std[t - 1]
                )
            Y_test_pred[t-1, count_station] = y_test_pred
        count_station = count_station + 1
    return Y_test_pred

def get_list_of_stations():
    """Get a list of the permanent stations in Norway."""
    stations = hlp.get_norwegian_station_names()
    stations.remove('NO_MSU')
    stations.remove('NO_NYA')
    return stations

def load_training_data(station):
    # Load the training data
    x_y_dir = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast'
        + '/operational/data/x_y/'
        + station 
        )

    x_y_file = (
        'x_y_' 
        + station 
        + '.pickle'
    )

    with open(x_y_dir + '/' + x_y_file, 'rb') as handle:
        b = pickle.load(handle)

    return b

def load_trained_model(station, t):
    """
     Load a trained model.
     
     Args:
     	 station: String name of the station
     	 t: Integer lead time of the model to load. 
     
     Returns: 
     	 Keras model that was
    """
    model_dir = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast'
        + '/operational/ml_models/'
        + station 
    )

    model_file = (
        'best_model_' 
        + station 
        + '_t' 
        + str(t) 
        + '.h5'
    )

    model = keras.models.load_model(model_dir + '/' + model_file)
    return model

def preprocess_test_data(station, datetime_predict):
    """
     Preprocess test data. This is a wrapper around : func : 
     `preprocess_input_predict`
     
     Args:
     	 station: station to predict on.
     	 datetime_predict: datetime of start of forecast. It is used to set 
         time_start_hourly
     
     Returns: 
     	 dict with preprocessed data
    """
    feature_stations = get_feature_stations_per_region(station)

    feature_vars = [
        'obs', 
        'tide_py', 
        'stormsurge_corrected', 
        'u10', 
        'v10'
        ] 
    label_var = 'stormsurge_corrected - (obs - tide_py)'

    window_width_past = [24, 24, 0, 0, 0]
    window_width_future = [-1, 60, 60, 60, 60]  # -1 does not include time t0
    horizon = 60  

    datetime_start_hourly = datetime_predict

    pp = PreprocessInputPredict(
        feature_vars = feature_vars,
        label_var = label_var,
        window_width_past = window_width_past,
        window_width_future = window_width_future,
        horizon = horizon,
        feature_stations = feature_stations,
        label_station = station,
        datetime_start_hourly = datetime_start_hourly,
        datetime_end_hourly = datetime_start_hourly,
        remove_nans = False,
        normalize=True,
        use_existing_files=False,
        past_forecast=24
        )
    pp.preprocess_data()

    b = pp.x_y_dict
    return b

def get_stations_per_region():
    """The same method as in main_run_models_for_all_stations.py. 
    TODO: Move to utils"""
    
    skagerak = [
        'NO_OSL', 
        'NO_OSC', 
        'NO_VIK', 
        'NO_HRO', 
        'NO_TRG'
        ]
    west_coast = [
        'NO_SVG', 
        'NO_BGO', 
        'NO_MAY', 
        'NO_AES', 
        'NO_KSU', 
        'NO_HEI', 
        'NO_TRD', 
        'NO_RVK'
        ]
    north = [
        'NO_BOO', 
        'NO_KAB', 
        'NO_NVK', 
        'NO_HAR', 
        'NO_ANX', 
        'NO_TOS', 
        'NO_HFT', 
        'NO_HVG', 
        'NO_VAW'
        ]
    
    return skagerak, west_coast, north

def get_feature_stations_per_region(station):
    """The same method as in main_run_models_for_all_stations.py. 
    TODO: Move to utils"""
    
    skagerak, west_coast, north = get_stations_per_region()

    features_skagerak = [
        'NO_OSC', 
        'NO_AES', 
        'NO_BGO', 
        'NO_VIK', 
        'NO_TRG'
        ]
    features_west_coast=[
        'NO_BGO', 
        'NO_AES', 
        'NO_OSC', 
        'NO_MAY', 
        'NO_SVG'
        ]
    features_north = [
        'NO_ANX', 
        'NO_BOO', 
        'NO_HFT', 
        'NO_KAB', 
        'NO_KSU'
        ]
    
    
    if station in skagerak:
        if station in features_skagerak:
            feature_stations = features_skagerak
        else: 
            feature_stations = [
                station,  
                'NO_AES', 
                'NO_BGO', 
                'NO_HEI', 
                'NO_KSU'
                ]
    elif station in west_coast:
        if station in features_west_coast:
            feature_stations = features_west_coast
        else:
            feature_stations = [
                station, 
                'NO_AES', 
                'NO_OSC', 
                'NO_MAY', 
                'NO_SVG'
                ]
    elif station in north:
        if station in features_north:
            feature_stations = features_north
        else:             
            feature_stations = [
                station, 
                'NO_ANX', 
                'NO_BOO', 
                'NO_HFT', 
                'NO_KAB', 
                'NO_KSU'
                ]
            
    return feature_stations

def inverse_norm(x_norm, mu, sigma):
    """
     Inverse normalization of data. Unnormalizes and adds mean and standard 
     deviation to the data
     
     Args:
     	 x_norm: Normalized data to be unnormalised
     	 mu: Mean of the data ( must be positive definite )
     	 sigma: Standard deviation of the data ( must be positive definite )
     
     Returns: 
     	 x ( numpy array ) : Array of normalized data
    """
    x = (x_norm * sigma) + mu
    return x

def get_lats_lons_of_stations():
    """
     Get latitudes and longitudes of stations. This function is used to get the 
     latitudes and longitudes of stations.
     
     Returns: 
     	 list of station latitudes, list of station longitudes.
    """
    stations = get_list_of_stations()
    lats = []
    lons = []
    # Add lat lon to the list of stations
    for station in stations:
        lat, lon = hlp.get_station_lat_lon(station)
        lats.append(lat)
        lons.append(lon)
    lats = np.array(lats)
    lons = np.array(lons)
    return lats, lons

def compute_stormsurge_corrected_ml(kyststasjoner_dataset, Y_test_pred):
    """
     Compute stormsurge_corrected_ml. This function is used to correct 
     Nordic4-SS with ML predictions of the residuals
     
     Args:
     	 kyststasjoner_dataset: dataset of kyststasjoner data.
     	 Y_test_pred: 2D array of length 60
     
     Returns: 
     	 kyststasjoner_dataset
    """
    # Y_test_pred has shape 60 x n_stations
    kyststasjoner_dataset['stormsurge_corrected_ml'] = (
        ["time", "ensemble_member", "dummy", "station"], 
        np.full(
            kyststasjoner_dataset['stormsurge_corrected'].shape, 
            np.nan
            )
    )

    len_time = len(kyststasjoner_dataset["time"])
    len_stations = len(kyststasjoner_dataset["station"])

    kyststasjoner_dataset['bias_ml'] = (
        ["time", "station"], 
        np.full(
            [len_time, len_stations], 
            np.nan
            )
    )

    # Add full horizon length to Y_test_pred
    stations = get_list_of_stations()
    count_stations = 0
    for station in stations:
        station_nr = get_station_id(kyststasjoner_dataset, station)
        print('station: ', station, 'station_nr: ', station_nr)

        # This function is used to get the Nordic4-SS predictions with ROMS
        # (corrected with the sliding error method) for each station. It is 
        # saved in the kyststasjoner files.
        for m in range(len(kyststasjoner_dataset['ensemble_member'])):
            stormsurge_corrected_station_m  = (
                kyststasjoner_dataset['stormsurge_corrected']
                .isel(
                    {
                        'station':station_nr, 
                        'ensemble_member':m, 
                        'dummy':0
                        }
                        ).values
            )

            # Fill in Y_test_pred with NaNs so that it has the same length as 
            # the ROMS forecast
            diff_length = (
                stormsurge_corrected_station_m.shape[0]
                - Y_test_pred.shape[0]
            )

            Y_test_pred_station_m = np.append(
                Y_test_pred[:,count_stations], 
                np.zeros(diff_length) + np.nan
                )

            # Add the bias computed with ML to the output from ROMS
            stormsurge_corrected_ml_station_m = (
                stormsurge_corrected_station_m
                - Y_test_pred_station_m
            )

            (kyststasjoner_dataset['stormsurge_corrected_ml']
                [:, m, 0, station_nr]) = stormsurge_corrected_ml_station_m

            (kyststasjoner_dataset['bias_ml']
                [:, station_nr]) = Y_test_pred_station_m
            
        count_stations = count_stations + 1
    return kyststasjoner_dataset

def get_station_id(data_file, station_name):
        """
        TODO: Move to utils - also used in read_kyststasjoner_hourly
        Get station id of a particular station in a particular file.

        Parameters
        ----------
        data_file : xarray.Dataset
            Dataset containing kyststasjoner data.
        station_name : str
            Station name (short) including land code. Only norwegian stations
            are valid.

        Returns
        -------
        station_id : int
            station_id

        """
        stations_id = decode_station_names(data_file)
        station_id = stations_id[station_name]
        return station_id

def decode_station_names(data_file):
        """
        TODO: Move to utils, also used in read_kyststasjoner_hourly
        Generate a dictionary that maps station names and IDs.

        Parameters
        ----------
        data_file : xarray.Dataset
            Dataset containnig kyststasjoner data.

        Returns
        -------
        stations_id : dict
            The keys of the dictionary are the station names and the values are 
            the IDs in the data_file. These ID are not the same in the all 
            files.

        """
        stations_id={}
        n_stations = len(data_file.station)
        for i in range(n_stations):
            name = data_file.station_name[0:3,i].values.astype(str).tolist()
            name = "".join(name)  # Concatenate letters in list 
            name = 'NO_' + name.upper()
            stations_id[name] = i
        return stations_id


def open_kyststasjoner_and_add_correction(Y_test_pred, datetime_predict):
    """
     Open Kyststasjoner file dataset and add correction.
     
     Args:
     	 Y_test_pred: array of shape [ 60 x nstations]
     	 datetime_predict: datetime of predicting date. YYYYMMDD
     
     Returns: 
     	 xarray dataset with stormsurge corrected ML ( same shape as Y_test_pred )
    """
    data_dir = '/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive'
    date_str = datetime_predict.strftime('%Y%m%d%H')
    file_root_name = '/kyststasjoner_norge.nc'
    path = (
        data_dir 
        + file_root_name 
        + date_str
        )
    
    dataset = xr.open_dataset(path)
    dataset_with_ml_correction = compute_stormsurge_corrected_ml(
        dataset, 
        Y_test_pred
        )
    
    return dataset_with_ml_correction

def write_nc(ds, datetime_predict):
    """
     Write dataset to netcdf file.
     
     Args:
     	 ds: Dataset to be written to netcdf file
     	 datetime_predict: datetime of predicting storm surge
    """
    data_dir = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace'
        + '/paulinast/ml_predict_storm_surge'
    )

    date_str = (
        str(datetime_predict)
        .split(':')[0]
        .replace('-', '')
        .replace(' ', '')
    )
    path = data_dir + '/' + date_str + '.nc'
    ds.to_netcdf(path)


def main(datetime_predict):
    """
     Add predicted residuals with ML to Nordic4-SS and write to netCDF file. 
     This is a wrapper.
     
     Args:
     	 datetime_predict: datetime. datetime object with prediction time
    """
    Y_test_pred = predict_residuals(datetime_predict)
    dataset_with_ml_correction = open_kyststasjoner_and_add_correction(
        Y_test_pred, 
        datetime_predict
        )
    write_nc(
        dataset_with_ml_correction,
        datetime_predict
        )

if __name__ == "__main__":
    datetime_predict = sys.argv[1]
    datetime_predict = (
        dt.datetime
        .strptime(datetime_predict, "%Y-%m-%d %H:%M")
        .replace(tzinfo=pytz.UTC)
    )

    main(datetime_predict)