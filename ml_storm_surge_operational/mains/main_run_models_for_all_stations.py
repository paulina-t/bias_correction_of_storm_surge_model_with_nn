import numpy as np
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import datetime as dt
import pickle
import os

from ml_storm_surge_operational.data_loader.preprocessor_operational_train_and_test import PreprocessInput
import ml_storm_surge_operational.utils.helpers as hlp

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

def get_stations_per_region():
    """ Get the three clusters of stations in Norway: Skagerrak, West coast, and
    Northern Norway"""
    
    skagerrak = [
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
    
    return skagerrak, west_coast, north

def get_feature_stations_per_region(station):
    """For each station where the residuals will be predicted, get the group of 
    stations from which the features will be extracted. Each cluster has one 
    group of feature stations. The stations where the error is predicted is 
    always included in the feature stations."""
    
    # List of stations in each cluster
    skagerrak, west_coast, north = get_stations_per_region()

    # Feature stations for each cluster
    features_skagerrak = [
        'NO_OSC', # To be replaced with the station where the error is predicted
        'NO_AES', 
        'NO_BGO', 
        'NO_VIK', 
        'NO_TRG'
        ]
    features_west_coast=[
        'NO_BGO', # To be replaced with the station where the error is predicted
        'NO_AES', 
        'NO_OSC', 
        'NO_MAY', 
        'NO_SVG'
        ]
    features_north = [
        'NO_ANX', # To be replaced with the station where the error is predicted
        'NO_BOO', 
        'NO_HFT', 
        'NO_KAB', 
        'NO_KSU'
        ]
    
    # Replace first station in the feature station lists with the station where
    # the error is predicted
    if station in skagerrak:
        if station in features_skagerrak:
            feature_stations = features_skagerrak
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

def ml_stormsurrge_model(t, x_train_norm, y_train_norm, x_test_norm, 
                         y_test_norm, y_train_mean, y_train_std, y_test, 
                         station):
    
    try:
        del history
        del multi_dense_model
    except:
        print('Variables do not exist')

    multi_dense_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.Dense(1)    
    ])

    model_dir = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast'
        + '/operational/ml_models/' 
        + station 
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_dir + "/best_model" + "_" + station + "_t" + str(t + 1) + ".h5",  # t0 is the first lead time
            save_best_only=True, 
            monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.5, 
            patience=20, 
            min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(
        monitor="val_loss", 
        patience=50, 
        verbose=1
        )
    ]

    multi_dense_model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.MeanSquaredError(),
        metrics=[tf.metrics.MeanAbsoluteError()]
    )


    history = multi_dense_model.fit(
        x_train_norm,
        y_train_norm[:, t],
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.3,
        verbose=1
    )

    test_loss, test_acc = multi_dense_model.evaluate(
        x_test_norm, y_test_norm[:, t]
        )

    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

    y_test_norm_pred = multi_dense_model.predict(x_test_norm)

    mae_scaled = np.mean(
        np.abs( y_test_norm[:, t] - y_test_norm_pred.transpose() )
        )
    print('MAE of normalized variables: ', mae_scaled)

    # Unnormalize values
    y_test_pred = (y_test_norm_pred * y_train_std[t]) + y_train_mean[t]

    rmse_scaled = np.sqrt(
        np.mean((y_test[:, t] - y_test_pred.transpose())**2)
        )   
    bias_scaled = np.mean( -(y_test[:, t] - y_test_pred.transpose()) ) 
    std_scaled = np.std( -(y_test[:, t] - y_test_pred.transpose()) ) 
 
    print('rmse_scaled: ', rmse_scaled)
    
    return y_test_pred, rmse_scaled, bias_scaled, std_scaled, history, multi_dense_model 
    
# ------ Run the models ------    
# ------ Declare variables ------
fig_dir = (
        '/lustre/storeB/project/IT/geout/'
        +'machine-ocean/workspace/paulinast/storm_surge_results/'
        + 'linear_model/station_selection'
        )

feature_vars = ['obs', 'tide_py', 'stormsurge_corrected', 'u10', 'v10'] 
label_var = 'stormsurge_corrected - (obs - tide_py)'

window_width_past = [24, 24, 0, 0, 0]
window_width_future = [-1, 60, 60, 60, 60] # -1 does not include time t0
horizon = 60  
    
datetime_start_hourly=dt.datetime(2018, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
datetime_end_hourly=dt.datetime(2021, 3, 31, 12, 0, 0, tzinfo=dt.timezone.utc)
datetime_split=dt.datetime(2020, 3, 31, 12, 0, 0, tzinfo=dt.timezone.utc)
    

stations = hlp.get_norwegian_station_names()
stations.remove('NO_MSU')
stations.remove('NO_NYA')

# ------ Linear model ------

OUT_STEPS = horizon
num_features = 1
batch_size =32
epochs=500

metric = 'mean_absolute_error'

results_dir = (
    '/lustre/storeB/project/IT/geout/machine-ocean/workspace'
    + '/paulinast/operational'
    )

for station in stations:
    keras.backend.clear_session()
    
    print( '------ ', station, '------')  
    
    feature_stations =  get_feature_stations_per_region(station)
        
    # Open preprocessed file with all variables and stations!

    hist_path = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace'
        + '/paulinast/storm_surge_results/ml/history'
    )
    best_model_path = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace'
        + '/paulinast/storm_surge_results/ml/best_models'
    )
    
    pp = PreprocessInput(
        feature_vars = feature_vars,
        label_var = label_var,
        window_width_past = window_width_past,
        window_width_future = window_width_future,
        horizon = horizon,
        feature_stations = feature_stations,
        label_station = station,
        datetime_start_hourly = datetime_start_hourly,
        datetime_end_hourly = datetime_end_hourly,
        datetime_split = datetime_split,
        remove_nans = False,
        imputation_strategy='most_frequent',
        missing_indicator=False,
        normalize=True,
        use_existing_files=True,
        past_forecast=24
        )
    
    pp.preprocess_data()
    
    len_past_features  = len(pp.feature_names_past)

    x_train = pp.x_y_dict['x_train']
    y_train = pp.x_y_dict['y_train']
    x_test = pp.x_y_dict['x_test']
    y_test = pp.x_y_dict['y_test']
    x_train_mean = pp.x_y_dict['x_train_mean']
    x_train_std = pp.x_y_dict['x_train_std']
    y_train_mean = pp.x_y_dict['y_train_mean']
    y_train_std = pp.x_y_dict['y_train_std']
    x_train_norm = pp.x_y_dict['x_train_norm']
    y_train_norm = pp.x_y_dict['y_train_norm']
    x_test_norm = pp.x_y_dict['x_test_norm']
    y_test_norm = pp.x_y_dict['y_test_norm']

    x_y_dir = results_dir + '/data/x_y/' + station 
    if not os.path.exists(x_y_dir):
        os.makedirs(x_y_dir)
    with open(x_y_dir +'/x_y_' + station + '.pickle', 'wb') as handle:
        pickle.dump(pp.x_y_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Model
    rmse_all_times = []
    bias_all_times = []
    std_all_times = []
    history_all_times = []
    models_all_times=[]
    y_test_pred_all_times=[]
    learning_curves = {metric:{}}
    learning_curves[metric]['member' + str(-1)]  = {}
    learning_curves[metric]['member' + str(-1)]['val']={}
    learning_curves[metric]['member' + str(-1)]['train']={}
    for t in range(60):
        print('Lead time: ', t +1)
        y_test_pred_t, rmse_t, bias_t, std_t, history_t, model_t = ml_stormsurrge_model(
            t, 
            x_train_norm, 
            y_train_norm, 
            x_test_norm, 
            y_test_norm, 
            y_train_mean, 
            y_train_std, 
            y_test, station
            )
        learning_curves[metric]['member' + str(-1)]['val']['t' + str(t)] = (
            history_t.history["val_" + metric]
        )
        learning_curves[metric]['member' + str(-1)]['train']['t' + str(t)] = (
            history_t.history[metric]
        )
        rmse_all_times.append(rmse_t)  
        bias_all_times.append(bias_t) 
        std_all_times.append(std_t)  
        history_all_times.append(history_t)
        models_all_times.append(model_t)
        y_test_pred_all_times.append(y_test_pred_t)
    
        
    stats_dict = {
        'rmse' : rmse_all_times, 
        'bias' : bias_all_times, 
        'std' : std_all_times, 
        'y_test_pred' : y_test_pred_all_times
        #'x_y_dict' : pp.x_y_dict
        }

    stats_dir = results_dir + '/results/stats/' + station

    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
        
    path = stats_dir +'/stats_' + station + '.pickle'
    print(path)
    
    with open(path, 'wb') as handle:
        pickle.dump(stats_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    
