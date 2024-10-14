import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import datetime as dt
import gc

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

from ml_storm_surge_operational.data_loader.preprocessor_new_version_2 import PreprocessInput
import ml_storm_surge_operational.utils.helpers as hlp

import tensorflow

"""
For each station where we want to predict the residuals, select the most 
relevant stations from where the features are retrieved.

Do not need to run this script in an operational context where the stations are
already selected.
"""

# Reset Keras Session
def reset_keras(model):
    sess = tensorflow.compat.v1.keras.backend.get_session()
    tensorflow.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tensorflow.compat.v1.keras.backend.get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    tensorflow.compat.v1.keras.backend.set_session(
        tensorflow.compat.v1.Session(config=config)
        )
    
import ml_storm_surge_operational.utils.helpers as hlp

def generate_indices(pp, len_past_features):
    """Define different indices for subsetting the featurure DataFrame. Not 
    needed, but useful to run different experiments without generating new 
    DataFrames."""
    stormsurge_indices_future = []
    not_stormsurge_indices_future = []
    i = 0
    for name in pp.feature_names_future:
        if name.startswith('stormsurge'):
            stormsurge_indices_future.append(i + len_past_features) # Only in the future, counting from the first feature in the past
        else:
            not_stormsurge_indices_future.append(i + len_past_features)
        i = i + 1

    past_forecast_indices = []
    not_past_forecast_indices = []
    for index, var_name in enumerate(pp.feature_names_future):
        if ('_-' in var_name):
            past_forecast_indices.append(index + len_past_features)
        else:
            not_past_forecast_indices.append(index + len_past_features)

    past_forecast_indices_24 = []
    not_past_forecast_indices_24 = []
    for index, var_name in enumerate(pp.feature_names_future):
        if ('_-24' in var_name):
            past_forecast_indices_24.append(index + len_past_features)
        else:
            not_past_forecast_indices_24.append(index + len_past_features)

    msl_indices = []
    not_msl_indices = []
    for index, var_name in enumerate(pp.feature_names_future):
        if ('msl' in var_name):
            msl_indices.append(index + len_past_features)
        else:
            not_msl_indices.append(index + len_past_features)
            

    u10_indices = []
    not_u10_indices = []
    for index, var_name in enumerate(pp.feature_names_future):
        if ('swh' in var_name):
            u10_indices.append(index + len_past_features)
        else:
            not_u10_indices.append(index + len_past_features)
            

    v10_indices = []
    not_v10_indices = []
    for index, var_name in enumerate(pp.feature_names_future):
        if ('mwd' in var_name):
            v10_indices.append(index + len_past_features)
        else:
            not_v10_indices.append(index + len_past_features)
            

    future_tide_indices = []
    not_future_tide_indices = []
    for index, var_name in enumerate(pp.feature_names_future):
        if ('tide' in var_name):
            future_tide_indices.append(index + len_past_features)
        else:
            not_future_tide_indices.append(index + len_past_features)
            
    obs_indices = []
    not_obs_indices = []
    for index, var_name in enumerate(pp.feature_names_past):
        if ('tide' in var_name):
            obs_indices.append(index)
        else:
            not_obs_indices.append(index)
    
    # TODO: Check which index we want
    mls_stormsurge_obs_indices = (
        msl_indices 
        + stormsurge_indices_future 
        + past_forecast_indices_24 
        + future_tide_indices 
        + list(np.arange(len_past_features)
               ) 
        )
    
    return mls_stormsurge_obs_indices 
        
def initial_set_of_stations():
    
    initial_selection = hlp.get_norwegian_station_names()
    return initial_selection
    

def ml_stormsurge_model_7(t, x_train_norm, y_train_norm, x_test_norm, y_test_norm, y_train_mean, y_train_std, y_test):
    """After testing several ML models, we have selected this one based on its 
    performance. It can be used to select the feature stations, but to run this
    script faster, we have used a linear model."""
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

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]

    multi_dense_model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.MeanSquaredError(),
        metrics=[tf.metrics.MeanAbsoluteError()],
    )

    history = multi_dense_model.fit(
        x_train_norm,
        y_train_norm[:, t],
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.3,
        verbose=1,
    )

    metric = 'mean_absolute_error'
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric + "\n Dense 32 + dense 16 + batch normalization + dropout 0.3")
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()

    test_loss, test_acc = multi_dense_model.evaluate(x_test_norm, y_test_norm[:, t])

    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

    y_test_norm_pred = multi_dense_model.predict(x_test_norm)

    mae_scaled = np.mean(np.abs(y_test_norm[:, t] - y_test_norm_pred.transpose()))
    print('MAE of normalized variables: ', mae_scaled)

    # Unnormalize values
    y_test_pred = (y_test_norm_pred * y_train_std[t]) + y_train_mean[t]

    # Compute the statistics on unnormalized values
    rmse_scaled = np.sqrt(np.mean((y_test[:, t] - y_test_pred.transpose())**2))   
    bias_scaled = np.mean( -(y_test[:, t] - y_test_pred.transpose()) ) 
    std_scaled = np.std( -(y_test[:, t] - y_test_pred.transpose()) ) 
 
    print(rmse_scaled)
    
    return y_test_pred, rmse_scaled, bias_scaled, std_scaled, history, multi_dense_model 

# ******************************************************************************
# Run script with the parameters of interest
#*******************************************************************************
    
# ------ Example ------

# ------ Declare variables ------
fig_dir = (
        '/lustre/storeB/project/IT/geout/'
        +'machine-ocean/workspace/paulinast/storm_surge_results/'
        + 'linear_model/station_selection'
        )

label_stations = ['NO_OSC']

initial_selection = initial_set_of_stations()
# Put label station first in the list so that we start iterating with it
initial_selection.remove('NO_OSC')
initial_selection = ['NO_OSC'] + initial_selection
initial_selection.remove('NO_MSU') # Not in all the files
initial_selection.remove('NO_NYA') # Too far

feature_vars = ['obs', 'tide_py', 'stormsurge_corrected', 'msl', 'u10', 'v10'] 
label_var = 'stormsurge_corrected - (obs - tide_py)'

window_width_past = [24, 24, 0, 0, 0, 0]
window_width_future = [-1, 60, 60, 60, 60, 60]  # -1 does not include time t
horizon = 60  
    
datetime_start_hourly=dt.datetime(2018, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
datetime_end_hourly=dt.datetime(2021, 3, 31, 12, 0, 0, tzinfo=dt.timezone.utc)
datetime_split=dt.datetime(2020, 3, 31, 12, 0, 0, tzinfo=dt.timezone.utc)

# ------ Linear model ------
OUT_STEPS = horizon
num_features = 1
batch_size =32
epochs=500

metric = 'mean_absolute_error'

count = 0
final_station_selection = label_stations  # To be updated in every iteration
for station in initial_selection:
    keras.backend.clear_session()
    
    print( '------ ', station, '------')  
    
    feature_stations = final_station_selection + [station]
        
    # Open preprocessed file with all variables and stations
    pp = PreprocessInput(
        feature_vars = feature_vars,
        label_var = label_var,
        window_width_past = window_width_past,
        window_width_future = window_width_future,
        horizon = horizon,
        feature_stations = feature_stations,
        label_stations = label_stations,
        datetime_start_hourly = datetime_start_hourly,
        datetime_end_hourly = datetime_end_hourly,
        datetime_split = datetime_split,
        use_station_data=True,
        run_on_ppi=True,
        remove_nans = False,
        seasonal_model=False,  # Need to change this, so that it only does it if the variables is provided 
        new_station_loc=True,
        era5_forcing=False,
        add_aux_variables=False,
        missing_indicator=False,
        imputation_strategy='most_frequent',
        normalize=True,
        forecast_mode=True,  # TODO: fix case forecast_mode = False
        hourly=False,
        #use_existing_files=False,
        use_existing_files=True,
        past_forecast=24
        )
    
    pp.preprocess_data()
    
    len_past_features  = len(pp.feature_names_past)

    mls_stormsurge_obs_indices = generate_indices(pp, len_past_features)
    
    x_train = np.take(pp.x_y_dict['x_train'], mls_stormsurge_obs_indices, 1)
    y_train = pp.x_y_dict['y_train']
    x_test = np.take(pp.x_y_dict['x_test'], mls_stormsurge_obs_indices, 1)
    y_test = pp.x_y_dict['y_test']
    x_train_mean = np.take(pp.x_y_dict['x_train_mean'], mls_stormsurge_obs_indices)
    x_train_std = np.take(pp.x_y_dict['x_train_std'], mls_stormsurge_obs_indices)
    y_train_mean = pp.x_y_dict['y_train_mean']
    y_train_std = pp.x_y_dict['y_train_std']
    x_train_norm = np.take(pp.x_y_dict['x_train_norm'], mls_stormsurge_obs_indices, 1)
    y_train_norm = pp.x_y_dict['y_train_norm']
    x_test_norm = np.take(pp.x_y_dict['x_test_norm'], mls_stormsurge_obs_indices, 1)
    y_test_norm = pp.x_y_dict['y_test_norm']

    # Define the linear model - we could in setad use ml_stormsurge_model_7
    callbacks = [
    keras.callbacks.ModelCheckpoint(
        'best_model_nstations_' + str(count) + '.h5', save_best_only=True, monitor="val_loss"
        ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    
    linear = tf.keras.Sequential([
    tf.keras.layers.Dense(OUT_STEPS*num_features)
    ])

    linear.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.MeanSquaredError(),
        metrics=[tf.metrics.MeanAbsoluteError()],
    )

    # Run the model
    history_linear= linear.fit(
        x_train_norm,
        y_train_norm,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.3,
        verbose=0,
    )

    test_loss, test_acc = linear.evaluate(x_test_norm, y_test_norm)
    print("Test accuracy", test_acc)
    print("Test loss", test_loss)
    
    # Update selection
    if count == 0:
        validation_mae = [test_acc]
    else:
        if test_acc < validation_mae[-1]:
            validation_mae.append(test_acc)
            final_station_selection.append(station)
        print(count, ' final station selection: , ', final_station_selection)

    reset_keras(linear)
    
    try:
        del linear
        del history_linear
    except:
        pass
    count = count + 1
    
print('Final selection: ', final_station_selection)
    
