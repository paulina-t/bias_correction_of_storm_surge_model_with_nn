"""
Call this fuction from `main_train_operational_1000_iter.py` to train the models
in operational mode multiple times.

Not needed in an operational context.
"""

import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K

import pickle

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

dir_rmse = (
    '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast/'
    + 'storm_surge_results/ml/rmse_log'
)

path_to_dict = (
    '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast/'
    + 'storm_surge_results/data_preprocess_input'
)

fname = 'dict_5_stations_v2_12_hours_past_forecasts_wwpast_24hrs.pickle'
with open(path_to_dict + '/' + fname, 'rb') as handle:
    pp = pickle.load(handle)
    
# ----------------------------------------------------------------------------    
# Indices
# TODO: This can be written in a function as it is used in several scripts
len_past_features = len(pp['metadata']['feature_names_past'])

stormsurge_indices_future = []
not_stormsurge_indices_future = []
i = 0
for name in pp['metadata']['feature_names_future']:
    if name.startswith('stormsurge'):
        stormsurge_indices_future.append(i + len_past_features) # Only in the future, counting from the first feature in the past
    else:
        not_stormsurge_indices_future.append(i + len_past_features)
    i = i + 1 

start  = len(pp['metadata']['feature_names_past'])
past_forecast_indices = []
not_past_forecast_indices = []
for index, var_name in enumerate(pp['metadata']['feature_names_future']):
    if ('_-' in var_name):
        past_forecast_indices.append(index + len_past_features)
    else:
        not_past_forecast_indices.append(index + len_past_features)
        
stormsurge_not_past_forecast_indices = list(
    set(stormsurge_indices_future)  
    & set(not_past_forecast_indices)
    )

start  = len(pp['metadata']['feature_names_past'])
past_forecast_indices_24 = []
not_past_forecast_indices_24 = []
for index, var_name in enumerate(pp['metadata']['feature_names_future']):
    if ('_-24' in var_name):
        past_forecast_indices_24.append(index + len_past_features)
    else:
        not_past_forecast_indices_24.append(index + len_past_features)

start  = len(pp['metadata']['feature_names_past'])
msl_indices = []
not_msl_indices = []
for index, var_name in enumerate(pp['metadata']['feature_names_future']):
    if ('msl' in var_name):
        msl_indices.append(index + len_past_features)
    else:
        not_msl_indices.append(index + len_past_features)

start  = len(pp['metadata']['feature_names_past'])
u10_indices = []
not_u10_indices = []
for index, var_name in enumerate(pp['metadata']['feature_names_future']):
    if ('u10' in var_name):
        u10_indices.append(index + len_past_features)
    else:
        not_u10_indices.append(index + len_past_features)

start  = len(pp['metadata']['feature_names_past'])
v10_indices = []
not_v10_indices = []
for index, var_name in enumerate(pp['metadata']['feature_names_future']):
    if ('v10' in var_name):
        v10_indices.append(index + len_past_features)
    else:
        not_v10_indices.append(index + len_past_features)

start  = len(pp['metadata']['feature_names_past'])
future_tide_indices = []
not_future_tide_indices = []
for index, var_name in enumerate(pp['metadata']['feature_names_future']):
    if ('tide' in var_name):
        future_tide_indices.append(index + len_past_features)
    else:
        not_future_tide_indices.append(index + len_past_features)

start  = len(pp['metadata']['feature_names_past'])
obs_indices = []
not_obs_indices = []
for index, var_name in enumerate(pp['metadata']['feature_names_past']):
    if ('tide' in var_name):
        obs_indices.append(index)
    else:
        not_obs_indices.append(index)

# ----------------------------------------------------------------------------
# Select variables for the model
# TODO: Writhe in a utils function
mls_stormsurge_obs_indices =  past_forecast_indices_24 + future_tide_indices + list(np.arange(len_past_features)) + u10_indices + v10_indices
x_train = np.take(pp['x_y_dict']['x_train'], mls_stormsurge_obs_indices, 1)
y_train = pp['x_y_dict']['y_train']
x_test = np.take(pp['x_y_dict']['x_test'], mls_stormsurge_obs_indices, 1)
y_test = pp['x_y_dict']['y_test']
x_train_mean = np.take(pp['x_y_dict']['x_train_mean'], mls_stormsurge_obs_indices)
x_train_std = np.take(pp['x_y_dict']['x_train_std'], mls_stormsurge_obs_indices)
y_train_mean = pp['x_y_dict']['y_train_mean']
y_train_std = pp['x_y_dict']['y_train_std']
x_train_norm = np.take(pp['x_y_dict']['x_train_norm'], mls_stormsurge_obs_indices, 1)
y_train_norm = pp['x_y_dict']['y_train_norm']
x_test_norm = np.take(pp['x_y_dict']['x_test_norm'], mls_stormsurge_obs_indices, 1)
y_test_norm = pp['x_y_dict']['y_test_norm']


# Define the model
# Move to utils
def ml_stormsurrge_model_7(t, x_train_norm, y_train_norm, x_test_norm, y_test_norm, y_train_mean, y_train_std, y_test):
    OUT_STEPS = 60
    num_features = 1
    batch_size =32
    epochs=500
    
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
        use_multiprocessing=True
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

    y_test_norm_pred.shape

    rmse_scaled = np.sqrt(np.mean((y_test[:, t] - y_test_pred.transpose())**2))
    rmse_scaled
    print(rmse_scaled)
    
    return rmse_scaled, history, multi_dense_model

def main_train_model(i):
    print('i: ', i)
    metric = 'mean_absolute_error'
    rmse_all_times = []
    history_all_times = []
    models_all_times=[]
    learning_curves = {metric:{}}
    learning_curves[metric]['member' + str(-1)]  = {}
    learning_curves[metric]['member' + str(-1)]['val']={}
    learning_curves[metric]['member' + str(-1)]['train']={}
    
    for t in range(60):
        print('Lead time: ', t +1)
        rmse_t, history_t, model_t = ml_stormsurrge_model_7(t, x_train_norm, y_train_norm, x_test_norm, y_test_norm, y_train_mean, y_train_std, y_test)
        learning_curves[metric]['member' + str(-1)]['val']['t' + str(t)] = history_t.history["val_" + metric]
        learning_curves[metric]['member' + str(-1)]['train']['t' + str(t)] = history_t.history[metric]
        rmse_all_times.append(rmse_t)  
        history_all_times.append(history_t)
        models_all_times.append(model_t)
        
        rmse_dict = {'rmse_all_lead_times_' + str(i) : rmse_all_times}

    fpath = (
        dir_rmse
        + '/rmse_operational_5_stations'
        +'/rmse_NO_OSC_5_stations_operational_wind_one_model_' 
        + str(i) 
        + '.pickle'
        ) 

    with open(fpath, 'wb') as handle:
        pickle.dump(rmse_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)