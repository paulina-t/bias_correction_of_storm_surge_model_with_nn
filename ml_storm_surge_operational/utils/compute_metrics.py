import numpy as np
import pandas as pd
import os.path


def mae_fun(y_true, y_pred):
    """
    

    Parameters
    ----------
    y_true : array
        Targets.
    y_pred : array
        Predictions.

    Returns
    -------
    mae : scalar
        Mean absolute error.

    """
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def mae_all_fun(y_true, y_pred):
    """
    

    Parameters
    ----------
    y_true : array
        Targets.
    y_pred : array
        Predictions.

    Returns
    -------
    mae : array
        Mean absolute error for each time step (e.g. each column).

    """
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    return mae

def rmse_fun(y_true, y_pred):
    """
    

    Parameters
    ----------
    y_true : array
        Targets.
    y_pred : array
        Predictions.

    Returns
    -------
    rmse : scalar
        Root mean square error.

    """
    rmse = np.sqrt(np.mean((y_true- y_pred)**2))
    return rmse

def rmse_all_fun(y_true, y_pred):
    """
    

    Parameters
    ----------
    y_true : array
        Targets.
    y_pred : array
        Predictions.

    Returns
    -------
    rmse : array
        Root mean square error for each time step (e.g. each column).

    """
    rmse = np.sqrt(np.mean((y_true- y_pred)**2, axis=0))
    return rmse

def print_metrics(y_test, y_train, y_test_norm, y_test_norm_pred, y_train_norm, y_train_norm_pred, y_train_mean, y_train_std):
    """
    Prints MAE and RMSE computed for all the time stamps and for each of them
    separately.

    Parameters
    ----------
    y_test : array
        
    y_train : array
        
    y_test_norm : array
        Normalized test targets
    y_test_norm_pred : array
        Normalized predicted test labels 
    y_train_norm : array
        Normalized train targets
    y_train_norm_pred : array
        Normalized predicted train labels 
    y_train_mean : array
        Average of train targets.
    y_train_std : array
        Standard deviation of train targets.

    Returns
    -------
    None.

    """
    # Unnormalize data 
    y_test_pred = (y_test_norm_pred * y_train_std) + y_train_mean
    y_train_pred = (y_train_norm_pred * y_train_std) + y_train_mean
    
    mae_scaled = mae_fun(y_test_norm, y_test_norm_pred)
    print('MAE of normalized variables (test): ', mae_scaled)
    
    mae_scaled = mae_fun(y_train_norm, y_train_norm_pred)
    print('MAE of normalized variables (train): ', mae_scaled)
    
    mae = mae_fun(y_test, y_test_pred)
    print('MAE of unnormalized variables (test): ', mae)
    
    mae = mae_fun(y_train, y_train_pred)
    print('MAE of unnormalized variables (train): ', mae)
    
    
    
    rmse_scaled = rmse_fun(y_test_norm, y_test_norm_pred)
    print('RMSE of normalized variables (test): ', rmse_scaled)
    
    rmse_scaled = rmse_fun(y_train_norm, y_train_norm_pred)
    print('RMSE of normalized variables (train): ', rmse_scaled)
    
    rmse_scaled = rmse_fun(y_test, y_test_pred)
    print('RMSE of unnormalized variables (test): ', rmse_scaled)
    
    rmse_scaled = rmse_fun(y_train, y_train_pred)
    print('RMSE of unnormalized variables (train): ', rmse_scaled)
    
    
    
   
    print('------ Compute metrics for every time step ------')
    
    mae_scaled = mae_all_fun(y_test_norm, y_test_norm_pred)
    print('MAE of normalized variables (test): ', mae_scaled)
    
    mae_scaled = mae_all_fun(y_train_norm, y_train_norm_pred)
    print('MAE of normalized variables (train): ', mae_scaled)
    
    mae = mae_all_fun(y_test,y_test_pred)
    print('MAE of unnormalized variables (test): ', mae)
    
    mae = mae_all_fun(y_train, y_train_pred)
    print('MAE of unnormalized variables (train): ', mae)
      
    
    
    rmse_scaled = rmse_all_fun(y_test_norm, y_test_norm_pred)
    print('RMSE of normalized variables (test): ', mae)
    
    rmse_scaled = rmse_all_fun(y_train_norm, y_train_norm_pred)
    print('RMSE of normalized variables (train): ', mae)
    
    rmse_scaled = rmse_all_fun(y_test, y_test_pred)
    print('RMSE of unnormalized variables (test): ', mae)
    
    rmse_scaled = rmse_all_fun(y_train, y_train_pred)
    print('RMSE of unnormalized variables (train): ', mae)
    
    
def save_rmse(y_test, y_test_norm_pred, y_train_mean, y_train_std, model_name: str):      
    data_dir = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast'
        + '/storm_surge_results/ml/rmse_log'
        )
    
    y_test_pred = (y_test_norm_pred * y_train_std) + y_train_mean
    rmse_1_model = rmse_all_fun(y_test, y_test_pred)
    
    header = ['rmse_' + model_name]  
    rmse_1_model = pd.DataFrame(rmse_1_model)
    rmse_1_model = rmse_1_model.set_axis(header, axis=1)
    
    if os.path.isfile(data_dir + '/rmse.csv'):
        print ('CSV already exists')
        rmse_all_models = pd.read_csv(data_dir + '/rmse.csv') 

        rmse_all_models =pd.concat(
            [rmse_all_models, pd.DataFrame(rmse_1_model)], 
            axis=1
            )
        rmse_all_models.to_csv(data_dir + '/rmse.csv', index=False) 
    else:
        print ('Genetating new file')
        rmse_1_model.to_csv(data_dir + '/rmse.csv', index=False) 