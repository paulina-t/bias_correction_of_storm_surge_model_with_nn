import pandas as pd

def save_all_labels(y_test, y_train, y_test_norm, y_test_norm_pred, y_train_norm, y_train_norm_pred,
                y_train_mean, y_train_std, horizon: int, model_name: str):   
    """
    Save all the labels in a CSV file. If the label arrays have different 
    length, they will be filled with NaNs to match the length of the longest
    array.

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
    model_name : str
        Name of the model used as file name.

    Returns
    -------
    None.

    """
    
    # Some variables could be computed here instead of being passed as args.
    # For instance, y_train_mean, y_train_std and all the derived variables
    data_dir = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast'
        +'/storm_surge_results/ml/labels_log'
        )

    y_test_pred= (y_test_norm_pred * y_train_std) + y_train_mean
    y_train_pred = (y_train_norm_pred * y_train_std) + y_train_mean

    header = (
        ['y_train_' + str(i+1) for i in range(horizon)]  
        + ['y_test_' + str(i+1) for i in range(horizon)] 
        + ['y_train_norm_' + str(i+1) for i in range(horizon)] 
        + ['y_test_norm_' + str(i+1) for i in range(horizon)] 
        + ['y_train_norm_pred_'  + str(i+1) for i in range(horizon)] 
        + ['y_test_norm_pred_' + str(i+1) for i in range(horizon)]  
        + ['y_train_pred_' + str(i+1) for i in range(horizon)]     
        + ['y_test_pred_' + str(i+1) for i in range(horizon)]    
        + ['y_train_mean']  # OBS! Vector horizon x 1
        + ['y_train_std']  # OBS! Vector horizon x 1
        )
        
    
    labels =pd.concat(
        [           
            pd.DataFrame(y_train), 
            pd.DataFrame(y_test), 
            pd.DataFrame(y_train_norm), 
            pd.DataFrame(y_test_norm), 
            pd.DataFrame(y_train_norm_pred),
            pd.DataFrame(y_test_norm_pred),
            pd.DataFrame(y_train_pred),
            pd.DataFrame(y_test_pred),
            pd.DataFrame(y_train_mean.transpose()), 
            pd.DataFrame(y_train_std.transpose())
            ], 
        axis=1)
    
    labels= labels.set_axis(header, axis=1)

    labels.to_csv(data_dir + '/labels_' + model_name + '.csv', index=False) 
    


def save_unnormalized_labels(y_test, y_train, y_test_norm, y_test_norm_pred, y_train_norm, y_train_norm_pred,
                y_train_mean, y_train_std, horizon: int, model_name: str):   
    """
    Save unnormalized labels in a CSV file. If the label arrays have different 
    length, they will be filled with NaNs to match the length of the longest
    array.

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
    model_name : str
        Name of the model used as file name.

    Returns
    -------
    None.

    """
    
    data_dir = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast'
        +'/storm_surge_results/ml/labels_log'
        )
    
    # Some variables could be computed here instead of being passed as args.
    # For instance, y_train_mean, y_train_std and all the derived variables
    
    y_test_pred= (y_test_norm_pred * y_train_std) + y_train_mean
    y_train_pred = (y_train_norm_pred * y_train_std) + y_train_mean

    header = (
        ['y_train_' + str(i+1) for i in range(horizon)]  
        + ['y_test_' + str(i+1) for i in range(horizon)] 
        + ['y_train_pred_' + str(i+1) for i in range(horizon)]     
        + ['y_test_pred_' + str(i+1) for i in range(horizon)]    
        )
        
    
    labels =pd.concat(
        [           
            pd.DataFrame(y_train), 
            pd.DataFrame(y_test), 
            pd.DataFrame(y_train_pred),
            pd.DataFrame(y_test_pred),
            ], 
        axis=1)
    
    labels= labels.set_axis(header, axis=1)

    labels.to_csv(data_dir + '/labels_unnormalized_' + model_name + '.csv', index=False) 