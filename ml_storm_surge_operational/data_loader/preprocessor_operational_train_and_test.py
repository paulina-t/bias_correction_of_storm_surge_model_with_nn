"""Prepare feature and label arrays.

Use the DataFrame generated with prepare_df_operational.py to create the 
features and the labels for the ML models.
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import os
import time
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from ml_storm_surge_operational.data_loader.prepare_df_operational import (
    PrepareDataFrames 
)
from ml_storm_surge_operational.utils.helpers import verboseprint1, verboseprint2
import ml_storm_surge_operational.utils.helpers as hlp

os.environ["TZ"] = "UTC"
time.tzset()

class PreprocessInput():
    """
    Generates feature and label arrays from station data and ROMS data.
    Selects common period without missing observations.

    Attributes:
    ----------
    max_window_width_past : int
        Number of data points before time t to save for feature arrays. The 
        default is 120.
        
    max_window_width_future : int
        Number of data points after time t to save for feature arrays. The 
        default is 120.
        
    horizon : int 
        Lead time only used for the labels.
    feature_stations : list of strings
        Stations to be used as predictors. Default is 
        ['NO_OSC', 'SW_2111', 'NL_Westkapelle']
        
    datetime_start_hourly : datetime
        Date indicating when the period of interest begins. The final period 
        depends selected depends on data availability. All dates without 
        observations will be removed. Default is 2002.01.01 00:00.
        
    datetime_start_hourly : datetime
        Date indicating when the period of interest begins. The final period 
        depends selected depends on data availability. All dates without 
        observations will be removed. Default is 2019.12.01 00:00.
        
    datetime_split : datetime
        Date indicating when to split the datasets in train and test samples.
        The default is 2017.04.26.12 and corresponds to the first 
        kystasjoner_norge.nc* file available.
        
    remove_nan: int, optional
        If provided, replaces missing data with the provided value. The default
        is None.
        
    correct_roms : bool, optional
        If true, removes the bias from the roms variable. The default is False.
    """
    def __init__(
            self,
            feature_vars,
            label_var, # Only one label variable
            window_width_past,  # list with the same length as feature_vars
            window_width_future, # list with the same length as feature_vars
            horizon,  # list with the same length as label_vars
            feature_stations=['NO_OSC', 'BGO'], 
            label_station='NO_OSC',
            datetime_start_hourly=dt.datetime(
                2002, 1, 1, 0, 0, 0, 
                tzinfo=dt.timezone.utc
                ),
            datetime_end_hourly=dt.datetime(
                2019, 12, 1, 0, 0, 0, 
                tzinfo=dt.timezone.utc
                ),
            datetime_split=dt.datetime(
                2017, 4, 26, 12, 0, 0, 
                tzinfo=dt.timezone.utc
                ),  # First kystasjoner-file
            remove_nans = True,
            imputation_strategy='',
            missing_indicator=False,
            normalize = True,
            verbose=1, 
            use_existing_files=True,
            past_forecast=0
            ):
        
        self.feature_vars = feature_vars
        self.label_var = label_var
        self.window_width_past = window_width_past
        self.window_width_future = window_width_future
        self.horizon = horizon
        self.feature_stations = feature_stations
        self.label_station = label_station
        self.datetime_start_hourly = datetime_start_hourly
        self.datetime_end_hourly = datetime_end_hourly
        self.datetime_split = datetime_split
        self.remove_nans=remove_nans
        self.imputation_strategy = imputation_strategy
        self.missing_indicator = missing_indicator
        self.normalize = normalize
        self.verbose = verbose
        self.use_existing_files = use_existing_files
        self.past_forecast = past_forecast

    def preprocess_data(self):
        """Main function that preprocesses the features and labels according to 
        the arguments passed.
        """
        feature_and_label_vars = list(set(self.feature_vars + [self.label_var])) 
        feature_and_label_stations = list(
            set(
            self.feature_stations 
            + [self.label_station]
            )
            )  
        
        # TODO: Remove? - already in prepare_df
        if 'stormsurge_corrected - (obs - tide_py)' in feature_and_label_vars:
            feature_and_label_vars.append('stormsurge_corrected')
            feature_and_label_vars.append('(obs - tide_py)')
        
        verboseprint1(self.verbose, 'Call PrepareDataFrames...')
        prep = PrepareDataFrames(
            variables = feature_and_label_vars,
            stations = feature_and_label_stations,
            datetime_start_hourly = self.datetime_start_hourly,
            datetime_end_hourly = self.datetime_end_hourly,
            horizon = self.horizon + 1, #1,  # horizon in prepare_df is an int +1!!! Lags are computed in self.generate_feature_array
            window_width_past=np.max(self.window_width_past),
            use_existing_files=self.use_existing_files,
            past_forecast=self.past_forecast,
            )        
        
        verboseprint1(self.verbose, 'Preparing feature and label df...')    
        self.df_features_and_labels = prep.prepare_features_labels_df()
        print('self.df_features_and_labels: ', self.df_features_and_labels)
        
        self.time = self.df_features_and_labels.index
        print(self.df_features_and_labels[self.time.duplicated()])
        
        verboseprint2(self.verbose, self.df_features_and_labels)
        verboseprint2(self.verbose, self.df_features_and_labels.shape)
                       
        self.features, self.labels = self.concatenate_arrays() 
             
        self.features_without_nans, self.labels_without_nans = self.remove_rows_with_nans(
            self.features,
            self.labels
            )
        
        self.x_train, self.y_train, self.x_test, self.y_test = self.split_train_test(
        features=self.features_without_nans, 
        labels=self.labels_without_nans
        )
        
        self.imputation()
        self.normalize_arrays()

    def generate_feature_array_future(self):
        """Generate feature array for lead times greater than 0.

        Returns:
            array: Array of features for lead times greater than 0.
        """
        verboseprint1(self.verbose,'Generating feature array for future times...')
        count_station = 0
        variable_names = []
        for station in self.feature_stations:
            count_var = 0
            for var in self.feature_vars:
                print('station: ', station, ' feature_var: ', var)            
                H  = self.window_width_future[count_var] # How many hours in the future? In the case of forecasts generated at t0, it is the lead time
                print('window_width_future: ', H)
                hz_str = ''        
                for h in range(0, H):    # Kyststatsjoner data starts counting at 0, but it is actually the first lead time.
                    print(h)
                    if h > 0:
                        hz_str = '_t' + str(h)    
                    vname = (
                        var 
                        + hz_str 
                        + '_' + self.feature_stations[count_station]
                      )  # TODO: Change so that only one label station is possible
                    variable_names.append(vname)
                
                # Add forecast variable names for runs starting before t0
                if var in ['stormsurge', 'stormsurge_corrected']:
                    if self.past_forecast > 0: # If adding data features from forecasts generated before t0
                        past_forecast_vnames = (
                            self.generate_var_names_past_forecast(
                            var, 
                            H, 
                            station)
                        )
                        variable_names = variable_names + past_forecast_vnames
                        
                count_var = count_var + 1
            count_station = count_station + 1
        
        # Check which columns are missing in df_features_and_labels
        variable_names_not_in_df = (
            np.setdiff1d(
            variable_names, 
            self.df_features_and_labels.columns
            ).tolist()
        )
        print('variable_names_not_in_df: ', variable_names_not_in_df)
        
        # Select variables that are in the DataFrame
        vars_to_select=  [
            x for x in variable_names if x in self.df_features_and_labels.columns
            ]
        features_future = self.df_features_and_labels[vars_to_select].to_numpy()
        self.feature_names_future = vars_to_select # variable_names        
        return features_future

    def generate_feature_array_past(self):

        """Generates the features array for the past by lagging the first prediction.

        Returns:
            array: Array of features for the past
        """
        verboseprint1(self.verbose,'Generating feature array for past times...')

        variable_names = []
        for station in self.feature_stations:
            count_var = 0
            for var in self.feature_vars:
                H  = self.window_width_past[count_var]
                # Subset main df and create df with past data
                hz_str = ''
                # Iterate over past time steps
                # Kyststatsjoner data starts counting at 0, but it is actually 
                # the first lead time.
                for h in range(-H, 0):    
                    if h != 0:
                        hz_str = '_t' +str(h)
                    vname = var + hz_str + '_' + station #self.feature_stations[count_station]
                    variable_names.append(vname)  # Store all the variable names for later!
                count_var = count_var + 1

        vars_to_select=  [
            x for x in variable_names if x in self.df_features_and_labels.columns
            ]
        features_past = self.df_features_and_labels[vars_to_select].to_numpy()
        self.feature_names_past = vars_to_select # variable_names 
        return features_past
        
    def generate_var_names_past_forecast(self, var, H, station):
        """Create a list of variables names for past Nordic4-SS forecasts for a 
        particular station.

        List of variable names used to create a feature array consisting of
        forecasts generated before the analysis time.

        Args:
            var (str): Variable name
            H (int): Window width for the future
            station (str): Station ID

        Returns:
            list of str: Variable names of past forecasts.
        """
        verboseprint1(
            self.verbose,
            'Generating list of variable names of past forecasts...'
            )
        variable_names = []
        for fhr in range(-self.past_forecast, 0, 12):
            past_forecast_str = '_' + str(fhr)
            hz_str = ''
            # Kyststatsjoner data starts counting at 0, but it is actually the 
            # first lead time.
            for h in range(fhr, H):    
                    print(h)
                    if h != 0:
                        hz_str = '_t' + str(h)    
                    vname = var + hz_str +  past_forecast_str +'_' + station  # TODO: Change so that only one label station is possible
                    variable_names.append(vname)
            
        return variable_names

    def generate_label_array(self):
        """
        Generates the label array from the full dataset in DataFrame format.
        
        Algorithm:
        1) Create empty arrays to store features and labels
        2) Iterate over feature_stations
          a) Fill in columns in labels array
          b) Iterate window_width*2 times and fill in columns in features array
        3) Remove rows with NaNs

        Returns
        -------
        
        labels : array 
            Labels are heights differences defined as 
            delta_zeta = (ROMS_output - Observations - tides)
        
        features : array
            Each column in the array is a feature. This columns represent 
            ROMS output in the time period [t - window_width, t + window_width]
            and observations in the time period [t - window_width, t-1] at 
            each station.
        """
        # Define parameters
        verboseprint1(self.verbose,'Generating label array...')
        hz_str = ''
        
        # Get variable names
        variable_names = []
        # Kyststatsjoner data starts counting at 0, but it is actually the first 
        # lead time.
        for h in range(0, self.horizon):    
            if h > 0:
                hz_str = '_t' + str(h)
        
            vname = self.label_var + hz_str + '_' + self.label_station  # TODO: Change so that only one label station is possible
            variable_names.append(vname)
        vars_to_select =  [
            x for x in variable_names if x in self.df_features_and_labels.columns
            ]

        # Subset DataFrame and convert to array
        labels = self.df_features_and_labels[vars_to_select].to_numpy()
        self.label_names = variable_names
        return labels
    
    
    def remove_rows_with_nans(self, features, labels): 
        """
        If remove_nans is True, this method removes rows from the feature and 
        label arrays if one of these arrays contain at least one NaN in that 
        specific row. If not, removes only from the label array.
        
        Parameters
        ----------
        
        features : array
        
        labels : array
        
        Returns
        -------
        
        features_without_nans : array
        
        labels_without_nans : array
        """
        verboseprint1(self.verbose, 'Removing nans...')
            
        # True where rows do not contain any NaNs, i.e., no feature/label is NaN
        # for that record
        bool_features = ~np.isnan(features).any(axis=1)
        bool_labels = ~np.isnan(labels).any(axis=1)
        
        verboseprint1(
            self.verbose, 
            'Shape of feature array before removing NaNs: ', 
            features.shape
            )
        verboseprint1(
            self.verbose, 
            'Shape of labels array before removing NaNs: ', 
            labels.shape
            )
            
        verboseprint1(
            self.verbose, 
            'Number of valid observations in feature array: ', 
            np.sum(bool_features)
            )
        verboseprint1(
            self.verbose, 
            'Number of valid observations in label array: ', 
            np.sum(bool_labels)
            )   
        #print('self.time.shape: ', self.time.shape)
        self.time = self.time[bool_labels]
        #print('self.time.shape after removing NaNs: ', self.time.shape)
        
        if self.remove_nans:
            # Remove missing labels and features
            self.bool_features_and_labels = np.logical_and(
                bool_features, 
                bool_labels
                )
            self.time_without_nans = self.time[self.bool_features_and_labels]
        else:
            # Always remove missing labels
            # ML models do not run when we have missing labels
            # TODO: Remove after filling in with nans, otherwise, we loose too 
            # many data
            self.bool_features_and_labels = bool_labels
        verboseprint1(
            self.verbose, 
            'Number of valid observations in feature and label arrays: ', 
            np.sum(self.bool_features_and_labels)
            )
        
        # Remove rows
        features_without_nans = features[self.bool_features_and_labels, :]
        labels_without_nans = labels[self.bool_features_and_labels]
        verboseprint1(
            self.verbose, 
            'Shape of feature array after removing NaNs: ', 
            features_without_nans.shape
            )
        verboseprint1(
            self.verbose, 
            'Shape of label array after removing NaNs: ', 
            labels_without_nans.shape
            )
        
        return features_without_nans, labels_without_nans
    
    def get_split_index(self):
        """Get index for splitting data into training and test datasets

        Returns:
            int: Train test split index
        """
        if self.remove_nans:
            print('Compute index from time_without_nans.')
            index = self.time_without_nans.get_loc(
                self.datetime_split, 
                method='backfill'
                )
        else:
            print('Compute index from time.')
            index = self.time.get_loc(
                self.datetime_split, 
                method='backfill'
                )
        return index
    
    def split_train_test(self, features, labels):
        """Split the data into training and test datasets.

        Args:
            features array: Preprocessed features
            labels array: Preprocessed labels

        Returns:
            _type_: _description_
        """
        index = self.get_split_index()
        x_train  = features[:index, :]
        y_train = labels[:index, :]
        x_test = features[index:, :]
        y_test = labels[index:, :]
        return x_train, y_train, x_test, y_test
    
    
    def concatenate_arrays(self):
        """Concatenate data of all variables in new arrays.
        
        Constructs a feature and a label array by concatenating the arrays 
        generated for each variable in self.feature_vars and self.label_vars.

        Returns
        -------
        features : array
            Concatenated feature arrays.
        labels : array
            Concatenated label arrays.
        """   
        verboseprint1(self.verbose, 'Concatenating arrays...')
        try:
            features_past = self.generate_feature_array_past()
            print('features_past.shape: ', features_past.shape)
        except:
            print('Could not generate feature_array_past.')
            del features_past

        features_future = self.generate_feature_array_future()
        
        if ('features_past' in locals()) and ('features_future' in locals()) :
            features = np.concatenate(
                (features_past, features_future), 
                axis=1
                )  # TODO: Check values!!
        elif 'features_past' in locals():
            print('Only features_past')
            features = features_past
        elif 'features_future' in locals():
            print('Only features_future')
            features = features_future
            
        labels = self.generate_label_array()
            
        return features, labels
    
    def generate_feature_names(self):
        """
        Generates a list with the names of the the features. For each station 
        and each label at time t, the features are the observations in the 
        interval [t - window_width, t - 1] and the predictions from the ROMS 
        model in the interval [t - window_width, t - window_width].

        This function is only used to store the metadata, it is not needed for
        constructing the feature and label arrays
        
        Returns
        -------
        feature_names : list of strings
            List with the names of the features. 
        """
        feature_names = self.feature_names_past + self.feature_names_future
        label_names = self.label_names
  
        return feature_names, label_names
    
    def imputation(self):    
        """Impute missing data in feature arrays and add missing indicator.
        
        Impute the missing values if the class argument imputation_strategy 
        was provided. Possible methods are 'iterative' for multivariate 
        imputation, or 'mean', 'median', 'most_frequent', 'constant', for 
        univariate imputation.
        """
        verboseprint1(self.verbose, 'Performing imputation...')
        if self.imputation_strategy:
            if self.imputation_strategy == 'iterative':
                 if self.missing_indicator:
                     transformer = FeatureUnion(
                        transformer_list=[
                            (
                                'features', 
                                IterativeImputer(max_iter=10, random_state=0)
                                ),
                            (
                                'indicators', 
                                MissingIndicator())

                            ]
                        )
                 else:
                    transformer = IterativeImputer(max_iter=10, random_state=0)
            else:
                if self.missing_indicator:
                    transformer = FeatureUnion(
                        transformer_list=[
                            (
                                'features', 
                                SimpleImputer(strategy=self.imputation_strategy)
                                ),
                            (
                                'indicators', MissingIndicator()
                                )
                            ]
                        ) 
                else:
                    transformer = SimpleImputer(
                        strategy=self.imputation_strategy
                        )                 
            try:
                transformer = transformer.fit(self.x_train, self.y_train) 
                self.x_train = transformer.transform(self.x_train) 
                self.x_test = transformer.transform(self.x_test) 
            except:
                print('Could not perform imputation')

    def normalize_arrays(self):
        """Normalize train and test data.
        
        Normalize the data after removing empty columns in feature arrays. The
        average and standard deviation values used in the normalization process
        are computed only with the training data.

        Returns
        -------
        None.

        """
        if self.normalize:
            # Remove columns that have nans in all the rows (mean is nan)
            nan_idx = np.argwhere(np.isnan(np.nanmean(self.x_train, axis=0)))
            self.x_train = np.delete(self.x_train, nan_idx, axis=1)
            self.x_test = np.delete(self.x_test, nan_idx, axis=1)
            
            # Compute average and std of the training data
            x_train_mean = np.nanmean(self.x_train, axis=0)
            x_train_std = np.nanstd(self.x_train, axis=0)
    
            y_train_mean = np.nanmean(self.y_train, axis=0)
            y_train_std = np.nanstd(self.y_train,axis=0)
            
            # Normalize
            x_train_norm = (self.x_train - x_train_mean) / x_train_std
            x_test_norm = (self.x_test - x_train_mean) / x_train_std
            y_train_norm = (self.y_train - y_train_mean) / y_train_std
            y_test_norm = (self.y_test - y_train_mean) / y_train_std
            
            self.x_y_dict = {
                'x_train' : self.x_train, 
                'y_train' : self.y_train, 
                'x_test' : self.x_test, 
                'y_test' : self.y_test,
                'x_train_mean' : x_train_mean, 
                'x_train_std' : x_train_std,
                'y_train_mean' : y_train_mean,
                'y_train_std' : y_train_std,
                'x_train_norm' : x_train_norm,
                'y_train_norm' : y_train_norm,
                'x_test_norm' : x_test_norm,
                'y_test_norm' : y_test_norm,
                'nan_idx' : nan_idx
                }          
        
if __name__ == "__main__":
    # **************************************************************************
    # Modify this part to run the experiments of interest
    # **************************************************************************

    # Long example with all the data needed
    hist_path = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast/'
        + 'storm_surge_results/ml/history'
    )
    best_model_path = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast/'
        + 'storm_surge_results/ml/best_models'
    )

    feature_stations = ['NO_OSC', 'NO_AES', 'NO_BGO', 'NO_HEI', 'NO_KSU']
    feature_stations = ['NO_OSC', 'NO_AES', 'NO_HRO', 'NO_OSL', 'NO_TRG']
    feature_stations = ['NO_OSC', 'NO_AES', 'NO_BGO', 'NO_VIK', 'NO_TRG'] # best
    feature_stations = ['NO_OSC', 'NO_AES', 'NO_HRO', 'NO_VIK', 'NO_TRG']
    
    #feature_stations=['NO_ANX', 'NO_AES', 'NO_BOO', 'NO_HFT', 'NO_HRO', 'NO_KAB', 'NO_KSU']
    #feature_stations=['NO_BGO', 'NO_AES', 'NO_BOO', 'NO_HAR', 'NO_OSC', 'NO_MAY', 'NO_SVG']
    feature_stations=['NO_BGO', 'NO_AES', 'NO_BOO',  'NO_MAY', 'NO_TRG']

    #label_station='NO_OSC'
    #label_station='NO_ANX'
    label_station='NO_BGO'
    
    feature_vars = ['obs', 'tide_py', 'stormsurge_corrected', 'msl', 'u10', 'v10']
    label_var = 'stormsurge_corrected - (obs - tide_py)'
    
    window_width_past = [24, 24, 0, 0, 0, 0]
    window_width_future = [-1, 60, 60, 60, 60, 60] #, 0, 0]  # -1 does not include time t -> this is for forecast mode
    horizon = 60    
    
    datetime_start_hourly=dt.datetime(
        2018, 1, 1, 0, 0, 0, 
        tzinfo=dt.timezone.utc
        )
    datetime_end_hourly=dt.datetime(
        2021, 3, 31, 12, 0, 0, 
        tzinfo=dt.timezone.utc
        )
    datetime_split=dt.datetime(
        2020, 3, 31, 12, 0, 0, 
        tzinfo=dt.timezone.utc
        )

    # Short test
    feature_stations=['NO_BGO', 'NO_AES']
    feature_vars = ['obs', 'tide_py', 'stormsurge_corrected']#, 'v10']
    #label_var = 'stormsurge_corrected'
    window_width_past = [4, 4, 0]#, 0]
    window_width_future = [-1, 6, 6]#, 6]  # -1 does not include time t -> this is for forecast mode
    horizon = 24
    datetime_start_hourly=dt.datetime(
        2021, 1, 1, 0, 0, 0, 
        tzinfo=dt.timezone.utc
        )
    datetime_end_hourly=dt.datetime(
        2021, 1, 30, 0, 0, 0, 
        tzinfo=dt.timezone.utc
        )
    datetime_split=dt.datetime(
        2021, 1, 10, 12, 0, 0, 
        tzinfo=dt.timezone.utc
        )
    
    pp = PreprocessInput(
        feature_vars = feature_vars,
        label_var = label_var,
        window_width_past = window_width_past,
        window_width_future = window_width_future,
        horizon = horizon,
        feature_stations = feature_stations,
        label_station = label_station,
        datetime_start_hourly = datetime_start_hourly,
        datetime_end_hourly = datetime_end_hourly,
        datetime_split = datetime_split,
        remove_nans = False,
        missing_indicator=False,
        imputation_strategy='most_frequent',
        normalize=True,
        use_existing_files=True,
        past_forecast=24
        )
    
    pp.preprocess_data()
    
    index = pp.get_split_index()
    x_train_norm =  pp.x_y_dict['x_train_norm'] 
    y_train_norm = pp.x_y_dict['y_train_norm']
    x_test_norm = pp.x_y_dict['x_test_norm']
    y_test_norm = pp.x_y_dict['y_test_norm']
    y_train_mean = pp.x_y_dict['y_train_mean']
    y_train_std = pp.x_y_dict['y_train_std']
    y_test = pp.x_y_dict['y_test']
        
    data = {
    'df_features_and_labels' : pp.df_features_and_labels,
    'x_y_dict' : pp.x_y_dict,
    'metadata' : {
        'feature_stations' : feature_stations,
        'label_station' : label_station,
        'feature_vars' :feature_vars,
        'label_var' : label_var,
        'window_width_past' : window_width_past,
        'window_width_future' : window_width_future,  # -1 does not include time t
        'horizon' : horizon,
        'datetime_start_hourly' : datetime_start_hourly,
        'datetime_end_hourly' : datetime_end_hourly,
        'datetime_split' : datetime_split,
        'notebook' : '',
        'index' : pp.get_split_index(),
        'time' : pp.time,
        'bool_features_and_labels' : pp.bool_features_and_labels,
        'generated_feature_and_label_names' : pp.generate_feature_names(),
        'feature_names_past' : pp.feature_names_past,
        'feature_names_future' : pp.feature_names_future,
        'label_names' : pp.label_names
        }
    }
    
    """
    print('Saving dict with preprocessed data...')
    data_dir = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast/'
        + 'storm_surge_results/data_preprocess_input/'
    )
    fname = (
    'dict_other5_stations_v2_12_hours_past_forecasts_'
    + 'wwpast_24hrs_NO_BGO_v3.pickle'
    )

    path_to_dict = data_dir + fname
    with open(path_to_dict, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    """