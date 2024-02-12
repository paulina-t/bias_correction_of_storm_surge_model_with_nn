import numpy as np
import pandas as pd
import datetime as dt
import xarray as xr
import copy
import os

"""
Compute the MAE for the predicted error at Oscarsborg station.
"""

# TODO: write metrics in utils
class ReadKyststasjoner():
    
    """
    Attributes
    ----------
    
    dt_start : datetime
        Date of the first file to read. Hours must be 00 or 12.
        
    dt_end : datetime
        Date of the last file to read. Hours must be 00 or 12.
        
    data_dir : str
        Directory where the data is stored.
    """
    
    def __init__(self, 
                 dt_start=dt.datetime(2017, 4, 26, 12, 0, 0, tzinfo=dt.timezone.utc), 
                 dt_end=dt.datetime(2020, 12, 31, 12, 0, 0, tzinfo=dt.timezone.utc),
                 ensemble_member=-1,
                 data_dir:str='/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive'):
        self.dt_start = dt_start  
        self.dt_end = dt_end  # Check if available
        self.ensemble_member = ensemble_member
        self.data_dir = data_dir

    # First new_kyststasjoner fie: new_kyststasjoner_norge.nc2018012712
    # Last new_syststasjoner file: 2018082012

    def mae(self, y_true, y_pred): 
        """
        Computes the Mean Absolute Error for the entire period.

        Parameters
        ----------
        y_true : array
        y_pred : array

        Returns
        -------
        scalar

        """
        return np.nanmean(np.abs(y_true - y_pred))
    
    def mae_all(self, y_true, y_pred): 
        """
        Computes the Mean Absolute Error for each time step (for each column).

        Parameters
        ----------
        y_true : array
        y_pred : array

        Returns
        -------
        array

        """
        return np.nanmean(np.abs(y_true - y_pred), axis=0)

    def ae(self, y_true, y_pred): 
        """
        Computes the Absolute Error for the entire period.

        Parameters
        ----------
        y_true : array
        y_pred : array

        Returns
        -------
        scalar

        """
        return np.abs(y_true - y_pred)
    
    def se(self, y_true, y_pred):
        """
        Computes the Squared Error for the entire period.

        Parameters
        ----------
        y_true : array
        y_pred : array

        Returns
        -------
        array

        """
        return (y_true - y_pred)**2    
    
    def rmse(self, y_true, y_pred):
        """
        Computes the Root Mean Squared Error for the entire period.

        Parameters
        ----------
        y_true : array
        y_pred : array

        Returns
        -------
        scalar

        """
        
        return np.sqrt(np.nanmean((y_true - y_pred)**2))
    
    def rmse_all(self, y_true, y_pred):
        """
        Computes the Root Mean Squared Error for each time step (for each column).

        Parameters
        ----------
        y_true : array
        y_pred : array

        Returns
        -------
        array

        """
        return np.sqrt(np.nanmean((y_true - y_pred)**2, axis=0))
      

    def make_path_list(self, valid_dates=[]):
        """
        Generates a list of paths to the kyststasjoner files taking into 
        account the version.
        
        Parameters
        ----------
        valid_dates : listof datetimes
            If provided, checks that the date of the kystasjoner files to is
            in valid_dates. The default is false.

        Returns
        -------
        path_list : list of str
            List with the paths to the kyststasjoner paths.

        """
        start_new_kyststasjoner = dt.datetime(
            2018, 1, 27, 12, 0, 0, 
            tzinfo=dt.timezone.utc
            )
        end_new_kyststasjoner = dt.datetime(
            2018, 8, 20, 12, 0, 0, 
            tzinfo=dt.timezone.utc
            )

        deltatime_days = (self.dt_end -  self.dt_start).days
        deltatime_hours = (self.dt_end -  self.dt_start).seconds/3600
        if ((deltatime_days == 0 ) 
            & (deltatime_hours >= 12) 
            & (self.dt_end.day !=  self.dt_start.day)):
            # Example:
            # self.dt_start = datetime.datetime(2020, 6, 30, 12, 0, tzinfo=datetime.timezone.utc)
            # self.dt_end = datetime.datetime(2020, 7, 1, 0, 0, tzinfo=datetime.timezone.utc)
            deltatime_days = deltatime_days + 1
            
        n_hours = deltatime_days*24   

        if self.dt_start.hour > self.dt_end.hour:
            n_hours = n_hours - 12
        elif self.dt_start.hour < self.dt_end.hour:
            n_hours = n_hours + 12
            
        if not valid_dates:
            valid_dates = [
                self.dt_start + dt.timedelta(hours=i) for i in range(n_hours + 1)
                ]
        
        path_list = []  
        # Iterate over all days in the preriod and generate a string every
        # 12 hours.
        for hr in range(0, n_hours + 1, 12):
            date_dt = (self.dt_start + dt.timedelta(hours=hr))
            date_str = date_dt.strftime('%Y%m%d%H')
            if date_dt in valid_dates: # valid_dates are the dates of the pp features without nans
                if (date_dt >= start_new_kyststasjoner 
                    and date_dt <= end_new_kyststasjoner):
                    file_root_name = '/new_kyststasjoner_norge.nc'
                else:
                    file_root_name = '/kyststasjoner_norge.nc'
                path_list.append(
                    self.data_dir 
                    + file_root_name 
                    + date_str
                    )
        return path_list
    
    
    # TODO: Update other methods!!!
    def make_data_arrays(self, var, station_name, path_list, hz):
        # Define parameters
        l = len(path_list)
        data = np.full([l*12, hz], fill_value=np.NaN)
        
        obs_vars =[
            'observed_at_chartdatum',
            'observed',
            'tide_at_chartdatum',
            'tide'
            ]
        
        # Construct arrays of shape (n_samples, n_outputs)
        for i in range(l):
            for j in range(12):  # For each file, fill in the next 12 hours
                try:
                    data_file_i = xr.open_dataset(path_list[i])
                    station_nr = self.get_station_id(data_file_i, station_name)
                    length = j + hz
                    if length > 121:  # We have only 120 preditions
                        length = 121
                    
                    if var == 'stormsurge_corrected' or var == 'stormsurge':
                        data[i * 12 + j, :length-j] = ( 
                            data_file_i[var]
                            .isel(
                                {
                                'station':station_nr, 
                                'time':slice(j, length), 
                                'ensemble_member':self.ensemble_member, 
                                'dummy':0
                                }
                                ).values
                            )
                    elif var == 'bias':
                        data[i * 12 + j, :hz]= ( 
                            np.tile(
                                (
                                    data_file_i['var']
                                    .isel(station=station_nr)
                                    .values
                                ), 
                                hz
                                )
                            )
                    elif var in obs_vars:
                        data[i * 12 + j, :length-j] = (
                            data_file_i[var]
                            .isel(
                                {'station':station_nr, 
                                 'time':slice(j, length), 
                                 'dummy':0}
                                 ).values
                            )
                except:
                    # Keep nans in the files
                    pass
        return data
    
    def make_hz_col_names(self, var, station, hz, member=''):
        col_names = []
        if var=='observed_at_chartdatum' or var == 'observed':
            var = 'obs'
        if var == 'tide' or var == 'tide_at_chartdatum':
            var ='tide_py'
            
        member_str = ''
        # Assuming there are 52 members.
        # TODO: check why there are 53 members after the 26th of Feb 2021
        if type(member)==int:
            if (member < 51) and (member >= 0):   
                member_str = '_m' + str(member)
            elif member < -1:  # When counting from the last position in a vector
                member = 52 + member
                member_str = '_m' + str(member)
        
            
        for h in range(hz): # TODO: change and start at time 1?
            if h == 0:
                col_name_h = var + member_str + '_' + station
                col_names.append(col_name_h)
            else:
                col_name_h = var + member_str + '_t' + str(h) + '_' + station
                col_names.append(col_name_h)
                
        return col_names

    def make_hz_col_names_past(self, var, station, window_width_past, member=''):
        col_names = []
        if var=='observed_at_chartdatum' or var == 'observed':
            var = 'obs'
        if var == 'tide' or var == 'tide_at_chartdatum':
            var ='tide_py'
            
        member_str = ''
        # Assuming there are 52 members.
        # TODO: check why there are 53 members after the 26th of Feb 2021
        if type(member)==int:
            if (member < 51) and (member >= 0):   
                member_str = '_m' + str(member)
            elif member < -1:  # When counting from the last position in a vector
                member = 52 + member
                member_str = '_m' + str(member)
            
        for h in range(-window_width_past, 0): # TODO: change and start at time 1?
            col_name_h = var + member_str + '_t' + str(h) + '_' + station
            col_names.append(col_name_h)
            
        return col_names
    
    def make_time_idx(self):
        delta = dt.timedelta(hours=1)
        date = self.dt_start
        date_list = []
        while date <= self.dt_end:
            date_list.append(date)
            date += delta  
        # Add the last 11 hours
        for i in range(11):
            date_list.append(date)
            date += delta 
        return date_list
                
    def array_to_df(self, array, index, col_names):
        df = pd.DataFrame(data=array, index=index, columns=col_names)
        return df
        

    def make_error_arrays(self, station_name, path_list, hz:int=48):
        """
        Makes arrays with the estimated errors and the true error in the 
        kyststasjoner files.
        
        The true error is estimated as stormsurge - (observed - tide).
        The estimated error is the bias variable.
        
        Parameters
        ----------
        station_name : str
            Station where the  error arrays are computed. 
        path_list : list of str
            List with the paths to the files to open.
        hz : int, optional
            Prediction horizon in hours. The default is 48 hrs.
                    
        Returns
        -------
        estimated error_arr : array
            Array with the estimated error (bias).
        true_error_arr : TYPE
            Array with the true error (stormsurge - (observed - bias)).
            
        """
        roms_var = 'stormsurge'

        # Define parameters
        l = len(path_list)
        estimated_error_arr = np.full([l*12, hz], fill_value=np.NaN)
        true_error_arr = np.full([l*12, hz], fill_value=np.NaN)
        
        # Construct arrays of shape (n_samples, n_outputs)
        for i in range(l):
            for j in range(12):  # For each file, fill in the next 12 hours
                try:
                    data_file_i = xr.open_dataset(path_list[i])
                    station_nr = self.get_station_id(data_file_i, station_name)
                     # The same value is used for all the predictions
                    bias = np.tile(
                        data_file_i['bias'].isel(station=station_nr).values, 
                        hz
                        )
                    estimated_error_arr[i * 12 + j, :len(bias)] = bias
                    
                    true_error =(
                        data_file_i[roms_var].isel(
                            {
                                'station':station_nr, 
                                'time':slice(j, j + hz), 
                                'ensemble_member':self.ensemble_member, 
                                'dummy':0
                                }
                            ).values
                        - data_file_i['observed_at_chartdatum'].isel(
                            {
                                'station':station_nr, 
                                'time':slice(j, j + hz), 
                                'dummy':0
                                }
                            ).values  # OBS!! The last day (24 hrs) of observations are missing
                        + data_file_i['tide_at_chartdatum'].isel(
                            {
                                'station':station_nr, 
                                'time':slice(j,  j + hz), 
                                'dummy':0
                                }
                            ).values
                        )
                    true_error_arr[i * 12 + j, :len(true_error)] = true_error
                except:
                    # Keep nans in the files
                    pass
               
            
        return estimated_error_arr, true_error_arr
    
    def make_obs_stormsurge_arrays(self, stormsurge_var, station_name, path_list, hz:int=48):
        """
        Makes arrays with the estimated errors and the true error in the 
        kyststasjoner files.
        
        The true error is estimated as stormsurge - (observed - tide).
        The estimated error is the bias variable.
        
        Parameters
        ----------
        stormsurge_var : str
            Variable to return in the stormsurge_array, either 'stormsurge' or 
            'stormsurge_corrected'.
        station_name : str
            Station where the  error arrays are computed. 
        path_list : list of str
            List with the paths to the files to open.
        hz : int, optional
            Prediction horizon in hours. The default is 48 hrs.
                    
        Returns
        -------
        obs tide_arr : array
            Array with the estimated observations ('observed' - 'tide').
        stormsurge_arr : TYPE
            Array with stormsurge_estimations, either 'stormsurge' or 
            'stormsurge_corrected'.
            
        """
        # TODO: this is not an updated function
        roms_var = stormsurge_var #'stormsurge'  
        
        # Define parameters
        l = len(path_list)
        estimated = np.full([l*12, hz], fill_value=np.NaN)
        true = np.full([l*12, hz], fill_value=np.NaN)
        
        # Construct arrays of shape (n_samples, n_outputs)
        for i in range(l):
            for j in range(12):  # For each file, fill in the next 12 hours
                try:
                    data_file_i = xr.open_dataset(path_list[i])
                    station_nr = self.get_station_id(data_file_i, station_name)
                    length = j + hz
                    if length > 121:  # We have only 120 preditions
                        length = 121
                        
                    estimated[i * 12 + j, : length-j] = ( 
                        data_file_i[roms_var]
                        .isel(
                            {
                            'station':station_nr, 
                            'time':slice(j, length), 
                            'ensemble_member':self.ensemble_member, 'dummy':0
                            } # 'ensemble_member':-52??
                            ).values
                        )
                    length
                    
                    true[i * 12 + j, : length-j] = (
                        
                        data_file_i['observed_at_chartdatum'].isel(
                            {
                                'station':station_nr, 
                                'time':slice(j, length), 
                                'dummy':0
                                }
                            ).values  # OBS!! The last day (24 hrs) of observations are missing
                        - data_file_i['tide_at_chartdatum'].isel(
                            {
                                'station':station_nr, 
                                'time':slice(j,  length), 
                                'dummy':0
                                }
                            ).values
                        )
                except:
                    # Keep nans in the files
                    pass
        return estimated, true
    
    def concatenate_files(self, path_list, save=True):
        """
        Prepare a single NetCDF file from the 12-hourly kyststasjoner files.
        
        Keep only variables of interest: stormsurge, stormsurge_corrected, tide.
        
        Since the files are concatenated along the time dimension and there is
        some overlap among files, i.e., the files contain predictions for the 
        next 5 days but they are generated every 12 hours, a new dimension is 
        created. This new dimension called horizon, is the original time 
        dimension, that indicated the prediction hours. A new time dimension is
        created based on the original forecast_reference_time, along which the 

        Parameters
        ----------
        path_list : list of str
            List of strings indicating the file names to open from dt_start to 
            dt_end.
        save : bool, optional
            If True, saves the new NetCDF file. The default is True.

        Returns
        -------
        concatenated_files : xr.dataset
            Preprocessed and concatenated kyststasjoner files.

        """        
        first_file = True
        for path in path_list: 
            try:
                # Check if the path exists and preprocess data in file. 
                single_file = xr.open_dataset(path)
                file_exists = True
            except:
                # If not possible, fill in concatenated DataSet with NaNs
                file_exists = False
                print('Cannot open file ', path)
                single_file = self.fill_dataset_with_nans(single_file) # Does not work if it is the first file
            
            if file_exists:
                # process the file and then concatenate
                single_file = self.add_horizon_coord(single_file)
                single_file = self.rename_station_index(single_file)  # modify station numbers
                single_file = self.subset_dataset(single_file)
                single_file = self.fill_next_11_hrs(single_file)
            try:
                history_single_file = single_file.attrs['history']
            except:
                single_file.attrs['history'] = ''
        
            if first_file:
                concatenated_files = single_file
                first_file=False
            else:
                try:        
                    history_concatenated = concatenated_files.attrs['history']
                except:
                    concatenated_files.attrs['history'] = ''
                try:
                    concatenated_files = xr.concat(
                        [concatenated_files, single_file], 
                        dim='time'
                        )
                    concatenated_files.attrs['history'] = (
                        history_concatenated 
                        + history_single_file
                        )
                except:
                    print('Cannot concatenate file ', path)
                    
        concatenated_files = self.add_stationid_var(concatenated_files)        
        concatenated_files = self.add_attributes(concatenated_files)                
        
        if save:
            data_dir = (
                '/lustre/storeB/project/IT/geout/machine-ocean/'
                + 'workspace/paulinast/data/kyststasjoner'
                )
            str_start_date = str(self.dt_start).split(' ')[0].replace('-', '')
            str_end_date = str(self.dt_end).split(' ')[0].replace('-', '')
            path = (
                data_dir 
                + '/kyststasjoner_' 
                + str_start_date 
                + '_' + str_end_date 
                + '.nc'
            )
            concatenated_files.to_netcdf(path)
                    
        return concatenated_files
    
    def add_attributes(self, dataset): 
        try:
            dataset.attrs['About'] = (
                dataset.attrs['About'] 
                + ' The files have been concatenated, a new dimension denoted by '
                + 'horizon has been created as well as a time dimension that '
                + 'references the original forecast_reference_time. A subset of '
                + 'variables has been selected.'
                ) 
        except:
            dataset.attrs['About'] = (
                + ' The files have been concatenated, a new dimension denoted by '
                + 'horizon has been created as well as a time dimension that '
                + 'references the original forecast_reference_time. A subset of '
                + 'variables has been selected.'
                )
        try:
            dataset.attrs['title'] = dataset.attrs['title']
        except:
            dataset.attrs['title'] = 'Concatenated kyststasjoner files.'
        
        new_history = (
            str(dt.datetime.utcnow()) 
            + ' Python.'
            + ' Subset variables stormsurge, stormsurge_corrected, and bias.'
            + ' Fill in the 11 hours of gap between the original 12-hourly files.'
            + ' Added horizon dimension generated from the original time dimension.'
            + ' Added time coordinate generated from the original forecast_reference_time variable.'
            + ' New station dimension with consistent values over time that match'
            + ' those in the station data file from the North Sea collected by Jean Rabault.'
            + ' New stationid variable.'
            )
        
        try: 
            dataset.attrs['history'] = dataset.attrs['history'] + new_history
        except:
            dataset.attrs['history'] = new_history
        
            
        dataset.attrs['institution'] = 'Norwegian Meteorological Institute'
        dataset.attrs['source'] = 'ROMS'
        dataset.attrs['Author'] = (
            'Paulina Tedesco. The author of the original files is Nils Melsom'
            + ' Kristensen.'
            )
        dataset.attrs['contact'] = 'paulinast@met.no'
        
        return dataset
    
    def add_horizon_coord(self, dataset):
        """
        Change forecast time dimensions to horizon, and set new unique time
        dimension 

        Parameters
        ----------
        dataset : xarray.DataSet
            Dataset to reorganize.

        Returns
        -------
        dataset : xarray.DataSet
            Reorganized dataset.

        """
        # change name and values of time dimension to horizon
        values = np.arange(len(dataset.time))
        dataset['time'] = values
        dataset = dataset.rename({'time':'horizon'})
        # Set forecast_reference_time as dimension and rename to time
        dataset = dataset.assign_coords(time=dataset.forecast_reference_time)
        dataset.expand_dims('time')

        return dataset
    
    def subset_dataset(self, dataset, 
                       list_vars=[
                           'stormsurge', 
                           'stormsurge_corrected', 
                           'observed_at_chartdatum', 
                           'tide_at_chartdatum', 
                           'bias', 
                           'krit1', 
                           'krit2', 
                           'krit3'
                           ]
                           ):
        """
        The deterministic ensemble member is the last one.

        Parameters
        ----------
        dataset : xarray.DataSet
            Dataset to reorganize.

        Returns
        -------
        dataset : xarray.DataSet
            Reorganized dataset.

        """
        # TODO: check if we need variable observed or observed_at_chartdatum
        # Fill with NaNs if extra variables that only depend on 'station'
        # do not exist in the file.
        extra_vars = list(set(list_vars + ['bias', 'krit1', 'krit2', 'krit3']))
        for var in extra_vars:
            if var not in list(dataset.keys()):
                # Add the variable as an array with NaNs
                array_with_nans = xr.DataArray(
                    np.full([len(dataset.station)], np.nan), 
                    coords=[dataset.station], 
                    dims=['station']
                    )
                dataset[var] = array_with_nans
            
        # Subset the dataset
        dataset = dataset[list_vars]
        dataset = dataset.isel({
            'ensemble_member':self.ensemble_member, 
            'dummy':0
            })
        return dataset
        
    def fill_next_11_hrs(self, data):
        """
        Since the predictions and the files are generated every 12 hours, the 
        hours in between two files are filled in by introducing a 1 hour lag in 
        the horizon dimension. 
         
        The files are then concatenated along the time dimension, which is 
        originally the forecast_reference_time + the number of lags.

        Parameters
        ----------
        dataset : xarray.DataSet
            Dataset to reorganize.

        Returns
        -------
        dataset : xarray.DataSet
            Reorganized dataset.

        """
        data_concatenated = copy.deepcopy(data)
        len_hz = len(data.horizon)
        for i in range(11):
            data_i = data.isel(horizon=slice(i+1, len_hz)) #i+1 :
            data_i['horizon'] = data_i['horizon'] - (i + 1) 
            # Update forecast reference time to next hour
            new_time = data['time'] + pd.to_timedelta(dt.timedelta(hours=i+1)) 
            data_i['time'] = new_time
            data_i = data_i.expand_dims('time')
            data_concatenated = xr.concat(
                [data_concatenated, data_i], 
                dim='time'
                )  
        return data_concatenated
    
    
    def rename_station_index(self, dataset):
        """
        Set the same station numbers in all files as in the observations NetCDF
        file prepared by Jean.
        
        From Jean's file with observations with 106 station in the North Sea,
        copy the variable stationid and match the numbers.

        Parameters
        ----------
        dataset : xarray DataSet
            Dataset with the station dimension as in the kyststasjoner file.

        Returns
        -------
        dataset : TYPE
            Dataset with new station dimension.

        """
        # 
        # Open csv with station nrs.
        path_stationid_obs = os.path.join( os.getcwd(), '..', 'data', 'station_ids_obs_roms_files.csv' )
        stationid_obs = pd.read_csv(path_stationid_obs)
        
        # Dictionary with names and number in kyststasjoner files
        station_dict_kyststasjoner = self.decode_station_names(dataset)
        
        new_station_dim = []
        for station_name, station_nr in station_dict_kyststasjoner.items():
            # find stationname and number in csv file
            row_in_stationid_obs = stationid_obs[
                stationid_obs['station_name']==station_name
                ]  
            try:
                new_station_nr = row_in_stationid_obs['station_nr'].values[0]  # The values in the pandas series is an array. 
                # store the new station_numbers
                new_station_dim.append(new_station_nr)
            except:
                print('Station ', station_name, ' ', station_nr, 'is not in station_ids_obs_roms_files.csv.')
                dataset1 = dataset.sel(station=slice(0, station_nr - 1))
                last_item = dataset.station[-1].values.item()
                dataset2 = dataset.sel(station=slice(station_nr + 1, last_item))
                dataset = dataset1.merge(dataset2)            
            
        dataset['station'] = np.array(new_station_dim)
        
        # Sort the dataset according to the station dimension
        dataset = dataset.reindex(station=sorted(dataset.station))
        
        # TODO: add stationid variable when concatenating with other files
        # And fill in missing station numbers, i.e., not Norwegian stations.
        return dataset
            

        
    def fill_dataset_with_nans(self, dataset):
        """
        Copy the dataset and fill the variables bias, stormsurge, and 
        stormsurge_corrected with NaNs. Then sum 12 hours to the time dimension.

        Parameters
        ----------
        dataset : xarray DataSet
            Original DataSet with 12 hours of data.

        Returns
        -------
        dataset_nan : xarray DataSet
            DataSet with nans shiftet 12 hours forward in time.

        """
        dataset_nan = copy.deepcopy(dataset)
        variables_to_fill = ['bias', 'stormsurge', 'stormsurge_corrected']
        for var in variables_to_fill:       
            dataset_nan[var] = xr.full_like(dataset_nan[var], np.nan)     
        new_time = dataset_nan['time'] + pd.to_timedelta(dt.timedelta(hours=12)) 
        dataset_nan['time'] = new_time
        return dataset_nan
    
    def add_stationid_var(self, dataset):
        
        # Open the dataset with the station data
        path = (
            '/lustre/storeB/project/IT/geout/machine-ocean/prepared_datasets'
            + '/storm_surge/aggregated_water_level_data'
            + '/aggregated_water_level_observations_with_pytide_prediction_dataset.nc'
            )
    
        reference_dataset = xr.open_dataset(path)

        stationid = reference_dataset.stationid
        stationid = stationid[dataset['station']]  # Select only the Norwegian stations
        dataset['stationid']  = stationid
        
        return dataset
    
    def decode_station_names(self, data_file):
        """
        TODO: Move to utils, also used in main_predict
        Generate a dictionary that maps station names and IDs.

        Parameters
        ----------
        data_file : xarray.Dataset
            Dataset containing kyststasjoner data.

        Returns
        -------
        stations_id : dict
            The keys of the dictionary are the station names and the values are 
            the IDs in the data_file. These ID are not the same in the all files.

        """
        stations_id={}
        n_stations = len(data_file.station)
        for i in range(n_stations):
            name = data_file.station_name[0:3,i].values.astype(str).tolist()
            name = "".join(name)  # Concatenate letters in list 
            name = 'NO_' + name.upper()
            stations_id[name] = i
        return stations_id

    def get_station_id(self, data_file, station_name):
        """
        TODO: Move to utils - also used in main_predict
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
        stations_id = self.decode_station_names(data_file)
        station_id = stations_id[station_name]
        return station_id
        
    
    def get_list_stations(self):
        stations=[
                'DE_AlteWeser', 'DE_Eider', 'DE_Husum', 'DE_Wittduen', 'DE_Buesum', 
        'DE_Helgoland', 'DE_Norderney', 'DE_Cuxhaven', 'DE_Hoernum', 
        'DE_Wangerooge', 'NL_Amelander Westgat platform', 
        'NL_Aukfield platform', 'NL_Brouwershavensche Gat 08', 
        'NL_Brouwershavensche Gat, punt 02', 'NL_Cadzand', 'NL_Delfzijl', 
        'NL_Den Helder', 'NL_Engelsmanplaat noord', 'NL_Euro platform', 
        'NL_F3 platform', 'NL_Haringvliet 10', 'NL_Haringvlietmond', 
        'NL_Harlingen', 'NL_Holwerd', 'NL_Horsborngat', 'NL_Huibertgat', 
        'NL_IJmuiden stroommeetpaal', 'NL_K13a platform', 'NL_K14 platform', 
        'NL_L9 platform', 'NL_Lauwersoog', 'NL_Lichteiland Goeree', 
        'NL_Maasmond stroommeetpaal', 'NL_Nes', 'NL_Noordwijk meetpost', 
        'NL_North Cormorant', 'NL_Oosterschelde 11', 'NL_Oosterschelde 13', 
        'NL_Oosterschelde 14', 'NL_Oosterschelde 15', 'NL_Oudeschild',
        'NL_Platform A12', 'NL_Platform D15-A', 'NL_Platform F16-A', 
        'NL_Platform Hoorn Q1-A', 'NL_Platform J6', 'NL_Schiermonnikoog', 
        'NL_Terschelling Noordzee', 'NL_Texel Noordzee', 
        'NL_Vlakte van de Raan', 'NL_Vlieland haven', 'NL_West-Terschelling', 
        'NL_Westkapelle', 'NL_Wierumergronden', 'WL_draugen', 'WL_ekofisk', 
        'WL_heidrun', 'WL_heimdal', 'WL_sleipner', 'WL_veslefrikk', 'DK_4203',
        'DK_4201', 'DK_4303', 'DK_5103', 'DK_5104', 'DK_5203', 'DK_7101', 
        'DK_7102', 'DK_6401', 'DK_6501', 'NO_AES', 'NO_ANX', 'NO_BGO', 
        'NO_BOO', 'NO_HAR', 'NO_HEI', 'NO_HFT', 'NO_HRO', 'NO_HVG', 'NO_KAB', 
        'NO_KSU', 'NO_MAY', 'NO_MSU', 'NO_NVK', 'NO_NYA', 'NO_OSC', 'NO_OSL', 
        'NO_RVK', 'NO_SVG', 'NO_TOS', 'NO_TRD', 'NO_TRG', 'NO_VAW', 'NO_VIK', 
        'SW_2130', 'SW_2109', 'SW_2111', 'SW_35104', 'SW_35144', 
        'UK_Sheerness', 'UK_Aberdeen', 'UK_Whitby', 'UK_Lerwick', 
        'UK_Newhaven', 'UK_Lowestoft', 'UK_Wick']
        
        norwegian_stations = [
            'NO_AES', 'NO_ANX', 'NO_BGO', 
            'NO_BOO', 'NO_HAR', 'NO_HEI', 'NO_HFT', 'NO_HRO', 'NO_HVG', 'NO_KAB', 
            'NO_KSU', 'NO_MAY', 'NO_MSU', 'NO_NVK', 'NO_NYA', 'NO_OSC', 'NO_OSL', 
            'NO_RVK', 'NO_SVG', 'NO_TOS', 'NO_TRD', 'NO_TRG', 'NO_VAW', 'NO_VIK'
            ]
            
        return norwegian_stations
    
    
    def get_warning_thresholds(self, file:str='kyststasjoner_norge.nc2020010100', save_path:str ='../data/thresholds.csv'):
        """Get warning theresholds.
        
        Get the warning thresholds at each Norwegian harbor. If the save_path 
        argument is not empty, also save the data in a CSV file.
        
        The thresholds, krit1, krit2, krit3 determine the warning levels yellow,
        orange, and red. They are aproximately the returnperiods for 1, 5, and 
        20 yrs.
        
        Parameters
        ----------
        file : str, optional
            Kyststasjoner file containing from which to retrieve the threshold
            data. The default is 'kyststasjoner_norge.nc2020010100'.
        save_path : str, optional
            File path, if an empty string is provided, the data with thresholds
            will not be saved. The default is '../data/thresholds.csv'.

        Returns
        -------
        threshold_data : DataSet
            Dataset containing the warning thresholds for the Norwegian harbors.

        """
        
        try:
            dataset = xr.open_dataset(self.data_dir + '/' + file)
        except:
            print('File ', file, ' is not in directory ', self.data_dir)
            
        dataset = self.rename_station_index(dataset)
        
        thresholds =  ['krit1', 'krit2', 'krit3']
        for t in thresholds:
            if t not in list(dataset.keys()):
                # Add the bias variable as an array with NaNs
                array_with_nans = xr.DataArray(
                    np.full([len(dataset.station)], np.nan), 
                    coords=[dataset.station], 
                    dims=['station']
                    )
                dataset[t] = array_with_nans
                
        threshold_data =  dataset[thresholds]
        
        if save_path:
            # TODO: Add station_name
            df = threshold_data.to_dataframe()
            df.to_csv('../data/thresholds.csv')
        
        return threshold_data
        
        
        

if __name__ == "__main__":  
    
# =============================================================================
#     
#     ##########################################################################
#     #
#     # ------ Concatenate NetCDF files into one single file ------
#     #
#     ##########################################################################
#     
#     # Run the next part of the code for creating a single NetCDF file from all
#     # 12-hourly operational files.
#     
#     dt_start=dt.datetime(2017, 4, 26, 12, 0, 0, tzinfo=dt.timezone.utc) 
#     # dt_end=dt.datetime(2019, 12, 31, 12, 0, 0, tzinfo=dt.timezone.utc)
#     dt_end=dt.datetime(2021, 6, 30, 12, 0, 0, tzinfo=dt.timezone.utc)
#     data_dir = '/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive'
#     r = ReadKyststasjoner(dt_start, dt_end, data_dir)
#  
#     path_list = r.make_path_list()
#     
#     stations = r.get_list_stations()
#     
#     concatenated_files = r.concatenate_files(path_list, save=True)
#     
#     thresholds = r.get_warning_thresholds()   
# =============================================================================
     
    
    
    
    
    ##########################################################################
    #
    # ------ Compute RMSE ------
    #
    ##########################################################################
    # Run the next part of the code for computing the rmse of the error for all 
    # the stations stations
    
    print('--- Compute errors for all the files ---')
    #dt_start=dt.datetime(2017, 4, 26, 12, 0, 0, tzinfo=dt.timezone.utc)
    dt_start=dt.datetime(2020, 4, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    dt_end=dt.datetime(2021, 3, 31, 12, 0, 0, tzinfo=dt.timezone.utc)
    data_dir = '/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive'
    
    rmse_dir = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace/'
        +'paulinast/storm_surge_results/rmse_operational'
        )
    
    stormsurge_var = 'stormsurge_corrected'
    
    print('Compute RMSE for variable ', stormsurge_var)
    for member in range(52):
        print('Member: ', member)
        r = ReadKyststasjoner(dt_start, dt_end, member, data_dir)
        path_list = r.make_path_list()
        stations = r.get_list_stations()
        rmse_list = []
        rmse_list_all =[]
        for s in stations:
             y_pred, y_true = r.make_obs_stormsurge_arrays(
                 stormsurge_var=stormsurge_var,
                 station_name=s, 
                 path_list=path_list, 
                 hz=120-12
                 )
             rmse = r.rmse(y_true, y_pred)
             rmse_all = r.rmse_all(y_true, y_pred)
             print('RMSE at ', s, ': ', rmse)
             rmse_list.append(rmse)
             rmse_list_all.append(rmse_all)
             
        rmse_df_all = pd.DataFrame(rmse_list_all).transpose()
        rmse_df_all = rmse_df_all.set_axis(stations, axis=1)
        rmse_df_all.to_csv(
            ( 
                rmse_dir
                + '/rmse_' 
                + stormsurge_var 
                +'_108hr_20200401_20210331_m' 
                + str(member)
                + '.csv'
                ), 
            index=False
            ) 
        print(rmse_df_all)