import numpy as np
import pandas as pd
import datetime as dt
import xarray as xr

import ml_storm_surge_operational.utils.helpers as hlp

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
                 data_dir:str='/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive'):
        self.dt_start = dt_start  
        self.dt_end = dt_end  # Check if available
        self.data_dir = data_dir

    # There are two versions (different model setup?)
    # First new_kyststasjoner fie: new_kyststasjoner_norge.nc2018012712
    # Last new_syststasjoner file: 2018082012

    def make_path_list(self, valid_dates=[]):
        """
        Generates a list of paths to the kyststasjoner files taking into 
        account the version.
        
        Parameters
        ----------
        valid_dates : listof datetimes
            If provided, checks that the date of the kystasjoner files to is
            in valid_dates. The deafault is false.

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
        if (deltatime_days == 0 ) & (deltatime_hours >= 12) & (self.dt_end.day !=  self.dt_start.day):
            # Example:
            # self.dt_start = datetime.datetime(2020, 6, 30, 12, 0, tzinfo=datetime.timezone.utc)
            # self.dt_end = datetime.datetime(2020, 7, 1, 0, 0, tzinfo=datetime.timezone.utc)
            deltatime_days = deltatime_days + 1
            
        n_hours = deltatime_days*24   

        if self.dt_start.hour > self.dt_end.hour:
            n_hours = n_hours - 12
        elif self.dt_start.hour < self.dt_end.hour:
            n_hours = n_hours + 12
        
        """
        n_hours = ((self.dt_end -  self.dt_start).days)*24   
        
        if self.dt_start.hour > self.dt_end.hour:
            n_hours = n_hours + 24

        if self.dt_end.hour == 12:
            n_hours = n_hours + 12
        """

        if not valid_dates:
            valid_dates = [
                self.dt_start + dt.timedelta(hours=i) for i in range(n_hours + 1)
                ]
        
        path_list = []  
        # Iterate over all days in the period and generate a string every
        # 12 hours.
        for hr in range(0, n_hours + 1, 12):
            date_dt = (self.dt_start + dt.timedelta(hours=hr))
            date_str = date_dt.strftime('%Y%m%d%H')
            # valid_dates are the dates of the pp features without nans
            if date_dt in valid_dates: 
                if date_dt >= start_new_kyststasjoner and date_dt <= end_new_kyststasjoner:
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
        data = np.full([l, hz], fill_value=np.NaN)
        
        obs_vars =[
            'observed_at_chartdatum',
            'observed',
            'tide_at_chartdatum',
            'tide'
            ]
        
        # Construct arrays of shape (n_samples, n_outputs)
        for i in range(l):
                print(path_list[i])
                try:
                    data_file_i = xr.open_dataset(path_list[i])
                    station_nr = self.get_station_id(data_file_i, station_name)

                    if var == 'stormsurge_corrected' or var == 'stormsurge':
                        data[i, :hz] = ( 
                            data_file_i[var]
                            .isel(
                                {
                                'station':station_nr, 
                                'time':slice(0, hz), 
                                'ensemble_member':-1, 'dummy':0
                                }
                                ).values
                            )
                    elif var == 'bias':
                        data[i, :hz]= ( 
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
                        data[i, :hz] = (
                            data_file_i[var].isel(
                            {
                                'station':station_nr, 
                                'time':slice(0, hz), 
                                'dummy':0
                                }
                            ).values
                            )
                except:
                    # Keep nans in the files
                    print('File ', path_list[i], ' is not available.' )
                    pass
        return data
    
    def make_hz_col_names(self, var, station, hz, fhr):
        col_names = []
        if var=='observed_at_chartdatum' or var == 'observed':
            var = 'obs'
        if var == 'tide' or var == 'tide_at_chartdatum':
            var ='tide_py'

        past_forecast_str = ''
        if fhr != 0:
            past_forecast_str = '_' + str(fhr)
            
        for h in range(fhr, hz): # TODO: start at time 1?
            if h == 0:
                col_name_h = var + past_forecast_str + '_' + station
                col_names.append(col_name_h)
            else:
                col_name_h = (
                    var 
                    + '_t' 
                    + str(h) 
                    + past_forecast_str 
                    + '_' 
                    + station
                )
                col_names.append(col_name_h)
                
        return col_names
    
    def make_time_idx(self):
        delta = dt.timedelta(hours=12)
        date = self.dt_start     
        date_list = []
        while date <= self.dt_end:
            date_list.append(date)
            date += delta           
        return date_list   
                
    def array_to_df(self, array, index, col_names):
        df = pd.DataFrame(
            data=array, 
            index=index, 
            columns=col_names
            )
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
        estimated_error_arr = np.full(
            [l, hz], 
            fill_value=np.NaN
            )
        true_error_arr = np.full(
            [l, hz], 
            fill_value=np.NaN
            )
        
        # Construct arrays of shape (n_samples, n_outputs)
        for i in range(l):
                try:
                    data_file_i = xr.open_dataset(path_list[i])
                    station_nr = self.get_station_id(
                        data_file_i, 
                        station_name
                        )
                     # The same value is used for all the predictions
                    bias = np.tile(
                        (
                            data_file_i['bias']
                            .isel(station=station_nr)
                            .values
                        ),
                        hz
                        )
                    estimated_error_arr[i, :len(bias)] = bias
                    
                    true_error =(
                        data_file_i[roms_var].isel(
                            {
                                'station':station_nr, 
                                'time':slice(0, hz), 
                                'ensemble_member':-1, 
                                'dummy':0
                                }
                            ).values
                        - data_file_i['observed_at_chartdatum'].isel(
                            {
                                'station':station_nr, 
                                'time':slice(0, hz), 
                                'dummy':0
                                }
                            ).values  # OBS!! The last day (24 hrs) of observations are missing
                        + data_file_i['tide_at_chartdatum'].isel(
                            {
                                'station':station_nr, 
                                'time':slice(0, hz), 
                                'dummy':0
                                }
                            ).values
                        )
                    true_error_arr[i, :len(true_error)] = true_error
                except:
                    # Keep nans in the files
                    print('File ', path_list[i], ' is not available.' )
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
        estimated = np.full([l, hz], fill_value=np.NaN)
        true = np.full([l, hz], fill_value=np.NaN)
        
        # Construct arrays of shape (n_samples, n_outputs)
        for i in range(l):
                try:
                    data_file_i = xr.open_dataset(path_list[i])
                    station_nr = self.get_station_id(
                        data_file_i, 
                        station_name
                        )
                    estimated[i, : hz] = ( 
                        data_file_i[roms_var]
                        .isel(
                            {
                            'station':station_nr, 
                            'time':slice(0, hz), 
                            'ensemble_member':-1, 'dummy':0
                            }
                            ).values
                        )
                    
                    true[i, : hz] = (
                        
                        data_file_i['observed_at_chartdatum'].isel(
                            {
                                'station':station_nr, 
                                'time':slice(0, hz), 
                                'dummy':0
                                }
                            ).values  # OBS!! The last day (24 hrs) of observations are missing
                        - data_file_i['tide_at_chartdatum'].isel(
                            {
                                'station':station_nr, 
                                'time':slice(0,  hz), 
                                'dummy':0
                                }
                            ).values
                        )
                except:
                    # Keep nans in the files
                    print('File ', path_list[i], ' is not available.' )
                    pass
               
            
        return estimated, true
   
    
    def decode_station_names(self, data_file):
        """
        Generate a dictionary that maps station names and IDs.

        Parameters
        ----------
        data_file : xarray.Dataset
            Dataset containnig kyststasjoner data.

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
        Get station id of a particular station in a particular file.

        Parameters
        ----------
        data_file : xarray.Dataset
            Dataset containnig kyststasjoner data.
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
       
    # **************************************************************************
    # Modify this part to run the experiments of interest
    # **************************************************************************
    
    ##########################################################################
    #
    # ------ Compute RMSE ------
    #
    ##########################################################################
    # Run the next part of the code for computing the rmse of the error for all 
    # the stations stations
    
    print('--- Compute errors for all the files ---')
    dt_start=dt.datetime(2017, 12, 30, 0, 0, 0, tzinfo=dt.timezone.utc)
    dt_end=dt.datetime(2021, 3, 31, 12, 0, 0, tzinfo=dt.timezone.utc)
    data_dir = '/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive'
    
    rmse_dir = (
        '/lustre/storeB/project/IT/geout/machine-ocean/workspace/'
        +'paulinast/storm_surge_results/rmse_operational'
        )
    
    stormsurge_var = 'stormsurge_corrected'
    
    print('Compute RMSE for variable ', stormsurge_var)
    member=-1
    r = ReadKyststasjoner(dt_start, dt_end, data_dir)
    path_list = r.make_path_list()
    stations = hlp.get_norwegian_station_names()
    rmse_list = []
    rmse_list_all =[]
    for s in stations:
            y_pred, y_true = r.make_obs_stormsurge_arrays(
                stormsurge_var=stormsurge_var,
                station_name=s, 
                path_list=path_list, 
                hz=120-12
                )
            rmse = hlp.rmse(y_true, y_pred)
            rmse_all = hlp.rmse_all(y_true, y_pred)
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
            +'_108hr_20171230_20210331_m' 
            + str(member)
            + '.csv'
            ), 
        index=False
        ) 
    print(rmse_df_all)