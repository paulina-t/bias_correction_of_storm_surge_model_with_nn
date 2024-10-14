"""Correct ROMS radial plots or pressure plots.

Correct the output from the ROMS model (operational model or  hindcast),
using the bias from the radial plots or pressure plots, and make look-up tables.
"""

import numpy as np
import pandas as pd
import datetime as dt
import pickle
from scipy import interpolate
import copy
import os
import time
import pickle

import ml_storm_surge_operational.utils.helpers as hlp
from ml_storm_surge_operational.utils.eda import EDA
from ml_storm_surge_operational.data_loader.prepare_df import PrepareDataFrames 

print("Put the interpreter in UTC, to make sure no TZ issues...")
os.environ["TZ"] = "UTC"
time.tzset()


class CorrectROMS():
    """Correct the ROMS predictions."""
    
    def __init__(self, datetime_start_hourly, datetime_end_hourly, datetime_split, 
                 df=None, hindcast_forcing='NORA3ERA5', stations=[], variables=[], 
                 horizon = 120, ensemble_members=[-1], function_names = ['rmse', 'mean', 'std'],
                 train_on = ['hindcast'], test_on = ['hindcast', 'operational'],
                 interpolation_methods=['', 'cubic'], save=True, plot=False, use_existing_dataframes=False):
                     
        self.datetime_start_hourly = datetime_start_hourly
        self.datetime_end_hourly = datetime_end_hourly
        self.datetime_split = datetime_split
        self.df = df
        self.hindcast_forcing = hindcast_forcing
        self.stations = stations
        self.variables = variables
        self.horizon = horizon #range(121),
        self.ensemble_members = ensemble_members
        self.function_names = function_names
        self.train_on = train_on
        self.test_on = test_on
        self.interpolation_methods = interpolation_methods
        self.save=save
        self.plot=plot
        self.use_existing_dataframes = use_existing_dataframes
        
        self.data_dir = (
            '/lustre/storeB/project/IT/geout/'
            +'machine-ocean/workspace/paulinast/storm_surge_results/eda'
            )
        
        self.setup()
        
    def setup(self):
        
        if not self.stations:
            self.stations = hlp.get_norwegian_station_names()
            self.stations.remove('NO_MSU')  # This station is not in the kyststasjoner file  
            
        if not self.variables:
            self.variables = [
        'obs', 'tide_py', 'roms', 'msl', 'u10', 'v10', 'stormsurge_corrected', 
        'bias', 'wind_speed', 'wind_dir', '(roms - biasMET) - (obs - tide_py)', 
        '(obs - tide_py)', '(roms - biasMET)'
        ]
            
        self.roms_hind_var='(roms - biasMET)'
        self.roms_oper_var='stormsurge_corrected'
        self.obs_var='(obs - tide_py)'
        
        if self.hindcast_forcing=='ERA5':
            era5_forcing=True
        else:
            era5_forcing=False 

        
        print('Pickle dataframe for deterministic model.') 
        if self.df is None:
            prep = PrepareDataFrames(
                variables = self.variables,
                stations = self.stations,
                ensemble_members=[-1],
                datetime_start_hourly = self.datetime_start_hourly,
                datetime_end_hourly = self.datetime_end_hourly,
                horizon = self.horizon + 1,  
                use_station_data = True,
                run_on_ppi = True,
                new_station_loc=True,
                era5_forcing=era5_forcing,
                add_aux_variables=False,
                data_from_kartverket=True
                )
            self.df = prep.prepare_features_labels_df()   
        # Dump df with pickle and delete variable
        self.df_path = (
            '/lustre/storeB/project/IT/geout/machine-ocean/workspace/paulinast'
            + '/storm_surge_results/tmp_df_for_correction.pkl'
            )



    def append_horizon_to_str(self):
        roms_oper_var_horizons = []
        for h in range(self.horizon + 1):
            if h == 0:
                roms_oper_var_horizons.append(self.roms_oper_var)
            else:
                roms_oper_var_horizons.append(self.roms_oper_var + '_t' + str(h))
                
        return roms_oper_var_horizons
    
    
    def open_RP_and_PP_data(self, fun):
        " Make dicts with dataframes using get_data_and_make_radial_plots and get_data_and_make_pressure_plots"
        
        aggregated_dicts = {} 
        aggregated_dicts['RP'] = {}
        aggregated_dicts['PP'] = {}
        sorted_dicts = {}
        sorted_dicts['RP'] = {}
        sorted_dicts['PP'] = {}
        
        # TODO: Pass horizon!!!!
        
        print('self.train_on: ', self.train_on)
        print('self.test_on: ', self.test_on)
        print('self.roms_hind_var: ', self.roms_hind_var)
        print('self.roms_oper_var: ', self.roms_oper_var)
        
        # Hindcast 
        if (('hindcast' in self.train_on) or ('hindcast' in self.test_on)):
            # ------ Radial plot data ------
            print('Make Radial Plots with hindcast data.')
            aggregated_dicts['RP'][self.roms_hind_var], sorted_dicts['RP'][self.roms_hind_var] = self.get_data_and_make_radial_plots(
                fun,
                self.roms_hind_var,
                plot=self.plot
                )
            # ------ Pressure plot data ------
            print('Make Pressure Plots with hindcast data.')
            aggregated_dicts['PP'][self.roms_hind_var], sorted_dicts['PP'][self.roms_hind_var] = self.get_data_and_make_pressure_plots(
                fun,
                self.roms_hind_var,
                plot=self.plot
                )
            
        # Operational
        if (('operational' in self.train_on) or ('operational' in self.test_on)):
                print('Make Radial Plots with Operationaldata.')
                # ------ Radial plot data ------
                aggregated_dicts['RP'][self.roms_oper_var], sorted_dicts['RP'][self.roms_oper_var] = self.get_data_and_make_radial_plots(
                    fun,
                    self.roms_oper_var,
                    plot=self.plot
                    )
                
                print('Make Pressure Plots with operational data.')
                # ------ Pressure plot data ------
                aggregated_dicts['PP'][self.roms_oper_var], sorted_dicts['PP'][self.roms_oper_var] = self.get_data_and_make_pressure_plots(
                    fun,
                    self.roms_oper_var,
                    plot=self.plot
                    )
            
        if ('operational' in self.train_on):
            #loop_over_all_horizons
            pass
            
        return aggregated_dicts, sorted_dicts
        
    def correct_roms(self):
        map_datasets_and_var = {
            'hindcast':self.roms_hind_var,
            'operational': self.roms_oper_var
            }
        
        # ------ Prepare data dicts ------
        for fun in self.function_names: # TODO: It makes no sense to iterate over all func if we only do something when fun is mean.
            aggregated_dicts, sorted_dicts = self.open_RP_and_PP_data(fun) 
            for train in self.train_on:
                train_var = map_datasets_and_var[train]
                
                for test in self.test_on:
                    # Do not want to train on operational and test on hidcast
                    if (train == 'operational') and (test == 'hindcast'):  
                        print('Train is operational and test is hindcast.')
                    else:
                        test_var = map_datasets_and_var[test]
                        #if test_var==self.roms_oper_var:
                        #    test_var = self.append_horizon_to_str()
                        
                        #print('fun: ', fun)
                        #print('train_var: ', train_var)
                        #print('test_var: ', test_var)
    
                        # ------ Radial plot data ------
                        if  fun=='mean':
                            print('Function: ', fun)
                            # ------ Correct with pressure plots ------
                            self.correct_using_pressure_plots(
                                aggregated_dict_with_corrections=aggregated_dicts['PP'][train_var], 
                                sorted_dict_to_correct=sorted_dicts['PP'][test_var], 
                                test_var = test_var,  # TODO: Loop over names inside correct using radial plots
                                train_var = train_var,
                                save=self.save
                                )
                            
                            for interp in self.interpolation_methods:
                                print('Interpolation method: ', self.interpolation_methods)
                                self.correct_using_radial_plots(
                                    aggregated_dict_with_corrections=aggregated_dicts['RP'][train_var], 
                                    sorted_dict_to_correct=sorted_dicts['RP'][test_var], 
                                    test_var = test_var,  # TODO: Loop over names inside correct using radial plots
                                    train_var = train_var,
                                    interpolation_method = interp,
                                    save=self.save
                                    )
                                 

    def create_lookup_table(self, test_var, train_var, plot_var, interpolation_method=''):
        """
        Make an empty nested dictionary to use as lookup-table.

        Returns
        -------
        None.

        """
        lookup_table = {} 
        
        # --- Metadata ---
        lookup_table['metadata'] = {}
        lookup_table['metadata']['datetime_start'] = self.datetime_start_hourly
        lookup_table['metadata']['datetime_split'] = self.datetime_split
        lookup_table['metadata']['datetime_end'] = self.datetime_end_hourly
        lookup_table['metadata']['forcing'] = self.hindcast_forcing
        
        if interpolation_method:  #Pressure plots do not use interp method
            lookup_table['metadata']['interpolation_method'] = interpolation_method
            interpolation_str = '_' + interpolation_method
        else:
            interpolation_str = ''
            
        
        lookup_table['metadata']['train_on'] = train_var
        lookup_table['metadata']['test_on'] = test_var
        
        lookup_table['metadata']['description'] = (
            'The correctionRP is computed as'
            + ' mean(' + self.roms_hind_var + ' - ' + self.obs_var +')'
            + ' for each Norwegian station.'
            + self.roms_hind_var +' is computed for the hindcast forced with ' 
            + self.hindcast_forcing + '.'
            + ' This values can be used to correct the operational model stored as'
            + self.roms_oper_var + '.'
            )
        
        if plot_var == 'pressure':
            correction_str = 'correctionPP'
        elif plot_var == 'wind':
            correction_str = 'correctionRP'
        
        lookup_table['rmse'] = {}
        lookup_table['data'] = {}
        lookup_table['data'][correction_str] = {}
        #lookup_table['data']['correctionPP'] = {}
        
        lookup_table['rmse']['rmse_'+ hlp.crop_strings(test_var) + '_' + correction_str + interpolation_str] = {}
        lookup_table['rmse']['rmse_' + hlp.crop_strings(test_var)] = {}
        lookup_table['rmse']['rmse_'+ hlp.crop_strings(train_var) + '_' + correction_str + interpolation_str] = {}
        lookup_table['rmse']['rmse_' + hlp.crop_strings(train_var)] = {}
        
        
        for station in self.stations:
            lookup_table['data'][correction_str][station] = {}
            lookup_table['rmse']['rmse_'+ hlp.crop_strings(test_var) + '_' + correction_str + interpolation_str][station] = {}
            lookup_table['rmse']['rmse_' + hlp.crop_strings(test_var)][station] = {}
            
        self.lookup_table = lookup_table
    
    
    def interpolate_correction(self, aggregated_df, sorted_df, station, method):
        """Interpolate the corrections.
        
        Interolate the aggregated_df with correction at the original coordinate 
        points (e.g. wind speed and wind direction) and add the new column to 
        the sorted DataFrame.

        Parameters
        ----------
        aggregated_df : DataFrame
            DataFrame that has been aggregated by bins.
        sorted_df : DataFrame
            DataFrame that has been sorted by coordinate values.
        station : str
            Station ID.
        method : str
            Interpolation method. Example: 'cubic', or '' for default the 
            linear method.

        Returns
        -------
        sorted_df : DataFrame
            DESCRIPTION.

        """
        # Aggregated dict has the original corrections in bins
        # Sorted df has all the data in chronological order        
        aggregated_df = aggregated_df.interpolate(method ='linear') # TODO: 'linear' or method?
        z = np.array(aggregated_df.unstack(0) ) # Correction data, rows are wind_dir, columns are wind_speed
        x = np.array(aggregated_df.index.unique(level='wind_speed_bin_' + station) ) # TODO: generalize for wave coordinates?
        y = np.array(aggregated_df.index.unique(level='wind_dir_bin_' + station) )

        f = interpolate.interp2d(x, y, z, kind=method) # From aggregated hind
        xnew = np.array(sorted_df['wind_speed_' + station])
        ynew = np.array(sorted_df['wind_dir_' + station])
        
        # Interpolated values
        znew = []
        for i in range(len(xnew)):   
            znew.append(f(xnew[i], ynew[i]))        
       
        sorted_df['correctionRP' + method + '_' + station] = np.array(znew)
        return sorted_df
    
    
    def make_fname(self, correction_type:str, interpolation_method:str, train_var:str, test_var:str, ):
        """
        Make file name

        Parameters
        ----------
        correction_type : str
            'RP' or 'PP'.
        interpolation_method : str
            Interpolation method, e.g., 'cubic'. The default for linear 
            interpolation is ''.
        var_to_correct : str
            'stormsurge_corrected' or (roms - biasMET).

        Returns
        -------
        fname : str
            File name.

        """         
        correction_type_str = ''    
        if correction_type == 'PP':
            correction_type_str = '_pressure_correction'
        fname = (
                 hlp.crop_strings(self.hindcast_forcing)
                + '_train_on_' + hlp.crop_strings(train_var)
                + '_test_on_' + hlp.crop_strings(test_var)
                + '_' + hlp.crop_strings(self.obs_var)
                + '_' + interpolation_method
                + correction_type_str
                + '_' + str(self.datetime_start_hourly).split(' ')[0]
                + '_' + str(self.datetime_split).split(' ')[0]
                + '_' + str(self.datetime_end_hourly).split(' ')[0]
                )
        
        return fname
    
        
    def correct_using_radial_plots(self,  aggregated_dict_with_corrections, sorted_dict_to_correct,
                             train_var, test_var, interpolation_method='', save=True, fname=''):
        """Correct predictions with radial plots and fill in lookup-table.
        
        Correct the predictions generated with the model ROMS (operational or 
        hindcast), and fill in the lookup-table.
        
        Parameters
        ----------
        aggregated_dict_with_corrections : DataFrame
            DataFrame that has been aggregated by bins and contains the 
            corrections computed from the radial plots.
        sorted_dict_to_correct : DataFrame
            DataFrame that has been sorted by coordinate values and contains
            the corrections computed from the radial plots.
        correct_operational : bool, optional
            If true, correct the operational data, otherwise, correct the 
            hindcast data. The default is True.
        interpolation_method : str, optional
            Interpolation method, e.g., 'cubic'. The default for linear 
            interpolation is ''.
        save : bool, optional
            If True, save the dictionary. The default is True.
        fname : str, optional
            Path to the saved dictionary. The default is ''.

        Returns
        -------
        None.

        """
        if not fname:
            fname = self.make_fname(
                correction_type='RP', 
                interpolation_method=interpolation_method,
                train_var=train_var,
                test_var=test_var
                )
            
        print('fname radial plots: ', fname)
        # Create lookup_table dict
        self.create_lookup_table(test_var, train_var, plot_var='wind', interpolation_method=interpolation_method)

        fun = 'mean'
        rmse_var = 'rmse_'+ hlp.crop_strings(test_var) + '_correctionRP' + interpolation_method

        # ------ Correct the data with the output from the radial plots ------
        for station in self.stations:  
            print('Correcting data with radial plots for station ', station)
            
            if test_var == '(roms - biasMET)':
                var_to_correct = [test_var]
                horizon_for_loop = 0   
            else:
                var_to_correct = self.append_horizon_to_str()
                horizon_for_loop = self.horizon
                
            h_aggreg = 0
            rmse_corrected = []
            rmse_not_corrected = [] 
        
            for h in range(horizon_for_loop + 1):
                var_to_correct_h = var_to_correct[h]
                
                print('var_to_correct_h: ', var_to_correct_h)
                
                corr_col = 'correctionRP' + '_' + station
                obs_col = self.obs_var + '_' + station
                if h > 0:
                    if train_var == 'stormsurge_corrected':
                        corr_col = 'correctionRP_t' + str(h) + '_' + station
                        h_aggreg = h
                    obs_col = self.obs_var + '_t' + str(h) + '_' + station
                    
                #print(" sorted_dict_to_correct[fun]['test'][station].keys()",  sorted_dict_to_correct[fun]['test'][station].keys())
                #if (('t' + str(h)) in sorted_dict_to_correct[fun]['test'][station].keys()):# and (('t' + str(h)) in aggregated_dict_with_corrections[fun]['train'][station].keys()):

                # ------ DF with correction from train set aggregated by bins ------
                aggregated_df = copy.deepcopy(
                    aggregated_dict_with_corrections[fun]['train'][station]['t' + str(h_aggreg)][corr_col].fillna(0)
                    )
                
                # ------ DF with data to correct from test set sorted by bins ------
                sorted_df = copy.deepcopy(
                    sorted_dict_to_correct[fun]['test'][station]['t' + str(h)]
                    )
                
                # ------ Sorted DF with corrections ------
                sorted_df_with_corrections = (
                    sorted_df
                    .join(
                        aggregated_df, 
                        on=['wind_speed_bin_' + station, 'wind_dir_bin_' + station]
                        )
                    )
    
                # ------ Interpolate ------
                if interpolation_method:
                    if h > 0:
                        corr_col = 'correctionRP_t' + str(h) + interpolation_method + '_' + station
                    else:
                        corr_col = 'correctionRP_' + interpolation_method + station         
                    
                    sorted_df_with_corrections = self.interpolate_correction(
                        aggregated_df,
                        sorted_df_with_corrections, 
                        station, 
                        interpolation_method
                        )
                    
    
                # ------ Store correction in the lookup table ------
                if h == 0:
                    # Add level in dict for hz
                    self.lookup_table['data']['correctionRP'][station] =  (
                        aggregated_df
                        .to_numpy()
                        )
                # ------ Correct the test data ------
                correction_col = '(' + var_to_correct_h + ' - correctionRP' + interpolation_method + ')_' + station  
                
                sorted_df_with_corrections[correction_col] = (
                    sorted_df_with_corrections[var_to_correct_h + '_' + station] 
                    - sorted_df_with_corrections[corr_col] #['correctionRP' + interpolation_method + '_' + station]
                    )
                
                if save:
                # Pickle is too slow, can't load the data on jupyter. Use protocol=4
                # OBS: Saves data corresponding to the test period (not train) # TODO: Join all the df and then save them
                    sorted_df_with_corrections.sort_index().to_csv(
                        (
                            self.data_dir 
                            + '/dfs/df_' 
                            + fname
                            + '_'  + station 
                            + '_t' + str(h)
                            +'.csv'
                            
                            )
                        ) 

                # ------ RMSE ------            
                rmse_corrected.append(
                    hlp.rmse(
                        sorted_df_with_corrections[correction_col], 
                        sorted_df_with_corrections[obs_col]
                        )
                    )              
                rmse_not_corrected.append(
                    hlp.rmse(
                        sorted_df_with_corrections[var_to_correct_h + '_' + station], 
                        sorted_df_with_corrections[obs_col]
                        )
                    )
            print('rmse_corrected: ', rmse_corrected)   
            print('rmse_not_corrected: ', rmse_not_corrected)                        
            self.lookup_table['rmse'][rmse_var][station] = rmse_corrected
            self.lookup_table['rmse']['rmse_' + hlp.crop_strings(test_var)][station] = rmse_not_corrected
                
                     
        if save:
            print('Saving lookup-table...')
            self.save_lookup_table(fname)


    def initiate_dict(self):
        my_dict = {}
        my_dict['mean'] = {}
        for dataset in ['train', 'test']:
            my_dict['mean'][dataset] = {}
            for station in self.stations:
                my_dict['mean'][dataset][station] = {}
        return my_dict


    def get_data_and_make_radial_plots(self, fun, var1:str, plot=True):
        """Get the train and test data for each station and make radial plots.
        
        At each station, get the train and test data needed for making the 
        radial plots. If fun is 'mean', store the aggregated and sorted 
        DataFrames in dictionaries.

        Parameters
        ----------
        fun : str
            Aggregation method.
        var1 : str
            Name of a variable in the DataFrame to be used in the radial plots. 
        var2 : str, optional
            Name of a variable in the  DataFrame to be used in the radial plots.
            The default is ''.
        df : DataFrame
            DataFrame with the data to plot.
        plot : bool, optional
            If true, make the radial plots. The default is True.

        Returns
        -------
        aggregated_dict : dict
            Dictionary containing the aggregated train and test DataFramse 
            computed with the method mean. If fun is not 'mean', the dict is 
            empty.
        sorted_dict : dict
            Dictionary containing the sorted train and test DataFramse 
            computed with the method mean. If fun is not 'mean', the dict is 
            empty.

        """
        var2 = self.obs_var
        aggregated_dict =  self.initiate_dict()
        sorted_dict =  self.initiate_dict()

        horizon = 0
        if var1 == self.roms_oper_var:
            horizon = self.horizon
        
        ######################################################################
        # Compute radial plot dataset for each combination of ensemble member 
        # and horizon, for all stations, for both train and test data.
        ######################################################################
        
        # --- Loop over datasets ---
        
        # --- Loop over stations ---
        # TODO: define variable for both h and m here, not only m.
        # Here we define a variable for each member - var1_ and inside eda
        # We select the horizon.
        for station in self.stations:
            # --- Loop over horizons ---
            for h in range(horizon + 1):
                df_train, df_test = hlp.split_df(self.df, self.datetime_split)
                
                for df_to_plot in [df_train, df_test]:
                    if df_to_plot.equals(df_train):
                        dataset = 'train'
                    else:
                        dataset = 'test'
                    print('station: ', station, ', h:, ', h, ', dataset: ', dataset)
                    #print('df_to_plot: ', df_to_plot)
                    eda = EDA(station, self.data_dir, df_to_plot)
                    
                    #for fun in function_names: #['rmse', 'mean', 'std', 'corr']:
                    # ------ Radial plots ------
                    sorted_df, aggregated_df, c, theta, rho = eda.radial_plot_data(
                        function_name=fun,  
                        var1=var1, 
                        var2=var2, 
                        radial_coord_name='wind_speed',  # swh
                        angular_coord_name='wind_dir',  # mwd
                        h=h
                        ) 
                    
                    if fun == 'mean':
                        aggregated_dict['mean'][dataset][station]['t' + str(h)] = aggregated_df
                        sorted_dict['mean'][dataset][station]['t' + str(h)] = sorted_df
                        #print("aggregated_dict['mean'][dataset][station]['t' + str(h)]: ",
                        #      aggregated_dict['mean'][dataset][station]['t' + str(h)])
                    
                    # Plot only train data
                    if plot:
                        if c is not None:
                            if df_to_plot.equals(df_train):
                                abs_max = np.abs(np.max(np.nanquantile(c, 0.95)))
                                abs_min = np.abs(np.min(np.nanquantile(c, 0.05)))
                                vmax = max(abs_max, abs_min)
                                vmin = -vmax
                                
                                path = eda.make_radial_plot_path(
                                    fun, 
                                    station, 
                                    var1, 
                                    var2, 
                                    roms_dataset='hindcast_' + self.hindcast_forcing, 
                                    polar_coordinates='wind',
                                    h=h
                                    )
                                
                                print('Path polar plots: ', path)
                                
                                eda.radial_plot(
                                    c, 
                                    np.transpose(theta), 
                                    np.transpose(rho), 
                                    fun, 
                                    var1, 
                                    var2, 
                                    vmax=vmax, 
                                    vmin=vmin,
                                    h=h,
                                    path=path
                                    )
                

        return aggregated_dict, sorted_dict
    
                    
    def get_data_and_make_pressure_plots(self, fun, var1:str, plot=True):
        """Get the train and test data for each station and make pressure plots.
        
        At each station, get the train and test data needed for making the 
        pressure plots. If fun is 'mean', store the aggregated and sorted 
        DataFrames in dictionaries.
        

        Parameters
        ----------
        fun : str
            Aggregation method.
        var1 : str
            Name of a variable in the DataFrame to be used in the pressure 
            plots. 
        var2 : str, optional
            Name of a variable in the  DataFrame to be used in the pressure 
            plots. The default is ''.
        plot : bool, optional
            If true, make the pressure plots. The default is True.


        Returns
        -------
        train : DataFramse
            Train data.
        test : DataFrame
            Test data.

        """
        
        var2 = self.obs_var
        
        aggregated_dict =  self.initiate_dict()
        sorted_dict =  self.initiate_dict()
        
        horizon = 0
        if var1 == self.roms_oper_var:
            horizon = self.horizon

        for station in self.stations:            
                # --- Loop over horizons ---
                for h in range(horizon + 1):
                    
                    df_train, df_test = hlp.split_df(self.df, self.datetime_split)                        
                    for df_to_plot in [df_train, df_test]:
                        if df_to_plot.equals(df_train):
                            dataset = 'train'
                        else:
                            dataset = 'test'
                        eda = EDA(station, self.data_dir, df_to_plot)#df_to_plot)
                        print('station', station, 'h: ', h, 'dataset: ', dataset)
                        #print('df_to_plot: ', df_to_plot)
                        
                        #for fun in function_names: #['rmse', 'mean', 'std', 'corr']:
                        # ------ Pressure plots ------
                        if df_to_plot.equals(df_train):
                            path = None
                            if plot:
                                path = eda.make_pressure_plot_path(
                                    fun, 
                                    station, 
                                    roms_dataset='hindcast_' + self.hindcast_forcing,
                                    var1=var1, 
                                    var2=var2,
                                    h=h
                                    )

                        sorted_df, aggregated_df =  eda.plot_error_against_pressure(
                                fun, 
                                var1, 
                                var2, 
                                h=h,
                                path=path
                                )

                        if fun == 'mean':
                            aggregated_dict['mean'][dataset][station]['t' + str(h)] = aggregated_df
                            sorted_dict['mean'][dataset][station]['t' + str(h)] = sorted_df

        return aggregated_dict, sorted_dict
    
    def correct_using_pressure_plots(self,  aggregated_dict_with_corrections, sorted_dict_to_correct,
                             train_var, test_var, save=True, fname=''):
        """Correct predictions with pressure plots and fill in lookup-table.
        
        Correct the predictions generated with the model ROMS (operational or 
        hindcast), and fill in the lookup-table.
        
        

        Parameters
        ----------
        aggregated_dict_with_corrections : DataFrame
            DataFrame that has been aggregated by bins and contains the 
            corrections computed from the pressure plots.
        sorted_dict_to_correct : DataFrame
            DataFrame that has been sorted by coordinate values and contains
            the corrections computed from the pressure plots.
        correct_operational : bool, optional
            If true, correct the operational data, otherwise, correct the 
            hindcast data. The default is True.
        save : bool, optional
            If True, save the dictionary. The default is True.
        fname : str, optional
            Path to the saved dictionary. The default is ''.

        Returns
        -------
        None.

        """
        print('Correcting data using pressure plots...')
            
        self.create_lookup_table(test_var, train_var, plot_var='pressure')        
      
      
        fun = 'mean'
        rmse_var = 'rmse_'+ hlp.crop_strings(test_var) + '_correctionPP'
        
        # ------ Correct the data with the output from the radial plots ------
        
        for station in self.stations:
            #self.lookup_table['rmse'][rmse_var][station] = []
            #self.lookup_table['rmse']['rmse_' + hlp.crop_strings(test_var)][station]=[]
            
            if test_var == '(roms - biasMET)':
                var_to_correct = [test_var]
                horizon_for_loop = 0     
                members = [-1]
            else:
                var_to_correct = self.append_horizon_to_str()
                horizon_for_loop = self.horizon
                members = self.ensemble_members
            
            # ------ Loop over all horizons ------
            corr_col = 'correctionPP_' + station 
            obs_col = self.obs_var + '_' + station
            
            for m in members:                
                rmse_corrected = []
                rmse_not_corrected = [] 
                h_aggreg = 0
                
                
                for h in range(horizon_for_loop + 1):
                    var_to_correct_h = var_to_correct[h]
                    corr_col = 'correctionPP' + '_' + station
                    obs_col = self.obs_var + '_' + station
                    if h > 0:
                        if train_var == 'stormsurge_corrected':
                            corr_col = 'correctionPP_t' + str(h) + '_' + station
                            h_aggreg = h
                        obs_col = self.obs_var + '_t' + str(h) + '_' + station
                    #print('test_var: ', test_var)
                    #print('train_var: ', train_var)
                    #print('h_aggreg: ', h_aggreg)
                    #print("sorted_dict_to_correct[fun]['test'][station].keys(): ", sorted_dict_to_correct[fun]['test'][station].keys())
                    #print("aggregated_dict_with_corrections[fun]['train'][station].keys(): ", aggregated_dict_with_corrections[fun]['train'][station].keys())
                    #if (('t' + str(h)) in sorted_dict_to_correct[fun]['test'][station].keys()) and (('t' + str(h)) in aggregated_dict_with_corrections[fun]['train'][station].keys()): # Otherwise there might not be enough test data and nothing to correct
                        # ------ DF with correction from train set aggregated by bins ------
                    aggregated_df = copy.deepcopy(
                        aggregated_dict_with_corrections[fun]['train'][station]['t' + str(h_aggreg)][corr_col].fillna(0)
                        )
                    
                    # ------ DF with data to correct from test set sorted by bins ------
                    sorted_df = copy.deepcopy(
                        sorted_dict_to_correct[fun]['test'][station]['t' + str(h)]
                        )
                
                
                    # ------ Sorted DF with corrections ------
                    sorted_df_with_corrections = (
                        sorted_df
                        .join(
                            aggregated_df, 
                            on=['msl_bin_' + station]
                            )
                        )
                    
                    
                    # ------ Store correction in the lookup table ------
                    if h == 0:
                        # Add level in dict for hz
                        self.lookup_table['data']['correctionPP'][station] =  (
                            aggregated_df
                            .to_numpy()
                            )
                        
                     # ------ Correct the test data ------
                    correction_col = '(' + var_to_correct_h + ' - correctionPP' + ')_' + station  
                    
                    sorted_df_with_corrections[correction_col] = (
                        sorted_df_with_corrections[var_to_correct_h + '_' + station] 
                        - sorted_df_with_corrections[corr_col] #['correctionRP' + interpolation_method + '_' + station]
                        )
                    
                    # ------ RMSE ------            
                    rmse_corrected.append(
                        hlp.rmse(
                            sorted_df_with_corrections[correction_col], 
                            sorted_df_with_corrections[obs_col]
                            )
                        )              
                    rmse_not_corrected.append(
                        hlp.rmse(
                            sorted_df_with_corrections[var_to_correct_h + '_' + station], 
                            sorted_df_with_corrections[obs_col]
                            )
                        )
                
                self.lookup_table['rmse'][rmse_var][station] = rmse_corrected
                self.lookup_table['rmse']['rmse_' + hlp.crop_strings(test_var)][station] = rmse_not_corrected
              
        if save:
            #print('Saving lookup table...')
            if not fname:
                fname = self.make_fname(
                    correction_type='PP', 
                    interpolation_method='',
                    train_var = train_var,
                    test_var=test_var #var_to_correct
                    
                    )
            self.save_lookup_table(fname)
                    
                    
    def save_lookup_table(self, fname):
        """
        Save dictionary with lookup-table.

        Parameters
        ----------
        fname : str
            File name.

        Returns
        -------
        None.

        """
        file = (open(self.data_dir + '/lookup_table_' + fname + '.pkl', "wb"))
        print('Saving: ', file)
        pickle.dump(self.lookup_table, file)
        file.close() 
      
 
if __name__ == '__main__':
    
    # ------ Compute corrections training on hindcast ------
    print('Training on hindcast data...')
    
    stations = hlp.get_norwegian_station_names()
    stations.remove('NO_MSU')  # This station is not in the kyststasjoner file  
    
    variables = [
        'obs', 'tide_py', 'roms', 'msl', 'u10', 'v10', 'stormsurge_corrected', 
        'bias', 'wind_speed', 'wind_dir', '(roms - biasMET) - (obs - tide_py)', 
        '(obs - tide_py)', '(roms - biasMET)'
        ]
    datetime_start_hourly=dt.datetime(2001, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    datetime_end_hourly=dt.datetime(2019, 12, 31, 0, 0, 0, tzinfo=dt.timezone.utc)
    datetime_split=dt.datetime(2017, 12, 31, 0, 0, 0, tzinfo=dt.timezone.utc)
    
    function_names = ['rmse', 'mean', 'std']  # ['rmse', 'mean', 'std', 'corr']
    
    # ------ Compute corrections ------
    forcings  = ['NORA3ERA5', 'ERA5']
    for forcing in forcings:
        c = CorrectROMS(
            datetime_start_hourly, 
            datetime_end_hourly, 
            datetime_split, 
            hindcast_forcing=forcing, 
            stations=stations, 
            variables=variables, 
            function_names = ['rmse', 'mean', 'std'],
            train_on = ['hindcast'], 
            test_on = ['hindcast', 'operational'],
            interpolation_methods=[''],#, 'cubic'],
            save=True, 
            plot=False
            )
        
        c.correct_roms()

      
    # ------ Compute corrections training on operational ------  
    print('Training on operational data...')

    forcing = 'NORA3ERA5'
    
    datetime_start_hourly=dt.datetime(2018, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    datetime_end_hourly=dt.datetime(2020, 12, 31, 0, 0, 0, tzinfo=dt.timezone.utc)
    datetime_split=dt.datetime(2019, 12, 31, 0, 0, 0, tzinfo=dt.timezone.utc)
    
    variables = [
        'obs', 'tide_py', 'roms', 'msl', 'u10', 'v10', 'stormsurge_corrected', 
        'bias', 'wind_speed', 'wind_dir', '(obs - tide_py)', 
        'stormsurge_corrected - (obs - tide_py)'
        ]
    
    function_names = ['mean'] 
    
    c = CorrectROMS(
        datetime_start_hourly, 
        datetime_end_hourly, 
        datetime_split, 
        hindcast_forcing=forcing, 
        stations=stations, 
        variables=variables, 
        function_names = ['rmse', 'mean', 'std'],
        train_on = ['operational'], 
        test_on = ['operational'],
        interpolation_methods=[''], #, 'cubic'],
        save=True, 
        plot=False
        )
        
    c.correct_roms()