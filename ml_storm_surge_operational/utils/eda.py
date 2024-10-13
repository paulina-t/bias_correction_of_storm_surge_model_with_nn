"""Code for Exploratory Data Analysis."""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
from scipy.stats import norm
from scipy import stats
import warnings
import xarray as xr
#import cartopy.crs as ccrs
import ml_storm_surge_operational.utils.helpers as hlp
from ml_storm_surge_operational.data_loader.read_kyststasjoner import ReadKyststasjoner
from ml_storm_surge_operational.data_loader.prepare_df import PrepareDataFrames 

import os
import time

print("Put the interpreter in UTC, to make sure no TZ issues...")
os.environ["TZ"] = "UTC"
time.tzset()


warnings.filterwarnings('ignore')
# mpl.rcParams['figure.figsize'] = (8, 6)
#mpl.rcParams['axes.grid'] = False


class EDA():
    """Exploratory Data Analysis."""
   
    def __init__(self, station, data_dir, df_station):
        self.station = station
        self.data_dir = data_dir
        self.lat, self.lon = hlp.get_station_lat_lon(self.station)
        self.df_station = df_station
    
        self.coord_str = (
            '(' + str(np.round(self.lat)) + 'N, ' 
            + str(np.round(self.lon)) + 'E)'
            )
        
        self.subset_station_df()
        
    def subset_station_df(self):
        """
        Select the columns associated with the station passed as argument.

        Returns
        -------
        None.

        """
        self.df_station = self.df_station.filter(
            regex=(self.station + '$'), 
            axis=1
            )
        
    def summary(self):
        """
        Print the summary.

        Returns
        -------
        None.

        """
        summary = self.df_station.describe()
        print(summary)
           
        
    def count_missing_data(self):
        """Print the number and ratio of missing data.
        
        Print the number and the ratio of missing observations for each 
        variable in df_station.

        Returns
        -------
        None.

        """
        total = (
            self.df_station
            .isnull()
            .sum()
            .sort_values(ascending=False)
            )
        percent = (
            (
                self.df_station.isnull().sum()
                /self.df_station.isnull().count()
            )
            .sort_values(ascending=False)
            )
        
        missing_data = pd.concat(
            [total, percent], 
            axis=1,
            keys=['Total', 'Percent']
            )
        print('missing_data: ', missing_data)
        
    def plot_series(self):
        """
        Plot time series of each variable in df_station and saves the figures.

        Returns
        -------
        None.

        """
        for col in self.df_station:
            plt.figure()
            graph = sns.lineplot(
                data=self.df_station, 
                x=self.df_station.index, 
                y=col
                )
            
            # Plot the average as a horizontal line
            average = np.mean(self.df_station[col])
            graph.axhline(average, color='red')
            
            
            plt.tight_layout()
            
            plt.savefig(
                self.data_dir 
                + '/time_series/time_series_' 
                + hlp.crop_strings(col)
                + '.png'
                )
            
            plt.show()
            
    def plot_distribution(self):
        """Plot histograms and boxplots.
        
        Plot histograms and boxplots of each variable in df_station in one 
        figure and probability plots in another, and saves the figures.

        Returns
        -------
        None.

        """
        for col in self.df_station:    
            #histogram and normal probability plot
            plt.figure()
            fig, ax = plt.subplots(nrows=1, ncols=2)
            sns.distplot(self.df_station[col], fit=norm, ax=ax[0]);
            sns.boxplot(y=self.df_station[col], ax=ax[1])
            plt.tight_layout()
            plt.savefig(
                self.data_dir 
                + '/distributions/hist_and_boxplots/hist_and_boxplot_' 
                + hlp.crop_strings(col)
                + '.png'
                )            
            plt.show()
            
            plt.figure()
            stats.probplot(self.df_station[col], plot=plt)
            plt.tight_layout()
            plt.savefig(
                self.data_dir 
                +'/distributions/probabilities/probability_plot_'
                + hlp.crop_strings(col)
                + '.png'
                )
            plt.show()
            
    def norm_parameters(self):  
        """
        Print skewness and kurtosis.

        Returns
        -------
        None.

        """
        for col in self.df_station:
            print('------------------------')
            print(col)
            print("Skewness: %f" % self.df_station[col].skew())
            print("Kurtosis: %f" % self.df_station[col].kurt())
        
    def correlation_matrix(self, triangular=False):
        """Make correlation matrix.
        
        Compute the correlations between variables in df_station and saves 
        the figure with the correlation matrix.

        Parameters
        ----------
        triangular : bool, optional
            If true, generates a triangular correlation matrix. The default is 
            False.

        Returns
        -------
        None.

        """
        df_corr = self.df_station.copy()
        print('Columns in df_corr: ', df_corr.columns)
        
        # TODO: Remove this when bias is computed in preprocessor.py
        vars_to_drop = [] # ['(roms - biasMET)_' + self.station]
        if vars_to_drop:
            for v in vars_to_drop:
                if v in df_corr.columns:
                    df_corr = df_corr.drop(v, axis=1)
        
        corrmat = (
            self.df_corr
            .corr()
            )

        f, ax = plt.subplots(figsize=(12, 9))
        # Mask where values are True or missing
        if triangular:        
            matrix = np.triu(corrmat)
        else:
            matrix = np.zeros(corrmat.shape)
            
        colnames_list = list(self.df_station.columns) 
        labels = [i.split('_', 1)[0] for i in colnames_list]

        sns.heatmap(
            corrmat, 
            annot = True, 
            vmin=-1, 
            vmax=1, 
            center= 0, 
            cmap= 'coolwarm', 
            mask=matrix, 
            xticklabels=labels,
            yticklabels=labels
            )
        
        plt.tight_layout()
        plt.title(self.station)
        plt.savefig(
            self.data_dir 
            + '/corr_mat/corr_mat_' 
            + hlp.crop_strings(self.station)
            + '.png'
            )
        plt.show()
        
    def pairplot(self):
        """Make figure with histograms and scatter plots across variables.
        
        Plot a matrix with histograms and scatter plots for each pair of 
        variables in df_Station, and saves the figure.

        Returns
        -------
        None.

        """
        # Histogram and scatterplot
        df_pairplot = self.df_station.copy()
        # TODO: Remove this when bias is computed in preprocessor.py
        vars_to_drop = [] # ['(roms - biasMET)_' + self.station]
        if vars_to_drop:
            for v in vars_to_drop:
                if v in self.df_station.columns:
                    df_pairplot = df_pairplot.drop(v, axis=1)
                    
        colnames_list = list(self.df_station.columns) 
        labels = [i.split('_', 1)[0] for i in colnames_list]
                
        plt.figure()

        g = sns.pairplot(
            df_pairplot, 
            size = 2.5, 
            xticklabels=labels, 
            yticklabels=labels
            )
        g.map_lower(
            sns.histplot, 
            evels=4, 
            )
        
        plt.tight_layout()
        plt.savefig(
            self.data_dir 
            + '/distributions/pairplots/pairplot_hist_and_scatter_'
            + hlp.crop_strings(self.station)
            + '.png'
            )
        plt.show()
        
    def my_count(self, df_diff):
        """Define count function to be used in aggregation aggregation.
        
        Counts the number of observations of var1 in df.

        Parameters
        ----------
        df : DataFrame
            DataFrame with a column that is the difference between the variables
            that the RMSE is computed for.

        Returns
        -------
        scalar
            Number of observations of var1 in df.

        """
        return df_diff.count()

    def my_rmse(self, df_diff):
        """Define RMSE function to be used in aggregation aggregation.
        
        Computes the Root Mean Squared Error.

        Parameters
        ----------
        df_diff : DataFrame   
            DataFrame with a column that is the difference between the variables
            that the RMSE is computed for.

        Returns
        -------
        scalar
            RMSE.

        """
        return np.sqrt(np.nanmean((df_diff)**2))
    
    def aggregate_df(self, sorted_df, function_name, list_grouping_variables, var1, var2=''):
        """Aggregate DataFrame.
        
        Aggregate var1 and var2 of an already sorted dataframe using the method
        provided by the argument function_name.

        Parameters
        ----------
        sorted_df : TYPE
            DESCRIPTION.
        function_name : TYPE
            DESCRIPTION.
        list_grouping_variables : TYPE
            DESCRIPTION.
        var1 : TYPE
            DESCRIPTION.
        var2 : TYPE, optional
            DESCRIPTION. The default is ''.

        Returns
        -------
        aggregated_df : TYPE
            DESCRIPTION.

        """    
        dispatcher = {
            'mean' : np.mean,
            'std' : np.std, 
            'corr' : np.corrcoef, 
            'count' : self.my_count, 
            'rmse' : self.my_rmse
            }
        fun = dispatcher[function_name]
        col1 = var1 + '_' + self.station
        col2 = var2 + '_' + self.station
        
        if var2:
            gb = (
                sorted_df
                .groupby(list_grouping_variables)
                [[col1, col2]]
                )
            series = (
                gb
                .apply( lambda grp: fun(grp[col1] - grp[col2]) )
                )
                     
            if function_name == 'corr':           
                series = series.iloc[0::2,-1]  # Select only one value from 2x2 matrix for corr
                aggregated_df = series.to_frame()  
                aggregated_df = aggregated_df.reset_index(level=2, drop=True)  # corr generates one more index level since the 2x2 matrix has one row for each var
            else:   # count, std, mean, or rmse
                try:
                    aggregated_df = series.to_frame()  
                except:
                    print('Cannot convert series to dataframe.')
                    print(series)
                    aggregated_df = series       
                
        else:  # Not var2 
            if fun == 'corr':
                print('OBS: only one column has been provided. The correlation is 1.')
            if fun == 'rmse':
                print('OBS: only one column has been provided. The RMSE is computed WRT a vector of zeros.')
            gb = (
                sorted_df
                .groupby(list_grouping_variables)
                [[col1]]
                )
            series = (
                gb
                .apply( lambda grp: fun(grp[col1]) )
                )
            
            aggregated_df = series.to_frame() 
            
        return aggregated_df
    
    
    def plot_error_against_pressure(self, function_name, var1, var2='', h=0, m=None, path=None):
        """
        Plot the bias/error against the pressure.

        Parameters
        ----------
        function_name : str
            The function names must be, 'mean', 'std', 'rmse', or 'count'
        var1 : str
            Name of a variable in the df_station DataFrame. 
        var2 : str, optional
            Name of a variable in the df_station DataFrame. The default is ''. 
        horizon : int
            Horizons.
        path : str, optional
            If provided, the figure will be saved with this path. The default is None.

        Returns
        -------
        sorted_df : DataFrame
            df_dataframe sorted according to pressure bins.
        aggregated_df : DataFrame
            Aggregated DataFrame according to function_name.

        """
        # Define bins (round values)
        p_name = 'msl' + '_bin_' + self.station
        
        self.df_station[p_name] = np.round(
            (self.df_station['msl' + '_' + self.station] / 500) 
            )*500
        
        if h > 0:
            var1_h = var1 + '_t' + str(h)
            var2_h = var2 + '_t' + str(h)
            corr_col = 'correctionPP_t' + str(h) + '_' + self.station
        else:
            # If h==0 or h==None
            var1_h = var1
            var2_h = var2
            corr_col = 'correctionPP_' + self.station  
        
        # Sort data according to new pressure variable
        sorted_df = self.df_station.sort_values([p_name])
        
        aggregated_df_h = self.aggregate_df(
            sorted_df=sorted_df, 
            function_name=function_name, 
            list_grouping_variables=[p_name],
            var1=var1_h, 
            var2=var2_h
            )
        
        #col_name = function_name + '(' + var1_h + ', ' + var2_h + ')'
        aggregated_df = aggregated_df_h.rename(columns={0: corr_col})
        
        var_str = self.make_var_str(var1, var2)
        y_label = function_name + var_str + ' [m]'
        
        if not aggregated_df.empty:
            #print('aggregated_df: ', aggregated_df)
            m_str = ''  # m is the last one
            if type(m) == int:
                if (m>=0) and (m<51):
                    m_str = 'Ensemble member: '  + str(m)
                elif m<-1:
                    m_str = 'Ensemble member: '  + str(52 + m)
                
            h_str = ''
            if h>0:
                h_str = 'Lead time: ' + str(h)
            plt.figure()
            ax = sns.lineplot(
                x=aggregated_df.index, 
                y=corr_col, # 'correctionPP_' + self.station, #col_name
                data=aggregated_df
                )
            #ax.set_title(self.station + ' ' + self.coord_str +  '\n' + m_str + ' ' + h_str)
            ax.set(
                xlabel='Pressure [Pa]',
                ylabel=y_label#(col_name + ' [m]')
                )
            ax.grid()
            plt.tight_layout()
            if path:
                print('Saving pressure plots...')
                print(path)
                plt.savefig(path, bbox_inches='tight')
            plt.show()
        else: 
            print('aggregated_df is empty.')

        return sorted_df, aggregated_df
        
    def make_pressure_plot_path(self, fun, station, roms_dataset, var1, var2=None, h=None, m=None):
        """
        Generate a path for the pressure figures.

        Parameters
        ----------
        fun : str
            Aggregation method.
        station : str
            Station ID
        var1 : str
            Name of a variable in the df_station DataFrame. 
        var2 : str, optional
            Name of a variable in the df_station DataFrame. The default is ''.
        roms_dataset : str
            Specify the tye of data used, e.g., 'hindcast_NORA3ERA5'.
        h : int
            Horizon
        m : int
            Ensemble member

        Returns
        -------
        path : str
            Path to the pressure figure.

        """
        var2_str = ''
        h_str = ''
        m_str = ''
    
        if var2:
            var2_str = '_' + hlp.crop_strings(var2)
        
        if h:
            h_str = '_' + str(h)
        
            
        if type(m)==int:
            if (m < 51) and (m >= 0):   
                m_str = '_m' + str(m)
            elif m < -1:  # When counting from the last position in a vector
                m = 52 + m
                m_str = '_m' + str(m)

           
        new_dir = (
            self.data_dir 
            + '/pressure_plots'
            + '/' + fun
            + '/' + hlp.crop_strings(var1) 
            + var2_str
            ) 
        
        hlp.make_dir(new_dir)
        
        path = (
            new_dir 
            +'/pressure_plot_'
            + '_' + roms_dataset
            + '_' + hlp.crop_strings(station)
            + '_' + fun
            + '_' + hlp.crop_strings(var1)
            + '_' + hlp.crop_strings(var2)
            + '_' + str(self.df_station.index[0]).split(' ')[0]
            + '_' + str(self.df_station.index[-1]).split(' ')[0]
            + h_str
            + m_str
            + '.png'
            )
        return path
           
    
    def radial_plot_data(self, function_name,  var1, var2='', 
                        radial_coord_name='wind_speed', angular_coord_name='wind_dir',
                        h=0, min_obs_bin=1):
        """Generate radial plot data.
        
        Generates and saves radial plot figures (pcolor in polar coordinates), 
        where the radius is indicated by the wind speed variable and the the 
        angle represents the direction from where the wind blows.

        Parameters
        ----------
        function_name : str
            Aggregation function to represent in the radial plots, e.g. my_mean.
            It must be a method of the class EDA.
        var1 : str
            Name of a variable in the df_station DataFrame. 
        var2 : str, optional
            Name of a variable in the df_station DataFrame. The default is ''.
        radial_coord_name = str
            The variable name in df_station of the variable to be used as the
            radial coordinate. The default is 'wind_speed'.  
        angular_coord_name = str
            The variable name in df_station of the variable to be used as the
            angular coordinate. The default is 'wind_speed'.              
        operational : bool, optional
            If True, makes the plot for the operational roms data.
        h : int
            One specific horizon.

        Returns
        -------
        None.

        """        
        # Define bins (round values)
        rho_name = radial_coord_name + '_bin_' + self.station
        theta_name = angular_coord_name + '_bin_' + self.station
        
        #print('eda self.df_Station.colums: ', self.df_station.columns)
        #for c in self.df_station.columns:
        #    print(c)
            
        # Rho (wind speed)
        self.df_station[rho_name] = np.round(
            self.df_station[radial_coord_name + '_' + self.station]
            .shift(-h)
            )
        # Theta (wind dir)
        self.df_station[theta_name] = np.round(
            (
                self.df_station[angular_coord_name + '_' + self.station]
                .shift(-h) 
                / 10
                ) 
            )*10
        
        if h > 0:
            var1_h = var1 + '_t' + str(h)
            var2_h = var2 + '_t' + str(h)
            corr_col = 'correctionRP_t' + str(h) + '_' + self.station
        else:
            # If h==0 or h==None
            var1_h = var1
            var2_h = var2
            corr_col = 'correctionRP_' + self.station  
                    
        # Sort data according to rho and theta
        sorted_df = (
            self.df_station[[
                var1_h + '_' + self.station, 
                var2_h +'_' + self.station, 
                rho_name, 
                theta_name
                ]]
            .sort_values([
                rho_name, 
                theta_name
                ])
            )
                    
        aggregated_df_h = self.aggregate_df(
            sorted_df=sorted_df, 
            function_name=function_name, 
            list_grouping_variables=[rho_name, theta_name],
            var1=var1_h, 
            var2=var2_h
            )
        aggregated_df = aggregated_df_h.rename(columns={0: corr_col})  
        print('aggregated_df: ', aggregated_df)
        
        # Aggregate df accorfing to count function and remove data in bins with 
        # only one observation
        
        aggregated_df_h_count = self.aggregate_df(
            sorted_df=sorted_df, 
            function_name='count', 
            list_grouping_variables=[rho_name, theta_name],
            var1=var1_h, 
            var2=var2_h
            )
        aggregated_df_count = aggregated_df_h_count.rename(columns={0: 'count_' + self.station})
        print('aggregated_df_count: ', aggregated_df_count)
        
        aggregated_df.loc[aggregated_df_count['count_' + self.station] < min_obs_bin] = np.nan
        print('aggregated_df: ', aggregated_df)

        # Fill in missing directions and speeds with NaN 
        if not aggregated_df.empty:
            new_index = pd.MultiIndex.from_product(aggregated_df.index.levels, names=aggregated_df.index.names)
            aggregated_df = aggregated_df.reindex(new_index)#.rename(columns={0: 'correctionRP'})
                
            # Construct 2D arrays
            unique_directions = np.radians(aggregated_df.index.unique(level=theta_name))
            unique_speeds = aggregated_df.index.unique(level=rho_name)
            theta, rho = np.meshgrid(unique_directions, unique_speeds)
            c = aggregated_df.unstack(0).to_numpy()
        else:
            print('Aggregated df is empty.')
            c = None
            theta = None
            rho = None
        
        return sorted_df, aggregated_df, c, theta, rho
        
        
        #cmap = plt.cm.Spectral_r
        # Plot
        # http://blog.rtwilson.com/producing-polar-contour-plots-with-matplotlib/
        # https://stackoverflow.com/questions/44423694/plotting-windrose-making-a-pollution-rose-with-concentration-set-to-color
        
    def make_var_str(self, var1, var2):
        if (var1 == 'stormsurge_corrected') or (var1 == '(stormsurge_corrected - correctionML)'):
            var1_s = 'storm_surge'
            var1_s = 'R' # Residual
        elif var1 == '(obs - tide_py)':
            var1_s = '(obs - tide)'
            var1_s = '(Z - T)'  # Observed height - Tide
        elif var1 == '(roms - biasMET)':
            var1_s = 'storm_surge'
            var1_s = 'R'
        
            
        if (var1 == 'stormsurge_corrected') or (var1 == '(stormsurge_corrected - correctionML)'):
            var2_s = 'storm_surge'
            var2_s = 'R'
        elif var2 == '(obs - tide_py)':
            var2_s = '(obs - tide)'
            var2_s = '(Z - T)'
        elif var2 == '(roms - biasMET)':
            var2_s = 'storm_surge'
            var2_s = 'R'
            

        if var2:
            var_str = '(' + var1_s + ' - ' + var2_s + ')'
        else:
            var_str = '(' + var1_s + ')'
            
        return var_str
        
    
    def radial_plot(self, c, theta, rho, function_name, var1, var2='', vmax=None, vmin=None, path=None, h=0, m=None):
        """Make radial plots.
        
        Plot the data generated with the function radialplot_data.
        
        Parameters
        ----------
        c : DataFrame
            Unstacked dataframe containing the data to plot.
        theta : array
            The angular coordinate.
        rho : array
            The radial coordinate.
        function_name : str
            The function names must be, 'mean', 'std', 'rmse', or 'count'.
        coord_var : str
            Name of the polar coordinates, e.g., 'wind' or 'wave'.
            The default is 'wind.'
        var1 : str
            Name of a variable in the df_station DataFrame. 
        var2 : str, optional
            Name of a variable in the df_station DataFrame. The default is ''.

        Returns
        -------
        None.
        """
 
        m_str = ''
        if type(m)==int:
            if (m>=0) and (m<51):
                m_str = 'Ensemble member: '  + str(m)
            elif m<-1:
                m_str = 'Ensemble member: '  + str(52 + m)
            
        h_str = ''
        if h>0:
            h_str = 'Lead time: ' + str(h)
            
        var_str = self.make_var_str(var1, var2)
        cbar_label = function_name + var_str + ' [m]'
        fig, ax = plt.subplots(subplot_kw={"projection":"polar"})
        if function_name == 'corr':
            cmap = plt.get_cmap('Spectral_r')
            cmap.set_bad(color = 'w', alpha = 0)
            p = ax.pcolormesh(theta, rho, c, cmap=cmap, vmin=-1, vmax=1)
            cb = fig.colorbar(p)
            cb.set_label(cbar_label)
        elif function_name == 'mean':
            abs_max = np.abs(np.max(np.nanquantile(c, 0.95)))
            abs_min = np.abs(np.min(np.nanquantile(c, 0.05)))
            if not vmax:
                vmax = max(abs_max, abs_min)
            if not vmin:
                vmin = -vmax
            cmap = plt.get_cmap('Spectral_r')
            cmap.set_bad(color = 'w', alpha = 0)
            p = ax.pcolormesh(theta, rho, c, cmap=cmap,vmin=vmin, vmax=vmax)
            cb = fig.colorbar(p)
            cb.set_label(cbar_label)
        elif function_name == 'std' or function_name == 'rmse':
            p = ax.pcolormesh(theta, rho, c, cmap='viridis_r')
            cb = fig.colorbar(p)
            cb.set_label(cbar_label)
        else:
            p = ax.pcolormesh(theta, rho, c, cmap='viridis_r')
            cb = fig.colorbar(p)
            cb.set_label(cbar_label)
            
        #ax.set_title(self.station + ' ' + self.coord_str + '\n' + m_str + ' ' + h_str)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(135)
        
        if path:
            print('Saving polar plots...')
            print(path)
            plt.savefig(path, bbox_inches='tight')
        plt.show()
        
    def make_radial_plot_path(self, fun, station, roms_dataset:str, var1, var2=None, polar_coordinates='wind', h=None, m=None):
        """
        Generate a path for the radial plots.
        
        Parameters
        ----------
        fun : str
            Function name used.
        station : str
            Station ID
        var1 : str
            Name of a variable in the df_station DataFrame. 
        var2 : str, optional
            Name of a variable in the df_station DataFrame. The default is ''.
        roms_dataset : str
            Specify the tye of data used, e.g., 'hindcast_NORA3ERA5'.
        polar_coordinates : str, optional
            Specify whether to use wind or wave data as coordinates. The 
            default is 'wind'.
        h : int
            Horizon
        m : int
            Ensemble member

        Returns
        -------
        path : str
            Path to the pressure figure.

        """
        # roms_dataset: 'hindcast_NORA3', 'hindcast_ERA5', 'Operational'
        # The same as in correct_roms_using_radial_plots.py
        
        var2_str = ''
        h_str = ''
        m_str = ''
    
        if var2:
            var2_str = '_' + hlp.crop_strings(var2)
        
        if h:
            h_str = '_' + str(h)

        if type(m)==int:
            if (m < 51) and (m >= 0):   
                m_str = '_m' + str(m)
            elif m < -1:  # When counting from the last position in a vector
                m = 52 + m
                m_str = '_m' + str(m)
        
        new_dir = (
            self.data_dir 
            + '/radial_plots'
            + '/' + fun
            + '/' + hlp.crop_strings(var1) 
            + var2_str
            + '/radial_plot_'
            + polar_coordinates
            ) 
        
        hlp.make_dir(new_dir)
        
        path = (
            new_dir
            +'/radial_plot_'
            + polar_coordinates
            + '_' + roms_dataset
            + '_' + hlp.crop_strings(station)
            + '_' + fun
            + '_' + hlp.crop_strings(var1)
            + '_' + hlp.crop_strings(var2)
            + '_' + str(self.df_station.index[0]).split(' ')[0]
            + '_' + str(self.df_station.index[-1]).split(' ')[0]
            + h_str
            + m_str
            + '.png'
            )
      
        return path

        
    # ------ For the heatmaps ------    
    def weights(self):
        """Compute weights.
        
        Compute the weights used to correct the storm surge predictions as in 
        MET's operational model.

        Returns
        -------
        w : array
            Weights as computed in MET's operational model with the method 
            'linear'.

        """
        # Do we need this function???
        offset = (
            self.df_station['roms_' + self.station].shift(120)
            - (
                self.df_station['obs_' + self.station].shift(120)
                -self.df_station['tide_' + self.station].shift(120)
                )
            ) 
        # Weigh the last obs more:
        w = np.arange(np.float(offset.shape[0]))
        w = w/w.sum()           
        return w
    
    def compute_lags(self, df_with_lags, nlags:int=120):
        """Compute and add lags.
        
        Method that adds lags of a certain variable 't + 0 ', 
        e.g. 'roms - (obs - tide)' at time t, from t + 1 to t + nlags.

        Parameters
        ----------
        df_with_lags : DataFrame
            DataFrame containing 'observations' and data of another variable at
            't + 0' at a specific location.
        nlags : int, optional
            Number of lags to be computed. The default is 120.

        Returns
        -------
        df_with_lags : DataFrame
            DataFrame containing observations and roms - (obs-tide) at a 
            specific location and the lagged data.

        """
        for lag in range(1, nlags+1):
            new_col_name = 'roms - (obs - tide) t' + str(lag)
            df_with_lags[new_col_name] = df_with_lags['t + 0']  
            
        return df_with_lags
    
    def generate_df_with_lags(self):
        """Generate DataFrame wiith lags.
        
        Method that generates a DataFrame for a specific station including
        observations and lags of the variable 'roms - (obs - tide)' for 120 
        hours.

        Returns
        -------
        df_with_lags : DataFrame
            DataFrame with observations and lags of the variable 
            'roms - (obs - tide)' from t to t+120 hours.

        """
        df_with_lags = self.df_station[['obs_' + self.station, 'roms - (obs - tide)_' + self.station]]
        df_with_lags.columns = ['obs', 't + 0']        
        df_with_lags = self.compute_lags(df_with_lags, nlags=120)        
        return df_with_lags
     
    def group_df(self, df_with_lags, step):     
        
        # Do we need this?
        bins = np.arange(0, np.ceil(df_with_lags.obs.max()), step)
        labels = np.arange(1, len(bins))  # Num labels must be one fewer than num bin edges
        df_with_lags['binned'] = pd.cut(
            df_with_lags['obs'], 
            bins=bins, 
            labels=labels
            )
        self.df_grouped = df_with_lags.groupby(pd.cut(df_with_lags['obs'], bins=bins)).mean()

        
    def prepare_kyststasjon_data(self):
        # TODO: Investigate how to use the data prepared with prepare_df.py or read_kyststasjoner.py
        dt_start=dt.datetime(2018, 1, 27, 12, 0, 0, tzinfo=dt.timezone.utc)
        dt_end=dt.datetime(2020, 12, 31, 12, 0, 0, tzinfo=dt.timezone.utc)
        data_dir = '/lustre/storeB/project/fou/hi/stormsurge_eps/2dEPS_archive'
        hz = 120-11  # 5 days - 12 hr to fill gap between each prediction + 1 hr since file has 121 days
        
        r = ReadKyststasjoner(dt_start, dt_end, data_dir)

        path_list = r.make_path_list()
        

        l = len(path_list)
        self.stormsurge_corrected = np.full([l*12, hz], fill_value=np.NaN)
        self.delta = np.full([l*12, hz], fill_value=np.NaN)
        self.tide = np.full([l*12, hz], fill_value=np.NaN)
        self.observed = np.full([l*12, hz], fill_value=np.NaN)
        
        for i in range(l):
            try:
                data_file_i = xr.open_dataset(path_list[i])
                station_nr = r.get_station_id(data_file_i, self.station)
                for j in range(12):  # For each file, fill in next 12 hours
                    stormsurge_corrected_i =(
                        data_file_i['stormsurge_corrected']
                        .isel(
                            {
                                'station':station_nr, 
                                'time':slice(j, j + hz), 
                                'ensemble_member':-52, 
                                'dummy':0}
                            )
                        .values
                    )
                    
                    self.stormsurge_corrected[i * 12 + j, :] = stormsurge_corrected_i
                    
                    observed_i =(
                        data_file_i['observed']
                        .isel(
                            {
                                'station':station_nr, 
                                'time':slice(j, j + hz),
                                'dummy':0
                                }
                            )
                        .values
                    )
        
                    self.observed[i * 12 + j, :] = observed_i
                    
                    tide_i =(
                        data_file_i['tide']
                        .isel(
                            {
                                'station':station_nr, 
                                'time':slice(j, j + hz), 
                                'dummy':0
                                }
                            )
                        .values
                    )
                    
                    self.tide[i * 12 + j, :] = tide_i
                    
                    self.delta[i * 12 + j, :] = stormsurge_corrected_i - (observed_i - tide_i)      
            except:
                # Keep nans in the files
                #print('File ', path_list[i], ' is not available.' )
                pass
    
    def bin_data_for_error_heatmap(self):  
        """
        Generate bins for each horizon using variable 'observed-tide'.

        Returns
        -------
        None.

        """
        hz = 120-11
        bins = np.arange(
            np.round(np.nanquantile(self.observed - self.tide, 0), 1), 
            np.round(np.nanquantile(self.observed -self.tide, 1), 1),
            0.2
            )
        labels = np.arange(1, len(bins))
        for h in range(hz):
            df_h = np.hstack([
                (self.observed[:, h] - self.tide[:, h])[:, np.newaxis],  # TODO: check definition of self.observed, self.tide, etc
                self.delta[:, h][:, np.newaxis]
                ])
            df_h = pd.DataFrame(
                data=df_h, 
                columns = ['observed-tide', 't_' + str(h)]
                )
            df_h['binned'] = pd.cut(
                df_h['observed-tide'],
                bins=bins, 
                labels=labels
                )
            df_grouped_h = (
                df_h
                .groupby(
                    pd.cut(
                        df_h['observed-tide'], 
                        bins=bins
                        )
                    )
                .mean()
                )
                
            df_grouped_h = df_grouped_h.drop(['observed-tide'], axis=1)
            if h==0:
                self.df_grouped = df_grouped_h['t_' + str(h)]
            else:
                self.df_grouped = pd.concat(
                    [
                        self.df_grouped, 
                        df_grouped_h['t_' + str(h)]
                        ], 
                    axis=1
                    )
                
                
    
    def bin_data_for_error_boxplot(self):
        """
        Generate bins for each horizon in [0, 48, 96] using variable 'observed'.

        Returns
        -------
        None.

        """
        bins = np.arange(
            np.round(np.nanquantile(self.observed-self.tide, 0)*2)/2, 
            np.round(np.nanquantile(self.observed-self.tide, 1)*2)/2 + 0.1, 
            0.5
            )
        labels = np.arange(1, len(bins))  # Num labels must be one feewer than num bin edges
        for h in [0, 48, 96]:
            hour = np.ones([self.observed.shape[0], 1])*h
            df_h = np.hstack([
                (self.observed[:, h] - self.tide[:, h])[:, np.newaxis], 
                self.delta[:, h][:, np.newaxis], 
                hour
                ])
            df_h = pd.DataFrame(
                data=df_h, 
                columns = [
                    'observed', 
                    'stormsurge_corrected - (obs-tide) [m]', 
                    'horizon (hr)'
                    ]
                )
            # TODO; this function bins the data according to 'observed' whereas the other function uses 'observed-tide')
            df_h['binned'] = pd.cut(
                df_h['observed'], 
                bins=bins, 
                labels=labels
                )
            
            if h==0:
                self.df_long = df_h
            else:
                self.df_long = pd.concat([self.df_long, df_h], axis=0)
        
        self.df_long = self.df_long.drop(['observed'], axis=1) 
        
        bins_str = bins.astype(str)
        self.x_tick_labels = []
        for i in range(len(bins_str)-1):
            label_i = '[' + bins_str[i] + ', ' + bins_str[i+1] + ')'
            self.x_tick_labels.append(label_i)
        
    
    def error_heatmap(self):
        """
        Generate heatmaps of the error.

        Returns
        -------
        None.

        """
        self.bin_data_for_error_heatmap()
        abs_max=np.max([
            np.abs(np.quantile(self.df_grouped, 0.99)), 
            np.abs(np.quantile(self.df_grouped, 0.01))
            ])
        plt.figure()
        sns.heatmap(self.df_grouped.iloc[::-1], cmap='coolwarm', vmax=abs_max, vmin=-abs_max)
        plt.title('(stormsurge - biasMET) - (observed - tide)')
        plt.savefig(
            self.data_dir 
            + '/error_heatmap'
            + '/error_heatmap_'
            + hlp.crop_strings(self.station)
            + '.png'
            )
        plt.show()

    
    def error_boxplot(self):
        """
        Generate boxplot of the error.

        Returns
        -------
        None.

        """
        self.bin_data_for_error_boxplot()
        plt.figure()
        sns.boxplot(x='binned', hue='horizon (hr)', y='stormsurge_corrected - (obs-tide) [m]', data=self.df_long)
        nticks = len(self.x_tick_labels)
        xticks = list(np.arange(nticks))
        plt.xticks(xticks,  self.x_tick_labels) # ['(-1, -0.5]', '(-0.5, 0]', '(0, 0.5]', '(0.5, 1]', '(1, 1.5]'])
        plt.xlabel('observed [m]')
        plt.title(self.station)
        plt.savefig(
            self.data_dir 
            + '/error_boxplot'
            + '/error_boxplot_' 
            + hlp.crop_strings(self.station)
            + '.png'
            )
        plt.show()

        
        
 

class EDAMaps():
    
    def __init__(self, preprocessed_input, stations, data_dir): 
        self.pp = preprocessed_input
        self.stations = stations
        self.data_dir = data_dir
        
    
    def get_station_lon_lat(self, station_names:str):
        """Get latitude and longitude.
        
        Get station latitude and longitude from station_metadata.csv. 
        station_metadata.csv was generated from the WAM3 NORA3-ERA5 station
        file using station_metadata_to_csv.py.

        Parameters
        ----------
        station_name : str
            Station name (stationid), either a feature or a label station.

        Returns
        -------
        lat : float
            Latitude of the station.
        lon : float
            Longitude of the station

        """
        # TODO: update same function in preprocessor.py so that it reads a list of stations instead of a single station
        # Check if returning an array instead of a value does not cause any issues
        
        station_metadata = pd.read_csv('../data_loader/station_metadata.csv')
        subset = (
            station_metadata
            .loc[station_metadata['stationid'].isin(station_names)]
            )
        lats = subset['latitude_station'].to_numpy()
        lons = subset['longitude_station'].to_numpy()
        return lats, lons
    
    def scatter_map(self, var1:str='obs', var2:str='msl'): 
        """Make scatter map.
        
        Make and save a scatter map of the correlation between the variables 
        var1 and var2.

        Parameters
        ----------
        var1 : str, optional
            One of the variables used to compute the correlations. The default 
            is 'obs'.
        var2 : str, optional
            The other variable used to compute the correlations. The default is 
            'msl'.

        Returns
        -------
        None.

        """
        correlations = np.full([len(self.stations)], np.nan)
        count = 0
        for s in self.stations:
            col1 = var1 + '_' + s
            col2 = var2 + '_' + s
            correlations[count] = (
                self.pp.df_features_and_labels[[col1, col2]]
                .corr()
                .iloc[0,1]
                )
            count = count + 1
        
        lats, lons = self.get_station_lon_lat(self.stations)
        
        # Make a Mercator map of the data using Cartopy
        #plt.figure(figsize=(8, 6))
        plt.figure()
        ax = plt.axes(projection=ccrs.Mercator())
        # Plot the air temperature as colored circles and the wind speed as vectors.
        plt.scatter(
            lons,
            lats,
            c=correlations,
            s=100,
            cmap='Spectral_r',
            vmin=-1,
            vmax=1,
            transform=ccrs.PlateCarree(),
        )
        plt.colorbar().set_label('corr' + '(' + var1 + ', ' + var2 + ')')
        ax.coastlines()
        plt.savefig(
            self.data_dir 
            + '/scatter_maps'
            + '/scatter_map'
            + '_' + 'corr'
            + '_' + var1
            + '_' + var2
            + '.png'
            )

        plt.tight_layout()
        plt.show()

    
if __name__ == '__main__':
    import datetime as dt
    import sys
    import os
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
    sys.path.insert(2, os.path.join(sys.path[1], '..'))
    #from ml_models.data_loader.preprocessor import PreprocessInput
    
    # data_dir = '../figures/eda'
    data_dir = (
        '/lustre/storeB/project/IT/geout/'
        +'machine-ocean/workspace/paulinast/storm_surge_results/eda'
        )
                
# =============================================================================
# 
# 
#     ###########################################################################  
#     #          
#     # ------ Print and plot data ------
#     #
#     ###########################################################################
# 
#     feature_stations = hlp.get_all_station_names()
#     label_stations=['NO_OSC']
#     horizon = [1]   
#         
#     datetime_start_hourly=dt.datetime(2002, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
#     datetime_end_hourly=dt.datetime(2019, 12, 31, 0, 0, 0, tzinfo=dt.timezone.utc)
#     
# 
#       
#     feature_vars = ['obs', 'tide', 'roms', 'msl', 'u10', 'v10', 'wind_speed', 'wind_dir', 'swh', 'mwd']
#     label_vars = ['roms - (obs - tide)']
#     # TODO: Add bias to feature vars 
# 
#     prep = PrepareDataFrames(
#         feature_vars = feature_vars,
#         label_vars = label_vars,
#         horizon = horizon,
#         feature_stations = feature_stations,
#         label_stations = label_stations,
#         datetime_start_hourly = datetime_start_hourly,
#         datetime_end_hourly = datetime_end_hourly,
#         use_station_data = True,
#         run_on_ppi = True,
#         correct_roms=True,
#         new_station_loc=True,
#         era5_forcing=False
#         )
#         
#     df = prep.prepare_features_labels_df()
#     for station in feature_stations:
#         try:
#             eda = EDA(station, data_dir, df)
#             eda.summary()
#             eda.count_missing_data()
#             eda.plot_series()
#             eda.plot_distribution()
#             eda.norm_parameters()
#             eda.correlation_matrix(triangular=False)
#             eda.pairplot()  
# # =============================================================================
# #             #eda.radialplot_wind(function_name='count',  var1='roms', var2='(obs - tide)')
# #             for var1 in ['(roms - biasMET)']: #['roms', 'msl', 'u10', 'v10', 'wind_speed', 'wind_dir']:
# #                 for fun in ['rmse']: #['mean', 'std', 'corr']:
# #                     eda.radialplot_wind(function_name=fun,  var1=var1, var2='(obs - tide)')        
# # =============================================================================
#                     
#         except:
#             print('An error occurred while generating results for station ', station)
# =============================================================================




# =============================================================================
#     ###########################################################################  
#     #          
#     # ------ Make heatmaps and boxplots ------
#     #
#     ###########################################################################
# 
#     
#     feature_stations = hlp.get_norwegian_station_names()
#     feature_stations.remove('NO_MSU')  # This station is not in the kyststasjoner file    
#     label_stations=['NO_OSC']
#     
#     feature_vars = ['obs', 'tide', 'roms']
#     label_vars = ['roms - (obs - tide)']
#     # TODO: Add bias to feature vars 
#     
#     datetime_start_hourly=dt.datetime(2000, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
#     datetime_end_hourly=dt.datetime(2019, 12, 31, 0, 0, 0, tzinfo=dt.timezone.utc)
#     
#     horizon = [12] # TODO: Check!! Do we need this argument in the class?? Seems like it's not used
#     
#     prep = PrepareDataFrames(
#         feature_vars = feature_vars,
#         label_vars = label_vars,
#         horizon = horizon,
#         feature_stations = feature_stations,
#         label_stations = label_stations,
#         datetime_start_hourly = datetime_start_hourly,
#         datetime_end_hourly = datetime_end_hourly,
#         use_station_data = True,
#         run_on_ppi = True,
#         correct_roms=True,
#         new_station_loc=True,
#         era5_forcing=False
#         )
#     
#     
#     df = prep.prepare_features_labels_df()
#     
#     for station in feature_stations:
#         print(station)
#         eda = EDA(station, data_dir, df)
# 
#         eda.prepare_kyststasjon_data()
#         eda.error_heatmap()
#         eda.error_boxplot()
# =============================================================================

# =============================================================================
    ###########################################################################  
    #          
    # ------ Make radial plots of the error ------
    # ------ Plot the error against pressure ------
    #
    ###########################################################################

    stations = hlp.get_norwegian_station_names()
    stations.remove('NO_MSU')  # This station is not in the kyststasjoner file
    stations = ['NO_BGO', 'NO_OSC']
    stations = ['NO_OSC']
    
    #stations = hlp.get_all_station_names()
    
    len_stations = len(stations)
    # stations = ['NO_AES', 'NO_OSC', 'NO_BGO']
    variables = [
    'obs', 'tide_py', 'roms', 'msl', 'u10', 'v10', 
    'stormsurge_corrected', 'bias', 'wind_speed', 'wind_dir',
    'roms - (obs - tide_py)', '(roms - biasMET)', '(obs - tide_py)',
    'swh', 'mwd'
    ]

    horizon = 120
    #horizon = 1
    # feature_stations=['NO_OSC'] #, 'SW_2111', 'NL_Westkapelle']

    datetime_start_hourly=dt.datetime(2000, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    #datetime_end_hourly=dt.datetime(2019, 12, 31, 0, 0, 0, tzinfo=dt.timezone.utc)
    datetime_end_hourly=dt.datetime(2021, 3, 31, 0, 0, 0, tzinfo=dt.timezone.utc)
    
    # Iterate over station list slices because of memory issues 
    # Make slices of 6 stations
    
    for n in range( int( np.ceil(len_stations/6) ) ):
        station_list_slice = stations[n*6: n*6 + 6]
        print('station list slice: ', station_list_slice)
    
        prep = PrepareDataFrames(
            variables = variables,
            horizon = horizon+1,
            stations = station_list_slice,
            ensemble_members = [-1],
            datetime_start_hourly = datetime_start_hourly,
            datetime_end_hourly = datetime_end_hourly,
            use_station_data = True,
            run_on_ppi = True,
            new_station_loc=True,
            era5_forcing=False,
            data_from_kartverket=False
            )
        df = prep.prepare_features_labels_df()
        
        var2 = '(obs - tide_py)'
        print(df.columns)
        print(df.head())
        for station in station_list_slice:
            print(station)
            eda = EDA(station, data_dir, df)
            #['(roms - biasMET)', 'roms', 'msl', 'u10', 'v10', 'wind_speed', 'wind_dir']:
                
        
            for fun in ['std', 'mean']: #['rmse', 'mean', 'std', 'corr']:
                for var1 in ['(roms - biasMET)', 'stormsurge_corrected']:
                    if var1 == '(roms - biasMET)':
                        horizon_loop = 0   # Otherwise it crashes in line 643
                    else:
                        horizon_loop = 120
                    for h in range(horizon_loop + 1):
                        #eda.radialplot_wind(function_name=fun,  var1=var1, var2='(obs - tide_py)')  
                        sorted_df, aggregated_df, c, theta, rho = eda.radial_plot_data(
                            function_name=fun,  
                            var1=var1, 
                            var2=var2, 
                            radial_coord_name='wind_speed', #'swh', #'wind_speed',  # swh
                            angular_coord_name='wind_dir',  #'mwd', # wind_dir',  # mwd
                            h=h,
                            min_obs_bin=5
                            ) 
                        
                        path_rp = eda.make_radial_plot_path(
                            fun=fun, 
                            station=station, 
                            roms_dataset='hindcast_NORA3ERA5', 
                            var1=var1, 
                            var2=var2, 
                            polar_coordinates='wind', 
                            h=h,
                            m=None
                            )
        
                        #if var1 =='stormsurge_corrected':
                        abs_max = np.abs(np.max(np.nanquantile(c, 0.95)))
                        abs_min = np.abs(np.min(np.nanquantile(c, 0.05)))
                        vmax = max(abs_max, abs_min)
                        vmin = - vmax
                        
                        eda.radial_plot(
                            c, 
                            np.transpose(theta), 
                            np.transpose(rho), 
                            fun, 
                            var1, 
                            var2, 
                            vmax=vmax,
                            vmin=vmin,
                            path=path_rp,
                            h=h
                            )
                        
                        path_pp = eda.make_pressure_plot_path(
                            fun=fun, 
                            station=station, 
                            roms_dataset='hindcast_NORA3ERA5',
                            var1=var1, 
                            var2=var2,
                            h=h
                            )
                        eda.plot_error_against_pressure(
                            function_name=fun, 
                            var1=var1,
                            var2=var2,
                            h=h,
                            path=path_pp
                            )
                        
# =============================================================================
#     ###########################################################################  
#     #          
#     # ------ Make plots of pressure field ------
#     #
#     ###########################################################################
#     stations = hlp.get_norwegian_station_names()
#     stations.remove('NO_MSU')  # This station is not in the kyststasjoner file
#         
#     variables = [
#         'obs', 'tide_py', 'roms', 'msl', 'u10', 'v10', 
#         'stormsurge_corrected', 'bias', 'wind_speed', 'wind_dir',
#         'roms - (obs - tide_py)'
#         ]
#     horizon = 121
#     # feature_stations=['NO_OSC'] #, 'SW_2111', 'NL_Westkapelle']
# 
# 
#     datetime_start_hourly=dt.datetime(2000, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
#     #datetime_end_hourly=dt.datetime(2019, 12, 31, 0, 0, 0, tzinfo=dt.timezone.utc)
#     datetime_end_hourly=dt.datetime(2021, 3, 31, 0, 0, 0, tzinfo=dt.timezone.utc)
#     
#     prep = PrepareDataFrames(
#         variables = variables,
#         horizon = horizon,
#         stations = stations,
#         ensemble_members = [-1],
#         datetime_start_hourly = datetime_start_hourly,
#         datetime_end_hourly = datetime_end_hourly,
#         use_station_data = True,
#         run_on_ppi = True,
#         new_station_loc=True,
#         era5_forcing=False,
#         data_from_kartverket=True
#         )
#             
#     df = prep.prepare_features_labels_df()
#     
#     var2 = ''
#     #print(df.columns)
#     print('df.head(): ', df.head())
#     for station in stations:
#         print(station)
#         eda = EDA(station, data_dir, df)
#         for fun in ['mean', 'std']:
#             for var1 in ['msl']:
#                 sorted_df, aggregated_df, c, theta, rho = eda.radial_plot_data(
#                     function_name=fun,  
#                     var1=var1, 
#                     var2=var2, 
#                     radial_coord_name='wind_speed',  # swh
#                     angular_coord_name='wind_dir'  # mwd
#                     ) 
#                 
# # =============================================================================
# #                 path_rp = eda.make_radial_plot_path(
# #                     fun, 
# #                     station, 
# #                     var1, 
# #                     var2, 
# #                     roms_dataset='hindcast_NORA3ERA5', 
# #                     polar_coordinates='wind'
# #                     )
# # 
# #               
# #                 vmax = np.abs(np.max(np.nanquantile(c, 0.95)))
# #                 vmin = np.abs(np.min(np.nanquantile(c, 0.05)))
# #                 
# #                 eda.radial_plot(
# #                     c, 
# #                     np.transpose(theta), 
# #                     np.transpose(rho), 
# #                     fun, 
# #                     var1, 
# #                     var2, 
# #                     vmax=vmax,
# #                     vmin=vmin,
# #                     path=path_rp
# #                     )
# #                 
# # =============================================================================
#                 path_pp = eda.make_pressure_plot_path(
#                     fun=fun, 
#                     station=station, 
#                     roms_dataset='hindcast_NORA3ERA5',
#                     var1=var1, 
#                     var2=var2
#                     )
#                 eda.plot_error_against_pressure(
#                     function_name=fun, 
#                     var1=var1,
#                     var2=var2,
#                     path=path_pp
#                     )
# 
# 
# =============================================================================
