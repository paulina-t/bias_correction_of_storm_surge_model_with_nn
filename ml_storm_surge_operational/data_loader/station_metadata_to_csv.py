import xarray as xr

"""
Generate station metadata file containing stationid, latitude, longitude, 
latitude_station, and longitude_station as in the ERA5 station files generated 
by Martin Lilleeng Sætra.

Run this file if `station_metadata.csv` is not in this folder.
"""

data_dir_station='/home/paulinast/Desktop/data/wam3_nora3era5/martinls'

# Open one of the files <- the results do not depend on the chosen variable.
station_file = xr.open_mfdataset(
            data_dir_station 
            + '/aggregated_era5_msl.nc'
            )

station_data = station_file.stationid.to_dataframe()
station_data['latitude_station'] = station_file.latitude_station.to_pandas()
station_data['longitude_station'] = station_file.longitude_station.to_pandas()

station_data.to_csv('station_metadata.csv')