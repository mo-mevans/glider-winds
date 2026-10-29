# name: post_process
# date: 27/07/2026
# author: Toby Alexander

# contributor: Matt Evans
# date: 01/10/2026

# description: 
# Post processing for glider data analysis. This script reads in the results of the non-linear fitting of the glider data, and performs further analysis and visualisation of the results.
# Includes data validation against soundings and ERA5 reanalysis data.

# required modules:
#   - Python 3.6+ (created with 3.10.19)
#   - cartopy
#   - cdsapi (https://cds.climate.copernicus.eu/how-to-api) for downloading ERA5 reanalysis data
#   - matplotlib
#   - numpy
#   - pandas
#   - sklearn (https://pypi.org/project/scikit-learn/) for DBSCAN clustering
#   - xarray (https://pypi.org/project/xarray/)


# # 0.1 Imports
import cartopy.crs as ccrs 
import cartopy.feature as cfeature
import cdsapi   # for downloading ERA5 reanalysis data
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
from sklearn.cluster import DBSCAN  # DBSCAN - Density-Based Spatial Clustering of Applications with Noise
import xarray as xr     # working with netCDF files
from datetime import timedelta


# # 0.2 Global Constants and Configuration

# Global Constants

# Create a dictionary of sonde launch sites with station number, lat and lon
# Dicionary of UK radiosondes with [station_no, lat, lon] values
radiosondes_uk = {
    "Camborne": ["03808", 50.218, -5.326],
    "Herstmonceux": ["03882", 50.891, 0.317],
    "Watnall": ["03554", 53.006, -1.250],
    "Albemarle": ["03238", 55.019, -1.878],
    "Castorbay": ["03918", 54.501, -6.339],
    "Larkhill": ["03743", 51.202, -1.806],
    "Lerwick": ["03005", 60.139, -1.185]
}

# Set lat and long of radiosonde launch site
UK_SONDE = "Herstmonceux"   # CHNAGE THIS for the UK sonde site of interest
STATION_SONDE = radiosondes_uk[UK_SONDE][0]
LAT_SONDE = radiosondes_uk[UK_SONDE][1]
LON_SONDE = radiosondes_uk[UK_SONDE][2]


# Set Timestamp of radiosonde launch
TIMESTAMP_SONDE = pd.Timestamp('2026-09-25 23:15:00') # CHANGE THIS for the UK sonde launch time of interest
# TIMESTAMP_SONDE = pd.Timestamp('2026-09-26 00:00:00')

# Derive decimal time of radiosonde launch from timestamp
TIME_SONDE = int(TIMESTAMP_SONDE.hour) + (int(TIMESTAMP_SONDE.minute) / 60)



# # 1. Load in fitting results and radiosonde data

# Glider File Configuration
gliderroot = "./"
gliderfile = "nonlinfit_results_FLRD029BF.csv" # CHANGE THIS for the glider file of interest
gliderpath = gliderroot + gliderfile
print(f"file id: {gliderfile}")


# Radiosonde File Configuration
sonderoot = "./"
sondeymd = TIMESTAMP_SONDE.strftime('%Y%m%d')
sondehr = TIMESTAMP_SONDE.strftime('%H')


# radiosonde sample selected
sondefile = f"{sondeymd+sondehr}-{STATION_SONDE}.csv"
sondepath = sonderoot + sondefile


def load_data(filepath):
    '''
    Read in a .csv file and store in a pandas DataFrame.

    :param str filepath: Filepath of the .csv file location

    :return: A pandas DataFrame containing the contents of the injested .csv file
    :rtype: pandas.DataFrame
    '''

    df = pd.read_csv(filepath)

    return df



# # 1.2 Pre-process Radiosonde Data

def preprocess_sonde(df):
    '''
    Pre-process the radiosonde DataFrame.

    :param pandas.DataFrame df: Raw radiosonde data

    :return: Radiosone wind speed, altitude, wind direction and pressure data selected below 2000 metres.
    :rtype: pandas.DataFrame
    '''

    df = df.drop(index=0) # drop first entry - always seems to be incomplete

    dict_preproc = dict.fromkeys(['magU', 'alt', 'direction', 'pres'])
    sonde_cols = ["wind speed_m/s", "geopotential height_m", "wind direction_degree", "pressure_hPa"]

    # Select only data below 2km altitude, as this is the range of interest for the glider data.
    mask_2km = df["geopotential height_m"] <= 2000
    for k, i in zip(dict_preproc.keys(), sonde_cols):
        df_param = df[i].values.astype(float)
        dict_preproc[k] = df_param[mask_2km]


    df_preproc = pd.DataFrame(dict_preproc)

    return df_preproc


# # 1.3 Optional: Inspect how the Period and zdot vary over each thermal

def stats_period_zdot(df):
    """
    Investigate how period and zdot change over each thermal; returns mean values and standard deviations.

    :param pandas.DataFrame df: Glider estimated parameters for each bin within each thermal

    :return: mean values and standard deviations for the period and zdot for each thermal
    :rtype: pandas.DataFrame
    """
    results = []

    for (flight_id, therm_idx), thermal_df in df.groupby(['id', 'thermidx']):
        # separate lists of indices for each different thermal
        indices = thermal_df.index.tolist()
        # obtaining the omega values, grouping by the flight ID and thermal index
        omega = df.loc[indices, 'omegaest'].values
        P = abs(2 * np.pi / omega)
        # obtaining zdot values, grouping by the flight ID and thermal index
        zdot = df.loc[indices, 'zdot'].values

        results.append({
            'id': flight_id,
            'thermidx': therm_idx,
            'nbins': len(indices),
            'P_mean': P.mean(),
            'zdot_mean': zdot.mean(),
            'P_std': P.std(),
            'zdot_std': zdot.std()
        })

    return pd.DataFrame(results)



# # 2. Estimate upper bound of uncertainty in wind estimates from inter-bin variation

def calculate_upper_bound_errors(df):
    '''
    Calculate weighted upper bound errors for discrete adjacent bin pairs
    within each thermal. Assumes an even number of bins per thermal.
    Effectively calculates the disagreement between adjacent bins
    in the same thermal soaring window, in an attempt to quantify the uncertainty in the wind estimates.
    '''
    for col in ['u_upper', 'v_upper', 'magU_upper', 'dbUdeg_upper']:
        df[col] = np.nan
    
    params = [
        ('uest',  'ustd',    'u_upper'),
        ('vest',  'vstd',    'v_upper'),
    ]

    # Group by flight ID and thermal index, then iterate through each group.
    for (flight_id, therm_idx), thermal_df in df.groupby(['id', 'thermidx']):
        indices = thermal_df.index.tolist()
        print(indices)
        for i in range(0, len(indices) - 1, 2):
            idx1 = indices[i]
            idx2 = indices[i + 1]
            row1 = df.loc[idx1]
            row2 = df.loc[idx2]

            for est_col, std_col, out_col in params:
                print(est_col, std_col, out_col)
                print(f"est_col: {est_col}")
                print(row1[est_col])
                print(row2[est_col])
                # find the difference of the estimated wind speeds between the neighbouring rows
                delta = abs(row1[est_col] - row2[est_col])
                # determining the error of the difference between the two row estimated winds
                sigma_upper = delta / np.sqrt(2)  # Assuming n=2 for two measurements        

                df.at[idx1, out_col] = sigma_upper
                df.at[idx2, out_col] = sigma_upper

    return df


def propagate_wind_uncertainty(df):
    '''
    Propagate errors to calculate glider wind magnitude and direction error estimates.

    :param pandas.DataFrame df: Glider DataFrame with u and v measurements and std devs:
        from fitting and interbin methods

    :return: Interbin error and a maximum error in the glider wind magnitude and direction measurements.
        Maximum ('combined') error obtained by effectively taking the maximum of either the model fitting or interbin error.
    :rtype: pandas.DataFrame
    '''

    # propagate errors to obtain magnitude and direction estimates
    df['magU_upper'] = np.sqrt(df['uest']**2 * df['u_upper']**2 + df['vest']**2 * df['v_upper']**2) / df['magU']
    df['dbUdeg_upper'] = np.sqrt(df['vest']**2 * df['u_upper']**2 + df['uest']**2 * df['v_upper']**2) / df['magU']**2

    # Take the maximum of the standard deviation from fitting and interbin variation to get a combined error estimate for each parameter.
    df['u_combined']    = np.maximum(df['ustd'],    df['u_upper'])
    df['v_combined']    = np.maximum(df['vstd'],    df['v_upper'])
    df['magU_combined'] = np.sqrt(df['uest']**2 * df['u_combined']**2 + df['vest']**2 * df['v_combined']**2) / df['magU']
    df['dbUdeg_combined'] = np.sqrt(df['vest']**2 * df['u_combined']**2 + df['uest']**2 * df['v_combined']**2) / df['magU']**2

    return df


def plot_combined_wind_std_distribution(df, u_col='u_combined', v_col='v_combined',
                              bins=20, value_range=(0, 3), title='Distribution of standard deviations of wind estimates'):
    '''
    Plot the (maximum of either the model fitting or interbin bin variation) glider u and v wind std devs.
    '''

    plt.figure(figsize=(10, 6))
    plt.hist(df.u_combined, bins=bins, range=value_range, color='blue', alpha=0.7, label=u_col)
    # plt.hist(df.magU_upper, bins=20, range=[0, 3], color='orange', alpha=0.7, label='magU_upper')
    plt.hist(df.v_combined, bins=bins, range=value_range, color='red', alpha=0.7, label=v_col)
    plt.xlabel('std (m/s)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    # plt.suptitle(f"ICAO24: {fid}")
    plt.grid(True)
    plt.show()


def plot_mag_wind_std_cdf(df):
    '''
    Construct CDF plots for the model standard deviations of the wind estimates and the interbin variations.
    '''
    fig, axes = plt.subplots(1, 2)

    for ax, col, label in zip(axes, ['stdmagU', 'magU_upper'], ['Fit error', 'Upper bound']):
        sorted_vals = np.sort(df[col].dropna())
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf)
        ax.set_xlabel('std (m/s)')
        ax.set_ylabel('CDF')
        ax.set_title(label)
        ax.grid(True)
    
    plt.show()


def filter_clean_wind_bins(df, stdmagU_max=1.0, magU_upper_max=2.5):
    '''
    Retain only bins that pass the standard deviation wind thresholds:

    1. A standard deviation of 1 m/s or greater from the model fitting was characterised by poor fits.
    2. Interbin variation of 2.5 m/s or greater was considered unphysical when compared to the
        wind shear observed by the colocated radiosonde data.

    :param pandas.DataFrame df: Unfiltered glider wind data.
    :param float stdmagU_max: model fitting wind standard deviation maximum threshold.
        Defaults to 1.0.
    :param float magU_upper_max: interbin standard deviation maximum threshold.
        Defaults to 2.5.

    :return: Glider wind estimates which are below the model fit and interbin standard deviation wind limits.
    :rtype: pandas.DataFrame
    '''

    mask = (df['stdmagU'] < stdmagU_max) & (df['magU_upper'] < magU_upper_max)

    return df.loc[mask].copy()


def plot_qc_diagnostics(df, df_clean, fit_col='stdmagU', upper_col='magU_upper', 
                         alt_col='zhat', flight_col='id', thermal_col='thermidx'):
    '''
    Plot quality control diagnostics for the wind estimates, comparing retained and discarded bins based on the cleaning criteria.
    '''

    discarded = df[~df.index.isin(df_clean.index)]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Discarded bins by flight
    ax = axes[0, 0]
    flight_counts = df.groupby(flight_col).size()
    flight_counts_clean = df_clean.groupby(flight_col).size()
    flight_counts_discarded = flight_counts - flight_counts_clean.reindex(flight_counts.index, fill_value=0)
    x = np.arange(len(flight_counts))
    ax.bar(x, flight_counts, label='Retained', color='steelblue')
    ax.bar(x, flight_counts_discarded, label='Discarded', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(flight_counts.index, rotation=45, ha='right')
    ax.set_xlabel('Flight ID')
    ax.set_ylabel('Number of bins')
    ax.set_title('Discarded bins by flight')
    ax.legend()

    # 2. Discarded bins by thermal
    ax = axes[0, 1]
    therm_counts = df.groupby(thermal_col).size()
    therm_counts_clean = df_clean.groupby(thermal_col).size()
    therm_counts_discarded = therm_counts - therm_counts_clean.reindex(therm_counts.index, fill_value=0)
    x = np.arange(len(therm_counts))
    ax.bar(x, therm_counts, label='Retained', color='steelblue')
    ax.bar(x, therm_counts_discarded, label='Discarded', color='tomato')
    ax.set_xlabel('Thermal index')
    ax.set_ylabel('Number of bins')
    ax.set_title('Discarded bins by thermal')
    ax.legend()

    # 3. Fit error vs upper bound (retained vs discarded)
    ax = axes[0, 2]
    ax.scatter(df_clean[fit_col], df_clean[upper_col], 
               alpha=0.6, label='Retained', color='steelblue', s=20)
    ax.scatter(discarded[fit_col], discarded[upper_col], 
               alpha=0.6, label='Discarded', color='tomato', s=20)
    lim = max(df[fit_col].max(), df[upper_col].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='1:1 line')
    ax.set_xlabel('Fit error (m/s)')
    ax.set_ylabel('Upper bound (m/s)')
    ax.set_xlim(0, df[fit_col].max() * 1.05)
    ax.set_ylim(0, df[upper_col].max() * 1.05)
    ax.set_title('Fit error vs upper bound')
    ax.legend()

    # 4. Wind speed profile
    ax = axes[1, 0]
    ax.errorbar(df_clean['magU'], df_clean[alt_col], 
                xerr=df_clean[upper_col], fmt='o', alpha=0.5, 
                color='steelblue', ecolor='lightblue', capsize=2, markersize=3,
                label='Retained')
    ax.scatter(discarded['magU'], discarded[alt_col], 
               color='tomato', alpha=0.5, s=20, label='Discarded')
    ax.set_xlabel('Wind speed (m/s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Wind speed profile')
    ax.legend()

    # 5. Wind direction profile
    ax = axes[1, 1]
    ax.errorbar(df_clean['bUdeg'], df_clean[alt_col],
                xerr=df_clean['dbUdeg_upper'], fmt='o', alpha=0.5,
                color='steelblue', ecolor='lightblue', capsize=2, markersize=3,
                label='Retained')
    ax.scatter(discarded['bUdeg'], discarded[alt_col],
               color='tomato', alpha=0.5, s=20, label='Discarded')
    ax.set_xlabel('Wind direction (degrees)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Wind direction profile')
    ax.legend()

    # 6. Fit error vs upper bound ratio
    ax = axes[1, 2]
    ratio = df_clean[upper_col] / df_clean[fit_col]
    ax.hist(ratio, bins=20, color='steelblue', edgecolor='white')
    ax.axvline(1, color='tomato', linestyle='--', label='Upper = fit error')
    ax.set_xlabel('Upper bound / fit error')
    ax.set_ylabel('Frequency')
    ax.set_title('Ratio of upper bound to fit error\n(retained bins)')
    ax.legend()

    fig.tight_layout()
    plt.show()



# # 3. Visualise distribution of measurements, and cluster measurements by spatial and temporal distance

def plot_glider_locations(df, lat_col='lat0', lon_col='lon0',
                      sonde_lat=None, sonde_lon=None, sonde_time=23.25,
                      show_grid_labels=False):
    '''
    Plot the spatial distribution of wind estimates on a map.
    Optionally, include the location of a radiosonde launch.

    Colour code the wind estimate locations according to the time (of the glider flight segment/bin).
    '''
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
    ax.add_feature(cfeature.RIVERS, linewidth=0.3)
    
    margin = 0.2
    # expand extent to include sonde location
    all_lons = list(df[lon_col]) + ([sonde_lon] if sonde_lon is not None else [])
    all_lats = list(df[lat_col]) + ([sonde_lat] if sonde_lat is not None else [])
    
    lon_min = min(all_lons) - margin
    lon_max = max(all_lons) + margin
    lat_min = min(all_lats) - margin
    lat_max = max(all_lats) + margin
    # crs=PlateCarree(): interpret the four passed extent co-ordinates as long and lats in degrees
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # workaround added until a new version of Cartopy is available (for pixi env)
    # current version has troubles with adding the grid lines to lat/lon plot
    if show_grid_labels:
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')

    # t_hours = df['timestamp'] % (3600 * 24) / 3600
    t_seconds_per_day = df['timestamp'] % (3600 * 24)
    t_hours = t_seconds_per_day / 3600
    vmin, vmax = t_hours.min(), t_hours.max()

    scatter = ax.scatter(df[lon_col], df[lat_col],
                         c=t_hours, cmap='viridis', s=20, alpha=0.7,
                         vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree(),
                         label='Glider measurements')

    # optionally add radiosonde launch point
    if sonde_lat is not None and sonde_lon is not None:
        ax.scatter(sonde_lon, sonde_lat,
                   c=[sonde_time], cmap='viridis', vmin=vmin, vmax=vmax,
                   s=200, marker='+', linewidths=2,
                   transform=ccrs.PlateCarree(),
                   label='Radiosonde launch', zorder=5)
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Time (hr of day)')
    ax.set_title('Spatial distribution of wind estimates')
    ax.legend(loc='upper right')
    fig.tight_layout()
    plt.show()


def cluster_measurements(df, eps_km=50, min_samples=5, 
                          lat_col='lat', lon_col='lon'):
    '''
    Cluster measurements by spatial proximity using DBSCAN.
    eps_km: maximum distance between points in the same cluster (km)
    min_samples: minimum number of points to form a cluster

    DBSCAN performs well in handling real-world irregularly shaped clusters.
    i.e. clusters don't have to be circular or convex.
    '''
    lat_col = "lat0"
    lon_col = "lon0"
    # convert lat/lon to radians for haversine metric
    coords = np.deg2rad(df[[lat_col, lon_col]].values)
    # eps in radians (earth radius ~6371km)
    eps_rad = eps_km / 6371.0
    
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
    df['cluster'] = db.fit_predict(coords)

    # above step combines below two steps to fit the model and find cluster labels in one step
    # clustering = db.fit(coords)
    # df['cluster'] = clustering.labels_
    
    # -1 means noise (not assigned to any cluster)
    n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'].values else 0)
    n_noise = (df['cluster'] == -1).sum()
    print(f"Found {n_clusters} clusters, {n_noise} unassigned points")
    
    return df


def plot_clusters_map(df, lat_col='lat0', lon_col='lon0',
                      sonde_lat=None, sonde_lon=None, show_grid_labels=False):
    '''
    Plot the wind estimate (glider) cluster locations and colour code according to cluster number.
    Optionally plot the location of the radiosonde launch.
    '''
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
    })

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
    ax.add_feature(cfeature.RIVERS, linewidth=0.3)
    
    margin = 0.2
    # expand extent to include sonde location
    all_lons = list(df[lon_col]) + ([sonde_lon] if sonde_lon is not None else [])
    all_lats = list(df[lat_col]) + ([sonde_lat] if sonde_lat is not None else [])

    lon_min = min(all_lons) - margin
    lon_max = max(all_lons) + margin
    lat_min = min(all_lats) - margin
    lat_max = max(all_lats) + margin
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    
    if show_grid_labels:
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')

    # plot clusters
    clusters = sorted(df['cluster'].unique())
    cmap = plt.get_cmap('tab10', len(clusters))
    
    for i, cluster_id in enumerate(clusters):
        mask = df['cluster'] == cluster_id
        label = f'Cluster {cluster_id}' if cluster_id != -1 else 'Unassigned'
        color = 'grey' if cluster_id == -1 else cmap(i)
        ax.scatter(df[mask][lon_col], df[mask][lat_col],
                   color=color, label=label, s=20, alpha=0.7,
                   transform=ccrs.PlateCarree())

    # optionally add radiosonde launch point
    if sonde_lat is not None and sonde_lon is not None:
        ax.scatter(sonde_lon, sonde_lat,
                   color='black', s=200, marker='+', linewidths=2,
                   transform=ccrs.PlateCarree(),
                   label='Radiosonde launch', zorder=5)

    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
    ax.set_title('Spatial clusters of wind estimates')
    fig.tight_layout()
    plt.show()



def plot_wind_profiles(df, df_sonde, alt_col='zhat', speed_col='magU', dir_col='bUdeg',
                       speed_err_col='magU_combined', dir_err_col='dbUdeg_combined',
                       cluster_col='cluster', alt_sonde='alt', speed_sonde='magU',
                       dir_sonde='direction'):
    '''
    Plot wind speed and direction profiles for each cluster of measurements, including error bars and comparison with radiosonde data.
    '''
    clusters = sorted([c for c in df[cluster_col].unique() if c != -1])
    
    for cluster_id in clusters:
        cluster_df = df[df[cluster_col] == cluster_id].copy()

        plt.rcParams.update({
        'font.size': 14,          # default for everything
        'axes.titlesize': 14,     # subplot titles
        'axes.labelsize': 14,     # x and y labels
        'xtick.labelsize': 14,    # tick labels
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
        fig.suptitle(f'Wind profile — Cluster {cluster_id} (n={len(cluster_df)})')
        
        # wind speed
        ax1.errorbar(cluster_df[speed_col], cluster_df[alt_col],
                     xerr=cluster_df[speed_err_col],
                     fmt='s', markersize=6, capsize=3, linewidth=2,
                     color='steelblue', ecolor='k',
                     alpha=0.8)
        ax1.plot(df_sonde[speed_sonde], df_sonde[alt_sonde], label='Radiosonde (albemarle)', color='darkorange')
        ax1.set_xlabel('Wind speed (m/s)')
        ax1.set_ylabel('Altitude (m)')
        ax1.set_title('Wind speed')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)

        # wind direction
        ax2.errorbar(cluster_df[dir_col], cluster_df[alt_col],
                     xerr=cluster_df[dir_err_col],
                     fmt='s', markersize=6, capsize=3, linewidth=2,
                     color='darkorange', ecolor='k',
                     alpha=0.8)
        ax2.plot(df_sonde[dir_sonde], df_sonde[alt_sonde], label='Radiosonde (albemarle)', color='darkorange')
        ax2.set_xlabel('Wind direction (degrees)')
        ax2.set_title('Wind direction')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 360)
        ax2.set_xticks([0, 90, 180, 270, 360])
        ax2.set_xticklabels(['N', 'E', 'S', 'W', 'N'])

        fig.tight_layout()
        plt.show()


def plot_cluster_temporal_coverage(df, cluster_id=0, alt_col='zhat',
                                    cluster_col='cluster', speed_col='magU'):
    '''
    Plot:
    1. Temporal distribution of measurements in a given cluster 
    2. Scatter of altitude vs time, colour coded by wind speed
        to see temporal coverage at each height.
    '''
    cluster_df = df[df[cluster_col] == cluster_id].copy()
    cluster_df['timestamp_hr'] = cluster_df['timestamp'] % (3600 * 24) / 3600

    seconds_since_midnight = cluster_df['timestamp'] % (3600 * 24)
    cluster_df['timestamp_hr'] = seconds_since_midnight / 3600

    plt.rcParams.update({
        'font.size': 16,          # default for everything
        'axes.titlesize': 16,     # subplot titles
        'axes.labelsize': 14,     # x and y labels
        'xtick.labelsize': 14,    # tick labels
        'ytick.labelsize': 14,
        'legend.fontsize': 16,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # time distribution
    axes[0].hist(cluster_df['timestamp_hr'], bins=20, color='steelblue', edgecolor='white')
    axes[0].set_xlabel('Time of day (hr)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Temporal distribution of Cluster {cluster_id}')
    axes[0].grid(True, alpha=0.3)

    # scatter of time vs altitude to see temporal coverage at each height
    scatter = axes[1].scatter(cluster_df['timestamp_hr'], cluster_df[alt_col],
                               c=cluster_df[speed_col], cmap='viridis', s=20, alpha=0.7)
    cbar = fig.colorbar(scatter, ax=axes[1])
    cbar.set_label('Wind speed (m/s)')
    axes[1].set_xlabel('Time of day (hr)')
    axes[1].set_ylabel('Altitude (m)')
    axes[1].set_title('Altitude vs time coverage')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    print(f"Cluster {cluster_id}: {len(cluster_df)} points")
    print(f"Time range: {cluster_df['timestamp_hr'].min():.1f} - {cluster_df['timestamp_hr'].max():.1f} hr")
    print(f"Altitude range: {cluster_df[alt_col].min():.0f} - {cluster_df[alt_col].max():.0f} m")

    plt.show()



def subcluster_by_time(df, cluster_id, breaks, cluster_col='cluster'):
    '''
    Split a cluster into time-based sub-clusters using manual time breaks.
    breaks: list of hour values defining the boundaries e.g. [12.5, 16.0]
    Example: cluster_id=0, breaks=[12.5, 16.0]
    '''
    # Ensure the 'timestamp_hr' column exists
    if 'timestamp_hr' not in df.columns:
        df['timestamp_hr'] = (df['timestamp'] % (24 * 3600)) / 3600

    # Keep only the rows in the target cluster
    mask = df[cluster_col] == cluster_id

    # Preparing new 'subcluster' column in DataFrame
    df['subcluster'] = pd.Series(np.nan, index=df.index, dtype='object')

    # If the cluster is present, assign subcluster labels
    if mask.any():
        labels = [f'{cluster_id}a', f'{cluster_id}b', f'{cluster_id}c']
        df.loc[mask, 'subcluster'] = pd.cut(
            df.loc[mask, 'timestamp_hr'],
            bins=[-np.inf] + list(breaks) + [np.inf],
            labels=labels
        )

    return df


# # 4. Quantitative comparison between glider measurements and radiosonde

def derive_sonde_wind_components(df_sonde, speed_col='magU', dir_col='direction'):
    '''
    Convert wind speed and direction measured by sonde to u and v components for comparison with glider estimates.
    '''
    theta_rad = np.deg2rad(df_sonde[dir_col])  # met convention, degrees
    df_sonde['u'] = -df_sonde[speed_col] * np.sin(theta_rad)
    df_sonde['v'] = -df_sonde[speed_col] * np.cos(theta_rad)


    # reconstruct magU and direction from converted u/v and compare to original
    magU_check = np.sqrt(df_sonde['u']**2 + df_sonde['v']**2)
    dir_check = (270 - np.rad2deg(np.arctan2(df_sonde['v'], df_sonde['u']))) % 360

    print(np.allclose(magU_check, df_sonde[speed_col]))        # should be True
    print(np.allclose(dir_check, df_sonde[dir_col]))    # should be True

    return df_sonde.copy()



def haversine_km(lat1, lon1, lat2, lon2):
    '''
    Calculate the great-circle (Haversine) distance between two lat/lon points on a sphere.

    :param float lat1: glider latitude
    :param float lon1: glider longitude
    :param float lat2: radiosonde latitude
    :param float lon2: radiosonde longitude

    :return: The Haversine distance between the glider and radiosonde location.
    :rtype: float
    '''
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))



def compare_to_radiosonde(df_glider, df_sonde,
                           alt_col='alt', zhat_col='zhat',
                           u_col='uest', v_col='vest',
                           u_err_col='u_combined', v_err_col='v_combined',
                           glider_lat_col='lat0', glider_lon_col='lon0',
                           sonde_lat=None, sonde_lon=None,
                           sonde_time=None):
    '''
    Compare glider wind estimates to nearest radiosonde measurement by altitude.
    Treats radiosonde as truth. Reports fraction of measurements within 1 and 2 sigma.

    :param pandas.DataFrame df_glider: Glider data.
    :param pandas.DataFrame df_sonde: Radiosonde data.
    :param str alt_col: Radiosonde altitude.
        Optional, defaults to 'alt'.
    :param str zhat_col: Glider altitude.
        Optional, defaults to 'zhat'.
    :param str u_col: Glider u-wind.
        Optional, defaults to 'uest'.
    :param str v_col: Glider v-wind.
        Optional, defaults to 'vest'.
    :param str u_err_col: Glider u-wind error.
        Optional, defaults to 'u_combined'.
    :param str v_err_col: Glider v-wind error.
        Optional, defaults to 'v_combined'.
    :param str glider_lat_col: Glider latitude.
        Optional, defaults to 'lat0'.
    :param str glider_lon_col: Glider longitude.
        Optional, defaults to 'lon0'.
    :param float sonde_lat: Radiosonde latitude.
        Optional, defaults to None.
    :param float sonde_lon: Radiosonde longitude.
        Optional, defaults to None.
    :param float sonde_time: Radiosonde launch time.
        Optional, defaults to None.

    :return: The wind residuals, altitude difference, horizontal and total distance and time
        separation between the glider and radiosonde data at the nearest altitude match between
        the glider and radiosonde data.
    :rtype: pandas.DataFrame
    '''
    results = []
    for _, row in df_glider.iterrows():
        # find nearest radiosonde measurement by altitude
        alt_diff = np.abs(df_sonde[alt_col] - row[zhat_col])
        nearest = df_sonde.loc[alt_diff.idxmin()]

        # compute normalised residuals
        du = row[u_col] - nearest['u']
        dv = row[v_col] - nearest['v']

        # normalise by glider error (treating sonde as truth)
        # Note: we don't have ERA5 error, therefore only include glider error
        # if we had ERA5 error, divide du and dv by the glider and ERA5 errors added in qudrature
        # e.g. zu = du / np.sqrt(row[u_err_col]**2 + row[u_era5_err_col]**2)
        zu = du / row[u_err_col]
        zv = dv / row[v_err_col]

        # horizontal distance from glider to sonde launch point (km)
        if sonde_lat is not None and sonde_lon is not None:
            horiz_dist_km = haversine_km(row[glider_lat_col], row[glider_lon_col],
                                          sonde_lat, sonde_lon)
            # total 3D distance in metres (combining horizontal and vertical separation)
            vert_diff_m = abs(row[zhat_col] - nearest[alt_col])
            total_dist_m = np.sqrt((horiz_dist_km * 1000)**2 + vert_diff_m**2)
        else:
            horiz_dist_km = np.nan
            total_dist_m  = np.nan

        # time separation in hours
        if sonde_time is not None:
            time_sep_hr = abs(row['timestamp_hr'] - sonde_time)
        else:
            time_sep_hr = np.nan

        results.append({
            'alt':           row[zhat_col],
            'du':            du,
            'dv':            dv,
            'zu':            zu,
            'zv':            zv,
            'sonde_alt':     nearest[alt_col],
            'alt_diff':      alt_diff.min(),
            'horiz_dist_km': horiz_dist_km,
            'total_dist_m':  total_dist_m,
            'time_sep_hr':   time_sep_hr,
        })

    results_df = pd.DataFrame(results)

    # compute fractions within 1 and 2 sigma
    for comp, z_col in [('u', 'zu'), ('v', 'zv')]:
        within_1s = (np.abs(results_df[z_col]) <= 1).mean()
        within_2s = (np.abs(results_df[z_col]) <= 2).mean()
        within_3s = (np.abs(results_df[z_col]) <= 3).mean()
        within_5s = (np.abs(results_df[z_col]) <= 5).mean()
        print(f"{comp}: {within_1s*100:.1f}% within 1σ, {within_2s*100:.1f}% within 2σ, "
              f"{within_3s*100:.1f}% within 3σ, {within_5s*100:.1f}% within 5σ")

    print(f"\nExpected (Gaussian): 68.3% within 1σ, 95.4% within 2σ")
    print(f"\nMean horizontal separation: {results_df['horiz_dist_km'].mean():.1f} km")
    print(f"Mean total 3D separation:   {results_df['total_dist_m'].mean():.0f} m")
    print(f"Mean time separation:       {results_df['time_sep_hr'].mean():.2f} hr")

    return results_df



def plot_residuals_vs_separation(results_df):
    '''
    Plot absolute residuals of u and v components against time separation and horizontal distance from radiosonde.
    We are trying to see if the disagreement between glider and radiosonde wind estimates increases
    with time or distance from the radiosonde launch point.
    '''
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey='row')
    fig.suptitle('Absolute residuals vs separation from radiosonde')

    for col, (ax1, ax2), label in zip(
        ['du', 'dv'],
        [(axes[0, 0], axes[0, 1]), (axes[1, 0], axes[1, 1])],
        ['|du| (m/s)', '|dv| (m/s)']
    ):
        abs_resid = results_df[col].abs()

        # vs time separation
        ax1.scatter(results_df['time_sep_hr'], abs_resid,
                    alpha=0.6, color='steelblue', s=20)
        ax1.set_xlabel('Time separation (hr)')
        ax1.set_ylabel(label)
        ax1.grid(True, alpha=0.3)

        # add linear trend
        mask = results_df['time_sep_hr'].notna() & abs_resid.notna()
        if mask.sum() > 2:
            z = np.polyfit(results_df.loc[mask, 'time_sep_hr'], abs_resid[mask], 1)
            x = np.linspace(results_df['time_sep_hr'].min(),
                            results_df['time_sep_hr'].max(), 100)
            ax1.plot(x, np.polyval(z, x), 'r--', linewidth=1.5, label=f'Trend: {z[0]:.2f} m/s/hr')
            ax1.legend()

        # vs horizontal distance
        ax2.scatter(results_df['horiz_dist_km'], abs_resid,
                    alpha=0.6, color='darkorange', s=20)
        ax2.set_xlabel('Horizontal separation (km)')
        ax2.grid(True, alpha=0.3)

        # add linear trend
        mask = results_df['horiz_dist_km'].notna() & abs_resid.notna()
        if mask.sum() > 2:
            # obtaining best-fit parameters for the straight-line fit
            z = np.polyfit(results_df.loc[mask, 'horiz_dist_km'], abs_resid[mask], 1)
            x = np.linspace(results_df['horiz_dist_km'].min(),
                            results_df['horiz_dist_km'].max(), 100)
            # plotting the straight-line fit, using the best-fit parameters derived above
            ax2.plot(x, np.polyval(z, x), 'r--', linewidth=1.5, label=f'Trend: {z[0]:.2f} m/s/km')
            ax2.legend()

    fig.tight_layout()
    plt.show()


def plot_sonde_comparison_norm(results_df):
    '''
    Plot normalised u/v wind residual (glider minus radiosonde) distributions.
    '''
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    # normalised residual distributions
    for ax, z_col, label in zip(axes[:2], ['zu', 'zv'], ['u', 'v']):
        ax.hist(results_df[z_col], bins=20, density=True,
                color='steelblue', edgecolor='white', alpha=0.7)
        # overlay standard normal for reference
        x = np.linspace(-5, 5, 100)
        ax.plot(x, np.exp(-x**2/2) / np.sqrt(2*np.pi),
                'r--', label='Standard normal')
        ax.axvline(-1, color='grey', linestyle='--', alpha=0.5)
        ax.axvline( 1, color='grey', linestyle='--', alpha=0.5, label='±1σ')
        ax.axvline(-2, color='grey', linestyle=':',  alpha=0.5)
        ax.axvline( 2, color='grey', linestyle=':',  alpha=0.5, label='±2σ')
        ax.set_xlabel(f'Normalised {label} residual (z)')
        ax.set_ylabel('Density')
        ax.set_title(f'{label} residuals')
        ax.legend()

    # residuals vs altitude
    axes[2].plot(results_df['du'], results_df['alt'], 'o',
                 color='steelblue', alpha=0.6, markersize=4, label='u')
    axes[2].plot(results_df['dv'], results_df['alt'], 'o',
                 color='darkorange', alpha=0.6, markersize=4, label='v')
    axes[2].axvline(0, color='k', linewidth=0.5)
    axes[2].set_xlabel('Residual (m/s)')
    axes[2].set_ylabel('Altitude (m)')
    axes[2].set_title('Residuals vs altitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()



def plot_sonde_comparison_abs(results_df):
    '''
    Plot absolute u/v wind residual (glider minus radiosonde) distributions.
    '''
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    # absolute residual distributions
    for ax, z_col, label in zip(axes[:2], ['du', 'dv'], ['u', 'v']):
        ax.hist(results_df[z_col], bins=20, density=True,
                color='steelblue', edgecolor='white', alpha=0.7)
        # overlay standard normal for reference
        x = np.linspace(-5, 5, 100)
        # ax.plot(x, np.exp(-x**2/2) / np.sqrt(2*np.pi),
        #         'r--', label='Standard normal')
        # ax.axvline(-1, color='grey', linestyle='--', alpha=0.5)
        # ax.axvline( 1, color='grey', linestyle='--', alpha=0.5, label='±1σ')
        # ax.axvline(-2, color='grey', linestyle=':',  alpha=0.5)
        # ax.axvline( 2, color='grey', linestyle=':',  alpha=0.5, label='±2σ')
        ax.set_xlabel(f'{label} residual (m/s)')
        ax.set_ylabel('Density')
        ax.set_title(f'{label} residuals')
        # ax.legend()

    # residuals vs altitude
    axes[2].plot(results_df['du'], results_df['alt'], 'o',
                 color='steelblue', alpha=0.6, markersize=4, label='u')
    axes[2].plot(results_df['dv'], results_df['alt'], 'o',
                 color='darkorange', alpha=0.6, markersize=4, label='v')
    axes[2].axvline(0, color='k', linewidth=0.5)
    axes[2].set_xlabel('Residual (m/s)')
    axes[2].set_ylabel('Altitude (m)')
    axes[2].set_title('Residuals vs altitude')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()




# # 5. Quantitative comparison between glider measurements and ERA5

def calculate_era5_times(timestamp):
    '''
    Return year, month and day strings for the timestamp and following day.
    '''
    nextday = timestamp + timedelta(days=1)

    return {
        'current_day': {
            'year': timestamp.strftime('%Y'),
            'month': timestamp.strftime('%m'),
            'day': timestamp.strftime('%d'),
        },
        'next_day': {
            'year': nextday.strftime('%Y'),
            'month': nextday.strftime('%m'),
            'day': nextday.strftime('%d'),
        },
    }


def calculate_era5_bbox(df_glider, sonde_lat, sonde_lon,
    lat_col='lat0', lon_col='lon0', padding=1):
    '''
    Dynamically creating the lat/lon ERA5 bounding box.

    Return padded, truncated ERA5 bounds in (North, West, South, East) order.
    '''
    # Extend the glider and radiosonde bounds by one degree, then truncate.
    north = int(max(df_glider[lat_col].max(), sonde_lat) + padding)
    south = int(min(df_glider[lat_col].min(), sonde_lat) - padding)
    west = int(min(df_glider[lon_col].min(), sonde_lon) - padding)
    east = int(max(df_glider[lon_col].max(), sonde_lon) + padding)

    return (north, west, south, east)


def retrieve_era5_winds(client, year, month, day, times, area, pressure_levels, output_path):
    '''
    Download ERA5 u/v wind components at given pressure levels for a single date, saving to NetCDF.

    :param cdsapi.Client client: Connected CDS API client.
    :param str year: Year of the request, e.g. '2026'.
    :param str month: Month of the request, e.g. '03'.
    :param str day: Day of the request, e.g. '22'.
    :param list[str] times: Hours to retrieve, e.g. ['14:00', '15:00'].
    :param list[float] area: Bounding box as (North, West, South, East).
    :param list[str] pressure_levels: Pressure levels (hPa) to retrieve.
    :param str output_path: Local filename/path to save the retrieved NetCDF file to.

    :return: None. The retrieved data is saved to output_path.
    :rtype: None
    '''
    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping ERA5 retrieval")

        return None
    else:
        print(f"Retrieving ERA5 data and saving to {output_path}")
        client.retrieve('reanalysis-era5-pressure-levels', {
            'variable': ['u_component_of_wind', 'v_component_of_wind'],
            'pressure_level': pressure_levels,
            'year': year, 'month': month, 'day': day,
            'time': times,
            'area': area,  # N/W/S/E bounding box
            'data_format': 'netcdf'
        }, output_path)

        return None


def load_nc(filepath):
    '''
    Read in a .nc file and store in an xarray DataArray.

    :param str filepath: Filepath of the .nc file location

    :return: A xarray DataArray containing the contents of the injested .nc file
    :rtype: xarray.core.dataset.Dataset
    '''

    return xr.open_dataset(filepath)


def compare_glider_era5(df_glider, ds, alt_levels=None,
                            lat_col='lat0', lon_col='lon0',
                            time_col='timestamp', alt_col='zhat',
                            u_col='uest', v_col='vest',
                            u_err_col='u_combined', v_err_col='v_combined',
                            max_alt_diff=100):
    '''
    Compare glider wind estimates to nearest ERA5 match by altitude.
    Treats ERA5 as truth. Reports fraction of measurements within 1 and 2 sigma.

    :param pandas.DataFrame df_glider: Glider data.
    :param pandas.DataFrame ds: ERA5 data.
    :param dict alt_levels: ERA5 pressure:altitude pairs.
        Optional, defaults to None.
    :param str lat_col: Glider latitude.
        Optional, defaults to 'lat0'.
    :param str lon_col: Glider longitude.
        Optional, defaults to 'lon0'.
    :param str time_col: Glider timestamp.
        Optional, defaluts to 'timestamp'.
    :param str alt_col: Glider altitude.
        Optional, defaults to 'zhat'.
    :param str u_col: Glider u-wind.
        Optional, defaults to 'uest'.
    :param str v_col: Glider v-wind.
        Optional, defaults to 'vest'.
    :param str u_err_col: Glider u-wind error.
        Optional, defaults to 'u_combined'.
    :param str v_err_col: Glider v-wind error.
        Optional, defaults to 'v_combined'.
    :param int max_alt_diff: Maximum altitude allowed between glider altitude and nearest ERA5 altitude.
        Optional, defaults to 100 m

    :return: DataFrame containing glider and matched ERA5 altitudes, their altitude
        difference, u/v wind component residuals (glider minus ERA5), u/v wind residuals
        normalized by the glider wind uncertainties (z-scores) and glider and ERA5 u/v
        wind components.
    :rtype: pandas.DataFrame
    '''

    # need to define alt_levels mapping pressure levels to approximate altitudes (m) for comparison.
    if alt_levels is None:
        # alt_levels = {975.1: 399.0, 950.0: 618.0, 925.0: 845.0, 900.0: 1074.0, 875.0: 1309.0, 850.0: 1544.0, 825.0: 1784.0, 804.5: 1992.0}
        alt_levels = {975.0: 301, 950.0: 528, 925.0: 751, 900.0: 975, 875.0: 1215.0, 850.0: 1450, 825.0: 1696, 804.5: 1943.0}

    results = []

    for _, row in df_glider.iterrows():

        # find nearest ERA5 altitude level
        nearest_pressure = min(alt_levels, 
                               key=lambda p: abs(alt_levels[p] - row[alt_col]))
        nearest_alt = alt_levels[nearest_pressure]

        # skip if too far from any ERA5 level
        if abs(nearest_alt - row[alt_col]) > max_alt_diff:
            continue
        
        # spatially interpolate to glider location
        # interpolate ERA5 model point (at closest glider altitude) to glider lat/lon position and time
        era5_point = ds.sel(pressure_level=nearest_pressure).interp(
            latitude=row[lat_col],
            longitude=row[lon_col],
            valid_time=np.datetime64(pd.Timestamp(row[time_col], unit='s')),
            method='linear'
        )

        u_era5 = float(era5_point['u'].values)
        v_era5 = float(era5_point['v'].values)

        du = row[u_col] - u_era5
        dv = row[v_col] - v_era5
        zu = du / row[u_err_col]
        zv = dv / row[v_err_col]

        results.append({
            'alt':          row[alt_col],
            'era5_alt':     nearest_alt,
            'alt_diff':     abs(nearest_alt - row[alt_col]),
            'du':           du,
            'dv':           dv,
            'zu':           zu,
            'zv':           zv,
            'u_era5':       u_era5,
            'v_era5':       v_era5,
            'u_glider':     row[u_col],
            'v_glider':     row[v_col],
        })

    print(f"Retained {len(results)}/{len(df_glider)} measurements within {max_alt_diff}m of an ERA5 level")
    return pd.DataFrame(results)


def plot_era5_residuals(results_df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('ERA5 residuals vs physical variables')

    plot_params = [
        ('alt',       'Glider Altitude (m)',       axes[0, 0], axes[1, 0]),
        ('u_glider',  'Glider wind speed u (m/s)', axes[0, 1], axes[1, 1]),
        ('v_glider',  'Glider wind speed v (m/s)', axes[0, 2], axes[1, 2]),
    ]

    for x_col, x_label, ax_u, ax_v in plot_params:
        x = results_df[x_col]

        # du
        ax_u.scatter(x, results_df['du'].abs(), alpha=0.6, color='steelblue', s=20)
        ax_u.set_xlabel(x_label)
        ax_u.set_ylabel('|du| (m/s)')
        ax_u.axhline(0, color='k', linewidth=0.5)
        ax_u.grid(True, alpha=0.3)

        # dv
        ax_v.scatter(x, results_df['dv'].abs(), alpha=0.6, color='darkorange', s=20)
        ax_v.set_xlabel(x_label)
        ax_v.set_ylabel('|dv| (m/s)')
        ax_v.axhline(0, color='k', linewidth=0.5)
        ax_v.grid(True, alpha=0.3)

        # add linear trend to both
        for ax, col in [(ax_u, 'du'), (ax_v, 'dv')]:
            abs_resid = results_df[col].abs()
            mask = x.notna() & abs_resid.notna()
            if mask.sum() > 2:
                # obtaining best-fit parameters for the straight-line fit
                z = np.polyfit(x[mask], abs_resid[mask], 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                # plotting the straight-line fit, using the best-fit parameters derived above
                ax.plot(x_line, np.polyval(z, x_line),
                        'r--', linewidth=1.5)

    fig.tight_layout()
    plt.show()



# New function added to plot the u, v wind profiles from the glider and ERA5 data - 29/10/26
def plot_glider_vs_era5_alt(results_df):
    '''Plot glider and (interpolated) ERA5 u, v wind profiles at each glider altitude..'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    fig.suptitle('Glider and ERA5 u, v winds vs altitude')

    # mark ERA5 pressure level altitudes
    era5_alts = results_df['era5_alt'].unique()

    for ax, col, label, color in [
        (ax1, 'u_glider', 'u (m/s)', 'steelblue'),
        (ax1, 'u_era5', 'u (m/s)', 'darkorange'),
        (ax2, 'v_glider', 'v (m/s)', 'steelblue'),
        (ax2, 'v_era5', 'v (m/s)', 'darkorange')
    ]:
        ax.scatter(results_df[col], results_df['alt'],
                   label=col, alpha=0.6, color=color, s=20)
        # ax.axvline(0, color='k', linewidth=0.5)

        # mark ERA5 level altitudes
        for era5_alt in sorted(era5_alts):
            ax.axhline(era5_alt, color='grey', linewidth=0.8,
                       linestyle='--', alpha=0.5)
            ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else -5,
                    era5_alt + 10, f'{era5_alt:.0f}m',
                    fontsize=8, color='grey')

        # # mean residual per ERA5 level
        # for era5_alt in sorted(era5_alts):
        #     mask = results_df['era5_alt'] == era5_alt
        #     mean_resid = results_df.loc[mask, col].mean()
        #     ax.plot(mean_resid, era5_alt, 'r*', markersize=12,
        #             label='Mean per (ERA5) level' if era5_alt == sorted(era5_alts)[0] else '')

        ax.set_xlabel(label)
        ax.set_ylabel('Altitude (m)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    plt.show()


def plot_era5_bias_vs_alt(results_df):
    '''Plot signed residuals vs altitude to show systematic bias at each ERA5 level.'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    fig.suptitle('ERA5 signed residuals vs altitude')

    # mark ERA5 pressure level altitudes
    era5_alts = results_df['era5_alt'].unique()

    for ax, col, label, color in [
        (ax1, 'du', 'du (m/s)', 'steelblue'),
        (ax2, 'dv', 'dv (m/s)', 'darkorange')
    ]:
        ax.scatter(results_df[col], results_df['alt'],
                   alpha=0.6, color=color, s=20)
        ax.axvline(0, color='k', linewidth=0.5)

        # mark ERA5 level altitudes
        for era5_alt in sorted(era5_alts):
            ax.axhline(era5_alt, color='grey', linewidth=0.8,
                       linestyle='--', alpha=0.5)
            ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else -5,
                    era5_alt + 10, f'{era5_alt:.0f}m',
                    fontsize=8, color='grey')

        # mean residual per ERA5 level
        for era5_alt in sorted(era5_alts):
            mask = results_df['era5_alt'] == era5_alt
            mean_resid = results_df.loc[mask, col].mean()
            ax.plot(mean_resid, era5_alt, 'r*', markersize=12,
                    label='Mean per (ERA5) level' if era5_alt == sorted(era5_alts)[0] else '')

        ax.set_xlabel(label)
        ax.set_ylabel('Altitude (m)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    plt.show()





def compare_sonde_era5(df_sonde, ds, alt_levels=None,
                        sonde_lat=LAT_SONDE, sonde_lon=LON_SONDE,
                        sonde_time=TIMESTAMP_SONDE):
    '''
    Compare radiosonde wind speed with ERA5 at matching pressure levels.

    :param pandas.DataFrame df_sonde: Radiosonde winds.
    :param xarray.core.dataset.Dataset ds: ERA5 winds.
    :param dict alt_levels: ERA5 pressure:altitude pairs.
    :param float sonde_lat: Radiosonde launch latitude.
    :param float sonde_lon: Radiosonde launch longitude.
    :param pandas.Timestamp sonde_time: Radiosonde launch timestamp.

    :return: DataFrame containing matched pressure levels and altitudes, radiosonde and
        ERA5 wind speeds and directions, ERA5 wind components, and the wind-speed difference
        (radiosonde minus ERA5).
    :rtype: pandas.DataFrame
    '''

    if alt_levels is None:
        alt_levels = {975.0: 301, 950.0: 528, 925.0: 751, 900.0: 975,
                      875.0: 1215.0, 850.0: 1450, 825.0: 1696, 800.0: 1943.0}

    results = []

    for pressure, alt in alt_levels.items():
        # get nearest radiosonde measurement by altitude
        idx      = (df_sonde['alt'] - alt).abs().idxmin()
        sonde_row = df_sonde.loc[idx]

        # get ERA5 at this pressure level, interpolated to sonde location
        # interpolate ERA5 model point (at closest radiosonde altitude) to radiosonde lat/lon position and time
        era5_point = ds.sel(pressure_level=pressure, method='nearest').interp(
            latitude=sonde_row['LAT'] if 'LAT' in df_sonde.columns else sonde_lat,
            longitude=sonde_row['LON'] if 'LON' in df_sonde.columns else sonde_lon,
            valid_time=np.datetime64(sonde_time),
            method='linear'
        )

        u_era5    = float(era5_point['u'].values)
        v_era5    = float(era5_point['v'].values)
        magU_era5 = np.sqrt(u_era5**2 + v_era5**2)
        dir_era5  = (270 - np.rad2deg(np.arctan2(v_era5, u_era5))) % 360

        # radiosonde wind in m/s
        magU_sonde = sonde_row['magU']

        results.append({
            'pressure':       pressure,
            'alt':            alt,
            'sonde_pressure': sonde_row['pres'],
            'sonde_alt':      sonde_row['alt'],
            'magU_sonde':     magU_sonde,
            'dir_sonde':      sonde_row['direction'] if 'direction' in df_sonde.columns else np.nan,
            'magU_era5':      magU_era5,
            'dir_era5':       dir_era5,
            'u_era5':         u_era5,
            'v_era5':         v_era5,
            'diff':           magU_sonde - magU_era5,
        })


        print(f"{pressure} hPa (~{alt:.0f}m): "
        f"Sonde={magU_sonde:.2f} m/s, "
        f"ERA5={magU_era5:.2f} m/s, "
        f"Diff={magU_sonde - magU_era5:.2f} m/s")

    results_df = pd.DataFrame(results)

    return results_df


def plot_sonde_era5_comparison(df_sonde, df_comparison):
    '''
    Comparison plots of the radiosonde data vs ERA5 data.
    Plots ERA5 wind speed and direction overlaid on the radiosonde wind profile.

    :param pandas.DataFrame df_sonde: DataFrame containing radiosonde data.
    :param pandas.DataFrame df_comparison: DataFrame containing the radiosonde and ERA5 comparison data.

    :return: None
    :rtype: NoneType
    '''

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    fig.suptitle('Radiosonde vs ERA5 wind profile')

    # wind speed
    ax1.plot(df_sonde['magU'], df_sonde['alt'],
             color='steelblue', linewidth=1.5, label='Radiosonde')
    ax1.scatter(df_comparison['magU_era5'], df_comparison['alt'],
                color='darkorange', s=80, marker='D', zorder=5,
                label='ERA5')
    ax1.set_xlabel('Wind speed (m/s)')
    ax1.set_ylabel('Altitude (m)')
    ax1.set_title('Wind speed')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # wind direction
    if 'direction' in df_sonde.columns:
        ax2.plot(df_sonde['direction'], df_sonde['alt'],
                 color='steelblue', linewidth=1.5, label='Radiosonde')
    ax2.scatter(df_comparison['dir_era5'], df_comparison['alt'],
                color='darkorange', s=80, marker='D', zorder=5,
                label='ERA5')
    ax2.set_xlabel('Wind direction (degrees)')
    ax2.set_title('Wind direction')
    ax2.set_xlim(0, 360)
    ax2.set_xticks([0, 90, 180, 270, 360])
    ax2.set_xticklabels(['N', 'E', 'S', 'W', 'N'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)   

    fig.tight_layout()
    plt.show()

    return


def inspect_wind_residuals(results_df, thresholds=[1, 2, 5, 10]):
    r'''
    Report what fraction of normalised wind residuals are within more relaxed thresholds.

    :param pandas.DataFrame results_df: DataFrame containing u/v wind residuals
        between glider and ERA5.
    :param list[int] thresholds: List of normalised thresholds in units of :math:`\sigma`.

    :return: None
    :rtype: NoneType
    '''

    for threshold in thresholds:
        frac_u = (results_df['zu'].abs() < threshold).mean()
        frac_v = (results_df['zv'].abs() < threshold).mean()
        print(f"|z| < {threshold}: u={frac_u*100:.1f}%, v={frac_v*100:.1f}%")

    # identify the outliers
    outliers = results_df[results_df['zu'].abs() > thresholds[-1]]
    print(f"\n{len(outliers)} outliers with |zu| > {thresholds[-1]}")
    # print(outliers[['alt', 'era5_alt', 'alt_diff', 'du', 'dv', 'u_glider', 'u_era5']])
    print(outliers)

    return



def main():

    ##############################################################################
    ### 1. Load Estimated Glider Params and Load + Pre-process Radiosonde Data ###
    ##############################################################################
    print(f"*****************************************************************************")
    print(f"*** Loading Glider Fitting Params and Loading + Pre-processing Sonde Data ***")
    print(f"*****************************************************************************")

    # Read in estimated parameters from the processed thermal bins from .csv file, straight into a pandas dataframe.
    df_glider = load_data(gliderpath)
    print(f"Estimated parameters obtained from glider data analysis:")
    print(df_glider)

    # Read in radiosonde data, sourced from the University of Wyoming radiosonde archive (http://weather.uwyo.edu/upperair/sounding.html). 
    # The data is in .csv format, and contains the following columns:
    # [time, longitude, latitude, pressure_hPa, geopotential height_m, temperature_C, dew point temperature_C, ice point temperature_C, relative humidity_%, humidity wrt ice_%, mixing ratio_g/kg, wind direction_degree, wind speed_m/s]
    df_rawsonde = load_data(sondepath)
    print(f"(Raw) Radiosonde data for {sondeymd} at {sondehr} Z:")
    print(df_rawsonde)
    print(df_rawsonde.columns)


    print(df_rawsonde['time'])

    # Filter sonde data for < 2000 m - this range is useful for glider flying altitudes
    df_sonde = preprocess_sonde(df_rawsonde)
    print(df_sonde)


    # Optional: Inspect how the period and the rate of change altitude changes over each glider thermal.
    stats = stats_period_zdot(df_glider)
    print(stats)


    ########################################################
    ### 2. Determine Upper Bound on Uncertainty in Winds ###
    ########################################################
    print(f"********************************************************************")
    print(f"*** Estmating Upper Bound Uncertainty in Glider Wind Meaurements ***")
    print(f"********************************************************************")

    # find the difference between adjacent thermal bins to estimate an upper bound on uncertainty in u, v winds
    df_glider = calculate_upper_bound_errors(df_glider)
    print(f"Glider dataframe updated to include upper bound wind errors:\n {df_glider}")
    print(df_glider.columns)

    # propagate errors to update glider dataframe to contain magnitude and direction estimates in U wind
    df_glider = propagate_wind_uncertainty(df_glider)
    print(f"Updated glider DataFrame to obtain maximum ('combined') uncertainty estimates:\n {df_glider}")
    print(df_glider.columns)

    # Optional plot to view the distribution of std devs of u and v wind estimates
    plot_combined_wind_std_distribution(df_glider)
    # Optional plot of CDF distribution for the std devs of the wind magnitude from model fitting and interbin variation
    plot_mag_wind_std_cdf(df_glider)


    # Clean outliers from the dataframe based on standard deviation thresholds for the wind estimates.
    df_glider_clean = filter_clean_wind_bins(df_glider)
    print(f"Glider data after filtering based on model and interbin std dev wind thresholds:\n {df_glider_clean}")

    # Compare the glider wind quantities before and after the standard deviation wind filters have been applied 
    plot_qc_diagnostics(df_glider, df_glider_clean)


    ################################################################
    ### 3. Cluster measurements by spatial and temporal distance ###
    ################################################################
    print(f"********************************************************************")
    print(f"*** Cluster glider measurements by spatial and temporal distance ***")
    print(f"********************************************************************")
    # define a cluster number to investigate (used for later plots)
    cluster_no = 0


    ## cluster glider wind measurements by spatial proximity using DBSCAN
    df_glider_clean = cluster_measurements(df_glider_clean, eps_km=20, min_samples=4)
    print(df_glider_clean)
    
    print(df_glider_clean['cluster'])

    # checking that a valid cluster of gliders exists, if not exit the script
    if not (df_glider_clean['cluster'] >= 0).any():
        print(
            "WARNING: No valid clusters were found."
            "All measurements were classified as noise. Stopping processing."
        )
        return


    # check counts
    print(f"**********")
    for cluster_id, cluster_size in df_glider_clean.groupby('cluster').size().items():
        print(f"Cluster {cluster_id}: {cluster_size} points")
    print(f"**********")


    ## cluster plots ##
    # plot spatial and temporal glider wind measurements and radiosonde launch
    # sondes in UK launched at 11:15 Z (Camborne & Lerwick only) and 23:15 Z
    plot_glider_locations(df_glider_clean, lat_col='lat0', lon_col='lon0',
                    sonde_lat=LAT_SONDE, sonde_lon=LON_SONDE, sonde_time=TIME_SONDE)
    # plot glider wind measurement locations, colour coded by cluster number
    # call with radiosonde location
    plot_clusters_map(df_glider_clean, lat_col='lat0', lon_col='lon0',
                    sonde_lat=LAT_SONDE, sonde_lon=LON_SONDE)
    # plot wind profiles for each wind cluster vs radiosonde wind profile
    plot_wind_profiles(df_glider_clean, df_sonde)
    # plot temporal coverage of measurements within cluster 0
    plot_cluster_temporal_coverage(df_glider_clean, cluster_id=cluster_no)


    # Split a cluster into time-based sub-clusters using manual time breaks.
    df_glider_clean['timestamp_hr'] = df_glider_clean['timestamp'] % (3600*24) / 3600
    df_glider_clean = subcluster_by_time(df_glider_clean, cluster_id=cluster_no, breaks=[12.5, 16.0])
    # check counts
    print(f"**********")
    for cluster_id, cluster_size in df_glider_clean.groupby('subcluster').size().items():
        print(f"Subcluster {cluster_id}: {cluster_size} points")
    print(f"**********")

    print(df_glider_clean.info())


    ## subclusters wind profile plots ##
    # plot glider wind profiles for subclusters within a specific cluster e.g. 0
    plot_wind_profiles(df_glider_clean[df_glider_clean['cluster'] == cluster_no], df_sonde,
                    cluster_col='subcluster')


    ###############################################################
    ### 4. Comparison of Glider Measurements Against Radiosonde ###
    ###############################################################
    print(f"************************************************")
    print(f"*** Compare Glider Winds to Radiosonde Winds ***")
    print(f"************************************************")
    # define a subcluster number to investigate (used for later plots)
    subcluster_no = '0b'

    df_sonde = derive_sonde_wind_components(df_sonde)
    print(df_sonde)

    # calculate residuals between glider and sonde winds at nearest height match
    sonde_residuals = compare_to_radiosonde(df_glider_clean[df_glider_clean['subcluster'] == subcluster_no], df_sonde,
                                        sonde_lat=LAT_SONDE, sonde_lon=LON_SONDE,
                                        sonde_time=TIME_SONDE) # 23:15 in decimal hours
    print(sonde_residuals)

    # plot absolute residuals of u and v winds against time separation and horizontal distance from radiosonde
    plot_residuals_vs_separation(sonde_residuals)
    # plot normalised u, v residuals distributions and u, v residual profiles
    plot_sonde_comparison_norm(sonde_residuals)
    # plot absolute u, v residuals distributions and u, v residual profiles
    plot_sonde_comparison_abs(sonde_residuals)

    print(sonde_residuals.info())


    ######################################################################
    ### 5. Download ERA5 Measurements Closest to Glider and Radiosonde ###
    ######################################################################
    print(f"**********************************")
    print(f"*** Download ERA5 Measurements ***")
    print(f"**********************************")

    # Connect to the Copernicus Climate Data Store (CDS) API to retrieve additional data if needed.
    c = cdsapi.Client()
    print("Connected successfully")

    # ERA5 pressure levels of interest (hPa)
    pressure_levels = ['800', '825', '850', '875', '900', '925', '950', '975', '1000']

    # ERA5 times of interest - derived from the glider time range
    era5_times = pd.to_datetime(df_glider['time'], format='ISO8601', utc=True)
    time_hours = [(str(t)+':00') for t in range(era5_times.dt.hour.min(), 24)]


    # ERA5 lat/lon Region of interest
    era5_bbox = calculate_era5_bbox(
        df_glider,
        sonde_lat=LAT_SONDE,
        sonde_lon=LON_SONDE,
        padding=1,
    )

    # ERA5 dates of interest
    era5_dates = calculate_era5_times(TIMESTAMP_SONDE)

    current_day = era5_dates['current_day']
    next_day = era5_dates['next_day']


    # Two example retrievals of ERA5 reanalysis data for wind components at various pressure levels.
    # Uncomment and modify as needed to retrieve data.
    # The file will be saved in the current working directory.
    if os.path.exists(f'era5_winds_{current_day['year']+current_day['month']+current_day['day']}.nc'):
        print(f"era5_{current_day['year']+current_day['month']+current_day['day']}.nc already exists, skipping ERA5 retrieval.")
    else:
        retrieve_era5_winds(c, year=current_day['year'], month=current_day['month'], day=current_day['day'],
                            times=time_hours,
                            area=era5_bbox,
                            pressure_levels=pressure_levels,
                            output_path=f'era5_winds_{current_day['year']+current_day['month']+current_day['day']}.nc')


    if os.path.exists(f'era5_winds_{next_day['year']+next_day['month']+next_day['day']}_00.nc'):
        print(f"era5_{next_day['year']+next_day['month']+next_day['day']}_00.nc already exists, skipping ERA5 retrieval.")
    else:
        retrieve_era5_winds(c, year=next_day['year'], month=next_day['month'], day=next_day['day'],
                            times=['00:00'],
                            area=era5_bbox,
                            pressure_levels=pressure_levels,
                            output_path=f'era5_winds_{next_day['year']+next_day['month']+next_day['day']}_00.nc')

    
    # Read in ERA5 wind profile .nc file(s), straight into a xarray DataArray(s).
    ds_day = load_nc(f'era5_winds_{current_day['year']+current_day['month']+current_day['day']}.nc')
    ds_nextday = load_nc(f'era5_winds_{next_day['year']+next_day['month']+next_day['day']}_00.nc')
    # Combine the two xarrays into a single dataset for interpolation.
    ds = xr.concat([ds_day, ds_nextday], dim='valid_time').sortby('valid_time')
    print(f"Downloaded ERA5 DataArray co-ordinates:\n {ds.coords}")
    print(f"Downloaded ERA5 DataArray u-wind dimensions:\n {ds['u'].dims}")



    #########################################################
    ### 6. Comparison of Glider Measurements Against ERA5 ###
    #########################################################
    print(f"******************************************")
    print(f"*** Compare Glider Winds to ERA5 Winds ***")
    print(f"******************************************")

    # DataFrame of horizontal wind residuals between glider and ERA5 data
    glider_era5_residuals = compare_glider_era5(df_glider_clean, ds)
    # define which columns to look for missing values and remove subsequent rows
    glider_era5_residuals = glider_era5_residuals.dropna(subset=['u_era5', 'v_era5'])
    print(f"Retained {len(glider_era5_residuals)} valid comparisons.\n")
    print(f"Nearest ERA5 u/v winds altitude match to gliders u/v winds:\n {glider_era5_residuals}")

    # plotting residuals between the glider and ERA5 data NOT the radiosonde data
    # the plot_sonde_comaprison function just plots the residuals - needs to be renamed
    plot_sonde_comparison_norm(glider_era5_residuals)
    plot_sonde_comparison_abs(glider_era5_residuals)

    # plot ERA5 residuals against physical variables: glider altitude and u, v wind speeds
    plot_era5_residuals(glider_era5_residuals)

    # plot glider vs (interpolated) ERA5 u,v wind profiles
    plot_glider_vs_era5_alt(glider_era5_residuals)

    # plot glider altitude vs ERA5 signed residuals
    plot_era5_bias_vs_alt(glider_era5_residuals)

    print(f"Glider ERA5 residuals:\n")
    print(glider_era5_residuals)



    #############################################################
    ### 7. Comparison of Radiosonde Measurements Against ERA5 ###
    #############################################################
    print(f"**********************************************")
    print(f"*** Compare Radiosonde Winds to ERA5 Winds ***")
    print(f"**********************************************")

    # find which altitude/pressure level of ERA5 corresponds to in the radiosonde
    print(f"Printing each ERA5 pressure level against the closest-matching radiosonde pressure level and altitude:")
    for p in pressure_levels:
        idx = (df_sonde['pres'] - int(p)).abs().idxmin()
        print(f"ERA5 pressure level {p} hPa --> "
              f"Radiosonde pressure level {df_sonde.loc[idx, 'pres']} hPa at altitude {df_sonde.loc[idx, 'alt']} m")


    # DataFrame containing radiosonde winds and interpolated ERA5 winds at closest matching pressure levels
    print(f"Sonde, ERA5 |U| Wind Comparison:")
    sonde_era5_comparison = compare_sonde_era5(df_sonde, ds)
    print(f"Sonde, ERA5 comparison:\n {sonde_era5_comparison}")
    
    # plot ERA5 wind speeds and directions overlaid on radiosonde wind profiles
    plot_sonde_era5_comparison(df_sonde, sonde_era5_comparison)


    ##############################################################
    ### 8. Inspect Normalised Glider minus ERA5 Wind Residuals ###
    ##############################################################
    print(f"************************************************")
    print(f"*** Inspect Glider minus ERA5 Wind Residuals ***")
    print(f"************************************************")

    # check what fraction of normalised wind (glider minus ERA5) residuals are inside more relaxed thresholds
    inspect_wind_residuals(glider_era5_residuals)

    # inspect the lat/lon grids
    print("ERA5 grid extent:")
    print(f"Latitude: {float(ds.latitude.min()):.2f} to {float(ds.latitude.max()):.2f}")
    print(f"Longitude: {float(ds.longitude.min()):.2f} to {float(ds.longitude.max()):.2f}")

    print("\nGlider data extent:")
    print(f"Latitude: {df_glider_clean['lat0'].min():.2f} to {df_glider_clean['lat0'].max():.2f}")
    print(f"Longitude: {df_glider_clean['lon0'].min():.2f} to {float(df_glider_clean['lon0'].max()):.2f}")



if __name__ == "__main__":
    main()
