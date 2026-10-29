# %%
# name: ogn_tcp
# date: 23/07/2026
# author: Toby Alexander

# description: 
# A simple TCP client for connecting to OGN servers. 
# Includes further parsing and filtering of the data stream to examine spatial distribution of messages and identify unique flights.

# required modules:
#   - Python 3.6+ (created with 3.10.19)
#   - cartopy
#   - matplotlib
#   - numpy
#   - pandas


# # 0.1. Imports

import cartopy.crs as ccrs # map plotting library for geospatial data
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import time

# # 0.2. Global Constants and Configuration

# location of streamed OGN APRS .parquet files
root = "/data/scratch/matt.evans/aprs_data/2026-09-25/"
# filenames = ["beacons_20260925_1445.parquet", "beacons_20260925_1450.parquet"]

files1 = sorted(list(Path(root).glob("beacons_20260925_14*")))
files2 = sorted(list(Path(root).glob("beacons_20260925_15*")))
filenames = files1 + files2
filenames = [Path(i).name for i in filenames]
print(filenames)



# helper function to obtain the lat/lon co-ords of the OGN receivers
def load_ogn_receiver_locations(filename):
    '''
    Loading in the lat/lon co-ordiantes of OGN receivers located at:
        - glider airfields
        - other locations
    '''
    ogn_locs = pd.read_parquet(filename)

    ogn_lats = ogn_locs['latitude']
    ogn_lons = ogn_locs['longitude']

    return {'latitude': ogn_lats, 'longitude': ogn_lons}





# # Code to get all OGN APRS files stored in a directory
# # globbing all the glider flight files and storing the file names in a list
# # leave commented out for now - will be needed when dealing with several glider flights
# glider_files = list(Path(root).glob("beacons_*.parquet"))
# glider_file_names = sorted([i for i in glider_files])


# # 1. Extract the UTC timestamps encoded in the input filenames.

def start_end_times(filenames):
    '''
    Extracting the UTC timestamps encoded in the filenames.
    These timestamps are saved as floats in epoch time format.
    '''
    file_times = []
    for filename in filenames:
        filename_timestamp = Path(filename).stem.removeprefix('beacons_')
        file_time = datetime.strptime(filename_timestamp, '%Y%m%d_%H%M')
        file_time = file_time.replace(tzinfo=timezone.utc)
        file_times.append(file_time)

    start_datetime = min(file_times)
    end_datetime = max(file_times)
    print(start_datetime)
    print(end_datetime)
    start_time = start_datetime.timestamp()
    # adding on 5 minutes to the end_datetime to account for the 5 mins of data contained in the file
    # i.e. the data ends 5 mins after the time stated on the parquet file description
    end_time = (end_datetime + timedelta(minutes=5)).timestamp()

    return (start_time, end_time)



# # 3. Filtering
# 
# Other aircraft use the FLARM system, so we need to filter for gliders only. Also seem to have an issue with receiving messages which are outside the specified APRS filter, so we can filter those out too.

def get_gliders(df):
    '''
    As specified in the OGN documentation (http://wiki.glidernet.org/wiki:ogn-flavoured-aprs):
    aircraft_type = 1 corresponds to gliders, so filter the DataFrame to only include those records for now.
    '''
    gliders_df = df[df['aircraft_type'] == 1].copy()

    return gliders_df


def create_unix_time_col(df):
    '''
    Create a new unix timestamp column (seconds since epoch) for easier filtering and plotting.
    '''

    # extract the unit of the 'timestamp' column
    timestamp_dtype = df['timestamp'].dtype
    timestamp_unit = getattr(timestamp_dtype, 'unit', None)
    if timestamp_unit is None and np.issubdtype(timestamp_dtype, np.datetime64):
        timestamp_unit = np.datetime_data(timestamp_dtype)[0]

    # save the timestamp unit
    timestamp_unit = np.datetime_data(df['timestamp'].dtype)[0]

    # check the timestamp unit
    # assigin correct division scale factor for conversion of timestamps to unix times
    if timestamp_unit == 'us':
        divisor = 10**6
    elif timestamp_unit == 'ns':
        divisor = 10**9
    else:
        raise ValueError(
            f"Unsupported timestamp dtype {df['timestamp'].dtype}; "
            "expected microseconds ('us') or nanoseconds ('ns')."
        )

    timestamp_unix = df['timestamp'].astype('int64') / divisor
    # Create a new column for the timestamp in unix time (seconds since epoch) for easier filtering and plotting.
    # Where the original timestamp is missing/NaT, put in NaN to keep same row dimensions in DataFrame
    df['timestamp_unix'] = timestamp_unix.where(
        df['timestamp'].notna()
    )

    print(f"Total records: {len(df)}, Glider records: {len(df)} ({len(df)/len(df)*100:.2f}%)")

    return df


def filter_loc_time(df, start, end):
    '''
    Locate glider measurements that are in the UK and within the start/end file times.

    :param pandas.DataFrame df: DataFrame containing OGN APRS data.
    :param float start: Start time of the data collection.
    :param float end: End time of the data collection.
    '''
    
    # find measurements outside the UK bounding box (49.5N to 61N, -8.5E to 2E)
    # For some reason many messages outside of the filtered range are still being received from EU and US, so check how many and which they are.

    # Define a square bounding box for the UK (approximate)
    uk_bb_lat = [49.5, 61]
    uk_bb_lon = [-8.5, 2]

    mask_outside = (
        (df['latitude'] < uk_bb_lat[0]) | (df['latitude'] > uk_bb_lat[1]) |
        (df['longitude'] < uk_bb_lon[0]) | (df['longitude'] > uk_bb_lon[1])
    )
    print(f"Messages outside bounding box: {mask_outside.sum()} ({mask_outside.sum() / len(df) * 100:.2f}%)")
    print(df[mask_outside][['latitude', 'longitude', 'timestamp', 'name']].head(5))

    # Similarly some messages outside of the time range are also being received, so check how many and which they are.
    mask_time = (df['timestamp_unix'] >= start) & (df['timestamp_unix'] <= end)
    print(f"Points outside time range: {(~mask_time).sum()} ({(~mask_time).sum() / len(df) * 100:.2f}%)")


    # Filter for messages within a bounding box around the UK and within the specified time range, and print how many points remain.
    # Create a new dataframe with only UK glider messages within the correct time range.
    mask_inside = (
        (df['latitude'] >= uk_bb_lat[0]) & (df['latitude'] <= uk_bb_lat[1]) &
        (df['longitude'] >= uk_bb_lon[0]) & (df['longitude'] <= uk_bb_lon[1]) &
        (df['timestamp_unix'] >= start) & (df['timestamp_unix'] <= end)
    )

    gliders_uk_df = df[mask_inside].copy()
    print(f"Total glider records before bounding box and time range filters: {len(df)}")
    print(f"Points inside bounding box and time range: {len(gliders_uk_df)} ({len(gliders_uk_df)/len(df)*100:.2f}%)")

    return gliders_uk_df




def plot_messages(df, lat_col='latitude', lon_col='longitude',
                  lat_lim=(49.5, 61), lon_lim=(-8.5, 3), *,
                  ogn_airfields, ogn_other_locations):
    '''
    Plot the spatial distribution of glider messages on a map, coloured by time of day.

    - df: DataFrame containing the glider messages, with columns for latitude, longitude, and timestamp.
    - lat_col, lon_col: names of the columns in df that contain latitude and longitude values, respectively.
    - lat_lim, lon_lim: fixed (min, max) map extent; defaults to the UK bounding box.
    - The function creates a map using Cartopy, plots the glider message locations as a scatter plot coloured by time of day, and displays the map with appropriate features and labels.
    - The timestamp is converted to hours of the day (0-24) for colouring the points, and a colorbar is added to indicate the time scale.
    '''

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })

    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained',
                           subplot_kw={'projection': ccrs.PlateCarree()})
    
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--')
    ax.add_feature(cfeature.RIVERS, linewidth=0.3)

    
    # UK radiosonde locations
    sonde_lats_23Z = [50.89, 53.00, 55.02, 54.50, 51.20]
    sonde_lons_23Z = [0.32, -1.25, -1.88, -6.34, -1.80]

    sonde_lats_11Z = [50.22, 60.14]
    sonde_lons_11Z = [-5.32, -1.18]


    df = df.dropna(subset=[lat_col, lon_col])

    ax.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]],
                  crs=ccrs.PlateCarree())
    
    # Regular ticks (not gridliner labels) so constrained layout reserves space for both axes on resize
    ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5, linestyle='--')
    ax.set_xticks(np.arange(np.ceil(lon_lim[0]), lon_lim[1] + 1e-9, 1), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(np.ceil(lat_lim[0]), lat_lim[1] + 1e-9, 1), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # timestamp_unix is unit-safe (timestamps may be us or ns)
    t_hours = df['timestamp_unix'] % (3600 * 24) / 3600
    vmin, vmax = t_hours.min(), t_hours.max()

    scatter = ax.scatter(df[lon_col], df[lat_col],
                         c=t_hours, cmap='viridis', s=20, alpha=0.7,
                         vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree(),
                         label='Glider messages')


    # plot OGN receiver lat/lon locations (glider airfields)
    ax.scatter(ogn_airfields[lon_col], ogn_airfields[lat_col],
        s=20, alpha=1.0, color='darkorange', label='Glider Airfields')
    # plot OGN receiver lat/lon locations (other locations)
    ax.scatter(ogn_other_locations[lon_col], ogn_other_locations[lat_col],
        s=20, alpha=1.0, color='grey', label='Other Locations')


    # plot UK sonde lat/lon launch locations (23:15Z launch)
    ax.scatter(sonde_lons_23Z, sonde_lats_23Z, s=20, alpha=1.0,
        color='dodgerblue', label='Radiosondes 23:15Z')
    # highlight sondes with a 11:15Z launch
    ax.scatter(sonde_lons_11Z, sonde_lats_11Z, s=20, alpha=1.0,
        color='orangered', label='Radiosondes 11:15Z')




    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Time (hr of day)')
    ax.set_title('Spatial distribution of glider messages')
    ax.legend(loc='upper right')
    plt.show()


def parse_times(t):
    '''
    Parsing function used to extract yyyymmdd and hhmmss from epoch time.

    Used to extract the start and hhmmss.
    These are used when saving the filtered gliders parquet file to ID the file.

    :param float t: epoch start or end time.

    :return: tuple containing the start/end yymmdd and hhmmss.
    :rtype: tuple(str)
    '''
    t_datetime = datetime.fromtimestamp(int(t), tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
    
    # extracting the yyyymmdd and hhmmss
    ymd, hms = t_datetime.split()

    # joining the yyyy-mm-dd to yyyymmdd format
    ymd = ymd.split('-')
    ymd = ''.join(ymd)

    # joining the hh:mm:ss to hhmmss format
    hms = hms.split(':')
    hms = ''.join(hms)

    return (ymd, hms)


def save_parquet(df, start, end, return_name=False):
    '''
    Save the filtered glider DataFrame (by loc and time) to a Parquet file for later analysis.
    Parquet is more efficient than CSV, and can be read directly in with pandas for analysis.

    :param pandas.DataFrame df: filtered DataFrame contiaining glider data
        within valid locations and times.
    :param float start: Start time (epoch time) of the data collection.
    :param float end: End time (epoch time) of the data collection.
    :param bool return_name: If True, return the output file name.

    :return: Name of the output parquet file if return_name is True, else None.
    :rtype: str or NoneType
    '''

    # converting the epoch time to datetime and extracting the yyyymmdd and hhmmss components
    start_ymd, start_hms = parse_times(start)
    end_ymd, end_hms = parse_times(end)

    output_name = f'aprs_data_{start_ymd}_T{start_hms}-{end_hms}_UKfilter.parquet'
    # modify filename as needed for your data stream
    df.to_parquet(output_name)

    if return_name:
        return output_name




# # 4. Grouping of messages

def group_glider_messages(df):
    '''
    Group the messages by glider ID and print the number of gliders that have >1000 messages.

    :param pandas.DataFrame df: filtered DataFrame contiaining glider data
        within valid locations and times.

    :return:
        A tuple containing:
        - message_counts: zipped list mapping each aircraft name to its message count.
        - unq_f: dictionary mapping each aircraft name to its OGN data DataFrame.
        Both dictionaries include all aircraft in df.
    :rtype: tuple[list[str, int], dict[str, pandas.DataFrame]]
    '''
    # Now with the messages filtered, we can group them into individual flight tracks.
    # Group the glider messages by aircraft (using the name identifier) and store in a dictionary 'unq_f', where the keys are the aircraft names and the values are DataFrames containing the messages for each aircraft.

    unq_f = {}      # unique flights
    grouped = df.groupby('name')      # Group rows in the DataFrame by the 'name' column, which identifies individual aircraft


    for icao, group in grouped:
        unq_f[icao] = group

    print(f"num of unique aircraft: {len(unq_f.keys())}")
    unq_f.keys()
    nmes = np.empty(len(unq_f.keys()),dtype=int)
    keys = np.empty(len(unq_f.keys()), dtype='U32')  # Use string dtype for keys


    i = 0
    for key in unq_f.keys():
        nmes[i], keys[i] = len(unq_f[key]), key     # nmes: number of messages for this aircraft; key: name of the aircraft
        i += 1

    # Sort keys and nmes together in descending order of nmes
    sorted_indices = np.argsort(nmes)[::-1]     # Get indices that would sort
    nmes = nmes[sorted_indices]                 # Sort nmes
    keys = keys[sorted_indices]                 # Sort keys using the same indices

    print(f"nmes sorted in descending order:\n {nmes}")

    # Looking for around 1000 messages or more for a flight track to be of sufficient length to observe thermal soaring.
    # Often partial tracks are picked up since we taken a snippet in time, so may catch only the beginning or end of a flight, and some gliders may fly beyond receiver range.
    # create a boolean array for when an aircraft has > 1000 messages.
    # sum this array to get the number of aircraft which have more than 1000 messages.
    print(f"number of flights with > 1000 messages: {np.sum(nmes > 1000)}")

    # return dictionaries of:
    #   aircraft ID:number of messages
    #   aircraft ID: OGN data
    # return dict(zip(keys, nmes)), unq_f
    return list(zip(keys, nmes)), unq_f


def select_glider(message_counts, unq_f, keyidx=0,
    output_dir='glider_files', return_df=True):
    '''
    Select and output data from a single glider to a .parquet file using keyidx.
    The flights are sorted by message length, so keyidx = 0 corresponds to the most active glider,
        keyidx = 1 the second most active, and so on.

    This .parquet file will be used in `glider_analysis.py`

    :param: dict[str, int] message_counts: Number of messages from each glider sorted in
        descending count order: i.e. the glider with the most messages appears first in the dict.
    :param: dict[str, pandas.DataFrame] unq_f: The OGN dataframe for each glider.
    :param: int keyidx: select a specific flight by setting the keyidx arg.
    :param: str output_dir: Directory in which to save the glider parquet file.
        Created if it does not exist.
    :param: bool return_df, optional: If True, return the selected glider's DataFrame.
        The parquet file is saved regardless of this setting.
    
    :return: The selected glider's DataFrame if return_df is True, otherwise None.
    :rtype: pandas.DataFrame or None
    '''

    if keyidx < 0 or keyidx >= len(message_counts):
        raise IndexError(f"keyidx must be between 0 and {len(keys) - 1}")

    gkey, message_count = message_counts[keyidx]
    print(f"Glider ID key: {gkey}, number of messages: {message_count}")

    g1df = unq_f[gkey]
    g1df = g1df.sort_values('timestamp')
    g1df = g1df.reset_index(drop=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save out the flight track for this glider to a Parquet file for later analysis.
    filename = output_dir / f"flarm_aprs_{gkey}.parquet"
    g1df.to_parquet(filename, index=False)

    if return_df:
        return g1df



# # 5. Examine Flight Path

def plot_flight_path(message_counts, g1df, keyidx=0):

    # Examine the flight path by looking at altitude vs time, and longitude vs latitude. 
    # A sawtooth pattern in altitude is a strong indicator of thermal soaring, as the glider climbs in a thermal and then glides down to the next thermal, and so on.
    gkey, message_count = message_counts[keyidx]

    fig = plt.figure(figsize=(4, 3))
    plt.plot(g1df.timestamp, g1df.altitude, marker='o', linestyle='', markersize=3, label = gkey)

    plt.legend()
    plt.xlabel('Time')  
    plt.ylabel('Altitude (m)')
    plt.title(f"Altitude vs Time for {gkey}")
    plt.show()

    fig = plt.figure(figsize=(6, 4))
    plt.plot(g1df.longitude[40:-1], g1df.latitude[40:-1], marker='o', linestyle='-', markersize=2, label = gkey, alpha=0.7)
    plt.legend()
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title(f"Flight path for {gkey}")
    plt.show()



def main():

    # start and end time of the collected OGN APRS parquet datafiles
    start_time, end_time = start_end_times(filenames)

    print(f"start: {start_time}")
    print(f"end: {end_time}")


    frames = []
    for f in filenames:
        f = root + f

        df = pd.read_parquet(f)
        frames.append(df)

    df3 = pd.concat(frames, ignore_index=True)



    # # trial code for globbing all parquet files in a directory
    # all_frames = []
    # for f in glider_file_names:
    #     df = pd.read_parquet(f)

    #     all_frames.append(df)

    # df_final = pd.concat(all_frames, ignore_index=True)
    # print(df_final)

    print(df3)


    ##################################
    ### 3. Filtering OGN APRS Data ###
    ##################################
    print("*******************************************************************")
    print("Filtering OGN APRS Data for Gliders Within UK and Valid Time Bounds")
    print("*******************************************************************")

    gliders_df = get_gliders(df3)
    gliders_df = create_unix_time_col(gliders_df)


    print(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time)))
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end_time)))


    gliders_uk_df = filter_loc_time(gliders_df, start_time, end_time)



    # UK OGN Receiver Locations
    ogn_airfields = load_ogn_receiver_locations(
    'ogn_receiver_airfields_active.parquet'
    )
    ogn_other_locations = load_ogn_receiver_locations(
        'ogn_receiver_other_locations_active.parquet'
    )
    # Visualise the distribution of glider messages over the UK for the filtered data, coloured by time of day.
    plot_messages(gliders_uk_df, ogn_airfields=ogn_airfields, ogn_other_locations=ogn_other_locations)



    # saving filtered glider data to parquet file and 
    output_filename = save_parquet(gliders_uk_df, start_time, end_time, return_name=True)
    print(f"Filtered gliders data stored at:\n{output_filename}")

    # Read back in the Parquet file if required. Otherwise, the gliders_uk_df DataFrame is already in memory and can be used for further analysis.
    # gliders_uk_df = pd.read_parquet(output_filename)

    print(gliders_uk_df)


    #########################
    ### 4. Group Messages ###
    #########################
    print("*************************************")
    print("Group Messages by Glider Flight Track")
    print("*************************************")

    message_counts, unq_f = group_glider_messages(gliders_uk_df)


    print(f"message_counts:\n {message_counts}")
    print(f"unq_f:\n {unq_f.keys()}")

    # create a loop to loop through all the most active gliders and output these to .parquet files
    keyidx_no = 0
    # storing the data from the most active glider into a pandas DataFrame
    g1df = select_glider(message_counts, unq_f, keyidx=keyidx_no)


    print(f"Retrieving Gliders with >1000 messages:")
    for i, j in message_counts:   
        if j > 1000:
            print(f"Retrieving Glider {i} as it has >1000 messages.")
            select_glider(message_counts, unq_f, keyidx=keyidx_no, return_df=False)
            keyidx_no += 1

    keyidx_sel = 0
    print(f"Selected glider dataframe for {message_counts[keyidx_sel][0]}:\n {g1df}")



    ##############################################
    ### 5. Examine Selected Glider Flight Path ###
    ##############################################
    print("*******************")
    print("Examine Flight Path")
    print("*******************")

    # Examine the flight path by looking at altitude vs time, and longitude vs latitude. 
    # A sawtooth pattern in altitude is a strong indicator of thermal soaring, as the glider climbs in a thermal and then glides down to the next thermal, and so on.
    plot_flight_path(message_counts, g1df, keyidx=keyidx_sel)

    # Visualise the selected glider track over the UK for the filtered data, coloured by time of day.
    plot_messages(
        g1df,
        ogn_airfields=ogn_airfields,
        ogn_other_locations=ogn_other_locations,
    )


if __name__ == '__main__':
    main()


exit()
