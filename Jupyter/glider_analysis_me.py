#!/usr/bin/env python
# coding: utf-8


# name: glider_analysis
# date: 27/07/2026
# author: Toby Alexander

# contributor: Matt Evans
# date: 22/09/2026

# description: 
# Analyse historical FLARM data queried from the OpenSky Network's Trino database (https://openskynetwork.github.io/opensky-api/trino.html)
# Includes further parsing and filtering of the data stream to examine spatial distribution of messages and identify unique flights.

# required modules:
#   - Python 3.6+ (created with 3.10.19)
#   - matplotlib
#   - numpy
#   - pandas
#   - scipy


# # 0.1 Imports
import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

import sys
from pathlib import Path


# # 0.2 Global Constants and Configuration

# Global Constants

TURN_RATE_MIN = 0.02
MIN_DURATION = 60

# Configuration

gliderroot = "../Sample Data/OGN_sample_2/"
gliderfile = "flarm_aprs_FLRDDE434.parquet"

# # globbing all the glider flight files and storing the file names in a list
# # leave commented out for now - will be needed when dealing with several glider flights
# glider_files = list(Path(gliderroot).glob("flarm_aprs_*.parquet"))
# glider_file_names = sorted([i for i in glider_files])

# [print(f) for f in glider_file_names]


gliderpath = gliderroot + gliderfile
# Extracting the flight ID and storing this as a global variable
fid = gliderpath.split('_')[-1].split('.')[0]
print(fid)

outdir = "."

# # 0.3 Shared Helper Functions

# def bearings(x, y):

# def angular_rate(t, b, win_smooth=9, polyorder=2):

# def w_comp(t, z):

# def b_err(u, uerr, v, verr):


# # 1. Load in data

def load_data(filepath):
    # Load in data from parquet file, straight into a pandas dataframe.

    fid = filepath.split('_')[-1].split('.')[0] # extract file id from filename. Assuming filename format is '_' delimited and file id is the last element before the file extension.
    print(f"file id: {fid}")    

    dfm = pd.read_parquet(filepath)

    return dfm, fid



# # 2. Extract parameters and examine flight tracks

def preprocess_track(df):
    '''
    Process glider dataframe to:
        - optionally select one flight from one aircraft: there could be multiple flights from one aircraft
        - drop duplicate timestamp entries

    :param pandas.DataFrame df: Glider flight OGN APRS DataFrame

    :return: useful parameters extracted from the processed glider DataFrame
    :rtype: numpy.ndarrays (containing floats)
    '''

    # OGN APRS supplies datetime objects, which need converting to timestamps.
    t0 = df['timestamp'].iloc[0].timestamp()
    t_m = df['timestamp'].apply(lambda x: x.timestamp())  

 
    # function to roughly project lat/lon coords onto x-y plane
    # uses Equirectangular projection method to achieve this
    def latlon_to_xy_equirect(lat, lon, lat0, lon0):
        R = 6.378137*1e6 # mean Earth radius in meters
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        lat0_rad = np.radians(lat0)
        lon0_rad = np.radians(lon0)
        x = R * (lon_rad - lon0_rad) * np.cos(lat0_rad)
        y = R * (lat_rad - lat0_rad)

        return x, y

    # Some files contain multiple flights from the same aircraft, occuring at different times across the day. 
    # Possible to create a mask to select only the first flight, or flights within a certain time window.
    # mask to select one flight only

    masking = False # True: apply mask to select one flight only. False: use all data in the file.
    start_time = 0 # seconds after start of file to start the mask. Reference from the first timestamp in the file, t0.
    end_time = 1500 # seconds after start of file to end the mask. Reference from the first timestamp in the file, t0.

    if masking:
        mask = (t_m >= t0 + start_time) & (t_m <= t0 + end_time)
        df = df[mask].copy()
    else:
        df = df

    
    # Create numpy arrrays of the relevant columns.

    # OGN APRS supplies datetime objects, which need converting to timestamps. 
    # OGN has duplicate timestamps, perhaps due to seconds rounding, which breaks thermal detection algorithm due to division by zero on gradient calculations.
    # For now, just drop duplicate timestamps.
    # Find rows where the timestamps match. Keep only the first occurrence of each unique timestamp.
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    t = df['timestamp'].apply(lambda x: x.timestamp()).to_numpy() # time, s


    lat = df['latitude'].to_numpy()    # latitude, deg
    lon = df['longitude'].to_numpy()   # longitude, deg
    alt_m = df['altitude'].to_numpy()  # GPS altitude, m


    # find reference points lat0 and lon0 from midpoint coords of covered area
    min_lat = np.min(lat)
    max_lat = np.max(lat)
    lat0 = 0.5*(max_lat+min_lat)

    min_lon = np.min(lon)
    max_lon = np.max(lon)
    lon0 = 0.5*(max_lon+min_lon)


    # project lat and lon onto x-y plane using function defined in the above cell
    x, y = latlon_to_xy_equirect(lat, lon, lat0, lon0)

    print(f"elapsed time: {(t[-1] - t[0])/3600:6.2f} hours")
    print(df.head())

    return t, lat, lon, alt_m, x, y


# Exploratory plots plotting properties of the pre-processed glider data

def smooth_track(x, y, window_length=15, polyorder=2):
    # Apply a savgol filter to smooth the x and y data. This isn't used for processing, only exploratory visualisation.
    sgx = savgol_filter(x, window_length, polyorder)
    sgy = savgol_filter(y, window_length, polyorder)

    return sgx, sgy


def plot_altitude(t, alt):
    # Examine altitude vs time. Periods of climb can indicate thermalling behaviour.

    plt.figure(figsize = (6, 4))
    plt.scatter(t - t[0], alt, s=1, alpha=0.5)
    plt.xlabel('flight time (s)')
    plt.ylabel('altitude (m)')

    plt.show()

def plot_time_intervals(t):

    # Examine time steps between adjacent messages to check for any large gaps in the data stream. Large gaps may indicate periods of no flight, or periods where the aircraft was not transmitting.
    time_diff = np.diff(t)
    # time_diff = np.concatenate([0, time_diff])

    plt.figure(figsize = (8, 6))

    plt.plot(t[1:] - t[0], time_diff, linestyle = '', marker = '.')

    plt.xlabel('time, s')
    plt.ylabel('time interval, s')
    plt.ylim([0, 10])
    print(t)
    print(f"Average time interval between adjacent messages: {np.mean(time_diff[time_diff < 1800]):.2f} s")

    plt.figure(figsize=(8, 6))
    plt.hist(time_diff, bins=100, range=[-0.5, 10], color='blue')
    plt.xlabel('Time interval (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of time intervals between messages')
    plt.suptitle(f"Flight ID: {fid}")
    plt.grid(True)
    plt.show()


def plot_lat_lon(lat, lon, x, y):
    # plot lat and lon of xy data

    plt.figure(figsize = (6, 4))

    plt.plot(lon, lat, linestyle = '', marker = 'o', markersize=0.25)
    plt.plot(lon[0], lat[0], linestyle = '', marker = 'o', markersize=5, color = 'g', label = 'start')
    plt.plot(lon[-1], lat[-1], linestyle = '', marker = 'o', markersize=5, color = 'r', label = 'end')
    
    plt.xlabel('longitude')
    plt.ylabel('latitude')

    # plt.scatter(lon, lat, s=10, alpha=0.5)
    # plt.xlabel('longitude')
    # plt.ylabel('latitude')
    
    plt.legend()

    # plot 2D projection of xy data
    plt.figure(figsize = (6, 4))

    plt.plot(x/1e3, y/1e3, linestyle = '', marker = 'o', markersize=0.25)
    plt.plot(x[0]/1e3, y[0]/1e3, linestyle = '', marker = 'o', markersize=5, color = 'g', label = 'start')
    plt.plot(x[-1]/1e3, y[-1]/1e3, linestyle = '', marker = 'o', markersize=5, color = 'r', label = 'end')
    plt.xlabel('x, km')
    plt.ylabel('y, km')
    plt.legend()
    plt.show()


    # Apply a savgol filter to smooth the x and y data. This isn't ultimately used for processing, only visualisation.
    sgx, sgy = smooth_track(x, y)


    plt.figure(figsize = (6, 4))

    plt.plot(sgx, sgy, linestyle = '', marker = 'o', markersize=0.25)
    plt.plot(sgx[0], sgy[0], linestyle = '', marker = 'o', markersize=5, color = 'g', label = 'start')
    plt.plot(sgx[-1], sgy[-1], linestyle = '', marker = 'o', markersize=5, color = 'r', label = 'end')
    plt.xlabel('Savgol Smoothed x (m)')
    plt.ylabel('Savgol Smoothed y (m)')
    # plt.xlim([-2500, 2500])
    # plt.ylim([-22500, -15000]) 
    plt.legend()
    plt.show()




# # 3. Detect thermal soaring events


## Code to detect circling windows based on angular rate thresholding

'''
Helper functions to be used in the downstream thermal detection functions. These calculate:

1. Bearings
2. Angular Rate
3. Vertical Speed
'''

def bearings(x, y):
    # returns bearing (radians) at each sample (atan2 of forward diff)
    dx = np.diff(x)
    dy = np.diff(y)
    b = np.arctan2(dy, dx)
    # append last bearing to keep same length
    return np.concatenate([b, b[-1:]])

def angular_rate(t, b, win_smooth=9, polyorder=2):
    # unwrap then compute derivative db/dt -> angular rate (rad/s)
    # unwrap to avoid discontinuities of phase at 2pi
    bn = np.unwrap(b)
    # optional smoothing to reduce GPS noise; window length must be odd and <= len(b)
    if win_smooth is not None and win_smooth >= 3 and win_smooth % 2 == 1 and win_smooth < len(bn):
        bn_s = savgol_filter(bn, win_smooth, polyorder)
    else:
        bn_s = bn

    db = np.gradient(bn_s, t)
    return db  # rad/s

def w_comp(t, z):
    # returns vertical speed (m/s) at each sample (gradient of z)
    dzdt = np.gradient(z, t)
    return dzdt



def detect_circling_windows(t, x, y, alt_m,
                            ang_rate_thresh=0.02,   # rad/s
                            min_duration=8.0,      # seconds
                            gap_merge=2.0,         # seconds, merge windows closer than this
                            win_smooth=9):
    """
    Detect the thermal circling/soaring windows during the glider flight.

    :param numpy.ndarray (floats) t: timestamp along glider flight (seconds since Epoch)
    :param numpy.ndarray (floats) x: x-position along glider flight (metres)
    :param numpy.ndarray (floats) y: y-position along glider flight (metres)
    :param numpy.ndarray (floats) alt_m: GPS altitude along glider flight (metres)

    :param float ang_rate_thresh: absolute angular rate threshold in rad/s
    :param float min_duration: minimum duration of a window to accept (seconds)
    :param float gap_merge: merge windows separated by gaps shorter than this (seconds)
    :param int win_smooth: length of the filter window (optional)

    :return: list of tuples containing (start_idx, end_idx) inclusive windows where circling is detected.
    :rtype: list
    """


    # requires minimum number of samples to run
    if len(t) < 3:
        return []

    # Calculate instantaneous bearings, angular rate, and vertical speed
    b = bearings(x, y)
    angrate = angular_rate(t, b, win_smooth)
    dzdt = w_comp(t, alt_m)

    # boolean mask where ang_rate and dzdt exceeds thresholds. True where thermal circling is detected (i.e., angular rate exceeds threshold and vertical speed is positive)
    mask = (np.abs(angrate) >= ang_rate_thresh) & (dzdt > 0)  # only consider circling when climbing (positive vertical speed)

    # Find contiguous True windows
    # Change boolean mask to int (0s and 1s) and take the difference between adjacent elements.
    dif = np.diff(mask.astype(int))   

    # The start of a window occurs when a 0 is followed by a 1 (i.e., the difference is +1).
    # And the end of a window occurs when a 1 is followed by a 0 (i.e., the difference is -1). 
    # The np.where function returns the indices where these conditions are met, and we add 1 to the start indices to account for the shift caused by the np.diff operation.  
    starts = np.where(dif == 1)[0] + 1
    ends = np.where(dif == -1)[0]
    # edge cases
    # if the first element is True, then the first window starts at index 0. 
    # The dif method will not capture this, so we manually add it to the starts array.
    if mask[0]:     
        starts = np.concatenate(([0], starts))
    # Similarly, if the last element is True, then the last window ends at the last index.
    if mask[-1]:
        ends = np.concatenate((ends, [len(mask)-1]))
    # convert start and end indices to durations of windows, filter by min_duration
    windows = []
    for s, e in zip(starts, ends):
        dur = t[e] - t[s]
        if dur >= min_duration:
            windows.append([s, e])
    if not windows:
        return []

    # Noisy data can cause the instantaneous angular rate to dip below the threshold, leading to detection drop outs across a circling event.
    # Therefore, merge windows separated by short gaps. They are almost certainly part of the same circling event if the gap is short enough.
    merged = [windows[0]]
    for s, e in windows[1:]:
        prev_s, prev_e = merged[-1]
        gap = t[s] - t[prev_e]  # measure time interval between the end of the previous window and the start of the current window.
        if gap <= gap_merge:
            # merge
            merged[-1][1] = e   # if the gap is less than the merge threshold, extend the end of the previous window to the end of the current window, effectively merging them into a single window.
        else:
            merged.append([s, e])   # if the gap is larger than the merge threshold, treat it as a separate window and append it to the merged list.

    merged_final = []

    # Also enforce min duration of 60s on merged windows before finalising append. 
    # Pilots may enter a promising thermal for a short time.
    # But ultimately decide the lift is insufficient and exit.
    # These are often too short to be worth processing further, so we can filter them out.
    for s, e in merged:
        if t[e] - t[s] >= 60.0:  
            merged_final.append([s, e])

    # return as list of tuples (start_idx, end_idx)
    return [(int(a), int(b)) for a, b in merged_final]



def plt_therm_res(t, x, y, alt_m, windows, ang_rate_thresh=0.02):
    # Plot the glider trajectories and angular rate, for each thermal window.

    # Plot 3D path
    fig = plt.figure(figsize=(8,6))
    tidx = 0
    ax = fig.add_subplot(111, projection = '3d')
    # ax.plot(x, y, alt_m, '-')
    for s,e in windows:
        ax.plot(x[s:e+1], y[s:e+1], alt_m[s:e+1], markersize=0.25, label=f"thermal {tidx}")
        tidx += 1

    plt.title('Trajectory (detected circling highlighted)')

    plt.plot(x[0], y[0], alt_m[0], linestyle = '', marker = '+', markersize=5, color = 'g', label = 'start')
    plt.plot(x[-1], y[-1], alt_m[-1], linestyle = '', marker = 'x', markersize=5, color = 'r', label = 'end')

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("altitude (m)")
    # ax.legend()
    # ax.legend(loc="center left", bbox_to_anchor=(-0.3, 0.5))
    fig.subplots_adjust(left=0.10)  # make room for the legend
    plt.tight_layout()
    plt.show()

    fig2 = plt.figure(figsize=(6,4))
    ax2 = fig2.add_subplot(111)
    b = bearings(x,y)
    angrate = angular_rate(t,b,win_smooth=9)
    ax2.scatter(t, angrate, s=5, label='angular rate (rad/s)')
    for s,e in windows:
        ax2.axvspan(t[s], t[e], color='orange', alpha=0.3)
    ax2.axhline(ang_rate_thresh, color='k', linestyle='--')
    ax2.axhline(-ang_rate_thresh, color='k', linestyle='--')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('angular rate (rad/s)')
    ax2.set_ylim([-1, 1])
    ax2.legend()
    plt.tight_layout()
    plt.show()

    fig3 = plt.figure(figsize=(8,6))
    ax3 = fig3.add_subplot(111)
    ax3.plot(x, y, linestyle = '', marker = '.', markersize=0.5)
    ax3.plot(x[0], y[0], linestyle = '', marker = '+', markersize=10, color = 'g', label = 'start')
    ax3.plot(x[-1], y[-1], linestyle = '', marker = 'x', markersize=10, color = 'r', label = 'end')  

    tidx = 0
    for s,e in windows:
        ax3.plot(x[s:e+1], y[s:e+1], marker = '.', markersize=0.5, label=f"thermal {tidx}")
        tidx += 1

    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    # ax3.legend()
    # ax3.legend(loc="center left", bbox_to_anchor=(1.1, 0.5))
    # fig3.subplots_adjust(right=0.3)  # make room for the legend
    plt.tight_layout()


def plt_window_height_gained(alt_m, windows):
    # Height ascended through for each thermal
    dz = []
    for s, e in windows:

        dz.append(alt_m[e] - alt_m[s])

    plt.figure(figsize=(6,4))
    plt.scatter(np.linspace(0, len(windows)-1, len(windows)), dz)
    plt.grid(True)
    plt.xlabel('window index')
    plt.ylabel('altitude gain (m)')
    plt.title(f"Altitude gain during detected thermalling windows")
    plt.show()



# # 4. Inspect detected thermal soaring events


def plot_window_alt_track(t, alt_m, x, y, windows, nwin=0):
    # Inspect individual windows
    # nwin = 0 # choose index of window to inspect
    s = windows[nwin][0]
    e = windows[nwin][1]

    print(f"start {s}")
    print(f"end {e}")

    print(f"dz: {alt_m[e] - alt_m[s]:.1f} m, duration: {t[e] - t[s]:.1f} s")

    fig = plt.figure(figsize=(16, 6)) 
    fig.fontsize = 14
    ax = fig.add_subplot(121)
    colors = plt.cm.viridis(np.linspace(0,1,len(windows)))

    ax.plot(x[s:e]-x[s], y[s:e]-y[s], color=colors[nwin])
    ax.scatter(x[s:e]-x[s], y[s:e]-y[s], color='k')
    ax.plot(x[s]-x[s], y[s]-y[s], color='g', marker='o', markersize=10, label='start')
    ax.plot(x[e]-x[s], y[e]-y[s], color='r', marker='o', markersize=10, label='end')

    ax.tick_params(labelsize=16)
    ax.set_xlabel("X (m)",fontsize=16)
    ax.set_ylabel("Y (m)", fontsize=16)
    ax.set_title(f"Detected Thermalling Circle for Window {nwin} of flight {fid}", fontsize=16)
    ax.legend(fontsize=14)

    ax2 = fig.add_subplot(122)

    colors = plt.cm.viridis(np.linspace(0,1,len(windows)))

    ax2.scatter(t[s:e]-t[s], alt_m[s:e], color=colors[nwin], label=f"Segment {nwin}")

    ax2.tick_params(labelsize=16)
    ax2.set_xlabel("time (s)", fontsize=16)
    ax2.set_ylabel("Altitude (m)", fontsize=16)
    ax2.set_title(f"Z Profile of Detected Thermalling Circle for Window {nwin} of flight {fid}", fontsize=16)
    # ax2.legend(fontsize=14)
    plt.tight_layout()
    plt.show()

    # fig.savefig(f"{outdir}/nonlinfitx_thermal_{ntherm}_fid_{fid}.png", dpi=600)


def trim_window(windows, nwin=0, start_trim=0, end_trim=0):
    """
    Trim a chosen number of data points from the start and end of a chosen window.
    This is useful in removing any initial or final segments of the window where the aircraft may not be fully engaged in thermalling behaviour.

    Unlikely to need this section but could be useful to remove transient tails entering into helical thermal soaring windows.

    :param list (tuples) windows: list of the (start_idx, end_idx) of windows
    :param int nwin: index number of window to trim
    :param int start_trim: number of data points to trim from start of the chosen window
    :param int end_trim: number of data points to trim off the end of the chosen window

    :return: list of tuples containing trimmed (start_idx, end_idx) inclusive windows where circling is detected.
    :rtype: list
    """

    s, e = windows[nwin]
    windows[nwin] = (s + start_trim, e - end_trim) # update start and end indices of the window to reflect the cut

    return windows


def plot_trimmed_window_xy_components(t, x, y, windows, nwin=0):
    # Inspect the x and y components of the trimmed window.
    # In reality, this function can also analyse un-trimmed windows too.

    # This will update the start and end indices of the thermal window to match the trimmed indices (above)
    s, e = windows[nwin]


    fig = plt.figure(figsize=(8, 4))
    fig.fontsize = 14

    ylim_top = np.max([x[s:e]-x[s], y[s:e]-y[s]])*1.05/1e3  # scale in km for plotting
    ylim_bot = np.min([x[s:e]-x[s], y[s:e]-y[s]])*1.05/1e3  # scale in km for plotting
    if ylim_bot > 0:
        ylim_bot = 0

    colors = plt.cm.viridis(np.linspace(0,1,len(windows)))
    ax = fig.add_subplot(121)

    ax.scatter(t[s:e]-t[s], (x[s:e]-x[s])/1e3, s=5, color=colors[nwin])

    ax.set_ylim([ylim_bot, ylim_top])
    ax.tick_params(labelsize=16)
    ax.set_xlabel("time (s)",fontsize=16)
    ax.set_ylabel("X (km)", fontsize=16)
    ax.set_title(f"X component for window {nwin}; id {fid}", fontsize=12)
    ax.grid(True)

    ax2 = fig.add_subplot(122)

    colors = plt.cm.viridis(np.linspace(0,1,len(windows)))

    ax2.scatter(t[s:e]-t[s], (y[s:e]-y[s])/1e3, s=5, color=colors[nwin])

    ax2.set_ylim([ylim_bot, ylim_top])
    ax2.tick_params(labelsize=16)
    ax2.set_xlabel("time (s)", fontsize=16)
    ax2.set_ylabel("Y (km)", fontsize=16)
    ax2.set_title(f"Y component for window {nwin}; id {fid}", fontsize=12)
    ax2.grid(True)
    plt.tight_layout()
    plt.show()




# # 5,6. Bin chosen thermal soaring window and fit to model.


# Attempt non linear method. Each window containing a thermal is binned into smaller bins and then each bin is modelled to extract linear drift, angular rate, and other parameters.
# The function was built to process multiple thermalling windows at once, but each window needs individual attention, so this level of automation is not currently possible.

'''
Helper functions to be used in the analysis functions later in the code:

1. b_err(): Determine normalised bearing angles and associated errors in degrees
2. bearings(): Compute bearing angles (defined earlier in script)
'''

def b_err(u, uerr, v, verr): 

    # project u and v onto a bearing in degrees, and propagate the uncertainties in u and v to uncertainty in bearing.

    brad = np.atan2(v,u) # bearing in radians
    bdeg = 270 - 180*brad/np.pi  # convert to deg, and adjust from math convention (0deg East, CCW, direction vector points *TO*) to meteorological bearing convention (0deg North, CW, direction winds travel *FROM*).
    nbdeg = bdeg % (360)   # wrap around to 0-360 deg

    # Standard error propagation formulae for atan2 function
    dbdu = (-1/v)*(1/(1+(u/v)**2))
    dbdv = (u/(u**2+v**2))

    dbrad = np.sqrt((dbdu*uerr)**2 + (dbdv*verr)**2)
    ndbrad = dbrad % (2*np.pi)
    dbdeg = 180*ndbrad/np.pi

    return nbdeg, dbdeg     # nbdeg: normalised bearing in degrees, dbdeg: uncertainty in bearing in degrees



def bin2(t_data, nz_data, sbin, P, Ptrim):
    '''
    Bin a thermalling window based on number of turns achieved.
    This is preferred since vertical speed varies a lot.

    :param numpy.ndarray (floats) t_data: Time data in seconds for selected thermal window
    :param numpy.ndarray (floats) nz_data: Vertical height data in meters normalised to start 0
    :param float sbin: Bin size in metres (not used)
    :param float P: Estimated period of circling
    :param float Ptrim: Number of periods to trim from the start of the data to avoid initial transients
        This is important since the initial part of the circling may not be representative of the steady-start circling behaviour.

    :return bidx, nbin: ``bidx`` is a list of tuples containing the bin edges
        ``(start_idx, end_idx)`` for each bin, and ``nbin`` is the number of bins created.
    :rtype: tuple[list(tuples), int]
    '''

    nP = 2 # number of periods/turns per bin

    # Check if there is a short time window
    if len(t_data) < 3:
        print(f"Skipping binning: window is too short to define a valid bin: {len(t_data)}")
        print(f"Therefore, cannot define a valid bin.")
        return [], 0

    # Check if the window duration is less than the minimum required time for one full bin (i.e. < 2 periods)
    if (t_data[-1] - t_data[0]) < (nP * P):
        print(f"Skipping binning: window is less than {nP} periods: {t_data[-1] - t_data[0]} < {nP * P}")
        print(f"Therefore, cannot define a valid bin.")
        return [], 0
    

    trim_idx = np.argmin(abs(t_data - t_data[0] - Ptrim*P))  # find the index for the time after N Ptrim periods
    t_data_trim = t_data[trim_idx:] - t_data[trim_idx]   # trim the data down and normalize time to start at 0 for better numerical stability
    nz_data_trim = nz_data[trim_idx:]

    # nbin = int((t_data[-1]-t_data[0])/(nP*P)) 
    # should nbins be based on t_data_trim rather than t_data ? Yes.
    nbin = int((t_data_trim[-1]-t_data_trim[0])/(nP*P)) # how many full bins can be made based on the number of periods in the thermal soaring data window. Each bin will contain np=2 full periods of circling.
    print(f"nbins: {nbin}")
    bidx = []

    prevedege = trim_idx # the start index for the first bin will be the first index after the trim.
    for s in range(1, nbin+1):
        # find the index for the time corresponding to a further 2 full periods
        edge = np.argmin(abs(t_data_trim - t_data_trim[0] - (nP*P*s))) + trim_idx + 1   # remember to add back the 'trim_idx' offset to get the correct index in the original data
        bidx.append((prevedege, edge))
        prevedege = edge

    # bin edges are the indices of the period start and stop times in the original times list
    print("bin edges:")
    print(bidx)

    return bidx, nbin


'''
Helper functions be used in the analysis functions later in the code:

1. x_model: model (osc + linear drift) for x position of glider motion
2. y_model: model (osc + linear drift) for y position of glider motion
3. linear_model: evaluate linear drift of model. Used to get residuals for initial guess of angular rate
'''

# Define models of thermal soaring
def x_model(t, omega, R, phi, u, x0):

    return R * np.cos(omega * t + phi) + u*t + x0

def y_model(t, omega, R, phi, v, y0):

    return R * np.sin(omega * t + phi) + v*t + y0

def linear_model(t, v, x0):
    return v * t + x0


def omega_guess(t_data, x_data, y_data, plot=True):
    '''
    Initial guess for angular rate based on linear regression of bearings against time (db/dt) (for the entire window).

    This is used to inform the binning process.
    A similar approach is used in the function 'nonlinmdl' to initialize the non-linear fitting procedure (per bin).

    :param numpy.ndarray (floats) t_data: Time data in seconds for given window
    :param numpy.ndarray (floats) x_data: x position data in meters for given window
    :param numpy.ndarray (floats) y_data: y position data in meters for given window
    :param bool plot, optional: If True, generate a diagnostic plot.

    :return omega0: Initial guess for the angular rate of the thermal soaring window in rad/s
    :rtype: float
    '''

    tn = t_data - t_data[0]  # normalize time to start at 0 for better numerical stability
    xn = x_data - x_data[0]  # normalize position to start at 0 for better numerical stability
    yn = y_data - y_data[0]
    T = tn[:-1, np.newaxis]**[0, 1]  # Design matrix for linear regression

    p_x = np.linalg.lstsq(T, xn[:-1])
    p_y = np.linalg.lstsq(T, yn[:-1])

    # Extract linear fit parameters for initial guess
    u0 = p_x[0][1]
    v0 = p_y[0][1]
    x00 = p_x[0][0]
    y00 = p_y[0][0]

    # evaluate linear model to get residuals for initial guess of angular rate
    x_eval_lin = linear_model(tn[:-1], u0, x00)
    y_eval_lin = linear_model(tn[:-1], v0, y00)

    # The residual will mostly contain the oscillatory component
    # This is used to estimate angular rate for the fitting procedure initial guess
    x_res_lin = xn[:-1] - x_eval_lin
    y_res_lin = yn[:-1] - y_eval_lin

    b = bearings(y_res_lin, x_res_lin)  # Compute bearings from residuals, exclude last point to match dimensions for angular rate calculation
    bn = np.unwrap(b)

    # Fit the model
    p_bn = np.linalg.lstsq(T, bn)
    omega0 = p_bn[0][1]  # Extract the slope (angular rate)

    
    # Optional debug/inspection plotting
    if plot:
        plot_omega_guess_diagnostics(
            tn, x_res_lin, y_res_lin, bn, T, p_bn, omega0, u0, v0
        )
    
    print(f"omega0: {omega0}")
    return omega0

'''
Plotting helper function to be used in omega_guess - optional.
'''

def plot_omega_guess_diagnostics(tn, x_res_lin, y_res_lin, bn, T, p_bn, omega0, u0, v0):
    # Optional plots for reviewing:
        # 1. x and y oscillatory components 
        # 2. bearing vs time for omega0
    # Plot fitting results from 'omega_guess' for visual inspection on a 1x3 subplot figure

    ylim_top = np.max([x_res_lin, y_res_lin])*1.05  # scale in m for plotting
    ylim_bot = np.min([x_res_lin, y_res_lin])*1.05  # scale in m for plotting
    if ylim_bot > 0:
        ylim_bot = 0

    # subplots: x position, y position, and bearing with linear fits overlaid
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.scatter(tn[:-1], x_res_lin, s=5, label='x component linear fit residuals (m)')
    plt.plot(tn[:-1], x_res_lin, alpha=0.5)
    plt.ylim([ylim_bot, ylim_top])
    plt.xlabel('time (s)')
    plt.ylabel('x position (m)')
    plt.title(f"x comp lin fit res (m). u0~{u0:.3f} m/s")
    plt.legend()
    plt.grid(True)

    plt.subplot(132)
    plt.scatter(tn[:-1], y_res_lin, s=5, label='y component linear fit residuals (m)')
    plt.plot(tn[:-1], y_res_lin, alpha=0.5)
    plt.ylim([ylim_bot, ylim_top])
    plt.xlabel('time (s)')
    plt.ylabel('y position (m)')
    plt.title(f"y comp lin fit res (m). v0~{v0:.3f} m/s")
    plt.legend()
    plt.grid(True)

    plt.subplot(133)
    plt.scatter(tn[:-1], bn, s=5, label='bearing (rad)')
    plt.plot(tn[:-1], T @ p_bn[0], color='r', label='linear fit')
    plt.xlabel('time (s)')
    plt.ylabel('bearing (rad)')
    plt.title(f"Estimated angular rate: omega0: {omega0:.3f} rad/s")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Combined residual function
def residuals(params, t, x_data, y_data):
    '''
    Calculate residuals for both x and y models given the parameters and data.

    :param np.ndarray params: array of parameters from the fitting procedure [omega, Rx, phix, u, x0, Ry, phiy, v, y0]
    :param numpy.ndarray (floats) t_data: Time data in seconds for given window
    :param numpy.ndarray (floats) x_data: x position data in meters for given window
    :param numpy.ndarray (floats) y_data: y position data in meters for given window
    
    returns: concatenated residuals for x and y models.
    rtype: np.ndarray
    '''

    # Unpack parameters
    omega = params[0]  # Shared angular rate
    Rx, phix, u, x0 = params[1], params[2], params[3], params[4]  # x-specific parameters
    Ry, phiy, v, y0 = params[5], params[6], params[7], params[8]  # y-specific parameters

    # Calculate residuals for both x and y
    x_residuals = x_data - x_model(t, omega, Rx, phix, u, x0)
    y_residuals = y_data - y_model(t, omega, Ry, phiy, v, y0)

    # Concatenate residuals
    return np.concatenate([x_residuals, y_residuals])


def cov_est(t, result):
    '''
    Estimate covariance matrix of fitted parameters from the Jacobian and residuals.
    
    :param numpy.ndarray (floats) t_data: Time data in seconds for given window 
    :param scipy.optimize.OptimizeResult result: The output object returned
        by :func:`scipy.optimize.least_squares`. It contains the best-fit parameters
        vector in ``result.x``, the residual vector in ``result.fun`` (measure between
        the observations and fitted model e.g. x(t) - x_model), and the
        Jacobian matrix in ``result.jac`` at the final iteration. This object is used
        to compute the covariance estimate from the Jacobian (``result.jac```)
        and residual variance (``result.fun``).
    
    returns out: a dictionary containing estimated parameters, covariance matrix,
        residual norm, and fitting success information.
    rtype: dict
    '''

    p_est = result.x    # extract parameter estimates from the optimization result

    j = result.jac
    # two pairs of residuals per time point: x and y. Therefore 2 is used in len(t)*2.
    dof = max(1, (len(t)*2 - len(p_est)))   # degrees of freedom
    ssr = 2 * np.sum(result.fun**2) / dof   # non-dimensional sum of squared residuals (normalized by degrees of freedom)
    try:
        cov = np.linalg.inv(j.T @ j) * ssr  # covariance approximation (J^T J)^-1 * residual_variance
        cond = np.linalg.cond(j.T @ j)      # condition number of the Jacobian matrix, which indicates how sensitive the solution is to changes in the input data. A high condition number suggests that the problem is ill-conditioned and small changes in the data can lead to large changes in the parameter estimates.
    except np.linalg.LinAlgError:
        cov = np.full((len(p_est), len(p_est)), np.nan)

    print(f"Condition number: {cond}")
    print(f"non dimensional sum of squared residuals: {ssr:.3f}, degrees of freedom: {dof}")
    s_physical = np.sqrt(ssr) * 100  # convert from normalized units to physical units (metres)
    print(f"Typical residual: {s_physical:.2f} m")  

    print("Scaled covariance matrix:")
    print(pd.DataFrame(cov).round(4))

    keys = ['omega', 'Rx', 'phix', 'u', 'x0', 'Ry', 'phiy', 'v', 'y0']
    out = {k: float(v) for k, v in zip(keys, p_est)}
    out['cov'] = cov
    out['residual_norm'] = result.cost
    out['success'] = result.success
    out['message'] = result.message

    return out


def nonlinmdl(fid, t_data, x_data, y_data, z_data, lat, lon, bidx, nbin, wins, ntherm, results, nax, globP, plot=True):
    '''
    Executes non-linear (circling model) fitting procedure for each bin of a given thermalling window.
    Parameters such as angular rate, turn radius, phase, linear drift, and initial positions are extracted: 
    ``[omega, Rx, phix, u, x0, Ry, phiy, v, y0]``
    
    The results are stored in the provided 'results' dictionary.
    This also handles initial signal processing for parameter initialisation.

    The fitting results and residuals for each bin are then displayed on a multi-subplot figure.

    :param int or str fid: Flight ID for the current thermalling window
    :param numpy.ndarray t_data: Time data in seconds for the current thermalling window
    :param numpy.ndarray x_data: x position data in meters for the current thermalling window
    :param numpy.ndarray y_data: y position data in meters for the current thermalling window
    :param numpy.ndarray z_data: vertical position data in meters for the current thermalling window
    :param numpy.ndarray lat: latitude data in degrees for the current thermalling window
    :param numpy.ndarray lon: longitude data in degrees for the current thermalling window
    :param list of tuple bidx: list of tuples containing bin edges ``(start_idx, end_idx)`` for the current thermalling window
    :param int nbin: number of bins created for the current thermalling window
    :param int wins: start index for the current thermalling window in the full dataset
    :param int ntherm: index of the current thermalling window being processed
    :param pandas.DataFrame results: dictionary to store the fitting results for each bin
    :param numpy.ndarray nax: number of subplots/axes needed to visualise all of the bins for the current thermalling window
    :param float globP: estimated time period of circling in seconds for the entire current thermalling window

    returns results: updated dictionary appended with the new fitting results for each bin of the current thermalling window
    rtype: pandas.DataFrame
    '''

    if plot:
        # define a figure in which the fits for each bin will be plotted on their own subplot
        fig = plt.figure(figsize=(4*nax[0], 3*nax[1]))
        fig2 = plt.figure(figsize=(4*nax[0], 3*nax[1]))

    fidarr = fid                        # flight ID for the current thermalling window
    refts = np.empty(nbin)              # reference timestamp for each bin, calculated as the midpoint of the time range for that bin
    reft = np.empty(nbin, dtype=object) # reference time as a datetime object for each bin, for easier human-readable output
    lat0 = np.empty(nbin)               # [deg] reference latitude for each bin, calculated as the midpoint of the latitude range for that bin 
    lon0 = np.empty(nbin)               # [deg] reference longitude for each bin, calculated as the midpoint of the longitude range for that bin
    var = np.empty(nbin)                # variance of the residuals for each bin from diag of cov matrix, used to estimate uncertainty
    omegaest = np.empty(nbin)           # [rad/s] estimated angular rate for each bin, extracted from the non-linear fitting procedure
    omegastd = np.empty(nbin)           # [rad/s] estimated uncertainty in angular rate for each bin, extracted from the covariance matrix of the non-linear fitting procedure
    Rxest = np.empty(nbin)              # [m] estimated turn radius from x component for each bin, extracted from the non-linear fitting procedure
    phixest = np.empty(nbin)            # [rad] estimated phase for x component for each bin, extracted from the non-linear fitting procedure
    Ryest = np.empty(nbin)              # [m] estimated turn radius from y component for each bin, extracted from the non-linear fitting procedure
    phiyest = np.empty(nbin)            # [rad] estimated phase for y component for each bin, extracted from the non-linear fitting procedure
    x0est = np.empty(nbin)              # [m] estimated initial x position for each bin, extracted from the non-linear fitting procedure
    y0est = np.empty(nbin)              # [m] estimated initial y position for each bin, extracted from the non-linear fitting procedure
    uest = np.empty(nbin)               # [m/s] estimated linear velocity for x component for each bin, extracted from the non-linear fitting procedure
    vest = np.empty(nbin)               # [m/s] estimated linear velocity for y component for each bin, extracted from the non-linear fitting procedure
    ustd = np.empty(nbin)               # [m/s] estimated uncertainty in linear velocity for x component for each bin, extracted from the covariance matrix of the non-linear fitting procedure
    vstd = np.empty(nbin)               # [m/s] estimated uncertainty in linear velocity for y component for each bin, extracted from the covariance matrix of the non-linear fitting procedure
    magU = np.empty(nbin)               # [m/s] estimated magnitude of linear velocity for each bin, extracted from the non-linear fitting procedure
    stdmagU = np.empty(nbin)            # [m/s] estimated uncertainty in magnitude of linear velocity for each bin, extracted from the covariance matrix of the non-linear fitting procedure
    bUdeg = np.empty(nbin)              # [deg] estimated bearing of linear velocity for each bin, extracted from the non-linear fitting procedure
    dbUdeg = np.empty(nbin)             # [deg] estimated uncertainty in bearing of linear velocity for each bin, extracted from the covariance matrix of the non-linear fitting procedure
    dz = np.empty(nbin)                 # [m] vertical displacement for each bin
    zhat = np.empty(nbin)               # [m] reference vertical position for each bin, calculated as the midpoint of the vertical position range for that bin
    zdot = np.empty(nbin)               # [m/s] vertical rise rate for each bin, extracted from the linear regression of vertical position against time
    thermidx = np.zeros(nbin, dtype=int)# thermalling window index for each bin, used to associate bins with their corresponding thermalling window in the results dictionary
    thermidx[:] = ntherm                # function is called for each thermalling window, so all bins processed in this call will have the same thermalling window index

    i = 0   # initialse row index for the results arrays, which will be incremented for each bin processed in the loop below.
    for s,e in bidx:

        print(f"\n---------- Bin {ntherm}-{i+1} ------------")      # print header for the current bin being processed.

        # Unpack desired data for the current bin from the full dataset, and normalize to start at 0 for better numerical stability in fitting.    
        tbin = t_data[s:e] - t_data[s]
        xbin = x_data[s:e] - x_data[s]
        ybin = y_data[s:e] - y_data[s]  
        zbin = z_data[s:e]

        refts[i] = 0.5 * (t_data[s] + t_data[e-1])  # reference timestamp for the current bin
        reft[i] = datetime.datetime.fromtimestamp(refts[i], tz=datetime.timezone.utc)   # reference time as a datetime object for easier human-readable output
        lat0[i] = 0.5 * (lat[s] + lat[e])  # reference lat for the bin
        lon0[i] = 0.5 * (lon[s] + lon[e])  # reference lon for the bin

        # Conduct linear regression on each component of position against time to extract linear drift and vertical rise rate. 
        # This is used to inform the initial guess for the non-linear fitting procedure.

        T = tbin[:-1, np.newaxis]**[0, 1]  # Design matrix for linear regression

        p_x = np.linalg.lstsq(T, xbin[:-1])
        p_y = np.linalg.lstsq(T, ybin[:-1])
        p_z = np.linalg.lstsq(T, zbin[:-1])
        zdot[i] = p_z[0][1]     # extract vertical rise rate as a secondary parameter

        # Extract linear fit parameters for initial guess
        u0 = p_x[0][1]
        v0 = p_y[0][1]
        x00 = p_x[0][0]
        y00 = p_y[0][0]

        # evaluate linear model to get residuals for initial guess of angular rate
        x_eval_lin = linear_model(tbin, u0, x00)
        y_eval_lin = linear_model(tbin, v0, y00)

        # The residual will mostly contain the oscillatory component
        # This is used to estimate angular rate for the fitting procedure initial guess
        x_res_lin = xbin - x_eval_lin
        y_res_lin = ybin - y_eval_lin

        # print(f"DEBUG: size of x_res_lin: {x_res_lin.shape}, size of y_res_lin: {y_res_lin.shape}")

        b = bearings(y_res_lin, x_res_lin)[:-1]  # Compute bearings from residuals, exclude last point to match dimensions for angular rate calculation
        bn = np.unwrap(b)   # Unwrap the bearings to avoid sawtooth discontinuities that can affect the linear regression for angular rate estimation

        # Fit the model
        p_bn = np.linalg.lstsq(T, bn)
        omega0 = p_bn[0][1]  # Extract the slope (angular rate)
        # print(f"initial guesses: om: {omega0:.3f}")

        # Turn radius estimation ~ sqrt(2)*RMS of oscillatory component, since the oscillatory component is a sine wave with amplitude R, and RMS of a sine wave is R/sqrt(2). So we multiply by sqrt(2) to get the amplitude from the RMS.
        Rx0 = np.sqrt(2*np.mean(x_res_lin**2))
        Ry0 = np.sqrt(2*np.mean(y_res_lin**2))

        # Phase estimation
        # find time indices corresponding to the first period, based on the omega estimation, to estimate the phase for initial guess

        # The period is 2*pi/omega0
        period = 2*np.pi/abs(omega0)
        print(f"estimated local period based on omega guess: {period:.1f} s")

        if period > 1.5* globP or period < globP * 0.5:  
            # if the estimated period is very different from the global period estimate, default to global period for the initial guess instead, to avoid issues with phase estimation and fitting convergence. 
            # Issues may arise due to noise preventing the bearings unwrapping smoothly.
            # The thresholds are somewhat arbitrary and can be adjusted based on expected variability in period.
            period = globP
            omega0 = 2*np.pi/globP
            print("Error finding local period - setting to global period for initial guess.")

        # Find the first time index that is greater than one period
        # tbin - tbin[0] > period is a boolean array of False (0) and True (1) values
        # np.argmax returns the index of the first True (1) value
        p1x_idx = np.argmax(tbin - tbin[0] > period)
        p1y_idx = np.argmax(tbin - tbin[0] > period)

        if p1x_idx == 0 or p1y_idx == 0:
            print("Error: period is longer than the duration of the data, cannot estimate phase. Setting phase to 0 for initial guess.")
            phix0 = 0
            phiy0 = 0
        else:
            # Estimate phases. Likely to have large error since we are using the omega estimation for this. But should help the fitting converge.
            x_idx = np.argmin(abs((x_res_lin - Rx0)[:p1x_idx]))
            y_idx = np.argmin(abs((y_res_lin - Ry0)[:p1y_idx]))

            tphix = tbin[x_idx] # time corresponding to when Rx0*cos(omega0*t + phix0) is cloes to Rx0
            tphiy = tbin[y_idx]

            # cosine for x term, sine for y term

            phix0 = -omega0*tphix % (2*np.pi)   # wrap to [0, 2pi]

            phiy0 = np.pi/2 - omega0*tphiy % (2*np.pi)  # wrap to [0, 2pi]


        # Now initialisation is complete, proceed with fitting.
        # Non-dimensionalize parameters for better numerical stability in fitting. We will re-dimensionalize after fitting to get the actual parameter estimates.

        A = 100     # Typical scale of motion in metres.

        omega0_nd = omega0 * period  # non-dimensionalize angular rate by multiplying by the estimated period, so that the initial guess is around 2*pi
        Rx0_nd = Rx0 / A  # non-dimensionalize radius by dividing by the overall scale of the oscillation
        Ry0_nd = Ry0 / A # same scale for x and y to avoid issues with different magnitudes
        u0_nd = u0 * period / A  # non-dimensionalize linear velocity by multiplying by period and dividing by scale, so that it is around the same order of magnitude as the oscillatory component
        v0_nd = v0 * period / A
        x00_nd = x00 / A  # non-dimensionalize offset by dividing by scale
        y00_nd = y00 / A        

        # Centre and non-dimensionalize time and position for better numerical stability in fitting.
        # The centering is not strictly necessary, but it can help with numerical stability, especially for the phase parameters, since it makes the time range symmetric around zero.
        # We fit in centred coordinates (t_centred) rather than the raw tbin values so the time axis is centred on zero; this improves conditioning for the intercept/phase terms.

        t_centred = tbin - tbin.mean()   # centre time to have mean of 0 for better numerical stability in fitting

        x_centred = xbin - xbin.mean()
        y_centred = ybin - ybin.mean()

        t_nd = t_centred / period  # normalize time to ~[-1,1] for better numerical stability in fitting
        x_nd = x_centred / A  # normalize x to [-1,1]
        y_nd = y_centred / A  # normalize y to [-1,1]


        # Initial guess: [omega, Rx, phix, u, x0, Ry, phiy,v, y0]
        # p0 = [omega0_nd, Rx0_nd, phix0, u0_nd, x00_nd, Ry0_nd, phiy0, v0_nd, y00_nd]
        p0 = [omega0_nd, Rx0_nd, phix0, u0_nd, x00_nd, Ry0_nd, phiy0, v0_nd, y00_nd]

        # Perform least squares fit
        # result = least_squares(residuals, p0, method = 'lm', args=(t_data, x_data, y_data))
        result = least_squares(residuals, p0, args=(t_nd, x_nd, y_nd)) # args are the actual obs/data the residual function needs

        # estimate errors on fitted parameters using covariance estimation
        out = cov_est(tbin, result)

        # Extract results and re-dimensionalise to get physical units. Basically undo all the operations we applied to non-dimensionalise.
        omegaest[i] = result.x[0] / period  # re-dimensionalize omega by dividing by the period
        Rxest[i], phixest[i], uest[i], x0est[i] = A* result.x[1], result.x[2], result.x[3] * A / period, result.x[4] * A
        Ryest[i], phiyest[i], vest[i], y0est[i] = A* result.x[5], result.x[6], result.x[7] * A / period, result.x[8] * A

        # Evaluate model with fitted parameters to get residuals.
        # teval is a dense time grid over the centred data range so the fitted curve can be plotted smoothly and compared to the binned track.
        teval = np.linspace(t_centred[0], t_centred[-1], int(tbin[-1]-tbin[0]))
        xevalplt = x_model(teval, omegaest[i], Rxest[i], phixest[i], uest[i], x0est[i]) + xbin.mean()  # for plotting smooth function
        yevalplt = y_model(teval, omegaest[i], Ryest[i], phiyest[i], vest[i], y0est[i]) + ybin.mean() # for plotting smooth function

        xeval = x_model(t_centred, omegaest[i], Rxest[i], phixest[i], uest[i], x0est[i]) + xbin.mean()
        xres = xbin - xeval # residuals between x values and x (cosine + linear drift) model data
        yeval = y_model(t_centred, omegaest[i], Ryest[i], phiyest[i], vest[i], y0est[i]) + ybin.mean()
        yres = ybin - yeval

        # Convert u and v components to magnitude and bearing
        magU[i] = np.sqrt(uest[i]**2+vest[i]**2)

        # Total vertical displacement over the full thermalling window.
        dz_window = z_data[-1] - z_data[0]

        # Estimate errors

        var = np.diag(out['cov'])
        sigma = np.sqrt(var)

        # Redimensionalise the standard deviations of the parameters to get them in physical units. 
        # The scaling is the same as for the parameters themselves, since the covariance matrix is in the same units as the parameters.
        sigma_omega     = sigma[0] / period   # omega_nd = omega * period, so sigma_omega = sigma_omega_nd / period
        sigma_Rx        = sigma[1] * A        # A_norm = A / A (trivial)
        sigma_phix      = sigma[2]            # phi is already dimensionless
        sigma_u         = sigma[3] * A / period               
        sigma_x0        = sigma[4] * A
        sigma_Ry        = sigma[4] * A        # A_norm = A / A (trivial)
        sigma_phiy      = sigma[5]            # phi is already dimensionless
        sigma_v         = sigma[6] * A / period               
        sigma_y0        = sigma[8] * A   


        print("standard deviations of parameter estimates:")
        print(f"omega: {sigma_omega:.3f} rad/s") 
        print(f"Rx: {sigma_Rx:.1f} m, phix: {sigma_phix:.3f} rad, u: {sigma_u:.3f} m/s")  
        print(f"Ry: {sigma_Ry:.1f} m, phiy: {sigma_phiy:.3f} rad, v: {sigma_v:.3f} m/s")    

        # Store results in their corresponding arrays for later use and output to the results dictionary.
        omegastd[i] = sigma_omega
        ustd[i] = sigma_u
        vstd[i] = sigma_v

        stdmagU[i] = np.sqrt( (uest[i]*ustd[i])**2 + (vest[i]*vstd[i])**2 ) / magU[i]
        bUdeg[i], dbUdeg[i] = b_err(uest[i], ustd[i], vest[i], vstd[i])
        # dz_bin is the height ascended through in this bin, i.e. the difference between the max and min altitude in the current bin.
        dz_bin = zbin[-1] - zbin[0]
        dz[i] = dz_bin

        # zhat is the representative altitude for this bin, which we take to be the midpoint between max and min altitude in the bin
        zhat[i] = 0.5 * ( np.max(zbin) + np.min(zbin) )

        print(f"initial guesses: om: {omega0:.3f} rad/s; Rx: {Rx0:.1f} m, Ry: {Ry0:.1f} m; u: {u0:.3f} m/s, v: {v0:.3f} m/s")
        print(f"fitted parameters: om={omegaest[i]:.3f} rad/s, Rx={Rxest[i]:.1f} m, Ry={Ryest[i]:.1f} m; u={uest[i]:.3f} m/s, v={vest[i]:.3f} m/s")
        print(f"wind speed: |U|={magU[i]:.3f} +- {stdmagU[i]:.3f} m/s, bearing: b={bUdeg[i]:.3f} +- {dbUdeg[i]:.3f} deg")
        print(f"height ascended through, dz={dz_bin:.1f} m")


        # Optional debug/inspection plotting
        if plot:
            # Plot the results for visual inspection.
            # Each bin will have its own subplot in a larger figure, with the fit and residuals plotted below each other.
            bin_coords = {"tbin": tbin, "xbin": xbin, "ybin": ybin,
                "teval": teval, "xevalplt": xevalplt, "yevalplt": yevalplt,
                "xres": xres, "yres": yres,
            }
            print(f"NAX: {nax}")
            plot_nonlin_bin_fit(nax, fig, fig2, ntherm, i, bin_coords)


        i += 1
# ------------ for loop iterating over bins ends here ---------------

    if plot:
        # Clean up figures and add titles, then save them to the output directory
        fig.suptitle(f"Non-linear fit of x for {fid}; thermal {ntherm}")
        fig2.suptitle(f"Non-linear fit of y for {fid}; thermal {ntherm}")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig2.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figs out to the output directory
        fig.savefig(f"{outdir}/nonlinfitx_thermal_{ntherm}_fid_{fid}.png", dpi=300)
        fig2.savefig(f"{outdir}/nonlinfity_thermal_{ntherm}_fid_{fid}.png", dpi=300)


    # Optional debug/inspection plotting
    if plot:
        # # Create a third figure, fig3, to show the track of the thermal in the x-y plane, with the bins highlighted, and the altitude profile over time. 
        fig3 = plt.figure(figsize=(9,3))

        plot_nonlin_track_summary(fig3, fid, ntherm, bidx, t_data, x_data, y_data, z_data)

        fig3.savefig(f"{outdir}/track_thermal_{ntherm}_fid_{fid}.png", dpi=600)

    # add back the window start index offset to retrieve indices in original data
    # this tells us the indices of the original data (with all thermal windows) that correspond to each bin.
    win = [(x + wins, y + wins) for x, y in bidx]

    # Append all the results from all of the bins for this thermalling window to the pandas dataframe created at the start of the cell.
    new_rows = pd.DataFrame({'id': fidarr, 'timestamp': refts, 'time': reft, 'lat0': lat0, 'lon0': lon0, 'thermidx': thermidx, 'win': win, 'omegaest': omegaest,'omegastd': omegastd,'Rxest': Rxest, 'phixest': phixest,'uest': uest, 'ustd': ustd, 'x0est': x0est, 'Ryest': Ryest, 'phiyest': phiyest, 'vest': vest, 'vstd': vstd, 'y0est': y0est, 'magU': magU, 'stdmagU': stdmagU, 'bUdeg': bUdeg, 'dbUdeg': dbUdeg, 'dz': dz, 'zhat': zhat, 'zdot': zdot})
    results = pd.concat([results, new_rows], ignore_index=True)

    return results


def plot_nonlin_bin_fit(nax, fig, fig2, ntherm, i, bin_result):
    '''
    fig, fig2: nspection plots for each of the thermal bins created from each thermal window.
    Plotting the x-fit and y-fit vs time and the x (x data - x model) and y residuals vs time.
    '''

    tbin = bin_result["tbin"]
    xbin = bin_result["xbin"]
    ybin = bin_result["ybin"]
    teval = bin_result["teval"]
    xevalplt = bin_result["xevalplt"]
    yevalplt = bin_result["yevalplt"]
    xres = bin_result["xres"]
    yres = bin_result["yres"]

    # Subplots on fig for x component fit for each bin
    # ax = fig.add_subplot(2*nax[1], nax[0], 2*i+1)   # double the columns in order to have space for residuals plot alongside each fit  
    ax = fig.add_subplot(nax[1], nax[0], 2*i+1)
    ax.plot(tbin, xbin, linestyle = '', marker = '.', markersize=4)
    # adding tbin.mean() back onto teval to display curve against the original data
    ax.plot(teval + tbin.mean(), xevalplt, linestyle = '-', color = 'k', alpha = 0.5, label = f"Bin {ntherm}-{i+1} x fit")

    ax.set_xlabel("t, s")
    ax.set_ylabel("x, m")
    ax.legend()

    # Subplots on fig for x component residuals for each bin
    # ax2 = fig.add_subplot(2*nax[1], nax[0], 2*i+2)   # double the columns in order to have space for residuals plot alongside each fit 
    ax2 = fig.add_subplot(nax[1], nax[0], 2*i+2) 
    # remember, xres = xbin - xeval
    # len(tbin) == len(t_centred) however, len(tbin) != len(teval)
    ax2.plot(tbin, xres, linestyle = '', marker = '.', markersize=4, label = f"Bin {ntherm}-{i+1} x residuals")
    ax2.set_xlabel("t, s")
    ax2.set_ylabel("x residual, m")
    ax2.legend()

    # Subplots on fig2 for y component fit for each bin
    ax3 = fig2.add_subplot(nax[1], nax[0], 2*i+1)   # double the rows in order to have space for residuals plot below each fit  
    ax3.plot(tbin, ybin, linestyle = '', marker = '.', markersize=4)
    ax3.plot(teval + tbin.mean(), yevalplt, linestyle = '-', color = 'k', alpha = 0.5, label = f"Bin {ntherm}-{i+1} y fit")

    ax3.set_xlabel("t, s")
    ax3.set_ylabel("y, m")
    ax3.legend()

    # Subplots on fig2 for y component residuals for each bin
    ax4 = fig2.add_subplot(nax[1], nax[0], 2*i+2)   # double the rows in order to have space for residuals plot below each fit  
    ax4.plot(tbin, yres, linestyle = '', marker = '.', markersize=4, label = f"Bin {ntherm}-{i+1} y residuals")
    ax4.set_xlabel("t, s")
    ax4.set_ylabel("y residual, m")
    ax4.legend()


def plot_nonlin_track_summary(fig, fid, ntherm, bidx, t, x, y, z):
    '''
    fig3: Inspection plots for each of the thermal bins created from each thermal window.
    Plotting the x vs y tracks and alitude vs t tracks for each of the bins in each thermal window.
    '''
    # fig = plt.figure(figsize=(9,3))
        
    ax5 = fig.add_subplot(121)
    ax5.plot((x-x[0])/1e3, (y-y[0])/1e3)
    ax5.scatter(0, 0, label = 'start', color = 'green')
    ax5.scatter((x[-1]-x[0])/1e3, (y[-1]-y[0])/1e3, label = 'end', color = 'red')

    binnum = 0
    for s, e in bidx:
        ax5.plot((x[s:e]-x[0])/1e3, (y[s:e]-y[0])/1e3, label = f"Bin {binnum}")
        binnum += 1

    ax5.set_xlabel("x (km)")
    ax5.set_ylabel("y (km)")
    # ax.set_zlabel("z (m)")
    ax5.legend(loc="center right", bbox_to_anchor=(-0.20, 0.5))

    ax6 = fig.add_subplot(122)
    ax6.plot(t-t[0], z)
    for s, e in bidx:
        ax6.axvline(x=(t[s]-t[0]), color='k', linestyle='--')
        ax6.axvline(x=(t[e-1]-t[0]), color='k', linestyle='--')
    # ax6.legend(loc="center left", bbox_to_anchor=(-0.5, 0.5))

    fig.subplots_adjust(left=0.3)  # make room for the legend
    plt.xlabel("t (s)")
    plt.ylabel("altitude (m)")

    fig.suptitle(f"Thermal {ntherm} track and altitude for {fid}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

    # fig.savefig(f"{outdir}/track_thermal_{ntherm}_fid_{fid}.png", dpi=600)

# # 7. Plot u and v components of wind estimates

def plot_uv_winds(df_main, windows):
    '''
    Plots the estimated u and v wind components for each thermal soaring bin processed.
    The thermal soaring bins are colour coded by the 'parent' thermal soaring window.

    :param pandas.DataFrame df_main: Estimated parameters for each thermal soaring bin processed.
    :param list (of tuples) windows: Bin edge indices as ``(start_idx, end_idx)`` for each thermal window.
    '''

    print(f"Len of windows: {len(windows)}")

    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(121)

    for i in range(len(windows)):
        if np.sum(df_main.magU[df_main.thermidx == i]) > 0: # color codes by thermalling window each estimate came from.
            ax1.errorbar(df_main.uest[df_main.thermidx == i], df_main.zhat[df_main.thermidx == i], xerr=df_main.ustd[df_main.thermidx == i], fmt='s', markersize = 5, capsize = 3, label = f"Thermal {i}")
    plt.xlabel("u (m/s)")
    plt.ylabel("Altitude (m)")
    plt.legend()
    plt.grid()
    plt.title(f"u vs alt for '{fid}'")

    ax2 = fig.add_subplot(122)

    for i in range(len(windows)):
        if np.sum(df_main.magU[df_main.thermidx == i]) > 0:
            ax2.errorbar(df_main.vest[df_main.thermidx == i], df_main.zhat[df_main.thermidx == i], xerr=df_main.vstd[df_main.thermidx == i], fmt='s', markersize = 5, capsize = 3, label = f"Thermal {i}")
    plt.xlabel("v (m/s)")
    plt.legend()
    plt.grid()
    plt.title(f"v vs alt for '{fid}'")

    plt.show()



def main():
    
    ####################################
    ### 1. Load OGN APRS Glider Data ###
    ####################################
    print(f"******************************")
    print(f"*** Loading in Glider Data ***")
    print(f"******************************")
    dfm, fid = load_data(filepath=gliderpath)

    # Check the DataFrame structure and column types
    print(dfm.info())


    ####################################
    ### 2. Pre-process Flight Tracks ###
    ####################################
    print(f"*******************************************")
    print(f"*** Pre-processing the Glider DataFrame ***")
    print(f"*******************************************")

    # Dropping entries with duplicate timestamps
    t, lat, lon, alt_m, x, y = preprocess_track(dfm)

    # Generating smoothed x and y co-ordinates, using a savgol filter, for plotting
    sgx, sgy = smooth_track(x, y, window_length=15, polyorder=2)

    # Exploration plots showing fields from the pre-processed glider DataFrame
    plot_altitude(t, alt_m)
    plot_time_intervals(t)
    plot_lat_lon(lat, lon, x, y)


    ########################################
    ### 3. Detect Thermal Soaring Events ###
    ########################################
    print(f"****************************************")
    print(f"*** Identify Thermal Soaring Windows ***")
    print(f"****************************************")

    # Identify thermal soaring windows, as ``(start_idx, end_idx)`` positions, within the glider flight
    windows = detect_circling_windows(t, x, y, alt_m,
                                        ang_rate_thresh=0.02,
                                        min_duration=3.0,
                                        gap_merge=20.0,
                                        win_smooth=None)
    
    print(f"Number of detected windows: {len(windows)}")
    print("Detected windows (start_idx, end_idx):", windows)
    windows_time = [(t[s], t[e]) for s, e in windows]
    print("Detected windows (start_time, end_time):", windows_time)
    win_dt = [t[e]-t[s] for s, e in windows]
    print("Window durations (s):", np.array(win_dt).astype(int))

    # Apply a savgol filter to smooth the x and y data: only used for visualisation.
    sgx, sgy = smooth_track(x, y)

    # Plotting thermal soaring events trajectories and angular rates
    plt_therm_res(t, sgx, sgy, alt_m, windows, ang_rate_thresh=0.02)
    # Plotting height gained per thermal soaring window event
    plt_window_height_gained(alt_m, windows)
        

    ##################################################
    ### 4. Inspect Detected Thermal Soaring Events ###
    ##################################################
    print(f"***************************************")
    print(f"*** Inspect Thermal Soaring Windows ***")
    print(f"***************************************")
    
    # Looping through the thermal soaring windows for inspection
    print(f"Windows before trim: {windows}")

    # OPTIONAL: trim thermal soaring window(s) to remove transient tails when entering/exiting thermal soaring window
    start_trim = 0
    end_trim = 0
    windows_trimmed = windows
    for nwin in range(0, len(windows)):
        # update window(s) to include trimming
        windows_trimmed = trim_window(windows, nwin, start_trim, end_trim)

        # optional inspection plots
        plot_window_alt_track(t, alt_m, x, y, windows, nwin)
        plot_trimmed_window_xy_components(t, x, y, windows_trimmed, nwin)

    # updating windows to reflect the trim
    windows = windows_trimmed

    print(f"Windows after trim: {windows}")


    #################################################################
    ### 5, 6. Selected Thermal Soaring Window Binned and Fit to Model
    #################################################################
    print(f"*****************************************************************")
    print(f"*** Bin the Selected Thermal Soaring Windows and Fit to Model ***")
    print(f"*****************************************************************")

    # Iterate through each thermal window and bin by altitude

    # indices = [nwin]    # pass in an array of the indices of the windows to process.
    # indices = [0, 1]
    indices = [i for i in range(0, len(windows))] # pass in an array of the indices of the windows to process.
    print(f"Array of indices of the windows to process: {indices}")


    # Construct dataframe
    results_nln = pd.DataFrame({'id': np.empty(0), 'timestamp': np.empty(0), 'time': np.empty(0), 'lat0': np.empty(0), 'lon0': np.empty(0), 'thermidx': np.empty(0), 'win': np.empty(0), 'omegaest': np.empty(0),'omegastd': np.empty(0), 'Rxest': np.empty(0), 'phixest': np.empty(0),'uest': np.empty(0),'ustd': np.empty(0), 'x0est': np.empty(0), 'Ryest': np.empty(0), 'phiyest': np.empty(0), 'vest': np.empty(0), 'vstd': np.empty(0), 'y0est': np.empty(0), 'magU': np.empty(0), 'stdmagU': np.empty(0), 'bUdeg': np.empty(0), 'dbUdeg': np.empty(0), 'dz': np.empty(0), 'zhat': np.empty(0), 'zdot': np.empty(0)})
    sbin = 100  # bin size in meters for altitude (used for vertical displacment binning only).

    # Initialise loop variables
    globnbin = 0    # keep a count of the total number of bins processed across all thermalling windows passed in from 'indices'.
    ntherm = 0      # thermalling window index, incremented for each for loop iteration, used to keep track of which thermalling window is being processed.
    for s, e in windows:
        if ntherm in indices:
        # if the current thermalling window is in the list of windows chosen to process

            wins = s        # keep track of the start index of the current thermalling window, used to offset the bin indices back to the original data indices for output.
            x_data = x[s:e]
            y_data = y[s:e]
            z_data = alt_m[s:e]
            t_data = t[s:e]

            print(f"----------------- Window {ntherm} -----------------")

            # Estimate global period for the current thermalling window using the omega_guess function, which analyzes the time series data to provide an initial estimate of the angular rate of rotation (omega). 
            # This is used to determine the period of oscillation for binning and fitting purposes.
            globomega0 = omega_guess(t_data, x_data, y_data, plot=False)
            globP = 2*np.pi/abs(globomega0)
            if globP > 1.5*30 or globP < 0.5*30:
                globP = 30
                globomega0 = 2*np.pi/30
                print("Error estimating global period - setting to 30s")

            print(f"global period estimation for window {ntherm}: P={globP:.1f} s based on omega_guess={globomega0:.3f} rad/s")

            # create bins of data with 2 turns each.
            nz_data = alt_m[s:e] - alt_m[s]     # altitude data normalised to start at 0 for binning purposes.

            Ptrim = 2   # choose to trim the start of the window by a fraction of a global period to exclude transient motion that may occur when entering thermals.

            # Create bins of data, each containing 2 global periods of motion. Returns bin start and end indices, and the number of bins created.
            bidx, nbin = bin2(t_data, nz_data, sbin, globP, Ptrim)

            # Guard against short time windows or time windows that do not exceed 2 periods (see bin2() for deets)
            if nbin <= 0:
                print(f"Skipping window {ntherm}")
                continue


            # Track how many bins have been processed across all thermalling windows.
            globnbin = globnbin + nbin

            # Create variables to store the number of rows and columns needed for the subplots.
            # Each bin gets its own row, with fit and residual side by side.
            nax = np.empty(2, dtype=int)
            nax[0] = 2  # number of columns for subplots (fit and residual)
            nax[1] = nbin  # one row per bin

            # Execute the non-linear fitting routine. Each bin is processed in turn within this function, and results are stored in the results_nln dataframe. 
            # The function also generates plots for each bin showing the fit and residuals, and saves them to the output directory.
            results_nln = nonlinmdl(fid, t_data, x_data, y_data, z_data, lat, lon, bidx, nbin, wins, ntherm, results_nln, nax, globP, plot=False)

        ntherm += 1

    # ---------- for loop iterating over thermalling windows ends here --------------- #


    #####################################################################################
    ### 7. View the Estimated Parameters from the Thermal Bins and Output Results to .csv
    #####################################################################################
    print(f"*********************************************************************************")
    print(f"*** View Estimated Params, Plot u,v Winds, Output Params for Each Bin to .csv ***")
    print(f"*********************************************************************************")

    # View results
    # contains the estimated parameters from all of the thermal soaring bins processed
    print(results_nln[['omegaest','uest','ustd','vest','vstd','magU','stdmagU','bUdeg','dz', 'zhat', 'zdot']])

    df_main = results_nln.copy()

    # De-duplicate rows so re-running downstream cells does not grow df_main with repeated entries.
    dedupe_cols = ["id", "timestamp", "thermidx", "win"]
    df_main = df_main.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)

    # Quick diagnostic: confirm no duplicate key rows remain in df_main.
    dup_count = df_main.duplicated(subset=dedupe_cols).sum()
    print(f"Duplicate rows in df_main (by {dedupe_cols}): {dup_count}")

    print(df_main[['id','thermidx','omegaest','uest','ustd','vest','vstd','magU','stdmagU','dz', 'zhat', 'zdot']])

    # Save out to csv
    df_main.to_csv(f"{outdir}/nonlinfit_results_{fid}.csv", index=False)

    # Plot the estimated u and v winds for each thermal soaring bin processed

    print(f"Windows: {windows}")
    plot_uv_winds(df_main, windows)


if __name__ == "__main__":
    main()
