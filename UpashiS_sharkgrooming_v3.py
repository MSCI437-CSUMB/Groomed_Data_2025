# %% Commenting and Imports
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# sharkdatagrooming.py

# Copyright (c) 2019, Steve Nixon
# Copyright (c) 2019, Monterey Bay Aquarium
# Produced at the Monterey Bay Aquarium
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Realign IMU data from Shark Camera Tag Data Loggers
The white shark reasearch group at the Monterey Bay Aquarium has been using shark tags produced by CATS, Inc. to
track the movements of white sharks. This program takes data from those tags and grooms those data to align the
accelerometer data so that the z axis is always align with the gravitational pull of the earth. The program uses
a series of 3D rotations and convolutions to identify points where the shark tag has become misaligned due to
the shark coming into contact with an object. Currently, it corrects the axes, trims the files to remove data from
prior to the tag being placed on the shark and after the tag come off, and downsamples the data by a factor of two.

Author:
  Steve Nixon - stnixon@gmail.com
  Dylan Moran - dmoran@csumb.edu

Organization:
  Monterey Bay Aquarium

License: MIT

Version: 2025.1.0

Requirements
------------
* Python 3.5+ - https://www.python.org
* Numpy 1.14 - https://www.numpy.org
* SciPy 1.3.1 - https://www.scipy.org
* Pandas - 0.25.1 - https://pandas.pydata.org/
* Matplotlib - 3.1.1 - https://matplotlib.org/
* Requests - 2.22.0 - https://requests.readthedocs.io/en/master/

Installation and setup
----------------------
* Install Python 3.7+ for your operating systems
* Install pip (Python Package Installer) for your operating system
* python -m pip install --user numpy scipy matplotlib pandas requests

Running the program
-------------------
#Batch process in the bash shell
#for file in *CC07*.csv; do python sharkdatagrooming.py "$file"; done

"""
# Imports
import sys, time, math, os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.signal import find_peaks
from argparse import ArgumentParser


# %%unction Definitions
# Functions

def data_read_in(filename):
    # Read the file into a pandas dataframe for processing. Only selects certain columns. If this fails
    # check to see if there has been a change to the file structure.
    global front_clip, end_clip
    # Provide default clip values if not defined by metadata
    front_clip = 0
    end_clip = None

    # CSV DOWNLOADED AS MOVEBANK COMPATIBLE
    df = pd.read_csv(filename, delimiter=";", header=0, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                     names=['TagID', 'Timestamp', 'accX', 'accY', 'accZ', 'magX', 'magY', 'magZ', 'Depth', 'Temp',
                            'Metadata'], encoding='ISO-8859–1',
                     dtype={'TagID': object, 'Timestamp': object, 'accX': float, 'accY': float, 'accZ': float,
                            'magX': float, 'magY': float, 'magZ': float, 'Depth': float, 'Temp': float,
                            'Metadata': object}, on_bad_lines='warn')

    # df.rename(columns={'Acc_X': 'Acc_Z','Acc_Y':'Acc_X','Acc_Z':'Acc_Y','Mag_X': 'Mag_Z','Mag_Y':'Mag_X','Mag_Z':'Mag_Y','Gyr_X':'Gyr_Z','Gyr_Y':'Gyr_X','Gyr_Z':'Gyr_Y'}, inplace=True) #Fix for the column header issue

    dcal = args.dcal
    # df['Depth'] = df['Depth']/dcal

    # Frequency stuff moved by DM
    # depth_offset = metadata.loc[metadata['FNKey_For_Archive'] == meta_file, 'Depth_Zero_Offset'].iloc[0]
    # File_object.write('Collection frequency is: '+str(frequency)+' Hz\n')
    #	df['Depth'] = df['Depth'] + float(depth_offset)
    #	File_object.write('Depth offset is ' +str(depth_offset) + '\n')
    if end_clip:
        df = df.iloc[front_clip:end_clip]
    else:
        df = df.iloc[front_clip:]
    print('Front clipped', front_clip, 'data points')
    print('End clipped', end_clip, 'data points')

    File_object.write('Front clipped ' + str(front_clip) + ' data points\n')
    File_object.write('End clipped ' + str(end_clip) + ' data points\n')

    return df  # Returns the dataframe to be processed.


def get_arguments():
    # Read the file name in from the command line prompt
    global filename, meta_file, frequency, args

    parser = ArgumentParser()
    parser.add_argument('-i', '--file', required=True,
                        help="Path to deployment CSV file (also used for calibration if --calib not given")
    parser.add_argument('-f', '--frequency', required=False, help="sampling frequency")
    parser.add_argument('-d', '--dcal', nargs='?', default=1, type=float, required=False, help="depth calbration")
    parser.add_argument('-t', '--tcal', nargs='?', default=1, type=float, required=False, help="temp calbration")
    parser.add_argument('-g', '--gcal', nargs='?', default=1, type=float, required=False, help="g calbration")
    parser.add_argument('-p1', '--peaks1', nargs='*', type=int, default=[], required=False, help="a1 peaks")
    parser.add_argument('-p2', '--peaks2', nargs='*', type=int, default=[], required=False, help="a2 peaks")
    parser.add_argument('-c', '--calib', required=False, help='Optional: separate calibration CSV file')
    parser.add_argument('--out', default='calibrated_output.csv', help='Output CSV file')

    args = parser.parse_args()
    filename = args.file

    meta_file = os.path.basename(os.path.splitext(filename)[0])

    try:
        os.mkdir(meta_file)
    except OSError:
        print("Creation of the directory %s failed. Directory already exists." % meta_file)
    else:
        print("Successfully created the directory %s " % meta_file)

    return filename  # returns the filename to be processed


# def param_read_in():
#     # Read in metadata to allow for clipping, depth correcting, etc...
#
#     global metadata
#
#     # r = requests.get('https://docs.google.com/spreadsheets/d/e/2PACX-1vROhld2eVHbEPTNbNagbyN5TAaZH5HnKljEV5B6ylUQM62siMb-uWdbTZ9dKi3b3s0jmC8c2pYjy5c3/pub?output=csv')
#
#     metadata = pd.read_csv("C:/Users/srupa/sharkdata/CC_CD_Tag_Metadata.csv", delimiter=',', on_bad_lines='warn',
#                            header=2, usecols=(4, 12, 14, 17))
#
#     return metadata  # reads the required columns from the parameters file into a pandas dataframe


def quality_check(dataframe, t):
    # stats on memory usage and runtime
    print('\n---- Quality Check -----')
    print('df Data Frame Info')
    df.info(memory_usage='deep', verbose=False)
    print('')
    print(str(time.time() - t) + "s")


def get_sensor_offsets(df):
    acc_offset = df[['accX', 'accY', 'accZ']].mean()
    mag_offset = df[['magX', 'magY', 'magZ']].mean()
    return acc_offset, mag_offset


def apply_offsets(dataframe, acc_offset, mag_offset):
    df = dataframe

    expected_columns = ['accX', 'accY', 'accZ', 'magX', 'magY', 'magZ']
    for col in expected_columns:
        if col not in df.columns:
            print(f"ERROR: Column '{col}' not found in the deployment file.")
            print("Available columns:", list(df.columns))
            sys.exit(1)

    df['accX'] -= acc_offset['accX']
    df['accY'] -= acc_offset['accY']
    df['accZ'] -= acc_offset['accZ']
    df['magX'] -= mag_offset['magX']
    df['magY'] -= mag_offset['magY']
    df['magZ'] -= mag_offset['magZ']

    # magnetometers may have a few values above 1000 because of magnet swipes or other interferences. Reset them to 0 for plot cleanliness. Ideally NaN but 0 works for now
    for col in expected_columns:
        df[col] = df[col].apply(lambda x: 0 if x > 1000 or x < -1000 else x)

    return df


def trim_data(df):
    # Compute acceleration magnitude
    df['acc_magnitude'] = np.sqrt(df['accX'] ** 2 + df['accY'] ** 2 + df['accZ'] ** 2)
    threshold = 0.25
    active = df[df['acc_magnitude'] > threshold]

    if active.empty:
        print("Warning: Could not find trim points based on acceleration magnitude.")
        return df

    print(f"Trimming to only rows where acc_magnitude > {threshold} ({len(active)} rows kept)")
    return active.reset_index(drop=True)


def plot_sensor_data(df, tag_id, suffix=""):
    mx = df['magX'].dropna()
    my = df['magY'].dropna()
    mz = df['magZ'].dropna()
    t = np.arange(len(df))
    mt = np.arange(len(mx))
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    axs[0].plot(t, df['accX'], 'r', label='accX')
    axs[0].plot(t, df['accY'], 'g', label='accY')
    axs[0].plot(t, df['accZ'], 'b', label='accZ')
    axs[0].set_title(f'Accelerometer {suffix}')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(mt, mx, c='r', label='magX')
    axs[1].plot(mt, my, c='g', label='magY')
    axs[1].plot(mt, mz, c='b', label='magZ')
    axs[1].set_title(f'Magnetometer {suffix}')
    axs[1].legend()
    axs[1].grid(True)

    if 'acc_magnitude' in df.columns:
        axs[2].plot(t, df['acc_magnitude'], 'k', label='acc_magnitude')
        axs[2].set_title(f'Acceleration Magnitude {suffix}')
        axs[2].legend()
        axs[2].grid(True)

    plt.tight_layout()
    filename = f'{tag_id}_sensor_plot{suffix}.png'
    plt.savefig(str(meta_file) + '/' + filename, dpi=300)
    plt.close()
    print(f"Saved plot: {filename}")
    print(f"Saved plot: {filename}")


def matrix_reshape(dataframe):
    global df, df1
    global axm, aym, azm
    global axtm, aytm, aztm
    global mxm, mym, mzm
    global mxtm, mytm, mztm
    global gxm, gym, gzm
    global gxtm, gytm, gztm
    global xyzd
    global windows, total_windows
    global frequency

    # frequency = int(args.frequency)                                            # DM wrote some lines to calculate frequency from the df rather than set it in metadata sheet
    # Determines the size in data points of the 15 minute bucket
    windows = int(get_windows(frequency))
    total_windows = len(df) // windows
    # Determines the number of rows
    columns = len(df.index) // windows
    elements = windows * columns
    initial_length = len(df.index)
    n = abs(initial_length - elements)

    df = df.drop(df.tail(n).index)
    df1 = df[['accX', 'accY', 'accZ', 'magX', 'magY', 'magZ']]

    xyzd = df1.to_numpy(copy=True)

    axtm = np.reshape(xyzd[:, 0], (len(xyzd) // windows, windows))
    axm = np.mean(axtm, axis=1)
    aytm = np.reshape(xyzd[:, 1], (len(xyzd) // windows, windows))
    aym = np.mean(aytm, axis=1)
    aztm = np.reshape(xyzd[:, 2], (len(xyzd) // windows, windows))
    azm = np.mean(aztm, axis=1)
    mxtm = np.reshape(xyzd[:, 3], (len(xyzd) // windows, windows))
    mxm = np.mean(mxtm, axis=1)
    mytm = np.reshape(xyzd[:, 4], (len(xyzd) // windows, windows))
    mym = np.mean(mytm, axis=1)
    mztm = np.reshape(xyzd[:, 5], (len(xyzd) // windows, windows))
    mazm = np.mean(mztm, axis=1)


def create_arrays(mean_array, full_array):
    global a1, x1, y1, z1, a2
    global ax1a, ay1a, az1a
    global ax2a, ay2a, az2a
    global mx1a, my1a, mz1a
    global mx2a, my2a, mz2a
    global gx1a, gy1a, gz1a
    global gx2a, gy2a, gz2a

    a1 = np.zeros_like(mean_array)
    a2 = np.zeros_like(mean_array)

    x1 = np.zeros_like(mean_array)
    y1 = np.zeros_like(mean_array)
    z1 = np.zeros_like(mean_array)

    ax1a = np.zeros_like(full_array)
    ax2a = np.zeros_like(full_array)
    ay1a = np.zeros_like(full_array)
    ay2a = np.zeros_like(full_array)
    az1a = np.zeros_like(full_array)
    az2a = np.zeros_like(full_array)

    mx1a = np.zeros_like(full_array)
    mx2a = np.zeros_like(full_array)
    my1a = np.zeros_like(full_array)
    my2a = np.zeros_like(full_array)
    mz1a = np.zeros_like(full_array)
    mz2a = np.zeros_like(full_array)


# gx1a = np.zeros_like(full_array)
# gx2a = np.zeros_like(full_array)
# gy1a = np.zeros_like(full_array)
# gy2a = np.zeros_like(full_array)
# gz1a = np.zeros_like(full_array)
# gz2a = np.zeros_like(full_array)

def get_windows(frequency):
    window = 15 * 60 * frequency

    return window


def temp_plot(dataframe, clip=False):
    t = dataframe['Temp'].dropna()
    f = t.index

    fig, ax = plt.subplots()
    ax.plot(f, t, color='black', label='t')

    ax.grid(True)
    fig.canvas.manager.set_window_title('Raw Temp Plot')
    ax.legend(loc='best')
    # legend = ax.legend(loc='best')
    if clip == True:
        plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_ClippedTempPlot.png'), dpi=600)
    if clip == False:
        plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_RawTempPlot.png'), dpi=600)


def depth_plot(dataframe):
    d = dataframe['Depth'].dropna()
    f = d.index

    fig, ax = plt.subplots()
    ax.plot(f, d, color='black', label='d')

    ax.grid(True)
    fig.canvas.manager.set_window_title('Raw Depth Plot')
    ax.legend(loc='best')
    # legend = ax.legend(loc='best')
    plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_RawDepthPlot.png'), dpi=600)


# plt.savefig('/Users/dylan/Desktop/sharkdata/diagnostic_plots/'+str(meta_file)+'_RawDepthPlot.png',dpi=600)

def depth_plot_clipped(dataframe):
    d = dataframe['Depth'].dropna()
    f = d.index

    fig, ax = plt.subplots()
    ax.plot(f, d, color='black', label='d')

    ax.grid(True)
    fig.canvas.manager.set_window_title('Raw Depth Plot')
    ax.legend(loc='best')
    # legend = ax.legend(loc='best')
    plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_ClippedDepthPlot.png'), dpi=600)


# plt.savefig('/Users/dylan/Desktop/sharkdata/diagnostic_plots/'+str(meta_file)+'_ClippedDepthPlot.png',dpi=600)

def accel_raw_plots(dataframe):
    f = dataframe.index
    x = dataframe['accX']
    y = dataframe['accY']
    z = dataframe['accZ']

    fig, ax = plt.subplots()

    ax.plot(f, x, 'r', label="x")
    ax.plot(f, y, 'g', label="y")
    ax.plot(f, z, 'b', label="z")

    ax.grid(True)
    fig.canvas.manager.set_window_title('Raw Acceleration Data Plot')
    legend = ax.legend(loc='best')
    plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_AccelerationRawDataPlot.png'), dpi=600)


# plt.savefig('/Users/dylan/Desktop/sharkdata/diagnostic_plots/'+str(meta_file)+'_AccelerationRawDataPlot.png',dpi=600)

def mag_raw_plots(dataframe):
    x = dataframe['magX'].dropna()
    y = dataframe['magY'].dropna()
    z = dataframe['magZ'].dropna()
    f = x.index

    fig, ax = plt.subplots()

    ax.plot(f, x, c='r', label="x")
    ax.plot(f, y, c='g', label="y")
    ax.plot(f, z, c='b', label="z")

    ax.grid(True)
    fig.canvas.manager.set_window_title('Raw Magnetometer Data Plot')
    legend = ax.legend(loc='best')
    plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_MagnetometerRawDataPlot.png'), dpi=600)


# plt.savefig('/Users/dylan/Desktop/sharkdata/diagnostic_plots/'+str(meta_file)+'_MagnetometerRawDataPlot.png',dpi=600)

def gyr_raw_plots(dataframe):
    f = dataframe.index
    x = dataframe['Gyr_X']
    y = dataframe['Gyr_Y']
    z = dataframe['Gyr_Z']

    fig, ax = plt.subplots()

    ax.plot(f, x, 'r', label="x")
    ax.plot(f, y, 'g', label="y")
    ax.plot(f, z, 'b', label="z")

    ax.grid(True)
    fig.canvas.manager.set_window_title('Raw Gyroscope Data Plot')
    legend = ax.legend(loc='best')
    plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_GyroscopeRawDataPlot.png'), dpi=600)


# plt.savefig('/Users/dylan/Desktop/sharkdata/diagnostic_plots/'+str(meta_file)+'_GyroscopeRawDataPlot.png',dpi=600)

def accel_clipped_raw_plots(dataframe):
    f = dataframe.index
    x = dataframe['accX']
    y = dataframe['accY']
    z = dataframe['accZ']

    fig, ax = plt.subplots()

    ax.plot(f, x, 'r', label="x")
    ax.plot(f, y, 'g', label="y")
    ax.plot(f, z, 'b', label="z")

    ax.grid(True)
    fig.canvas.manager.set_window_title('Clipped Raw Data Plot')
    legend = ax.legend(loc='best')
    plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_ClippedAccelerationRawDataPlot.png'), dpi=600)


# plt.savefig('/Users/dylan/Desktop/sharkdata/diagnostic_plots/'+str(meta_file)+'_ClippedAccelerationRawDataPlot.png',dpi=600)

def mag_clipped_raw_plots(dataframe):
    x = dataframe['magX'].dropna()
    y = dataframe['magY'].dropna()
    z = dataframe['magZ'].dropna()
    f = x.index

    fig, ax = plt.subplots()

    ax.plot(f, x, c='r', label="x")
    ax.plot(f, y, c='g', label="y")
    ax.plot(f, z, c='b', label="z")

    ax.grid(True)
    fig.canvas.manager.set_window_title('Raw Magnetometer Data Plot')
    legend = ax.legend(loc='best')
    plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_ClippedMagnetometerRawDataPlot.png'), dpi=600)


# plt.savefig('/Users/dylan/Desktop/sharkdata/diagnostic_plots/'+str(meta_file)+'_ClippedMagnetometerRawDataPlot.png',dpi=600)

def gyr_clipped_raw_plots(dataframe):
    f = dataframe.index
    x = dataframe['Gyr_X']
    y = dataframe['Gyr_Y']
    z = dataframe['Gyr_Z']

    fig, ax = plt.subplots()

    ax.plot(f, x, 'r', label="x")
    ax.plot(f, y, 'g', label="y")
    ax.plot(f, z, 'b', label="z")

    ax.grid(True)
    fig.canvas.manager.set_window_title('Raw Gyroscope Data Plot')
    legend = ax.legend(loc='best')
    plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_ClippedGyroscopeRawDataPlot.png'), dpi=600)


# plt.savefig('/Users/dylan/Desktop/sharkdata/diagnostic_plots/'+str(meta_file)+'_ClippedGyroscopeRawDataPlot.png',dpi=600)

def angle_diagnostic_plots(dataframe1, dataframe2, x_var, y1_var, y2_var, dy1_var, dy2_var):
    # Fancy diagnostic plots: this is both diagnostic and used to find the mean angle that will be applied to each segment
    # in the final rotation.
    fig1, ax1 = plt.subplots()

    ax1.plot(x_var, dataframe1, 'r', label="a1")
    ax1.plot(x_var, dataframe1, 'ro')
    ax1.plot(x_var, y1_var, 'b', label='smoothed a1')
    ax1.plot(x_var, y1_var, 'bo')
    ax1.plot(x_var, dy1_var, label='a1 derivative')

    ax1.plot(x_var, dataframe2, 'g', label="a2")
    ax1.plot(x_var, dataframe2, 'g+')
    ax1.plot(x_var, y2_var, 'y', label='smoothed a2')
    ax1.plot(x_var, y2_var, 'y+')
    ax1.plot(x_var, dy2_var, label='a2 derivative')

    ax1.grid(True)
    fig1.canvas.manager.set_window_title('Rotation Angle Diagnostic Plot')
    legend = ax1.legend(loc='best')
    plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_RotationAngles.png'), dpi=600)
    # plt.savefig('/Users/dylan/Desktop/sharkdata/diagnostic_plots/'+str(meta_file)+'_RotationAngles.png',dpi=600)


def inflection_diagnostic_plots(convolution1, angle1peaks, convolution2, angle2peaks):
    # Plot inflection point diagnostic plot
    fig1, ax1 = plt.subplots()
    ax1.plot(convolution1, 'r', label="a1")
    ax1.plot(convolution1, 'ro')
    ax1.plot(angle1peaks, convolution1[angle1peaks], "x")
    ax1.plot(convolution2, 'g', label="a2")
    ax1.plot(convolution2, 'go')
    ax1.plot(angle2peaks, convolution2[angle2peaks], "x")
    ax1.grid(True)
    fig1.canvas.manager.set_window_title('Inflection Points Based on Convolutions')
    legend = ax1.legend(loc='best')
    plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_InflectionPoints.png'), dpi=600)


# plt.savefig('/Users/dylan/Desktop/sharkdata/diagnostic_plots/'+str(meta_file)+'_InflectionPoints.png',dpi=600)

def accel_data_plot(array1, array2, array3, dataframe):
    s = np.arange(0, len(array1))
    d = dataframe['Depth'].dropna()  # drop the nas from depth because the axy records depth @ 1Hz
    f = range(1, len(d) + 1)  # seperate index vector for the depth plots

    fig, axs = plt.subplots(4, 2, sharex=False, sharey=False)

    axs[0, 0].plot(s, dataframe['accX'], color='red')
    axs[0, 0].set_xlim(0, len(array1))
    axs[0, 0].set_ylabel('x_raw')
    axs[0, 0].grid(True)

    axs[1, 0].plot(s, dataframe['accY'], color='green')
    axs[1, 0].set_xlim(0, len(array1))
    axs[1, 0].set_ylabel('y_raw')
    axs[1, 0].grid(True)

    axs[2, 0].plot(s, dataframe['accZ'], color='blue')
    axs[2, 0].set_xlim(0, len(array1))
    axs[2, 0].set_ylabel('z_raw')
    axs[2, 0].grid(True)

    axs[3, 0].plot(f, d, color='black')
    axs[3, 0].set_xlim(0, len(f))
    axs[3, 0].set_ylabel('depth')
    axs[3, 0].grid(True)

    axs[0, 1].plot(s, array1, color='red')
    axs[0, 1].set_xlim(0, len(array1))
    axs[0, 1].set_ylabel('x_corr')
    axs[0, 1].grid(True)

    axs[1, 1].plot(s, array2, color='green')
    axs[1, 1].set_xlim(0, len(array1))
    axs[1, 1].set_ylabel('y_corr')
    axs[1, 1].grid(True)

    axs[2, 1].plot(s, array3, color='blue')
    axs[2, 1].set_xlim(0, len(array1))
    axs[2, 1].set_ylabel('z_corr')
    axs[2, 1].grid(True)

    axs[3, 1].plot(f, d, color='black')
    axs[3, 1].set_xlim(0, len(f))
    axs[3, 1].set_ylabel('depth')
    axs[3, 1].grid(True)

    fig.canvas.manager.set_window_title('Final Accelerometer Comparison Plots')
    fig.tight_layout()
    plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_FinalComparisonAccelerometerPlots.png'), dpi=600)


# plt.savefig('/Users/dylan/Desktop/sharkdata/diagnostic_plots/'+str(meta_file)+'_FinalComparisonAccelerationPlots.png',dpi=600)

def mag_data_plot(array1, array2, array3, dataframe):
    # drop the NAs from depth + mags because the axy records these @ 1Hz
    x = dataframe['magX'].dropna()
    y = dataframe['magY'].dropna()
    z = dataframe['magZ'].dropna()
    d = dataframe['Depth'].dropna()

    # seperate index vectors for the plots (no reason for the letters being named this way)
    s = np.arange(0, len(array1))  # Transformed vectors
    g = range(1, len(x) + 1)  # NA-less magnetometers
    f = range(1, len(d) + 1)  # Depth

    fig, axs = plt.subplots(4, 2, sharex=False, sharey=False)

    axs[0, 0].plot(g, x, color='red')
    axs[0, 0].set_xlim(0, len(g))
    axs[0, 0].set_ylabel('x_raw')
    axs[0, 0].grid(True)

    axs[1, 0].plot(g, y, color='green')
    axs[1, 0].set_xlim(0, len(g))
    axs[1, 0].set_ylabel('y_raw')
    axs[1, 0].grid(True)

    axs[2, 0].plot(g, z, color='blue')
    axs[2, 0].set_xlim(0, len(g))
    axs[2, 0].set_ylabel('z_raw')
    axs[2, 0].grid(True)

    axs[3, 0].plot(f, d, color='black')
    axs[3, 0].set_xlim(0, len(f))
    axs[3, 0].set_ylabel('depth')
    axs[3, 0].grid(True)

    axs[0, 1].scatter(s, array1, color='red', s=2)
    axs[0, 1].set_xlim(0, len(array1))
    axs[0, 1].set_ylabel('x_corr')
    axs[0, 1].grid(True)

    axs[1, 1].scatter(s, array2, color='green', s=2)
    axs[1, 1].set_xlim(0, len(array1))
    axs[1, 1].set_ylabel('y_corr')
    axs[1, 1].grid(True)

    axs[2, 1].scatter(s, array3, color='blue', s=2)
    axs[2, 1].set_xlim(0, len(array1))
    axs[2, 1].set_ylabel('z_corr')
    axs[2, 1].grid(True)

    axs[3, 1].plot(f, d, color='black')
    axs[3, 1].set_xlim(0, len(f))
    axs[3, 1].set_ylabel('depth')
    axs[3, 1].grid(True)

    fig.canvas.manager.set_window_title('Final Magnetometer Comparison Plots')
    fig.tight_layout()
    plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_FinalComparisonMagentometerPlots.png'), dpi=600)


# plt.savefig('/Users/dylan/Desktop/sharkdata/diagnostic_plots/'+str(meta_file)+'_FinalComparisonPlots.png',dpi=600)

def gyr_data_plot(array1, array2, array3, dataframe):
    s = np.arange(0, len(array1))

    fig, axs = plt.subplots(4, 2, sharex=False, sharey=False)

    axs[0, 0].plot(s, dataframe['Gyr_X'], color='red')
    axs[0, 0].set_xlim(0, len(array1))
    axs[0, 0].set_ylabel('x_raw')
    axs[0, 0].grid(True)

    axs[1, 0].plot(s, dataframe['Gyr_Y'], color='green')
    axs[1, 0].set_xlim(0, len(array1))
    axs[1, 0].set_ylabel('y_raw')
    axs[1, 0].grid(True)

    axs[2, 0].plot(s, dataframe['Gyr_Z'], color='blue')
    axs[2, 0].set_xlim(0, len(array1))
    axs[2, 0].set_ylabel('z_raw')
    axs[2, 0].grid(True)

    # axs[3,0].plot(s, dataframe['Depth'],color='black')
    # axs[3,0].set_xlim(0, len(array1))
    # axs[3,0].set_ylabel('depth')
    # axs[3,0].grid(True)

    axs[0, 1].plot(s, array1, color='red')
    axs[0, 1].set_xlim(0, len(array1))
    axs[0, 1].set_ylabel('x_corr')
    axs[0, 1].grid(True)

    axs[1, 1].plot(s, array2, color='green')
    axs[1, 1].set_xlim(0, len(array1))
    axs[1, 1].set_ylabel('y_corr')
    axs[1, 1].grid(True)

    axs[2, 1].plot(s, array3, color='blue')
    axs[2, 1].set_xlim(0, len(array1))
    axs[2, 1].set_ylabel('z_corr')
    axs[2, 1].grid(True)

    # axs[3,1].plot(s, dataframe['Depth'],color='black')
    # axs[3,1].set_xlim(0, len(array1))
    # axs[3,1].set_ylabel('depth')
    # axs[3,1].grid(True)

    fig.canvas.manager.set_window_title('Final Gyroscope Comparison Plots')
    fig.tight_layout()
    plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_FinalComparisonGyroscopePlots.png'), dpi=600)


# plt.savefig('/Users/dylan/Desktop/sharkdata/diagnostic_plots/'+str(meta_file)+'_FinalComparisonPlots.png',dpi=600)

def accel_groomed_plots(xvector, yvector, zvector):
    s3 = np.arange(0, len(xvector))
    fig2, ax2 = plt.subplots()
    ax2.plot(s3, xvector, 'r', label="x")
    ax2.plot(s3, yvector, 'g', label="y")
    ax2.plot(s3, zvector, 'b', label="z")
    plt.grid(True)
    legend = ax2.legend(loc='best')
    fig2.canvas.manager.set_window_title('Groomed Accelerometer Plots')
    plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_GroomedAccelerationPlots.png'), dpi=600)


# plt.savefig('/Users/dylan/Desktop/sharkdata/diagnostic_plots/'+str(meta_file)+'_GroomedAccelerationPlots.png',dpi=600)

def mag_groomed_plots(xvector, yvector, zvector):
    s3 = np.arange(0, len(xvector))
    fig2, ax2 = plt.subplots()
    ax2.scatter(s3, xvector, c='r', label="x", s=10)
    ax2.scatter(s3, yvector, c='g', label="y", s=10)
    ax2.scatter(s3, zvector, c='b', label="z", s=10)
    plt.grid(True)
    legend = ax2.legend(loc='best')
    fig2.canvas.manager.set_window_title('Groomed Magnetometer Plots')
    plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_GroomedMagnetometerPlots.png'), dpi=600)


# plt.savefig('/Users/dylan/Desktop/sharkdata/diagnostic_plots/'+str(meta_file)+'_GroomedMagnetometerPlots.png',dpi=600)

def gyr_groomed_plots(xvector, yvector, zvector):
    s3 = np.arange(0, len(xvector))
    fig2, ax2 = plt.subplots()
    ax2.plot(s3, xvector, 'r', label="x")
    ax2.plot(s3, yvector, 'g', label="y")
    ax2.plot(s3, zvector, 'b', label="z")
    plt.grid(True)
    legend = ax2.legend(loc='best')
    fig2.canvas.manager.set_window_title('Groomed Gyroscope Plots')
    plt.savefig(os.path.join(str(meta_file), str(meta_file) + '_GroomedGyroscopePlots.png'), dpi=600)


# plt.savefig('/Users/dylan/Desktop/sharkdata/diagnostic_plots/'+str(meta_file)+'_GroomedGyroscopePlots.png',dpi=600)

def x_rotation(vector, theta):
    # Rotates 3-D vector around x-axis
    global xR

    xR = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

    return np.dot(xR, vector)


def y_rotation(vector, theta):
    # Rotates 3-D vector around y-axis
    global yR

    yR = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

    return np.dot(yR, vector)


def z_rotation(vector, theta):
    # Rotates 3-D vector around z-axis
    global zR

    zR = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    return np.dot(zR, vector)


def convolve(smoothed_array, array):
    angle = []
    angle -= np.average(smoothed_array)
    step = np.hstack((np.ones(len(smoothed_array)), -1 * np.ones(len(smoothed_array))))
    ang_step = np.convolve(array, step, mode='valid')

    return ang_step


def detect_peaks(anglestep):
    global rotation_chunks

    upper_peaks, _ = find_peaks(anglestep, prominence=1)  # , width=20)
    lower_peaks, _ = find_peaks(-anglestep, prominence=1)  # , width=20)
    peaks = np.sort(np.concatenate([upper_peaks, lower_peaks]))
    peaks = np.sort(np.append(peaks, 0))
    peaks = np.sort(np.append(peaks, len(anglestep) - 1))

    return peaks


# **********************************************************
# %%Main program
# Start the timer for computational purposes
t = time.time()
print('Ready....Set....Go!')  # The program has passed syntax checks

# Set your working directory
new_directory = "C:/Users/srupa/MSCI_437/Data"
os.chdir(new_directory)

# Verify the change
updated_directory = os.getcwd()
print(f"Updated working directory: {updated_directory}")

# %% Prepping the data
# Get the data
get_arguments()
File_object = open(str(meta_file) + '/' + str(meta_file) + '_log.txt', "w")
File_object.write(str(meta_file) + '\n')
File_object.write('Ready....Set....Go!\n')

# param_read_in()

df = data_read_in(filename)
print('Data loaded... ', str(time.time() - t) + "s")
File_object.write('Data loaded... ')
File_object.write(str(time.time() - t) + "s\n")

if args.calib:
    df_calib = pd.read_csv(args.calib, delimiter=';')
    print("Using separate calibration file.")
else:
    df_calib = pd.read_csv(args.file, delimiter=';').iloc[:1000]
    print("Using first 1000 rows of deployment file as calibration data.")

# print("describe" + str(df['magX'].describe()))
acc_offset, mag_offset = get_sensor_offsets(df_calib)
# print(acc_offset,mag_offset)
df = apply_offsets(df, acc_offset, mag_offset)
# print("describe" + str(df['magX'].describe()))

tag_id = os.path.splitext(os.path.basename(args.file))[0]
plot_sensor_data(df, meta_file, suffix="_raw")

# df = trim_data(df)
# df_trimmed.dropna(subset=['magX', 'magY', 'magZ', 'Temp. (?C)', 'Battery Voltage (V)'], inplace=True)
# plot_sensor_data(df, meta_file, suffix="_trimmed")

# df_trimmed.to_csv(args.out, index=False)
# print(f'Final trimmed and calibrated file written to: {args.out}')


# Calculate Sampling Frequency of AXY
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%Y/%m/%d %H:%M:%S.%f")  # Convert Timestamp to Datetime
time_diff = df.iloc[11, df.columns.get_loc('Timestamp')] - df.iloc[10, df.columns.get_loc(
    'Timestamp')]  # Difference in time between 11th and 10th rows (because first couple rows may be weird)
frequency = 1 / time_diff.total_seconds()  # do the maths
frequency = int(frequency)  # convert to a whole number
print('Collection frequency is: ' + str(frequency) + ' Hz\n')

# Plot the original data for comparision
temp_plot(df, clip=False)
depth_plot(df)
accel_raw_plots(df)
mag_raw_plots(df)
# gyr_raw_plots(df)
print('Raw plots generated... ', str(time.time() - t) + "s")
File_object.write('Raw plots generated... ')
File_object.write(str(time.time() - t) + "s\n")

# Clip data - This removes data that isn't from periods when the tag is attached to the shark
df = df[front_clip:end_clip]
df.reset_index(drop=True, inplace=True)
temp_plot(df, clip=True)
depth_plot_clipped(df)
accel_clipped_raw_plots(df)
mag_clipped_raw_plots(df)
# gyr_clipped_raw_plots(df)
print('File clipped... ', str(time.time() - t) + "s")
File_object.write('File clipped... ')
File_object.write(str(time.time() - t) + "s\n")

tcal = args.tcal
gcal = args.gcal

df['accX'] = df['accX'] / gcal
df['accY'] = df['accY'] / gcal
df['accZ'] = df['accZ'] / gcal
df['Temp'] = df['Temp'] / tcal

# This does a lot.
matrix_reshape(df)
print('The matrix has been reshaped, Shreya... ', str(time.time() - t) + "s")
File_object.write('The matrix has been reshaped, Mr. Moran... ')
File_object.write(str(time.time() - t) + "s\n")

create_arrays(axm, axtm)
print('Arrays initialized... ', str(time.time() - t) + "s")
File_object.write('Arrays initialized... ')
File_object.write(str(time.time() - t) + "s\n")
# *****************************************************************#
# %% Rotation angles
# Loop through to determine the required rotation angle for each set of values in the windows. Remember that
# this is being done on the mean in each 15 minute window and not ont the actual data. That comes later. The mean for
# X and Y will be set to zero in this process.
for d in range(0, len(axm)):
    # First rotation (around y axis)
    a1[d] = np.deg2rad(np.rad2deg(np.arctan2(azm[d], axm[d])) + 90)
    x1[d], y1[d], z1[d] = y_rotation(np.array([axm[d], aym[d], azm[d]]), a1[d])

    # Second rotation (around x axis)
    a2[d] = np.deg2rad(np.rad2deg(np.arctan2(z1[d], math.sqrt(y1[d] ** 2 + x1[d] ** 2))) + 90)

# Plot the angle diagnostic plot
a1df = pd.DataFrame(a1)
a2df = pd.DataFrame(a2)
# %% Diagnostic Plots
x = np.arange(0, len(a1))
y1 = np.array(a1df.rolling(math.floor(total_windows * 0.1)).apply(np.nanmean, raw=False)).flatten()
y2 = np.array(a2df.rolling(math.floor(total_windows * 0.1)).apply(np.nanmean, raw=False)).flatten()

dy1 = np.zeros(y1.shape, float)  # we know it will be this size
dy1[0:-1] = np.diff(y1) / np.diff(x)
dy1[-1] = (y1[-1] - y1[-2]) / (x[-1] - x[-2])

dy2 = np.zeros(y2.shape, float)  # we know it will be this size
dy2[0:-1] = np.diff(y2) / np.diff(x)
dy2[-1] = (y2[-1] - y2[-2]) / (x[-1] - x[-2])

angle_diagnostic_plots(a1df, a2df, x, y1, y2, dy1, dy2)

# Convert ahat to array from dataframe
ahat1 = a1df.to_numpy()
ahat2 = a2df.to_numpy()

# Find inflection points for average angles
ang1_step = convolve(y1, a1)
ang2_step = convolve(y2, a2)

peaks1 = args.peaks1
peaks1 = np.asarray(peaks1)
peaks2 = args.peaks2
peaks2 = np.asarray(peaks2)

if peaks1.size != 0:
    print("peaks1 not empty")
    print(peaks1)
else:
    print("peaks1 empty")
    peaks1 = detect_peaks(ang1_step)
    print(peaks1)

if peaks2.size != 0:
    print("peaks2 not empty")
    print(peaks2)
else:
    print("peaks2 empty")
    peaks2 = detect_peaks(ang2_step)
    print(peaks2)

File_object.write('a1 peaks ')
file_peaks1 = np.array2string(peaks1)
File_object.write(file_peaks1)
File_object.write('\na2 peaks ')
file_peaks2 = np.array2string(peaks2)
File_object.write(file_peaks2)

inflection_diagnostic_plots(ang1_step, peaks1, ang2_step, peaks2)

print('Fancy diagnostic plots generated... ', str(time.time() - t) + "s")
File_object.write('\nFancy diagnostic plots generated... ')
File_object.write(str(time.time() - t) + "s\n")

num_rows, num_cols = aztm.shape
# %% Apply transformation
# Apply transformation
for j in range(0, len(peaks1) - 1):
    a1_mean = np.mean(a1[peaks1[j]:peaks1[j + 1]])
    for e in range(peaks1[j], peaks1[j + 1]):  # Fix this range
        for d in range(0, num_cols):
            # First Rotation (y direction)
            ax1a[e][d], ay1a[e][d], az1a[e][d] = y_rotation(np.array([axtm[e][d], aytm[e][d], aztm[e][d]]), a1_mean)
            mx1a[e][d], my1a[e][d], mz1a[e][d] = y_rotation(np.array([mxtm[e][d], mytm[e][d], mztm[e][d]]), a1_mean)
        # gx1a[e][d], gy1a[e][d], gz1a[e][d] = y_rotation(np.array([gxtm[e][d],gytm[e][d],gztm[e][d]]), a1_mean)

for i in range(0, len(peaks1) - 1):
    a2_mean = np.mean(a2[peaks1[i]:peaks1[i + 1]])
    for e in range(peaks1[i], peaks1[i + 1]):  # Fix this one too
        for d in range(0, num_cols):
            # Second Rotation (x direction)
            ax2a[e][d], ay2a[e][d], az2a[e][d] = x_rotation(np.array([ax1a[e][d], ay1a[e][d], az1a[e][d]]), a2_mean)
            mx2a[e][d], my2a[e][d], mz2a[e][d] = x_rotation(np.array([mx1a[e][d], my1a[e][d], mz1a[e][d]]), a2_mean)
        # gx2a[e][d], gy2a[e][d], gz2a[e][d] = x_rotation(np.array([gx1a[e][d],gy1a[e][d],gz1a[e][d]]), a2_mean)

print('Rotation matrices applied... ', str(time.time() - t) + "s")
File_object.write('Rotation matrices applied... ')
File_object.write(str(time.time() - t) + "s\n")

# %% Final plots
# Generate final plots
ax_vect = np.transpose(ax2a.flatten())
ay_vect = np.transpose(ay2a.flatten())
az_vect = np.transpose(az2a.flatten())
mx_vect = np.transpose(mx2a.flatten())
my_vect = np.transpose(my2a.flatten())
mz_vect = np.transpose(mz2a.flatten())
# gx_vect = np.transpose(gx2a.flatten())
# gy_vect = np.transpose(gy2a.flatten())
# gz_vect = np.transpose(gz2a.flatten())

new_xyz = np.column_stack((ax_vect, ay_vect, az_vect, mx_vect, my_vect, mz_vect))
# ,gx_vect,gy_vect,gz_vect))

accel_groomed_plots(ax_vect, ay_vect, az_vect)
mag_groomed_plots(mx_vect, my_vect, mz_vect)
# gyr_groomed_plots(gx_vect,gy_vect,gz_vect)
print('Final corrected plots generated... ', str(time.time() - t) + "s")
File_object.write('Final corrected plots generated... ')
File_object.write(str(time.time() - t) + "s\n")

accel_data_plot(ax_vect, ay_vect, az_vect, df)
mag_data_plot(mx_vect, my_vect, mz_vect, df)
# gyr_data_plot(gx_vect,gy_vect,gz_vect,df)
print('Final comparison plots generated... ', str(time.time() - t) + "s")
File_object.write('Final comparison plots generated... ')
File_object.write(str(time.time() - t) + "s\n")

# %% building final dataframes
df['accX'] = pd.DataFrame(ax_vect)
df['accY'] = pd.DataFrame(ay_vect)
df['accZ'] = pd.DataFrame(az_vect)
df['magX'] = pd.DataFrame(mx_vect)
df['magY'] = pd.DataFrame(my_vect)
df['magZ'] = pd.DataFrame(mz_vect)
# df['Gyr_X'] = pd.DataFrame(gx_vect)
# df['Gyr_Y'] = pd.DataFrame(gy_vect)
# df['Gyr_Z'] = pd.DataFrame(gz_vect)

File_object.write('\nAcc_X Statistics\n')
File_object.write(df['accX'].describe().to_string())
File_object.write('\n\nAcc_Y Statistics\n')
File_object.write(df['accY'].describe().to_string())
File_object.write('\n\nAcc_Z Statistics\n')
File_object.write(df['accZ'].describe().to_string())
File_object.write('\n')
File_object.write('\nMag_X Statistics\n')
File_object.write(df['magX'].describe().to_string())
File_object.write('\n\nMag_Y Statistics\n')
File_object.write(df['magY'].describe().to_string())
File_object.write('\n\nAMag_Z Statistics\n')
File_object.write(df['magZ'].describe().to_string())
# File_object.write('\n')
# File_object.write('\nGyr_X Statistics\n')
# File_object.write(df['Gyr_X'].describe().to_string())
# File_object.write('\n\nGyr_Y Statistics\n')
# File_object.write(df['Gyr_Y'].describe().to_string())
# File_object.write('\n\nGyr_Z Statistics\n')
# File_object.write(df['Gyr_Z'].describe().to_string())
File_object.write('\n')

# %%Writes the groomed dataset to a csv
pd.DataFrame(df).to_csv(os.path.join(str(meta_file), str(meta_file) + '_groomed.csv'),
                        index=False)  # write the groomed dataframe out to a csv

if frequency == 20:
    print('No downsample needed.')
    File_object.write('No downsample needed.\n')
elif frequency == 100:
    df = df.iloc[::4]
    File_object.write('Downsampled by a factor of 4.\n')
else:
    df = df.iloc[::2]
    File_object.write('Downsampled by a factor of 2.\n')

pd.DataFrame(df).to_csv(os.path.join(str(meta_file), str(meta_file) + '_groomed_down.csv'),
                        index=False)  # write the groomed dataframe out to a csv
print('Data groomed and written to ', os.path.join(str(meta_file), str(meta_file) + '_groomed_down.csv'),
      ' new file at ', str(time.time() - t) + "s")
File_object.write('Data groomed and written to ')
File_object.write(os.path.join(str(meta_file), str(meta_file) + '_groomed_down.csv'))
File_object.write(' at ')
File_object.write(str(time.time() - t) + "s\n")

print('Fin.')
File_object.write('Fin.')
File_object.close()