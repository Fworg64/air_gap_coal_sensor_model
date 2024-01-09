# This file uses sklearn library to setup ML model comparison experiment with cross validation
# It will write out files to an 'out' directory that it will try to create
import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate

from scipy import signal
from scipy.signal import decimate

import argparse

import pdb

##
# Set Figures
##
SMALL_SIZE = 20
MEDIUM_SIZE = 24
SMALLER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

##
# Commnad line defaults
##

# Use decimated values instead of moving average
argument_use_decimated_vals = 0

# Use up to N consectutive samples for classification
argument_window_len = 1

# Don't Include NN regressions
argument_no_nn = False

# Do filter calcs
argument_filt = True

## Allow command line overrides
parser = argparse.ArgumentParser()

parser.add_argument("--decimated", default=argument_use_decimated_vals, type=int,
  help="Set to 1 or greater to use decimation")

# Making command line argument for window duration
parser.add_argument("--window_len", default=argument_window_len, type=int,
  help="Number of how many consecutive samples to give regressor")

parser.add_argument("--no_nn", default=argument_no_nn, type=bool,
  help="Set to True to omit NN from processing")

parser.add_argument("--filt", default=argument_filt, type=bool,
  help="Set to False to omit input/output filters")

args = parser.parse_args()

##
# Load all good csv files as indicated by index.csv
##

files_index = pd.read_csv("../data/coal_csvs/index.csv")

files_index.loc[files_index["Data is good"] == True]

good_files_list = files_index.loc[files_index["Data is good"] == True]["Filename"].to_list()

# Dictionary of dataframes
data_files_dict = {}
for filename in good_files_list:
    fullpath = "../data/coal_csvs/" + filename
    data_files_dict[filename] = pd.read_csv(fullpath)

print("Data Loaded.")
this_time = time.time()


chan_names_list = ["Freq. A (MHz)", "Freq. B (MHz)", "Freq. C (MHz)", "Freq. D (MHz)"]

# Moving average size in number of samples    
avg_len = 40

# Dictionary of dictionary of lists
data_list_dict = {}
# Preprocess data to have bias adjusted based on first 0.5s from sample
x_vals = []
y_vals = []

decimation_factor = 2
if args.decimated > 0:
  decimation_factor = args.decimated
  print(f"Decimating by factor of {decimation_factor}")

chan_key = ""

for filename in good_files_list:
    
    # Compute avg force values
    data_files_dict[filename]["Avg. Force (kN)"] = \
        np.convolve(data_files_dict[filename]["Force (kN)"], 
          np.ones(avg_len), 'same') / avg_len
    data_list_dict[filename] = {
        "Avg. Force (kN)": 
            data_files_dict[filename]["Avg. Force (kN)"].to_list()}

    # Compute decimated force values
    data_list_dict[filename]["Decimated Force (kN)"] = \
        decimate(data_files_dict[filename]["Force (kN)"],
          q=decimation_factor, n=40, ftype='fir', axis=0, zero_phase=True)

    # Compute 5 Hz, 10 Hz, and 20 Hz filtering of drag force
    sample_freq = 1.0 / 0.002475#1.0 / (data_list_dict[filename]["Time (s)"][1] - data_list_dict[filename]["Time (s)"][0])
    force_freqs = [5, 10, 20] # Hz at 10 in/s this is 5,10,20 goes to 2 in, 1 in, 0.5 in
    for force_freq in force_freqs:
      bc, ac = signal.butter(N=10, btype="low", Wn=force_freq, fs=sample_freq)
      data_list_dict[filename][f"Filt Force {force_freq} Hz Force (kN)"] = \
          signal.filtfilt(bc, ac, data_files_dict[filename]["Force (kN)"])
    
    # Compute bias compensated avg values
    chan_freqs = [10, 20, 40]
    for chan_name in chan_names_list:
      # Auto-Bias adjustment (subtract average of first 0.5s from sample)
      bias_adjustment_segment = data_files_dict[filename].loc[
          data_files_dict[filename]["Time (s)"] < 0.5]
      freq_bias_adjustment_val = np.max(bias_adjustment_segment[chan_name])
      data_files_dict[filename][f"Adj {chan_name}"] = \
          1000.0 * (freq_bias_adjustment_val - data_files_dict[filename][chan_name])

      # Compute avg sensor values
      data_files_dict[filename][f"Avg. Adj {chan_name}"] = \
          np.convolve(data_files_dict[filename][f"Adj {chan_name}"],
            np.ones(20), 'same') / 20
      data_list_dict[filename][f"Avg. Adj {chan_name}"] = \
          data_files_dict[filename][f"Avg. Adj {chan_name}"].to_list()

      # Add decimated values
      data_list_dict[filename][f"Decimated {chan_name}"] = \
          decimate(data_files_dict[filename][f"Adj {chan_name}"],
                   q=decimation_factor, n=40, ftype='fir', axis=0, zero_phase=True)

      # Compute filtered values
      for chan_freq in chan_freqs:
          bc, ac = signal.butter(N=10, btype="low", Wn=chan_freq, fs=sample_freq)
          data_list_dict[filename][f"Filt {chan_freq} Hz Adj {chan_name}"] = \
              signal.filtfilt(bc, ac, data_files_dict[filename][f"Adj {chan_name}"])

    chan_key = "Avg. Adj"
    force_key = "Avg."
    if (args.decimated > 0):
        chan_key = "Decimated"
        force_key = "Decimated"
    if (args.filt == True):
        chan_key = "Filt 10 Hz Adj"
        force_key = "Filt Force 10 Hz"

    # Glob data into X and Y lists, use decimated values if arg
    chan_vals = [data_list_dict[filename][f"{chan_key} {chan_name}"] 
                  for chan_name in chan_names_list] # list of each channel
    chan_time_list = [list(vals) for vals in zip(*chan_vals)] # transpose to each channel as column
    x_vals.extend(chan_time_list) # each entry is all 4 channels at same time
    y_vals.extend(data_list_dict[filename][f"{force_key} Force (kN)"])

# Add first order difference to sample
add_diffs = False

predictor_key = chan_key

##
# Window
##
window_len = args.window_len
# Window x values to windows of len N
# Generate polynomial expansions
poly = PolynomialFeatures(degree=2, include_bias=False)
# Flatten samples to [t0 t1 ...]
# Window the xvals
nn_window_size = window_len
nn_window_x_vals = []
nn_window_poly_x_vals = []
nn_window_y_vals = []
for index in range(len(x_vals)-nn_window_size + 1):
    chanelled_vals = x_vals[index:index+nn_window_size]
    if add_diffs and window_len > 1:
        plus = chanelled_vals[1:window_len]
        minus = chanelled_vals[0:window_len-1]
        diff_chanells = [
          [p[0] - m[0], p[1] - m[1], p[2] - m[2], p[3] - m[3]] 
          for p,m in zip(plus, minus)]
        chanelled_vals.extend(diff_chanells)
    # convert each row to poly expansion
    poly_chan_vals = poly.fit_transform(chanelled_vals)
    # Flatten X to be abcd abcd abcd...
    nn_window_x_vals.append([val for row in chanelled_vals for val in row])
    # Flatten polyX to be abcda2b2c2... abcda2b2c2.... ...
    nn_window_poly_x_vals.append([val for row in poly_chan_vals for val in row])
    nn_window_y_vals.append(y_vals[index+nn_window_size-1])

nn_window_y_vals = np.array(nn_window_y_vals) # add inner dim

num_samples = nn_window_y_vals.shape[0]
linear_input_dim = len(nn_window_x_vals[0])
poly_input_dim = len(nn_window_poly_x_vals[0])

print(f"Using decimated samples: {args.decimated}")
print(f"Regressor window len: {args.window_len}")

print(f"Total number of samples: {num_samples}")
# Should be num channels * window size
print(f"Linear input dim: {linear_input_dim}")
# Should be num channels combos * window size
print(f"Poly input dim: {poly_input_dim}")

that_time = time.time()
print("Data preprocessed in {0} sec; performing experiments".format(that_time - this_time),
      end='', flush=True)
this_time = time.time()

# Manually add Rozeboom 1987 adjusted value
scorings = ['r2','neg_root_mean_squared_error', 'neg_mean_absolute_error']

##
# Linear Model
##

# Fit data with least squares
reg = linear_model.LinearRegression()
reg.fit(x_vals, y_vals)
print(f"Linear coef: {reg.coef_}, Intercept: {reg.intercept_}")

y_hats = reg.predict(x_vals)

print("lin MSE: ")
print(mean_squared_error(y_vals, y_hats))
print("lin R2: ")
print(r2_score(y_vals, y_hats))

# Add model data to data dict
for filename in good_files_list:
    chan_vals = [data_list_dict[filename][f"{predictor_key} {chan_name}"]
                  for chan_name in chan_names_list]
    chan_time_list = [list(vals) for vals in zip(*chan_vals)]
    data_files_dict[filename]["Lin Est Force (kN)"] = \
      reg.predict(chan_time_list)

##
# Poly fit
##

poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly_vals = poly.fit_transform(x_vals)
poly_reg_model = linear_model.LinearRegression()
poly_reg_model.fit(x_poly_vals, y_vals)
y_poly_hats = poly_reg_model.predict(x_poly_vals)

print(f"Poly coef: {poly_reg_model.coef_}, Intercept: {poly_reg_model.intercept_}")
print("poly MSE: ")
print(mean_squared_error(y_vals, y_poly_hats))
print("poly R2: ")
print(r2_score(y_vals, y_poly_hats))

# Add model data to data dict
for filename in good_files_list:
    chan_vals = [data_list_dict[filename][f"{predictor_key} {chan_name}"]
                  for chan_name in chan_names_list]
    chan_time_list = [list(vals) for vals in zip(*chan_vals)]
    data_files_dict[filename]["Poly Fit Force (kN)"] = \
        poly_reg_model.predict(poly.fit_transform(
            np.array(chan_time_list)))

##
# FFNN fit
##
print("Fitting ffnn...")
# Window the xvals
nn_window_size = args.window_len
nn_window_x_vals = []
nn_window_y_vals = []
for index in range(len(x_vals)-nn_window_size + 1):
    chanelled_vals = x_vals[index:index+nn_window_size]
    # Flatten X to be abcd abcd abcd...
    nn_window_x_vals.append([val for row in chanelled_vals for val in row])
    nn_window_y_vals.append(y_vals[index+nn_window_size-1])

nn_window_y_vals = np.array(nn_window_y_vals) # add inner dim


ffnn = MLPRegressor((nn_window_size, nn_window_size, nn_window_size, nn_window_size), activation='relu', solver='adam',
          max_iter=800, verbose=False, random_state=12345).fit(
    nn_window_x_vals, nn_window_y_vals)

ffnn_y_hat = ffnn.predict(nn_window_x_vals)
print("ffnn MSE: ")
print(mean_squared_error(nn_window_y_vals, ffnn_y_hat))
print("ffnn R2: ")
print(r2_score(nn_window_y_vals, ffnn_y_hat))

## Add model data to data dict
for filename in good_files_list:
    # break file data into windows
    chan_vals = [data_list_dict[filename][f"{predictor_key} {chan_name}"]
                  for chan_name in chan_names_list]
    chan_time_list = [list(vals) for vals in zip(*chan_vals)]
    windowed_vals = []
    for index in range(len(chan_time_list) - nn_window_size +1):
        chanelled_vals = chan_time_list[index:index+nn_window_size]
        windowed_vals.append([val for row in chanelled_vals for val in row]) # flatten
    windowed_vals = [windowed_vals[0]] * (nn_window_size-1) + windowed_vals

    data_files_dict[filename]["FFNN Fit Force (kN)"] = \
      ffnn.predict(windowed_vals)


##
# Poly FFNN fit
##
print("Fitting Poly FFNN...")
# reuse nn_window_size and nn_window_y_vals
poly_nn_x_vals = []
for index in range(len(x_poly_vals) - nn_window_size + 1):
    chanelled_vals = x_poly_vals[index:index+nn_window_size]
    # Flatten X
    poly_nn_x_vals.append([val for row in chanelled_vals for val in row])

poly_ffnn = MLPRegressor((nn_window_size, nn_window_size, nn_window_size, nn_window_size), activation='relu', solver='adam',
                max_iter=800, verbose=False, random_state=54321).fit(
        poly_nn_x_vals, nn_window_y_vals)

poly_ffnn_y_hat = poly_ffnn.predict(poly_nn_x_vals)

print("poly ffnn MSE: ")
print(mean_squared_error(nn_window_y_vals, poly_ffnn_y_hat))
print("poly ffnn R2: ")
print(r2_score(nn_window_y_vals, poly_ffnn_y_hat))

## Add model data to data dict
for filename in good_files_list:
    # break file data into windows
    chan_vals = [data_list_dict[filename][f"{predictor_key} {chan_name}"]
                  for chan_name in chan_names_list]
    chan_time_list = [list(vals) 
                      for vals in zip(*chan_vals)]
    poly_chan_time_list = poly.fit_transform(chan_time_list)
    windowed_vals = []
    for index in range(len(chan_time_list) - nn_window_size +1):
        chanelled_vals = poly_chan_time_list[index:index+nn_window_size]
        windowed_vals.append([val for row in chanelled_vals for val in row]) # flatten
    windowed_vals = [windowed_vals[0]] * (nn_window_size-1) + windowed_vals

    data_files_dict[filename]["Poly FFNN Fit Force (kN)"] = \
      poly_ffnn.predict(windowed_vals)

print("Done fitting models!")

# Plot bias adjusted traces
num_plot_rows = 2
num_plot_cols = 2
fig1, axes1 = plt.subplots(num_plot_rows, num_plot_cols)
fig2, axes2 = plt.subplots(num_plot_rows, num_plot_cols)
fig3, axes3 = plt.subplots(num_plot_rows, num_plot_cols)
fig4, axes4 = plt.subplots(num_plot_rows, num_plot_cols)
fig5, axes5 = plt.subplots(num_plot_rows, num_plot_cols)
axes_list = [axes1, axes2, axes3, axes4, axes5]
  
# Plot time series data
fig_ts1, axes_ts1 = plt.subplots(num_plot_rows, num_plot_cols)
fig_ts2, axes_ts2 = plt.subplots(num_plot_rows, num_plot_cols)
fig_ts3, axes_ts3 = plt.subplots(num_plot_rows, num_plot_cols)
fig_ts4, axes_ts4 = plt.subplots(num_plot_rows, num_plot_cols)
fig_ts5, axes_ts5 = plt.subplots(num_plot_rows, num_plot_cols)
axes_ts_list = [axes_ts1, axes_ts2, axes_ts3, axes_ts4, axes_ts5]

# Plot row major
for idx, filename in enumerate(good_files_list):
    # Calculate indicies
    axes_dex = int(idx / (num_plot_rows * num_plot_cols))
    row_dex = int((idx) / num_plot_cols) % num_plot_rows
    col_dex = idx % num_plot_cols

    print(f"Axes {axes_dex}, row {row_dex}, col {col_dex}. {idx}. {filename}")
    # Plot channel freq vs avg'd lcm force
    axes_list[axes_dex][row_dex][col_dex].scatter(
            data_files_dict[filename]["Avg. Force (kN)"], 
            data_list_dict[filename][f"{predictor_key} Freq. A (MHz)"], 
            label="Chan A", marker='x', color="blue", s=2)
    axes_list[axes_dex][row_dex][col_dex].scatter(
            data_files_dict[filename]["Avg. Force (kN)"], 
            data_list_dict[filename][f"{predictor_key} Freq. B (MHz)"], 
            label="Chan B", marker='+', color="blue", s=2)
    axes_list[axes_dex][row_dex][col_dex].scatter(
            data_files_dict[filename]["Avg. Force (kN)"], 
            data_list_dict[filename][f"{predictor_key} Freq. C (MHz)"], 
            label="Chan C", marker='o', color="blue", s=2)
    axes_list[axes_dex][row_dex][col_dex].scatter(
            data_files_dict[filename]["Avg. Force (kN)"], 
            data_list_dict[filename][f"{predictor_key} Freq. C (MHz)"], 
            label="Chan D", marker='*', color="blue", s=2)
    # Plot model fits on twin'd x axis
    twin_force_axis = axes_list[axes_dex][row_dex][col_dex].twinx()
    twin_force_axis.scatter(
        data_files_dict[filename]["Avg. Force (kN)"], 
        data_files_dict[filename]["Lin Est Force (kN)"], c='red', label='Linear')
    twin_force_axis.scatter(
        data_files_dict[filename]["Avg. Force (kN)"], 
        data_files_dict[filename]["Poly Fit Force (kN)"], c='purple', label='Quadratic')
    twin_force_axis.scatter(
        data_files_dict[filename]["Avg. Force (kN)"], 
        data_files_dict[filename]["FFNN Fit Force (kN)"], c='black',
        marker='D', s=3, label='Neural Net')
    twin_force_axis.set_ylabel("Estimated Force")
    twin_force_axis.set_ylim(-5,50)

    axes_list[axes_dex][row_dex][col_dex].legend(loc="lower right")
    axes_list[axes_dex][row_dex][col_dex].set_ylabel("Freq. Deviation (kHz)")
    axes_list[axes_dex][row_dex][col_dex].set_xlabel("Force (kN)")

    axes_list[axes_dex][row_dex][col_dex].text(-115, 70, filename)
    axes_list[axes_dex][row_dex][col_dex].set_ylim(-10, 100)
    #axes_list[axes_dex][row_dex][col_dex].set_xlim(1.65, 1.85)
    axes_list[axes_dex][row_dex][col_dex].set_xlim(-5, 50)

    # plot time data
    axes_ts_list[axes_dex][row_dex][col_dex].plot(
        data_files_dict[filename]["Time (s)"], 
        data_files_dict[filename]["Force (kN)"], 
        color='lightsteelblue', label='Raw Force (kN)')
    axes_ts_list[axes_dex][row_dex][col_dex].plot(
        data_files_dict[filename]["Time (s)"], 
        data_files_dict[filename]["Avg. Force (kN)"], 
        color='black', label='Avg Force (kN)')
    axes_ts_list[axes_dex][row_dex][col_dex].plot(
        data_files_dict[filename]["Time (s)"], 
        data_list_dict[filename][f"{force_key} Force (kN)"], 
        color='magenta', label=force_key)
    axes_ts_list[axes_dex][row_dex][col_dex].plot(
        data_files_dict[filename]["Time (s)"], 
        data_files_dict[filename]["Lin Est Force (kN)"],
        color='orange', label="Linear Reg. Est.")
    axes_ts_list[axes_dex][row_dex][col_dex].plot(
        data_files_dict[filename]["Time (s)"], 
        data_files_dict[filename]["Poly Fit Force (kN)"],
        color='cyan', label="Poly Reg. Est.")
    axes_ts_list[axes_dex][row_dex][col_dex].plot(
        data_files_dict[filename]["Time (s)"], 
        data_files_dict[filename]["FFNN Fit Force (kN)"],
        color='forestgreen', label="Linear FFNN")
    #axes_ts_list[axes_dex][row_dex][col_dex].plot(
    #    data_files_dict[filename]["Time (s)"], 
    #    data_files_dict[filename]["Poly FFNN Fit Force (kN)"],
    #    color='magenta', label="Poly FFNN")
    axes_ts_list[axes_dex][row_dex][col_dex].set_ylim(-15, 85)
    axes_ts_list[axes_dex][row_dex][col_dex].set_xlabel("Time (s)")
    axes_ts_list[axes_dex][row_dex][col_dex].set_ylabel("Force (kN)")
    #axes_ts_list[axes_dex][row_dex][col_dex].text(1.00,70, filename)
    axes_ts_list[axes_dex][row_dex][col_dex].set_title(filename)
    axes_ts_list[axes_dex][row_dex][col_dex].legend(loc="upper center", bbox_to_anchor=(0.45, 0.99))
    

    # Twinning frequency axis
    #ax2 = axes_ts_list[axes_dex][row_dex][col_dex].twinx()
    #ax2.plot(
    #    data_files_dict[filename]["Time (s)"], 
    #    data_files_dict[filename]["Avg. Adj Freq. A (MHz)"], ls='--')
    #ax2.plot(
    #    data_files_dict[filename]["Time (s)"], 
    #    data_files_dict[filename]["Avg. Adj Freq. B (MHz)"], ls='--')
    #ax2.plot(
    #    data_files_dict[filename]["Time (s)"], 
    #    data_files_dict[filename]["Avg. Adj Freq. C (MHz)"], ls='--')
    #ax2.plot(
    #    data_files_dict[filename]["Time (s)"], 
    #    data_files_dict[filename]["Avg. Adj Freq. D (MHz)"], ls='--')
    #ax2.set_ylim(-30, 170)
    #ax2.set_ylabel("Freq. Deviation (kHz)")
#
plt.show(block=False)

input("Press Enter to close")
