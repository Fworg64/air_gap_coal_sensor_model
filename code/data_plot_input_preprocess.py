# This file takes all available input data and trains the regressors
# using the prescribed preprocess methods. 
# It displays time domain estimates for all data files as well as
# sensitivity plots for sensor channels and estimates
import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
SMALL_SIZE = 18
MEDIUM_SIZE = 26
SMALLER_SIZE = 18

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

# Don't Include NN regressions
argument_no_nn = False

# Do filter calcs
argument_filt = True

## Allow command line overrides
parser = argparse.ArgumentParser()

parser.add_argument("--no_nn", default=argument_no_nn, type=bool,
  help="Set to True to omit NN from processing")

parser.add_argument("--filt", default=argument_filt, type=bool,
  help="Set to False to omit input filters")

args = parser.parse_args()

##
# Load all good csv files as indicated by index.csv and preprocess
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

sample_freq = 1.0 / 0.002475

chan_names_list = ["Freq. A (MHz)", "Freq. B (MHz)", "Freq. C (MHz)", "Freq. D (MHz)"]

# Moving average size in number of samples for input   
avg_len = 5
chan_freqs = [10]

# Dictionary of dictionary of lists
data_list_dict = {}
# Preprocess data to have bias adjusted based on first 0.5s from sample
x_vals = []
y_vals = []

y_filt_b, y_filt_a = signal.butter(N=10, btype="low", Wn=10.0, fs=sample_freq)

chan_key = ""
force_key = "Filt Force <10 Hz"

for filename in good_files_list:

    # Compute 10 Hz filtering of drag force
    data_list_dict[filename] = {
        "Filt Force <10 Hz Force (kN)": signal.filtfilt(
            y_filt_b, y_filt_a, data_files_dict[filename][f"Force (kN)"])}
    
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
            np.ones(avg_len), 'same') / avg_len
      data_list_dict[filename][f"Avg. Adj {chan_name}"] = \
          data_files_dict[filename][f"Avg. Adj {chan_name}"].to_list()

      # Compute filtered values
      for chan_freq in chan_freqs:
          bc, ac = signal.butter(N=10, btype="low", Wn=chan_freq, fs=sample_freq)
          data_list_dict[filename][f"Filt {chan_freq} Hz Adj {chan_name}"] = \
              signal.filtfilt(bc, ac, data_files_dict[filename][f"Adj {chan_name}"])

    chan_key = "Avg. Adj"
    if (args.filt == True):
        chan_key = "Filt 10 Hz Adj"

    # Glob data into X and Y lists
    chan_vals = [data_list_dict[filename][f"{chan_key} {chan_name}"] 
                  for chan_name in chan_names_list] # list of each channel
    chan_time_list = [list(vals) for vals in zip(*chan_vals)] # transpose to each channel as column
    x_vals.extend(chan_time_list) # each entry is all 4 channels at same time
    y_vals.extend(data_list_dict[filename][f"{force_key} Force (kN)"])

predictor_key = chan_key

# Let x be an np array, as a treat
x_vals = np.array(x_vals)

# Generate polynomial expansions
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_x_vals = poly.fit_transform(x_vals)

# Add inner dim to y_vals
y_vals = np.array(y_vals).ravel() # add inner dim

num_samples = y_vals.shape[0]
linear_input_dim = x_vals.shape[1]
poly_input_dim = poly_x_vals.shape[1]

print(f"Total number of samples: {num_samples}")
# Should be num channels
print(f"Linear input dim: {linear_input_dim}")
# Should be num channels combos
print(f"Poly input dim: {poly_input_dim}")

that_time = time.time()
print("Data preprocessed in {0} sec; performing experiments".format(that_time - this_time),
      end='\n', flush=True)
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
print("lin MAE: ")
print(mean_absolute_error(y_vals, y_hats))
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

poly_reg_model = linear_model.LinearRegression()
poly_reg_model.fit(poly_x_vals, y_vals)
y_poly_hats = poly_reg_model.predict(poly_x_vals)

print(f"Poly coef: {poly_reg_model.coef_}, Intercept: {poly_reg_model.intercept_}")
print("poly MSE: ")
print(mean_squared_error(y_vals, y_poly_hats))
print("poly MAE: ")
print(mean_absolute_error(y_vals, y_poly_hats))
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

ffnn = MLPRegressor(
    (linear_input_dim,) * 3,
    activation='relu', solver='adam',
          max_iter=800, verbose=False, random_state=12345).fit(
    x_vals, y_vals)

ffnn_y_hat = ffnn.predict(x_vals)

print("ffnn MSE: ")
print(mean_squared_error(y_vals, ffnn_y_hat))
print("ffnn MAE: ")
print(mean_absolute_error(y_vals, ffnn_y_hat))
print("ffnn R2: ")
print(r2_score(y_vals, ffnn_y_hat))

## Add model data to data dict
for filename in good_files_list:
    # break file data into windows
    chan_vals = [data_list_dict[filename][f"{predictor_key} {chan_name}"]
                  for chan_name in chan_names_list]
    chan_time_list = [list(vals) for vals in zip(*chan_vals)]

    data_files_dict[filename]["FFNN Fit Force (kN)"] = \
      ffnn.predict(chan_time_list)


##
# Poly FFNN fit
##
print("Fitting Poly FFNN...")

poly_ffnn = MLPRegressor((poly_input_dim,) * 3, activation='relu', solver='adam',
                max_iter=800, verbose=False, random_state=54321).fit(
        poly_x_vals, y_vals)

poly_ffnn_y_hat = poly_ffnn.predict(poly_x_vals)

print("poly ffnn MSE: ")
print(mean_squared_error(y_vals, poly_ffnn_y_hat))
print("poly ffnn MAE: ")
print(mean_absolute_error(y_vals, poly_ffnn_y_hat))
print("poly ffnn R2: ")
print(r2_score(y_vals, poly_ffnn_y_hat))

## Add model data to data dict
for filename in good_files_list:
    # break file data into windows
    chan_vals = [data_list_dict[filename][f"{predictor_key} {chan_name}"]
                  for chan_name in chan_names_list]
    chan_time_list = [list(vals) 
                      for vals in zip(*chan_vals)]
    poly_chan_time_list = poly.fit_transform(chan_time_list)
    
    data_files_dict[filename]["Poly FFNN Fit Force (kN)"] = \
      poly_ffnn.predict(poly_chan_time_list)

that_time = time.time()
print("Models fit in {0} sec; plotting experiments".format(that_time - this_time),
      end='\n', flush=True)
this_time = time.time()

##
# Numerically estimate transfer function via cross-spectral density and psd
# Hbar = Pyx / Pxx; Do for poly ffnn
# Estimate linear system between output of ffnn and chosen regression target
##

psd_freqs, Pxx = signal.welch(poly_ffnn_y_hat, fs=sample_freq)
ypsd_freqs, Pyy = signal.welch(y_vals, fs=sample_freq)
csd_freqs, Pyx = signal.csd(y_vals, poly_ffnn_y_hat, fs=sample_freq)

# Get psd's for each channel 
chan_psd_freqs, chan_psd = signal.welch(x_vals, axis=0, fs=sample_freq)

#Hbar = np.divide(np.real(Pyx), Pxx)
Hbar = np.divide(Pyx, Pxx)

#fig0, axes0 = plt.subplots(2, 2)
fig0 = plt.figure()

gs = fig0.add_gridspec(2,2)
axes0 = []
axes0.append(fig0.add_subplot(gs[0, 0])) # Pyy and Pxx, top left
axes0.append(fig0.add_subplot(gs[1, 0])) # Pyx, bottom left
axes0.append(fig0.add_subplot(gs[:, 1])) # Hbar, right side

# Cross spectral density
axes0[0].plot(chan_psd_freqs, np.abs(chan_psd[:, 0]), 
    label="Chan A", linestyle=":", color="#1F0322")
axes0[0].plot(chan_psd_freqs, np.abs(chan_psd[:, 1]), 
    label="Chan B", linestyle=":", color="#8A1C7C")
axes0[0].plot(chan_psd_freqs, np.abs(chan_psd[:, 2]), 
    label="Chan C", linestyle=":", color="#DA4167")
axes0[0].plot(chan_psd_freqs, np.abs(chan_psd[:, 3]), 
    label="Chan D", linestyle=":", color="#F0BCD4")
axes0[0].plot(ypsd_freqs, np.abs(Pyy), 
    color="magenta", label="Pyy, Target Force")
axes0[0].plot(psd_freqs, np.abs(Pxx), 
    color="#3EA06D", label="Pxx, Poly FFNN Output")

axes0[0].set_title("Pyy and Pxx")
axes0[0].set_yscale("log")
axes0[0].set_ylim( 1e-6, 1e2)
axes0[0].legend(ncol=2, loc="upper right")
axes0[0].set_ylabel("Spectral Density")
axes0[0].set_xlabel("Frequency (Hz)")

#axes0[0,1].plot(psd_freqs, np.abs(Pxx), color="blue", label="Magnitude")
#axes0[0,1].legend()
#axes01_2 = axes0[0,1].twinx()
#axes01_2.plot(psd_freqs, np.angle(Pxx), color="orange", label="Phase")
#axes01_2.legend()
#axes0[0,1].set_title("Pxx")
#axes0[0,1].set_yscale("log")

axes0[1].plot(csd_freqs, np.abs(Pyx), color="purple", label="Magnitude")
axes0[1].legend(loc="upper center")
axes10_2 = axes0[1].twinx()
axes10_2.plot(csd_freqs, np.angle(Pyx), color="orange", label="Phase")
axes10_2.legend(loc="upper right")
axes10_2.set_ylabel("Phase (Rad.)")
axes0[1].set_title("Pyx: Sensor and Target Cross Spectrum")
axes0[1].set_yscale("log")
axes0[1].set_ylabel("Spectral Density")
axes0[1].set_xlabel("Frequency (Hz)")

axes0[2].plot(psd_freqs, np.abs(Hbar), color="purple", label="Magnitude")
axes0[2].legend(loc="upper center")
axes11_2 = axes0[2].twinx()
axes11_2.plot(psd_freqs, np.angle(Hbar), color="orange", label="Phase")
axes11_2.legend(loc="upper right")
axes11_2.set_ylabel("Phase (Rad.)")
axes0[2].set_title("Hbar = Pyx / Pxx. Sensor to Target TF")
axes0[2].set_yscale("log")
axes0[2].set_ylabel("Power Spectral Density Magnitude")
axes0[2].set_xlabel("Frequency (Hz)")

plt.suptitle("Finding Transfer Function between Sensor and Target")

# Plot bias adjusted traces
num_plot_rows = 2
num_plot_cols = 2
fig1, axes1 = plt.subplots(num_plot_rows, num_plot_cols)
fig2, axes2 = plt.subplots(num_plot_rows, num_plot_cols)
fig3, axes3 = plt.subplots(num_plot_rows, num_plot_cols)
fig4, axes4 = plt.subplots(num_plot_rows, num_plot_cols)
fig5, axes5 = plt.subplots(num_plot_rows, num_plot_cols)
fig6, axes6 = plt.subplots(num_plot_rows, num_plot_cols)
axes_list = [axes1, axes2, axes3, axes4, axes5, axes6]
  
# Plot time series data
fig_ts1, axes_ts1 = plt.subplots(num_plot_rows, num_plot_cols, sharex=True, sharey=True)
fig_ts2, axes_ts2 = plt.subplots(num_plot_rows, num_plot_cols, sharex=True, sharey=True)
fig_ts3, axes_ts3 = plt.subplots(num_plot_rows, num_plot_cols, sharex=True, sharey=True)
fig_ts4, axes_ts4 = plt.subplots(num_plot_rows, num_plot_cols, sharex=True, sharey=True)
fig_ts5, axes_ts5 = plt.subplots(num_plot_rows, num_plot_cols, sharex=True, sharey=True)
fig_ts6, axes_ts6 = plt.subplots(num_plot_rows, num_plot_cols, sharex=True, sharey=True)
axes_ts_list = [axes_ts1, axes_ts2, axes_ts3, axes_ts4, axes_ts5, axes_ts6]

# Plot row major
for idx, filename in enumerate(good_files_list):
    # Calculate indicies
    axes_dex = int(idx / (num_plot_rows * num_plot_cols))
    row_dex = int((idx) / num_plot_cols) % num_plot_rows
    col_dex = idx % num_plot_cols

    print(f"Axes {axes_dex}, row {row_dex}, col {col_dex}. {idx}. {filename}")
    # Plot channel freq vs avg'd lcm force
    axes_list[axes_dex][row_dex][col_dex].scatter(
            data_list_dict[filename][f"{force_key} Force (kN)"], 
            data_list_dict[filename][f"{predictor_key} Freq. A (MHz)"], 
            label="Chan A", marker='o', color="#1F0322", s=4)
    axes_list[axes_dex][row_dex][col_dex].scatter(
            data_list_dict[filename][f"{force_key} Force (kN)"], 
            data_list_dict[filename][f"{predictor_key} Freq. B (MHz)"], 
            label="Chan B", marker='o', color="#8A1C7C", s=4)
    axes_list[axes_dex][row_dex][col_dex].scatter(
            data_list_dict[filename][f"{force_key} Force (kN)"], 
            data_list_dict[filename][f"{predictor_key} Freq. C (MHz)"], 
            label="Chan C", marker='o', color="#DA4167", s=4)
    axes_list[axes_dex][row_dex][col_dex].scatter(
            data_list_dict[filename][f"{force_key} Force (kN)"], 
            data_list_dict[filename][f"{predictor_key} Freq. D (MHz)"], 
            label="Chan D", marker='o', color="#F0BCD4", s=4)
    # Plot model fits on twin'd x axis
    twin_force_axis = axes_list[axes_dex][row_dex][col_dex].twinx()
    twin_force_axis.scatter(
        data_list_dict[filename][f"{force_key} Force (kN)"], 
        data_files_dict[filename]["Lin Est Force (kN)"], c='#7E58A0', label='Lin. Reg.', marker="x", s=20)
    #twin_force_axis.scatter(
    #    data_list_dict[filename][f"{force_key} Force (kN)"], 
    #    data_files_dict[filename]["Poly Fit Force (kN)"], c='purple', label='Quadratic')
    #twin_force_axis.scatter(
    #    data_list_dict[filename][f"{force_key} Force (kN)"], 
    #    data_files_dict[filename]["FFNN Fit Force (kN)"], c='yellow',
    #    marker='D', s=3, label='Neural Net')
    twin_force_axis.scatter(
        data_list_dict[filename][f"{force_key} Force (kN)"], 
        data_files_dict[filename]["Poly FFNN Fit Force (kN)"], c='#3EA06D',
        marker='*', s=25, label='Poly FFNN')
    twin_force_axis.plot([0,50], [0,50], c="magenta", linestyle='-', linewidth=1.8, label="Target Force")
    twin_force_axis.set_ylabel("Estimated Force")
    twin_force_axis.set_ylim(-5,50)
    twin_force_axis.legend(loc="lower right")

    axes_list[axes_dex][row_dex][col_dex].legend(loc="upper left")
    axes_list[axes_dex][row_dex][col_dex].set_ylabel("Freq. Deviation (kHz)")
    axes_list[axes_dex][row_dex][col_dex].set_xlabel("Force (kN)")

    axes_list[axes_dex][row_dex][col_dex].set_title(filename)
    axes_list[axes_dex][row_dex][col_dex].set_ylim(-10, 100)
    #axes_list[axes_dex][row_dex][col_dex].set_xlim(1.65, 1.85)
    axes_list[axes_dex][row_dex][col_dex].set_xlim(-5, 50)

    # plot time data
    # Raw force from lcm (noisy)
    rawforce, = axes_ts_list[axes_dex][row_dex][col_dex].plot(
        data_files_dict[filename]["Time (s)"], 
        data_files_dict[filename]["Force (kN)"], 
        color='dimgrey', label='Force Data')
    # Sensor measurement regressions
    s1p, = axes_ts_list[axes_dex][row_dex][col_dex].plot(
        data_files_dict[filename]["Time (s)"], 
        data_files_dict[filename]["Lin Est Force (kN)"],
        color='#7E58A0', label="Lin. Reg.")
    s2p, = axes_ts_list[axes_dex][row_dex][col_dex].plot(
        data_files_dict[filename]["Time (s)"], 
        data_files_dict[filename]["Poly Fit Force (kN)"],
        color='#4883D6', label="Poly. Reg.")
    s3p, = axes_ts_list[axes_dex][row_dex][col_dex].plot(
        data_files_dict[filename]["Time (s)"], 
        data_files_dict[filename]["FFNN Fit Force (kN)"],
        color='#78BFE0', label="Lin. FFNN")
    s4p, = axes_ts_list[axes_dex][row_dex][col_dex].plot(
        data_files_dict[filename]["Time (s)"], 
        data_files_dict[filename]["Poly FFNN Fit Force (kN)"],
        color='#3EA06D', label="Poly. FFNN")
    # Filtered force from LCM strain gauges, regression target
    filtforce, = axes_ts_list[axes_dex][row_dex][col_dex].plot(
        data_files_dict[filename]["Time (s)"], 
        data_list_dict[filename][f"{force_key} Force (kN)"], 
        color='magenta', label="Filt. <10Hz")#label=force_key)
    #axes_ts_list[axes_dex][row_dex][col_dex].plot(
    #    data_files_dict[filename]["Time (s)"], 
    #    data_files_dict[filename]["Poly FFNN Fit Force (kN)"],
    #    color='magenta', label="Poly FFNN")
    axes_ts_list[axes_dex][row_dex][col_dex].set_ylim(-5, 85)

    # Only label outside axes
    if row_dex == 1:
      axes_ts_list[axes_dex][row_dex][col_dex].set_xlabel("Time (s)")
    if col_dex == 0:
      axes_ts_list[axes_dex][row_dex][col_dex].set_ylabel("Force (kN)")

    # Plot coal region as patch and text begin 1.2 s, solid 1.6 to 3.6, end 4
    
    # Solid Fill w/ gradient
    steps = 19
    grad_len = 0.4
    coal_color = "lightgrey"
    for width in range(steps):
      axes_ts_list[axes_dex][row_dex][col_dex].fill_between(
          (1.2 + width*grad_len/steps, 1.2 + (width+1)*grad_len/steps), 
          -100, 100, color=coal_color, alpha=width/steps, linewidth=0)
    axes_ts_list[axes_dex][row_dex][col_dex].fill_between((1.6, 3.6), -100, 100, color=coal_color)
    for width in range(steps):
      axes_ts_list[axes_dex][row_dex][col_dex].fill_between(
          (3.6 + width*grad_len/steps, 3.6 + (width+1)*grad_len/steps), 
          -100, 100, color=coal_color, alpha=1.0 - width/steps, linewidth=0)

    axes_ts_list[axes_dex][row_dex][col_dex].text(0.0, 5, "Concrete", color="black")
    axes_ts_list[axes_dex][row_dex][col_dex].text(2.4, 25, "Coal", color="black")
    axes_ts_list[axes_dex][row_dex][col_dex].text(4.0, 5, "Concrete", color="black")

    #axes_ts_list[axes_dex][row_dex][col_dex].text(1.00,70, filename)
    axes_ts_list[axes_dex][row_dex][col_dex].set_title(filename)
    leg1 = axes_ts_list[axes_dex][row_dex][col_dex].legend(handles=[s1p, s2p, s3p, s4p], loc="upper center", title="Cap. Sensor")
    axes_ts_list[axes_dex][row_dex][col_dex].add_artist(leg1)
    axes_ts_list[axes_dex][row_dex][col_dex].legend(handles=[rawforce, filtforce], loc="upper right", title="Strain Gauges")
    axes_ts_list[axes_dex][row_dex][col_dex].grid(True, which="minor", axis="both", color="black", linewidth=0.25)
    axes_ts_list[axes_dex][row_dex][col_dex].minorticks_on()
    axes_ts_list[axes_dex][row_dex][col_dex].tick_params(which="minor", bottom=False, left=False)
    axes_ts_list[axes_dex][row_dex][col_dex].grid(True, which="major", axis="both", linewidth=1, color='black')

    

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
