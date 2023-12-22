# This file fits the linear model to the data
# It shows plots of the model fit and error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

import pdb

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Load all good csv files as indicated by index.csv
files_index = pd.read_csv("../data/coal_csvs/index.csv")

files_index.loc[files_index["Data is good"] == True]

good_files_list = files_index.loc[files_index["Data is good"] == True]["Filename"].to_list()

data_files_dict = {}
for filename in good_files_list:
    fullpath = "../data/coal_csvs/" + filename
    data_files_dict[filename] = pd.read_csv(fullpath)

chan_names_list = ["Freq. A (MHz)", "Freq. B (MHz)", "Freq. C (MHz)", "Freq. D (MHz)"]

# Preprocess data to have bias adjusted based on first 0.5s from sample
x_vals = []
y_vals = []
y2_vals = []
for filename in good_files_list:
    
    # Compute avg force values
    avg_len = 40
    data_files_dict[filename]["Avg. Force (kN)"] = \
        np.convolve(data_files_dict[filename]["Force (kN)"], 
          np.ones(avg_len), 'same') / avg_len
    
    # Compute bias compensted avg values
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
      # TODO high pass filter

    # Glob data into X and Y lists
    chan_vals = [data_files_dict[filename][f"Avg. Adj {chan_name}"].to_list() 
                  for chan_name in chan_names_list]
    chan_time_list = [list(vals) for vals in zip(*chan_vals)]
    x_vals.extend(chan_time_list) # each entry is all 4 channels at same time
    y_vals.extend(data_files_dict[filename]["Force (kN)"].to_list())
    y2_vals.extend(data_files_dict[filename]["Avg. Force (kN)"].to_list())

##
# Linear Model
##

# Fit data with least squares
reg = linear_model.LinearRegression()
reg.fit(x_vals, y2_vals)
print(f"Linear coef: {reg.coef_}, Intercept: {reg.intercept_}")

y_hats = reg.predict(x_vals)

print("lin MSE: ")
print(mean_squared_error(y2_vals, y_hats))
print("lin R2: ")
print(r2_score(y2_vals, y_hats))

# Add model data to data dict
for filename in good_files_list:
    chan_vals = [data_files_dict[filename][f"Avg. Adj {chan_name}"].to_list() 
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
poly_reg_model.fit(x_poly_vals, y2_vals)
y2_poly_hats = poly_reg_model.predict(x_poly_vals)

print(f"Poly coef: {poly_reg_model.coef_}, Intercept: {poly_reg_model.intercept_}")
print("poly MSE: ")
print(mean_squared_error(y2_vals, y2_poly_hats))
print("poly R2: ")
print(r2_score(y2_vals, y2_poly_hats))

# Add model data to data dict
for filename in good_files_list:
    chan_vals = [data_files_dict[filename][f"Avg. Adj {chan_name}"].to_list() 
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
nn_window_size = 3
nn_window_x_vals = []
nn_window_y_vals = []
for index in range(len(x_vals)-nn_window_size + 1):
    chanelled_vals = x_vals[index:index+nn_window_size]
    # Flatten X to be abcd abcd abcd...
    nn_window_x_vals.append([val for row in chanelled_vals for val in row])
    nn_window_y_vals.append(y2_vals[index+nn_window_size-1])

nn_window_y_vals = np.array(nn_window_y_vals) # add inner dim


ffnn = MLPRegressor((nn_window_size, nn_window_size, int(1 + nn_window_size/2)), activation='relu', solver='adam',
          max_iter=400, verbose=False, random_state=12345).fit(
    nn_window_x_vals, nn_window_y_vals)
print("Done!")

ffnn_y_hat = ffnn.predict(nn_window_x_vals)
print("ffnn MSE: ")
print(mean_squared_error(nn_window_y_vals, ffnn_y_hat))
print("ffnn R2: ")
print(r2_score(nn_window_y_vals, ffnn_y_hat))

## Add model data to data dict
for filename in good_files_list:
    # break file data into windows
    chan_vals = [data_files_dict[filename][f"Avg. Adj {chan_name}"].to_list() 
                  for chan_name in chan_names_list]
    chan_time_list = [list(vals) for vals in zip(*chan_vals)]
    windowed_vals = []
    for index in range(len(chan_time_list) - nn_window_size +1):
        chanelled_vals = chan_time_list[index:index+nn_window_size]
        windowed_vals.append([val for row in chanelled_vals for val in row]) # flatten
    windowed_vals = [windowed_vals[0]] * (nn_window_size-1) + windowed_vals

    data_files_dict[filename]["FFNN Fit Force (kN)"] = \
      ffnn.predict(windowed_vals)


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
            data_files_dict[filename]["Avg. Adj Freq. A (MHz)"], 
            label="Chan A", marker='x', color="blue", s=2)
    axes_list[axes_dex][row_dex][col_dex].scatter(
            data_files_dict[filename]["Avg. Force (kN)"], 
            data_files_dict[filename]["Avg. Adj Freq. B (MHz)"], 
            label="Chan B", marker='+', color="blue", s=2)
    axes_list[axes_dex][row_dex][col_dex].scatter(
            data_files_dict[filename]["Avg. Force (kN)"], 
            data_files_dict[filename]["Avg. Adj Freq. C (MHz)"], 
            label="Chan C", marker='o', color="blue", s=2)
    axes_list[axes_dex][row_dex][col_dex].scatter(
            data_files_dict[filename]["Avg. Force (kN)"], 
            data_files_dict[filename]["Avg. Adj Freq. C (MHz)"], 
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
        data_files_dict[filename]["Force (kN)"])
    axes_ts_list[axes_dex][row_dex][col_dex].plot(
        data_files_dict[filename]["Time (s)"], 
        data_files_dict[filename]["Avg. Force (kN)"])
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
    # TODO add poly ffnn with magenta
    axes_ts_list[axes_dex][row_dex][col_dex].set_ylim(-15, 85)
    axes_ts_list[axes_dex][row_dex][col_dex].set_xlabel("Time (s)")
    axes_ts_list[axes_dex][row_dex][col_dex].set_ylabel("Force (kN)")
    axes_ts_list[axes_dex][row_dex][col_dex].text(1.00,70, filename)
    axes_ts_list[axes_dex][row_dex][col_dex].legend(loc="center", bbox_to_anchor=(0.45, 0.5))
    

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

