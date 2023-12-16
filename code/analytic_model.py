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

# Load all good csv files as indicated by index.csv
files_index = pd.read_csv("../data/coal_csvs/index.csv")

files_index.loc[files_index["Data is good"] == True]

good_files_list = files_index.loc[files_index["Data is good"] == True]["Filename"].to_list()

data_files_dict = {}
for filename in good_files_list:
    fullpath = "../data/coal_csvs/" + filename
    data_files_dict[filename] = pd.read_csv(fullpath)

# Preprocess data to have bias adjusted based on first 0.5s from sample
x_vals = []
y_vals = []
y2_vals = []
for filename in good_files_list:
    # Auto-Bias adjustment (subtract average of first 0.5s from sample)
    bias_adjustment_segment = data_files_dict[filename].loc[
        data_files_dict[filename]["Time (s)"] < 0.5]
    freq_bias_adjustment_val = np.max(bias_adjustment_segment["Freq. (MHz)"])
    data_files_dict[filename]["Adj Freq. (MHz)"] = \
        1000.0 * (data_files_dict[filename]["Freq. (MHz)"] - freq_bias_adjustment_val)

    # Compute avg force values
    avg_len = 50
    data_files_dict[filename]["Avg. Force (kN)"] = \
        np.convolve(data_files_dict[filename]["Force (kN)"], 
          np.ones(avg_len), 'same') / avg_len

    # Compute avg sensor values
    data_files_dict[filename]["Avg. Adj Freq. (MHz)"] = \
        np.convolve(data_files_dict[filename]["Adj Freq. (MHz)"],
          np.ones(avg_len), 'same') / avg_len

    # Glob data into X and Y lists
    x_vals.extend(data_files_dict[filename]["Avg. Adj Freq. (MHz)"].to_list())
    y_vals.extend(data_files_dict[filename]["Force (kN)"].to_list())
    y2_vals.extend(data_files_dict[filename]["Avg. Force (kN)"].to_list())

##
# Linear Model
##

# Reshape X so that each sample is its own row
x_vals = np.array(x_vals).reshape(-1,1)

# Fit data with least squares
reg = linear_model.LinearRegression()
reg.fit(x_vals, y2_vals)
lin_fit_coeff = reg.coef_[0]
print(reg.coef_)

y_hats = reg.predict(x_vals)

print("lin MSE: ")
print(mean_squared_error(y2_vals, y_hats))
print("lin R2: ")
print(r2_score(y2_vals, y_hats))

# Add model data to data dict
for filename in good_files_list:
    data_files_dict[filename]["Lin Est Force (kN)"] = \
      lin_fit_coeff * data_files_dict[filename]["Avg. Adj Freq. (MHz)"] 

##
# Poly fit
##

poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly_vals = poly.fit_transform(x_vals)
poly_reg_model = linear_model.LinearRegression()
poly_reg_model.fit(x_poly_vals, y2_vals)
y2_poly_hats = poly_reg_model.predict(x_poly_vals)

print(poly_reg_model.coef_)
print("poly MSE: ")
print(mean_squared_error(y2_vals, y2_poly_hats))
print("poly R2: ")
print(r2_score(y2_vals, y2_poly_hats))

# Add model data to data dict
for filename in good_files_list:
    data_files_dict[filename]["Poly Fit Force (kN)"] = \
        poly_reg_model.predict(poly.fit_transform(
            np.array(data_files_dict[filename]["Avg. Adj Freq. (MHz)"]).reshape(-1,1)))

##
# FFNN fit
##
print("Fitting ffnn...")
# Window the xvals
nn_window_size = 10
nn_window_x_vals = []
nn_window_y_vals = []
for index in range(len(x_vals)-nn_window_size + 1):
    nn_window_x_vals.append(
      x_vals[index:index+nn_window_size])
    nn_window_y_vals.append(y2_vals[index+nn_window_size-1])
nn_window_x_vals = np.array(nn_window_x_vals)[...,0] # drop last dim
nn_window_y_vals = np.array(nn_window_y_vals) # add inner dim


ffnn = MLPRegressor((nn_window_size, nn_window_size, int(1 + nn_window_size/2)), activation='relu', solver='adam',
          max_iter=400, verbose=False).fit(
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
    windowed_vals = []
    file_vals = data_files_dict[filename]["Avg. Adj Freq. (MHz)"].to_list()
    for index in range(len(file_vals) - nn_window_size +1):
        windowed_vals.append(file_vals[index:index+nn_window_size])
    windowed_vals = [windowed_vals[0]] * (nn_window_size-1) + windowed_vals
    
    data_files_dict[filename]["FFNN Fit Force (kN)"] = \
      ffnn.predict(windowed_vals)


# Plot bias adjusted traces
num_plot_rows = 2
num_plot_cols = 3
fig1, axes1 = plt.subplots(num_plot_rows, num_plot_cols)
fig2, axes2 = plt.subplots(num_plot_rows, num_plot_cols)
fig3, axes3 = plt.subplots(num_plot_rows, num_plot_cols)
fig4, axes4 = plt.subplots(num_plot_rows, num_plot_cols)
axes_list = [axes1, axes2, axes3, axes4]
  
# Plot time series data
fig_ts1, axes_ts1 = plt.subplots(num_plot_rows, num_plot_cols)
fig_ts2, axes_ts2 = plt.subplots(num_plot_rows, num_plot_cols)
fig_ts3, axes_ts3 = plt.subplots(num_plot_rows, num_plot_cols)
fig_ts4, axes_ts4 = plt.subplots(num_plot_rows, num_plot_cols)
axes_ts_list = [axes_ts1, axes_ts2, axes_ts3, axes_ts4]

# Plot row major
for idx, filename in enumerate(good_files_list):
    # Calculate indicies
    axes_dex = int(idx / (num_plot_rows * num_plot_cols))
    row_dex = int((idx) / num_plot_cols) % num_plot_rows
    col_dex = idx % num_plot_cols

    print(f"Axes {axes_dex}, row {row_dex}, col {col_dex}. {idx}. {filename}")
    # Plot XY data
    axes_list[axes_dex][row_dex][col_dex].scatter(data_files_dict[filename]["Adj Freq. (MHz)"], 
            data_files_dict[filename]["Force (kN)"], label="Raw Data")
    axes_list[axes_dex][row_dex][col_dex].scatter(data_files_dict[filename]["Adj Freq. (MHz)"], 
            data_files_dict[filename]["Avg. Force (kN)"], label="Avg Force")
    axes_list[axes_dex][row_dex][col_dex].scatter(data_files_dict[filename]["Avg. Adj Freq. (MHz)"], 
            data_files_dict[filename]["Avg. Force (kN)"], label="Avg Force and Avg Freq")
    axes_list[axes_dex][row_dex][col_dex].scatter(
        data_files_dict[filename]["Avg. Adj Freq. (MHz)"], 
        data_files_dict[filename]["Lin Est Force (kN)"], c='red', label='Linear')
    axes_list[axes_dex][row_dex][col_dex].scatter(
        data_files_dict[filename]["Avg. Adj Freq. (MHz)"], 
        data_files_dict[filename]["Poly Fit Force (kN)"], c='purple', label='Quadratic')
    axes_list[axes_dex][row_dex][col_dex].scatter(
        data_files_dict[filename]["Avg. Adj Freq. (MHz)"], 
        data_files_dict[filename]["FFNN Fit Force (kN)"], c='black',
        marker='x', s=3, label='Neural Net')
    axes_list[axes_dex][row_dex][col_dex].legend(loc="lower left")
    axes_list[axes_dex][row_dex][col_dex].set_xlabel("Freq. Deviation (kHz)")
    axes_list[axes_dex][row_dex][col_dex].set_ylabel("Force (kN)")

    left_freq = min(data_files_dict[filename]["Adj Freq. (MHz)"])
    axes_list[axes_dex][row_dex][col_dex].text(-115, 70, filename)
    axes_list[axes_dex][row_dex][col_dex].set_xlim(-120, 10)
    #axes_list[axes_dex][row_dex][col_dex].set_xlim(1.65, 1.85)
    axes_list[axes_dex][row_dex][col_dex].set_ylim(-15, 85)

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
        color='red')
    axes_ts_list[axes_dex][row_dex][col_dex].plot(
        data_files_dict[filename]["Time (s)"], 
        data_files_dict[filename]["Poly Fit Force (kN)"],
        color='purple')
    axes_ts_list[axes_dex][row_dex][col_dex].plot(
        data_files_dict[filename]["Time (s)"], 
        data_files_dict[filename]["FFNN Fit Force (kN)"],
        color='black')
    axes_ts_list[axes_dex][row_dex][col_dex].set_ylim(-15, 85)
    axes_ts_list[axes_dex][row_dex][col_dex].set_xlabel("Time (s)")
    axes_ts_list[axes_dex][row_dex][col_dex].set_ylabel("Force (kN)")
    

    # Twinning frequency axis
    ax2 = axes_ts_list[axes_dex][row_dex][col_dex].twinx()
    ax2.plot(
        data_files_dict[filename]["Time (s)"], 
        data_files_dict[filename]["Adj Freq. (MHz)"],
        color=(0.6, 0.2, 0.2, 0.5))
    ax2.plot(
        data_files_dict[filename]["Time (s)"], 
        data_files_dict[filename]["Avg. Adj Freq. (MHz)"],
        color=(0.6, 0.6, 0.2, 0.5))
    ax2.text(0,-.02, filename)
    ax2.set_ylim(-80, 10)
    ax2.set_ylabel("Freq. Deviation (kHz)")
#
plt.show(block=False)




# Split data into test/train

pdb.set_trace()
