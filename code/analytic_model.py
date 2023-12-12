# This file fits the linear model to the data
# It shows plots of the model fit and error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
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
for filename in good_files_list:
    # Auto-Bias adjustment (subtract average of first 0.5s from sample)
    bias_adjustment_segment = data_files_dict[filename].loc[
        data_files_dict[filename]["Time (s)"] < 0.5]
    freq_bias_adjustment_val = np.max(bias_adjustment_segment["Freq. (MHz)"])
    data_files_dict[filename]["Adj Freq. (MHz)"] = \
        data_files_dict[filename]["Freq. (MHz)"] - freq_bias_adjustment_val

    # Glob data into X and Y lists
    x_vals.extend(data_files_dict[filename]["Adj Freq. (MHz)"].to_list())
    y_vals.extend(data_files_dict[filename]["Force (kN)"].to_list())

# Reshape X so that each sample is its own row
x_vals = np.array(x_vals).reshape(-1,1)

# Plot bias adjusted traces
num_plot_rows = 3
num_plot_cols = 3
fig1, axes1 = plt.subplots(num_plot_rows, num_plot_cols)
fig2, axes2 = plt.subplots(num_plot_rows, num_plot_cols)
fig3, axes3 = plt.subplots(num_plot_rows, num_plot_cols)
fig4, axes4 = plt.subplots(num_plot_rows, num_plot_cols)
#


axes_list = [axes1, axes2, axes3, axes4]
  
# Plot row major
for idx, filename in enumerate(good_files_list):
    axes_dex = int(idx / (num_plot_rows * num_plot_cols))
    row_dex = int((idx) / num_plot_cols) % num_plot_rows
    col_dex = idx % num_plot_cols
    print(f"Axes {axes_dex}, row {row_dex}, col {col_dex}. {idx}. {filename}")
    axes_list[axes_dex][row_dex][col_dex].scatter(data_files_dict[filename]["Adj Freq. (MHz)"], 
            data_files_dict[filename]["Force (kN)"])
    left_freq = min(data_files_dict[filename]["Adj Freq. (MHz)"])
    axes_list[axes_dex][row_dex][col_dex].text(-0.1, 70, filename)
    axes_list[axes_dex][row_dex][col_dex].set_xlim(-0.12, 0.03)
    #axes_list[axes_dex][row_dex][col_dex].set_xlim(1.65, 1.85)
    axes_list[axes_dex][row_dex][col_dex].set_ylim(-15, 85)
#
plt.show(block=False)
input("Press Enter to close...")

# Split data into test/train

# Fit data with least squares
reg = linear_model.LinearRegression()
reg.fit(x_vals, y_vals)
print(reg.coef_)

y_hats = reg.predict(x_vals)

print("MSE: ")
print(mean_squared_error(y_vals, y_hats))
print("R2: ")
print(r2_score(y_vals, y_hats))
pdb.set_trace()
