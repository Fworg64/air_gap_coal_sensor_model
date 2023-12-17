# This file uses sklearn library to setup ML model comparison experiment with cross validation
# It will write out files to an 'out' directory that it will try to create
import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate

import pdb

# Load all good csv files as indicated by index.csv
files_index = pd.read_csv("../data/coal_csvs/index.csv")

files_index.loc[files_index["Data is good"] == True]

good_files_list = files_index.loc[files_index["Data is good"] == True]["Filename"].to_list()

data_files_dict = {}
for filename in good_files_list:
    fullpath = "../data/coal_csvs/" + filename
    data_files_dict[filename] = pd.read_csv(fullpath)

print("Data Loaded.")
this_time = time.time()


chan_names_list = ["Freq. A (MHz)", "Freq. B (MHz)", "Freq. C (MHz)", "Freq. D (MHz)"]

# Moving average size in number of samples    
avg_len = 50

# Preprocess data to have bias adjusted based on first 0.5s from sample
x_vals = []
y_vals = []
for filename in good_files_list:
    
    # Compute avg force values
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

      # TODO high pass filter (check if adding diff terms works first)

    # Glob data into X and Y lists
    chan_vals = [data_files_dict[filename][f"Avg. Adj {chan_name}"].to_list() 
                  for chan_name in chan_names_list]
    chan_time_list = [list(vals) for vals in zip(*chan_vals)]
    x_vals.extend(chan_time_list) # each entry is all 4 channels at same time
    y_vals.extend(data_files_dict[filename]["Avg. Force (kN)"].to_list())

number_parallel_jobs = 3

number_cross_validations = 3
my_test_size = 0.5

scorings = ['r2','neg_mean_squared_error']

# Add first order difference to sample
add_diffs = False

##
# Window
##
window_len = 4
# Window x values to windows of len N
# Flatten samples to [abcd abcd ...]
# Window the xvals
nn_window_size = window_len
nn_window_x_vals = []
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
    # Flatten X to be abcd abcd abcd...
    nn_window_x_vals.append([val for row in chanelled_vals for val in row])
    nn_window_y_vals.append(y_vals[index+nn_window_size-1])

nn_window_y_vals = np.array(nn_window_y_vals) # add inner dim

that_time = time.time()
print("Data preprocessed in {0} sec; performing experiments".format(that_time - this_time),
      end='', flush=True)
this_time = time.time()

# Preprocess steps poly or not
poly_preprocess = [
        ("Linear", None),
        ("Poly", PolynomialFeatures(degree=2, include_bias=False))]

# Scale data for numerical performance
data_scalings = [
        ("StandardScaler", StandardScaler()),
        ("RangeScaler", StandardScaler(with_mean=False)),
	("ScaleControl", None)]

# Classifiers list
classifiers = [
        ("Linear or Poly Fit", linear_model.LinearRegression()),
        ("FFNN (relu) (N,N,)", MLPRegressor(
            (nn_window_size, nn_window_size),
             activation='relu', solver='adam',
             max_iter=800, verbose=False))]

results_list = []
for ppp in poly_preprocess:
  for scale in data_scalings:
    for classy in classifiers:

      cross_val = ShuffleSplit(n_splits=number_cross_validations, test_size=my_test_size, 
                               random_state = 711711)

      my_pipeline = Pipeline([ppp, scale, classy])

      print(my_pipeline)

      scores = cross_validate(
        my_pipeline, nn_window_x_vals, nn_window_y_vals, 
        cv=cross_val, scoring=scorings, n_jobs=number_parallel_jobs)

      print(f"r2 scores: {scores['test_r2']}")
      print(f"MSE: {-scores['test_neg_mean_squared_error']}\n")

      results_list.append([
        ("num_cross_val", number_cross_validations),
        ("test_fraction", my_test_size),
        ("running avg sample length", avg_len),
        ("input data window length", window_len),
        ("diffs added", add_diffs),
        ("lp", my_pipeline.steps[0]),
        ("ds", my_pipeline.steps[1]),
        ("cl", my_pipeline.steps[2]),
        ("avg r2", scores["test_r2"].mean()),
        ("r2 std", scores["test_r2"].std()),
        ("avg MSE", -scores["test_neg_mean_squared_error"].mean()),
        ("MSE std", scores["test_neg_mean_squared_error"].std()),
        ])

print("Done.")

that_time = time.time()
print("Experiments completed in {0} sec; writing results file...".format(that_time - this_time),
      end='', flush=True)
this_time = time.time()

# Get list of column names from first entry in results list
result_columns = [item[0] for item in results_list[0]] 

# Get data rows from results list
result_rows = []
for row in results_list:
  name, values = zip(*row)
  result_rows.append(values)

results_frame = pd.DataFrame(data=result_rows, columns=result_columns)

print("results frame")
print(results_frame)

## Write file with timestring
os.makedirs('./out', exist_ok=True)
timestr = time.strftime("%Y%m%d_%H%M%Sresults.csv")
outfilename = './out/' + "COAL_2023_" + timestr
results_frame.to_csv(outfilename, index_label=False, index=False) 

print(f"Have a nice day!    file wrote to: {outfilename}")

