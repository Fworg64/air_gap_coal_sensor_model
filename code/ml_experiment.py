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

from scipy.signal import decimate

import argparse

import pdb

##
# Commnad line defaults
##

# Use decimated values instead of moving average
argument_use_decimated_vals = 0

# Use up to N consectutive samples for classification
argument_window_len = 1

# Don't Include NN regressions
argument_no_nn = False

# What percent of data to use for testing during test/train split
argument_test_frac = 0.5

# How many cross-validations to perform for Monte-Carlo performance distribution modelling
argument_num_cross_vals = 3

# How many cores to allow for computation
argument_parallel_jobs = 3

## Allow command line overrides
parser = argparse.ArgumentParser()

parser.add_argument("--decimated", default=argument_use_decimated_vals, type=int,
  help="Set to 1 or greater to use decimation")

# Making command line argument for window duration
parser.add_argument("--window_len", default=argument_window_len, type=int,
  help="Number of how many consecutive samples to give regressor")

parser.add_argument("--no_nn", default=argument_no_nn, type=bool,
  help="Set to True to omit NN from processing")

parser.add_argument("--test_frac", default=argument_test_frac, type=float,
  help="[0.0, 1.0) fraction to use for test during splits")

parser.add_argument("--num_cross_vals", default=argument_num_cross_vals, type=int,
  help="How many splits to run")

parser.add_argument("--num_parallel_jobs", default=argument_parallel_jobs, type=int,
  help="How many cores to use")

args = parser.parse_args()

my_test_size = args.test_frac
number_cross_validations = args.num_cross_vals

number_parallel_jobs = args.num_parallel_jobs



# Load all good csv files as indicated by index.csv
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
avg_len = 100

# Dictionary of dictionary of lists
data_list_dict = {}
# Preprocess data to have bias adjusted based on first 0.5s from sample
x_vals = []
y_vals = []

decimation_factor = 2
if args.decimated > 0:
  decimation_factor = args.decimated
  print(f"Decimating by factor of {decimation_factor}")

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
          q=decimation_factor, n=4, ftype='fir', axis=0, zero_phase=True)
    
    # Compute bias compensated avg values
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

      # Add decimated values
      data_list_dict[filename][f"Decimated {chan_name}"] = \
          decimate(data_files_dict[filename][f"Adj {chan_name}"],
                   q=decimation_factor, n=4, ftype='fir', axis=0, zero_phase=True)

    chan_key = "Avg. Adj"
    force_key = "Avg."
    if (args.decimated > 0):
        chan_key = "Decimated"
        force_key = "Decimated"

    # Glob data into X and Y lists, use decimated values if arg
    chan_vals = [data_list_dict[filename][f"{chan_key} {chan_name}"] 
                  for chan_name in chan_names_list] # list of each channel
    chan_time_list = [list(vals) for vals in zip(*chan_vals)] # transpose to each channel as column
    x_vals.extend(chan_time_list) # each entry is all 4 channels at same time
    y_vals.extend(data_list_dict[filename][f"{force_key} Force (kN)"])


# Add first order difference to sample
add_diffs = False

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

# Scale data for numerical performance
data_scalings = [
       	("ScaleControl", None),
        ("RangeScaler", StandardScaler(with_mean=False)),
        ("StandardScaler", StandardScaler())]

# Classifiers dictionary list 
classifiers = {
    "Linear": [
        ("Linear MSE Fit", linear_model.LinearRegression()),
        ("Linear MAE Fit", linear_model.SGDRegressor(loss='epsilon_insensitive', penalty=None, epsilon=0.0))],
    "Poly": [
        ("Poly MSE Fit", linear_model.LinearRegression()),
        ("Poly MAE Fit", linear_model.SGDRegressor(loss='epsilon_insensitive', penalty=None, epsilon=0.0))]}

# Extend with NNs
if (not args.no_nn):
  for width in [1,2,3,5,10]:
    for depth in [1,2,3,5,10]:
      classifiers["Linear"].append((f"Linear NN (ReLU) ({linear_input_dim*width},) * {depth}",
          MLPRegressor((linear_input_dim*width, ) * depth, 
               activation='relu', solver='adam',
               max_iter=800, verbose=False)))
      classifiers["Poly"].append((f"Poly NN (ReLU) ({poly_input_dim*width},) * {depth}",
          MLPRegressor((poly_input_dim*width, ) * depth, 
               activation='relu', solver='adam',
               max_iter=800, verbose=False)))

results_list = []
for xdata in [("Linear", nn_window_x_vals), ("Poly", nn_window_poly_x_vals)]:
  for scale in data_scalings:
    for classy in classifiers[xdata[0]]:

      cross_val = ShuffleSplit(n_splits=number_cross_validations, test_size=my_test_size, 
                               random_state = 711711)

      my_pipeline = Pipeline([scale, classy])

      print(my_pipeline)

      scores = cross_validate(
        my_pipeline, xdata[1], nn_window_y_vals, 
        cv=cross_val, scoring=scorings, n_jobs=number_parallel_jobs)

      print(f"r2 scores: {scores['test_r2']}")
      print(f"RMSE: {-scores['test_neg_root_mean_squared_error']}\n")
      print(f"MAE: {-scores['test_neg_mean_absolute_error']}\n")

      results_list.append([
        ("num_total_samples", num_samples),
        ("input_dim", len(xdata[1][0])),
        ("num_cross_val", number_cross_validations),
        ("test_fraction", my_test_size),
        ("running avg sample length", avg_len),
        ("decimated", args.decimated),
        ("input data window length", window_len),
        ("diffs added", add_diffs),
        ("lp", xdata[0]),
        ("ds", my_pipeline.steps[0]),
        ("cl", my_pipeline.steps[1]),
        ("avg r2", scores["test_r2"].mean()),
        ("r2 std", scores["test_r2"].std()),
        ("avg RMSE", -scores["test_neg_root_mean_squared_error"].mean()),
        ("RMSE std", scores["test_neg_root_mean_squared_error"].std()),
        ("avg MAE", -scores["test_neg_mean_absolute_error"].mean()),
        ("MAE std", scores["test_neg_mean_absolute_error"].std()),
        ])
      results_list[-1].extend([(f"r2 {idx}", score) for idx, score in enumerate(scores["test_r2"])])
      results_list[-1].extend([(f"rmse {idx}", -score) for idx, score in enumerate(scores['test_neg_root_mean_squared_error'])])
      results_list[-1].extend([(f"mae {idx}", -score) for idx, score in enumerate(scores['test_neg_mean_absolute_error'])])

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

