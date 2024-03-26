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

import argparse

import pdb

##
# Commnad line defaults
##

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

sample_freq = 1.0 / 0.002475

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


# Dictionary of dictionary of lists
data_list_dict = {}

# Preprocess data to have bias adjusted based on first 0.5s from sample
y_vals = []

# Filter data in each file, Filter regression target for <10 Hz changes 
y_filt_b, y_filt_a = signal.butter(N=10, btype="low", Wn=10.0, fs=sample_freq)

# Filter inputs with Bandstop for machine vibrations
xin_bs_b, xin_bs_a = signal.butter(N=10, btype="bandstop", Wn=[20,50], fs=sample_freq)

# Filter inputs with LPF or MA,
xin_lpf_freqs = [100, 50, 20, 10, 5, 2, 1]
ma_lens = [40, 20, 10, 5]

#input_filter_array = [('bandstop', 'control'), ('lowpass', 'moving avg', 'control')]
input_filter_array = [('control',), ('lowpass', 'control')]

filtered_file_data_list = []

for filename in good_files_list:
    
    # Compute force filtered to 10 Hz
    data_list_dict[filename] = {
        "Filt Force <10 Hz (kN)": signal.filtfilt(
            y_filt_b, y_filt_a, data_files_dict[filename][f"Force (kN)"])}


    # Setup dictionary of data lists with channel name keys, zip later
    x_data_dict_list = {chan_name: [] for chan_name in chan_names_list}

    # Compute bias compensated values
    for chan_name in chan_names_list:
      # Auto-Bias adjustment (subtract largest of first 0.5s from sample)
      bias_adjustment_segment = data_files_dict[filename].loc[
          data_files_dict[filename]["Time (s)"] < 0.5]
      freq_bias_adjustment_val = np.max(bias_adjustment_segment[chan_name])
      data_files_dict[filename][f"Adj {chan_name}"] = \
          1000.0 * (freq_bias_adjustment_val - data_files_dict[filename][chan_name])

      # Iterate and apply filters
      for bs_or_none in input_filter_array[0]:
        for lpf_or_ma_or_none in input_filter_array[1]:

          vals = np.array(data_files_dict[filename][f"Adj {chan_name}"])
          if bs_or_none == "bandstop":
            # bandstop val
            vals = signal.filtfilt(xin_bs_b, xin_bs_a, vals)

          if lpf_or_ma_or_none == 'lowpass':
            # lpf val and save for all lpf vals
            for lpf in xin_lpf_freqs:
              xin_lpf_b, xin_lpf_a = signal.butter(
                N=10, btype="low", Wn=lpf, fs=sample_freq)
              vals = signal.filtfilt(xin_lpf_b, xin_lpf_a, vals)

              x_data_dict_list[chan_name].append(
                  (f"{bs_or_none} and {lpf_or_ma_or_none} {lpf}", vals))
            
          elif lpf_or_ma_or_none == 'moving avg':
            # moving avg and save for all ma vals
            for ma in ma_lens:
              vals = np.convolve(vals, np.ones(ma), 'same') / ma

              x_data_dict_list[chan_name].append(
                  (f"{bs_or_none} and {lpf_or_ma_or_none} {ma}", vals))

          else:
            # save data anyway
            x_data_dict_list[chan_name].append(
                (f"{bs_or_none} and {lpf_or_ma_or_none}", vals))

    # Glob data from all chanells and all files into x and y
    # Y data, regression target
    y_vals.extend(data_list_dict[filename][f"Filt Force <10 Hz (kN)"])

    # X data, different filtered values for testing
    # prepare a dictionary for the different types of filters
    filt_dict = {filttype: [] for filttype, _ in x_data_dict_list[chan_names_list[0]]}
    for chan_name in chan_names_list:
        for filttype, vals in x_data_dict_list[chan_name]:
            # populate dictionary with list for each channel
            filt_dict[filttype].append(vals)

    filtered_file_data_list.append(filt_dict)

x_dict = {key: [] for key in filtered_file_data_list[0].keys()}

# glob different filter outputs for all files
for filter_data_dict in filtered_file_data_list:
  for key in x_dict.keys():
    # transpose to abcd abcd from aaa.. bbb.. ...
    chan_time_list = [list(vals) for vals in zip(*filter_data_dict[key])]
    x_dict[key].extend(chan_time_list)

y_vals = np.array(y_vals).ravel()

poly = PolynomialFeatures(degree=2, include_bias=False)

num_samples = y_vals.shape[0]
filter_options = list(x_dict.keys())
linear_input_dim = len(x_dict[filter_options[0]][0])
poly_input_dim = poly.fit_transform([x_dict[filter_options[0]][0]]).shape[1]

print(f"Total number of samples: {num_samples}")
# Should be num channels * window size
print(f"Linear input dim: {linear_input_dim}")
# Should be num channels combos * window size
print(f"Poly input dim: {poly_input_dim}")

print(f"Filter options tested: {filter_options}")

that_time = time.time()
print("Data preprocessed in {0} sec; performing experiments".format(that_time - this_time),
      end='', flush=True)
this_time = time.time()

# r2 and rmse will give same rankings
scorings = ['r2','neg_root_mean_squared_error', 'neg_mean_absolute_error']

# Scale data for numerical performance, Standard is best
data_scalings = [
       	#("ScaleControl", None),
        #("RangeScaler", StandardScaler(with_mean=False)),
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
  for width in [1]:
    for depth in [3]:
      classifiers["Linear"].append((f"Linear NN (ReLU) ({linear_input_dim*width},) * {depth}",
          MLPRegressor((linear_input_dim*width, ) * depth, 
               activation='relu', solver='adam',
               max_iter=800, verbose=False)))
      classifiers["Poly"].append((f"Poly NN (ReLU) ({poly_input_dim*width},) * {depth}",
          MLPRegressor((poly_input_dim*width, ) * depth, 
               activation='relu', solver='adam',
               max_iter=800, verbose=False)))

results_list = []
for name, x_vals in x_dict.items():
  for polly_or_not in [("Poly", poly), ("Linear", None)]:
    for classy in classifiers[polly_or_not[0]]:

      cross_val = ShuffleSplit(n_splits=number_cross_validations, test_size=my_test_size, 
                               random_state = 711711)

      scale = data_scalings[0] # standard is best for scaling

      my_pipeline = Pipeline([scale, polly_or_not, classy])

      print(name)
      print(my_pipeline)

      scores = cross_validate(
        my_pipeline, x_vals, y_vals, 
        cv=cross_val, scoring=scorings, n_jobs=number_parallel_jobs)

      print(f"r2 scores: {scores['test_r2']}")
      print(f"RMSE: {-scores['test_neg_root_mean_squared_error']}\n")
      print(f"MAE: {-scores['test_neg_mean_absolute_error']}\n")

      results_list.append([
        ("num_total_samples", num_samples),
        ("input_dim", len(x_vals[0])),
        ("num_cross_val", number_cross_validations),
        ("test_fraction", my_test_size),
        ("filter options", name),
        ("lp", polly_or_not[0]),
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

