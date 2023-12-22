# This file aggregates and plots the results from the experiment
# One figure for r2, one for mse
# Label with linear/poly scaling_type linear/ffnn

# Load all files in directory

from os import walk
import pandas as pd
import matplotlib.pyplot as plt

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

mypath = "../data/server_results/"
# List comprehension with nested fors to unpack walk() yielding its outputs
filelist = [filename for (dirpath, dirnames, filenames) in walk(mypath) if dirpath == mypath
              for filename in filenames]

print(filelist)

# Load all files
files_dict = {filename: pd.read_csv(mypath + filename) 
               for filename in filelist}

# Drop add_diff=True values for now
include_diffs = False

super_data_frame = pd.concat(list(files_dict.values()))

label_model_dict = {
    "Linear Regression": ["('Linear', None)", ["('Linear or Poly Fit', LinearRegression())"]],
    "Poly Regression": ["('Poly', PolynomialFeatures(include_bias=False))", 
                        ["('Linear or Poly Fit', LinearRegression())"]],
    "Linear FFNN": ["('Linear', None)", [
       "('FFNN (relu) (N,N,)', MLPRegressor(hidden_layer_sizes=(4, 4), max_iter=800))",
       "('FFNN (relu) (N,N,)', MLPRegressor(hidden_layer_sizes=(1, 1), max_iter=800))",
       "('FFNN (relu) (N,N,)', MLPRegressor(hidden_layer_sizes=(3, 3), max_iter=800))",
       "('FFNN (relu) (N,N,)', MLPRegressor(hidden_layer_sizes=(2, 2), max_iter=800))",
       "('FFNN (relu) (N,N,)', MLPRegressor(hidden_layer_sizes=(5, 5), max_iter=800))"]],
    "Poly FFNN": ["('Poly', PolynomialFeatures(include_bias=False))", [
       "('FFNN (relu) (N,N,)', MLPRegressor(hidden_layer_sizes=(4, 4), max_iter=800))",
       "('FFNN (relu) (N,N,)', MLPRegressor(hidden_layer_sizes=(1, 1), max_iter=800))",
       "('FFNN (relu) (N,N,)', MLPRegressor(hidden_layer_sizes=(3, 3), max_iter=800))",
       "('FFNN (relu) (N,N,)', MLPRegressor(hidden_layer_sizes=(2, 2), max_iter=800))",
       "('FFNN (relu) (N,N,)', MLPRegressor(hidden_layer_sizes=(5, 5), max_iter=800))"]]
}

label_n_list_dict = {key: [] for key in label_model_dict.keys()}

n_vals = list(pd.unique(super_data_frame["input data window length"]))
print(f"found n vals: {n_vals}")

for key, value in label_model_dict.items():
  for n in n_vals:
    label_n_list_dict[key].append(
        super_data_frame[
          (super_data_frame["lp"] == value[0]) &
          (super_data_frame["cl"].isin(value[1])) &
          (super_data_frame["input data window length"] == n) &
          (super_data_frame["diffs added"] == include_diffs)]
    )

# Split data by 4 classifier types and 3 normalization techniques
# Linear, Poly, Linear FFNN, Poly FFNN x None, Standard, dev only

# Split r2 and MSE
# Organize by window len N=1,2,3,4,5

# Get all r2 rows with N=1 and add_diff=False
# repeat for N=2...5

# plot for all methods with labels and color and marker

fig1, axs1 = plt.subplots()
fig2, axs2 = plt.subplots()

scalings = [
    "('StandardScaler', StandardScaler())",
    "('RangeScaler', StandardScaler(with_mean=False))",
    "('ScaleControl', None)"]

scalings_marker_dict = {
    scalings[0]: 'x',
    scalings[1]: '+',
    scalings[2]: '.'}

scalings_label_dict = {
    scalings[0]: '$\mu = 1, \sigma = 1$',
    scalings[1]: '$\sigma = 1$',
    scalings[2]: 'KHz'}

label_color_dict = {
    "Linear Regression": "orange",
    "Poly Regression": "cyan",
    "Linear FFNN": "forestgreen",
    "Poly FFNN": "magenta"}

label_offset_dict = {
    "Linear Regression": -0.3,
    "Poly Regression": -0.15,
    "Linear FFNN": 0.0,
    "Poly FFNN": 0.15}

for name, data_list in label_n_list_dict.items():
    x_vals = n_vals
    for scaledex, scaling in enumerate(scalings):
      r2_dists = []
      r2_vals = []
      r2_stds = []
      mse_dists = []
      mse_vals = []
      mse_stds = []
      for frame in data_list:
        r2_dist = []
        for index in range(int(frame[frame["ds"] == scaling]["num_cross_val"])):
          r2_dist.append(frame[frame["ds"] == scaling][f"r2 {index}"])
        r2_dists.append(r2_dist)

        mse_dist = []
        for index in range(int(frame[frame["ds"] == scaling]["num_cross_val"])):
          mse_dist.append(frame[frame["ds"] == scaling][f"mse {index}"])
        mse_dists.append(mse_dist)

        r2_vals.append(
            float(frame[frame["ds"] == scaling]["avg r2"]))
        r2_stds.append(
            float(frame[frame["ds"] == scaling]["r2 std"]))
        mse_vals.append(
            float(frame[frame["ds"] == scaling]["avg MSE"]))
        mse_stds.append(
            float(frame[frame["ds"] == scaling]["MSE std"]))

      for xx, r2_dist, mse_dist in zip(x_vals, r2_dists, mse_dists):
        xx_vals = [xx + scaledex * 0.04 + label_offset_dict[name]] * len(r2_dist)
        muh_label = "_no_legend_"
        if xx == x_vals[0]:
            muh_label = name + " (" + scalings_label_dict[scaling] + ")"
        axs1.scatter(xx_vals, r2_dist, 
              c=label_color_dict[name],
              marker=scalings_marker_dict[scaling],
              label=muh_label)
        axs2.scatter(xx_vals, mse_dist, 
              c=label_color_dict[name],
              marker=scalings_marker_dict[scaling],
              label=muh_label)
        # Get median values and plot in black
        r2_med = pd.DataFrame(r2_dist).median()
        mse_med = pd.DataFrame(mse_dist).median()
        axs1.scatter(xx + scaledex * 0.04 + label_offset_dict[name], r2_med, c='black',
              marker=scalings_marker_dict[scaling],
              label='_')
        axs2.scatter(xx + scaledex * 0.04 + label_offset_dict[name], mse_med, c='black',
              marker=scalings_marker_dict[scaling],
              label='_')
      #r2_lower = [a-b for a,b in zip(r2_vals, r2_stds)]
      #r2_upper = [a+b for a,b in zip(r2_vals, r2_stds)]
      #axs1.fill_between(x_vals, r2_lower, r2_upper,
      #      color = label_color_dict[name], alpha = 0.2)
      #axs1.plot(x_vals, r2_vals, label=name + " " + scaling, 
      #         ls='--', color=label_color_dict[name],
      #         marker=scalings_marker_dict[scaling])
      
      #mse_lower = [a-b for a,b in zip(mse_vals, mse_stds)]
      #mse_upper = [a+b for a,b in zip(mse_vals, mse_stds)]
      #axs2.fill_between(x_vals, mse_lower, mse_upper,
      #      color = label_color_dict[name], alpha = 0.2)
      #axs2.plot(x_vals, mse_vals, label=name + " " + scaling, 
      #         ls='--', color=label_color_dict[name],
      #         marker=scalings_marker_dict[scaling])

    axs1.set_ylim(0,1)
    axs1.legend(loc='lower left', ncol=2)
    fig1.suptitle("$R^2$ Values with Median vs Window Size", y=0.96)
    axs1.set_title("100 tests each with random 50:50 test:train split")
    axs1.set_xlabel("Window Size (Whole Numbers Only)")
    axs1.set_ylabel("$R^2$ Value, Higher is Better")

    fig2.suptitle("Base 10 log. plot of MSE Values with Median vs Window Size", y=0.96)
    axs2.set_title("100 tests each with random 50:50 test:train split")
    axs2.legend(loc='upper left', ncol=2)
    axs2.set_yscale("log")
    axs2.set_xlabel("Window Size (Whole Numbers Only)")
    axs2.set_ylabel("MSE with $Log_{10}$ scale, Lower is Better")

plt.show(block=False)
input("Press Enter to close...")


