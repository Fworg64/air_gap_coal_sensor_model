# This file aggregates and plots the results from the experiment
# One figure for r2, one for mse
# Label with linear/poly scaling_type linear/ffnn

# Load all files in directory

from os import walk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pdb

SMALL_SIZE = 22
MEDIUM_SIZE = 26
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Load experiment file 
mypath = "./out/"
filename = "COAL_2024_20240326_150438results.csv"

data_frame = pd.read_csv(mypath + filename) 
               
model_names_list = ["('Linear MSE Fit', LinearRegression())",
                    "('Poly MSE Fit', LinearRegression())",
                    "('Linear NN (ReLU) (4,) * 3', MLPRegressor(hidden_layer_sizes=(4, 4, 4), max_iter=1600))",
                    "('Poly NN (ReLU) (14,) * 3', MLPRegressor(hidden_layer_sizes=(14, 14, 14), max_iter=1600))"]

filter_names_list = [
    "control and control", 
    "control and lowpass 100", "control and lowpass 50", "control and lowpass 20", 
    "control and lowpass 10", "control and lowpass 5", "control and lowpass 2", "control and lowpass 1"]
filter_ticks_list = [
    "Control",
    "100 Hz", "50 Hz", "20 Hz", 
    "10 Hz", "5 Hz", "2 Hz", "1 Hz"]

model_print_names = {model_names_list[0]: "Linear Regression",
                     model_names_list[1]: "2nd Order Poly. Reg.",
                     model_names_list[2]: "Linear FFNN",
                     model_names_list[3]: "2nd Order Poly. FFNN"}

method_colors_dict = {model_names_list[0]: "#BBAA55",
                     model_names_list[1]:  "#9966BB",
                     model_names_list[2]: "#88CCEE",
                     model_names_list[3]: "#CC6677",}

method_shapes_dict = {model_names_list[0]: "D",
                     model_names_list[1]: "o",
                     model_names_list[2]: "X",
                     model_names_list[3]: "*"}

method_shapes_sizes_dict = {model_names_list[0]: "9",
                     model_names_list[1]: "15",
                     model_names_list[2]: "15",
                     model_names_list[3]: "15"}

model_names_list.reverse()

fig1, axs1 = plt.subplots()

# Sort methods into series wrt to filter freq
# Gather min max and median for plotting

r2_method_series_dict = {}
mae_method_series_dict = {}

num_vals = data_frame["num_cross_val"][0]

for model_name in model_names_list:
  r2_method_series_dict[model_name] = {"min": [], "max": [], "med": [], "q1": [], "q3": []}
  mae_method_series_dict[model_name] = {"min": [], "max": [], "med": [], "q1":[], "q3": []}
  for filter_name in filter_names_list:

    box_data = data_frame[data_frame["cl"] == model_name][data_frame["filter options"] == filter_name]
    r2_vals = []
    mae_vals = []
    for index in range(num_vals):
      r2_vals.append(float(box_data[f"r2 {index}"]))
      mae_vals.append(float(box_data[f"mae {index}"]))
    r2_method_series_dict[model_name]["min"].append(min(r2_vals))
    r2_method_series_dict[model_name]["max"].append(max(r2_vals))
    r2_method_series_dict[model_name]["med"].append(np.median(r2_vals))
    r2_method_series_dict[model_name]["q1"].append(np.quantile(r2_vals, 0.25))
    r2_method_series_dict[model_name]["q3"].append(np.quantile(r2_vals, 0.75))
    mae_method_series_dict[model_name]["min"].append(min(mae_vals))
    mae_method_series_dict[model_name]["max"].append(max(mae_vals))
    mae_method_series_dict[model_name]["med"].append(np.median(mae_vals))
    mae_method_series_dict[model_name]["q1"].append(np.quantile(mae_vals, 0.25))
    mae_method_series_dict[model_name]["q3"].append(np.quantile(mae_vals, 0.75))

# Plot R2 vals
xtickvals = list(range(len(filter_names_list)))
r2_legend_artists = []
r2_legend_labels = []
for model_name in model_names_list:
    axs1.fill_between(xtickvals,
        r2_method_series_dict[model_name]["min"],
        r2_method_series_dict[model_name]["max"],
        color=method_colors_dict[model_name],
        alpha = 0.25,
        zorder=0,
    )

    dashes, = axs1.plot(xtickvals, r2_method_series_dict[model_name]["med"], 
        label=model_name, color='k', linestyle='--', linewidth=1.8,
        zorder=10
    )
    
    q1dashes, = axs1.plot(xtickvals, r2_method_series_dict[model_name]["q1"], 
        label=model_name, color=method_colors_dict[model_name], 
        linestyle='--', linewidth=1.8,
        zorder=10
    )
    q3dashes, = axs1.plot(xtickvals, r2_method_series_dict[model_name]["q3"], 
        label=model_name, color=method_colors_dict[model_name], 
        linestyle='--', linewidth=1.8,
        zorder=10
    )

    dots, = axs1.plot(xtickvals, r2_method_series_dict[model_name]["med"], 
        color=method_colors_dict[model_name], 
        marker=method_shapes_dict[model_name], 
        markersize=method_shapes_sizes_dict[model_name], 
        linewidth=0,
        zorder=30
    )
    r2_legend_artists.append((dashes, dots))
    r2_legend_labels.append(model_print_names[model_name])

axs1.set_ylim(0.0,1)
plt.xticks(xtickvals, labels=filter_ticks_list)
axs1.grid(True, axis="both", which="minor")
axs1.minorticks_on()
axs1.tick_params(which="minor", bottom=False, left=False)
axs1.grid(True, axis="both", which="major", linewidth=2, color='k')
fig1.suptitle("$R^2$ Distributions with Median, Quartile, and Min/Max values", y=0.96)
axs1.set_title("100 tests each, random 70:30 test:train split")
axs1.set_ylabel("$R^2$ Value, 1.0 is Best")
axs1.set_xlabel("Lowpass Filter Cutoff Frequency")
axs1.legend(r2_legend_artists, r2_legend_labels, loc="upper right")

# Plot MAE vals
fig2, axs2 = plt.subplots()
legend_artists = []
legend_labels = []
for model_name in model_names_list:
    axs2.fill_between(xtickvals,
        mae_method_series_dict[model_name]["min"],
        mae_method_series_dict[model_name]["max"],
        color=method_colors_dict[model_name],
        alpha = 0.25,
        zorder=0,
    )

    dashes, = axs2.plot(xtickvals, mae_method_series_dict[model_name]["med"], 
        label=model_name, color='k', linestyle='--', linewidth=1.8,
        zorder=10
    )
    q1dashes, = axs2.plot(xtickvals, mae_method_series_dict[model_name]["q1"], 
        label=model_name, color=method_colors_dict[model_name], 
        linestyle='--', linewidth=1.8,
        zorder=10
    )
    q3dashes, = axs2.plot(xtickvals, mae_method_series_dict[model_name]["q3"], 
        label=model_name, color=method_colors_dict[model_name], 
        linestyle='--', linewidth=1.8,
        zorder=10
    )

    dots, = axs2.plot(xtickvals, mae_method_series_dict[model_name]["med"], 
        color=method_colors_dict[model_name], 
        marker=method_shapes_dict[model_name], 
        markersize=method_shapes_sizes_dict[model_name], 
        linewidth=0,
        zorder=30
    )
    legend_artists.append((dashes, dots))
    legend_labels.append(model_print_names[model_name])

axs2.set_ylim(0.0,10)
plt.xticks(xtickvals, labels=filter_ticks_list)
axs2.grid(True, axis="both", which="minor")
axs2.minorticks_on()
axs2.tick_params(which="minor", bottom=False, left=False)
axs2.grid(True, axis="both", which="major", linewidth=2, color='k')
fig2.suptitle("MAE Distributions with Median, Quartile, and Min/Max values", y=0.96)
axs2.set_title("100 tests each, random 70:30 test:train split")
axs2.set_ylabel("MAE Value, 0.0 is Best")
axs2.set_xlabel("Lowpass Filter Cutoff Frequency")
axs2.legend(legend_artists, legend_labels, loc="lower right")

plt.show(block=False)
input("Press Enter to close...")


