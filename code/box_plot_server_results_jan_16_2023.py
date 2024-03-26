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

# Load experiment file 
mypath = "./newout/"
filename = "COAL_2023_20240116_191818results.csv"

data_frame = pd.read_csv(mypath + filename) 
               
model_names_list = ["('Linear MSE Fit', LinearRegression())",
                    "('Poly MSE Fit', LinearRegression())",
                    "('Linear NN (ReLU) (4,) * 3', MLPRegressor(hidden_layer_sizes=(4, 4, 4), max_iter=800))",
                    "('Poly NN (ReLU) (14,) * 3', MLPRegressor(hidden_layer_sizes=(14, 14, 14), max_iter=800))"]

filter_names_list = ["control and lowpass 50", "control and lowpass 20", "control and lowpass 10", "control and lowpass 5"]

model_print_names = {model_names_list[0]: "Lin. Reg.",
                     model_names_list[1]: "Poly. Reg.",
                     model_names_list[2]: "Lin. FFNN",
                     model_names_list[3]: "Poly. FFNN"}

filter_print_names = {filter_names_list[3]: "  <5Hz Input",
                      filter_names_list[2]: " <10Hz Input",
                      filter_names_list[1]: " <20Hz Input",
                      filter_names_list[0]: " <50Hz Input"}

fig1, axs1 = plt.subplots()
fig2, axs2 = plt.subplots()

r2_box_data_array = []
mae_box_data_array = []
labels_list = []
num_vals = data_frame["num_cross_val"][0]

for model_name in model_names_list:
  for filter_name in filter_names_list:

    box_data = data_frame[data_frame["cl"] == model_name][data_frame["filter options"] == filter_name]
    labels_list.append(model_print_names[model_name] + filter_print_names[filter_name])
    r2_vals = []
    mae_vals = []
    for index in range(num_vals):
      r2_vals.append(float(box_data[f"r2 {index}"]))
      mae_vals.append(float(box_data[f"mae {index}"]))
    r2_box_data_array.append(r2_vals)
    mae_box_data_array.append(mae_vals)

axs1.boxplot(r2_box_data_array, labels=labels_list, whis=(0,100), vert=False)

axs1.set_xlim(0,1)
fig1.suptitle("$R^2$ Distributions with Median", y=0.96)
axs1.set_title("100 tests each, random 70:30 test:train split")
axs1.set_xlabel("$R^2$ Value, 1.0 is Best")
plt.grid(axis="both", which="both")

axs2.boxplot(mae_box_data_array, labels=labels_list, whis=(0,100), vert=False)

axs2.set_xlim(0,10)
fig2.suptitle("$MAE$ Distributions with Median", y=0.96)
axs2.set_title("100 tests each, random 70:30 test:train split")
axs2.set_xlabel("Mean Absolute Error, 0.0 is Best")
plt.grid(axis="both", which="both")

plt.show(block=False)
input("Press Enter to close...")


