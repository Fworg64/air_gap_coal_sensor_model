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

filter_names_list = ["control and lowpass 5", "control and lowpass 10", "control and lowpass 20", "control and lowpass 50"]

fig1, axs1 = plt.subplots()

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


pdb.set_trace()
box_data = data_frame[data_frame["cl"] == model_names_list[0]][data_frame["filter options"] == filter_names_list[0]]
r2_vals = []
mae_vals = []
for index in range(data_frame["num_cross_val"][0]):
  r2_vals.append(float(box_data[f"r2 {index}"]))
  mae_vals.append(float(box_data[f"mae {index}"]))

axs1.boxplot(r2_vals, whis=(0,100))

axs1.set_ylim(0,1)
axs1.legend(loc='lower left', ncol=2)
fig1.suptitle("$R^2$ Values with Median vs Window Size", y=0.96)
axs1.set_title("100 tests each with random 50:50 test:train split")
axs1.set_xlabel("Window Size (Whole Numbers Only)")
axs1.set_ylabel("$R^2$ Value, Higher is Better")


plt.show(block=False)
input("Press Enter to close...")


