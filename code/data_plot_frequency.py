# This file takes all available input data and trains the regressors
# using the prescribed preprocess methods. 
# It displays time domain estimates for all data files as well as
# sensitivity plots for sensor channels and estimates
import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate

from scipy import signal
from scipy.signal import decimate

import argparse

import pdb

##
#  FFT helper class
## 
class FFTMag:
    """
    Class for computing magnitude of right side of FFT of input signal,
    optional num_channels parameter causes computation to be broken down along chunks of signal
    optional power parameter transforms data with SQRT or SQUARE, SQUARE is approx PSD
    Input must be 2D np.array with second dimension multiple of number of channels
    First dim is samples, second dim is features of samples
    Transform returns the right side of the FFT magnitude, reducing features roughly by factor of 2
    """

    def __init__(self, num_channels=1, power=None, after=None):
      self.num_channels = num_channels
      self.power = power
      self.after = after
      self.recognized_powers = {  "SQRT": lambda x : np.sqrt(x), 
                                "SQUARE": lambda x : np.multiply(x,x),
                                 "OUTER": lambda x : np.multiply(np.expand_dims(x, axis=1), np.expand_dims(x,axis=2)).reshape(x.shape[0],-1),
                                  "DIFF": lambda x : np.diff(x, n=1),
                                 "DIFF2": lambda x : np.diff(x, n=2),
                                   "SUM": lambda x : np.cumsum(x, axis=-1),
                                    None: lambda x : x}
      if power not in self.recognized_powers:
        raise ValueError("power param must be in %s" % (str(self.recognized_powers)))
      if after not in self.recognized_powers:
        raise ValueError("after param must be in %s" % (str(self.recognized_powers)))

    def fit(self, x, y=None, **fit_params):
      dims = np.shape(x)
      if dims[1] % self.num_channels != 0:
        raise IndexError("Number of features must be divisable by number of channels!")
      return self


    def transform(self, x):
      z = np.array(x)
      if self.num_channels != 1:
        z = z.reshape((z.shape[0],self.num_channels,-1), order='F')
        z = np.abs(np.fft.rfft(z, axis=1)).reshape((z.shape[0], -1), order='F')
      else:
        z = np.abs(np.fft.rfft(z))
      z = self.recognized_powers[self.power](z)
      z = self.recognized_powers[self.after](z)
      return z
##
# End FFT helper class
##

##
# Windowing helper class
##
import scipy.signal.windows

def window_maker(shape, num_samples):
  """
  Returns a window suitable for use with the windowizer
  """
  window = scipy.signal.windows.get_window(shape, num_samples)
  return np.expand_dims(window, axis=-1)

class Windowizer:
  """
  Object to hold and apply window to data with specified overlap
  """
  def __init__(self, window_array, overlap_ratio):
    self.my_win = window_array
    self.overlap_ratio = overlap_ratio

  def windowize(self, data, labels, flatten=True):
    """
    Return windowed samples as feature vectors and corresponding labels
    Only return windows for which all data has the same labels
    """
    windowed_data = []
    new_labels = []
    step_size = int(len(self.my_win)*(1.0-self.overlap_ratio))
    for index in list(range(0,len(data)-len(self.my_win)+1,step_size)):
      # check for consistant labels over window
      label_check_fail = False
      for label_check_index in list(range(index,index+len(self.my_win))):
        if labels[index] != labels[label_check_index]:
          label_check_fail = True
          break # stop the check, its already failed
      if label_check_fail:
        continue # skip this window location
      glimpse = np.reshape(np.array(
                   data[index:index+len(self.my_win)], dtype=float), self.my_win.shape)
      windowed_datum = np.multiply(self.my_win, glimpse)
      if flatten:
        windowed_datum = windowed_datum.flatten('F')
      windowed_data.append(windowed_datum)
      new_labels.append(labels[index])
    return windowed_data, new_labels
##
# End Windowing helper class
##

##
# Set Figures
##
SMALL_SIZE = 22
MEDIUM_SIZE = 28
SMALLER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

##
# Commnad line defaults
##

# Don't Include NN regressions
argument_no_nn = False

# Do filter calcs
argument_filt = True

## Allow command line overrides
parser = argparse.ArgumentParser()

parser.add_argument("--no_nn", default=argument_no_nn, type=bool,
  help="Set to True to omit NN from processing")

parser.add_argument("--filt", default=argument_filt, type=bool,
  help="Set to False to omit input filters")

args = parser.parse_args()

##
# Load all good csv files as indicated by index.csv and preprocess
##

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

sample_freq = 1.0 / 0.002475

chan_names_list = ["Freq. A (MHz)", "Freq. B (MHz)", "Freq. C (MHz)", "Freq. D (MHz)"]

# Moving average size in number of samples for input   
avg_len = 5
chan_freqs = [10]

# Dictionary of dictionary of lists
data_list_dict = {}
# Preprocess data to have bias adjusted based on first 0.5s from sample
x_vals = []
y_vals = []

y_filt_b, y_filt_a = signal.butter(N=10, btype="low", Wn=10.0, fs=sample_freq)

chan_key = ""
force_key = "Filt Force <10 Hz"

for filename in good_files_list:

    # Compute 10 Hz filtering of drag force
    data_list_dict[filename] = {
        "Filt Force <10 Hz Force (kN)": signal.filtfilt(
            y_filt_b, y_filt_a, data_files_dict[filename][f"Force (kN)"])}
    
    # Compute bias compensated avg values
    chan_freqs = [10, 20, 40]
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

      # Compute filtered values
      for chan_freq in chan_freqs:
          bc, ac = signal.butter(N=10, btype="low", Wn=chan_freq, fs=sample_freq)
          data_list_dict[filename][f"Filt {chan_freq} Hz Adj {chan_name}"] = \
              signal.filtfilt(bc, ac, data_files_dict[filename][f"Adj {chan_name}"])

    chan_key = "Avg. Adj"
    if (args.filt == True):
        chan_key = "Filt 10 Hz Adj"

    # Glob data into X and Y lists
    chan_vals = [data_list_dict[filename][f"{chan_key} {chan_name}"] 
                  for chan_name in chan_names_list] # list of each channel
    chan_time_list = [list(vals) for vals in zip(*chan_vals)] # transpose to each channel as column
    x_vals.extend(chan_time_list) # each entry is all 4 channels at same time
    y_vals.extend(data_list_dict[filename][f"{force_key} Force (kN)"])

predictor_key = chan_key

# Let x be an np array, as a treat
x_vals = np.array(x_vals)

# Generate polynomial expansions
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_x_vals = poly.fit_transform(x_vals)

# Add inner dim to y_vals
y_vals = np.array(y_vals).ravel() # add inner dim

num_samples = y_vals.shape[0]
linear_input_dim = x_vals.shape[1]
poly_input_dim = poly_x_vals.shape[1]

print(f"Total number of samples: {num_samples}")
# Should be num channels
print(f"Linear input dim: {linear_input_dim}")
# Should be num channels combos
print(f"Poly input dim: {poly_input_dim}")

that_time = time.time()
print("Data preprocessed in {0} sec; performing experiments".format(that_time - this_time),
      end='\n', flush=True)
this_time = time.time()

# Manually add Rozeboom 1987 adjusted value
scorings = ['r2','neg_root_mean_squared_error', 'neg_mean_absolute_error']

##
# Linear Model
##

# Fit data with least squares
reg = linear_model.LinearRegression()
reg.fit(x_vals, y_vals)
print(f"Linear coef: {reg.coef_}, Intercept: {reg.intercept_}")

y_hats = reg.predict(x_vals)

print("lin MSE: ")
print(mean_squared_error(y_vals, y_hats))
print("lin MAE: ")
print(mean_absolute_error(y_vals, y_hats))
print("lin R2: ")
print(r2_score(y_vals, y_hats))

# Add model data to data dict
for filename in good_files_list:
    chan_vals = [data_list_dict[filename][f"{predictor_key} {chan_name}"]
                  for chan_name in chan_names_list]
    chan_time_list = [list(vals) for vals in zip(*chan_vals)]
    data_files_dict[filename]["Lin Est Force (kN)"] = \
      reg.predict(chan_time_list)

##
# Poly fit
##

poly_reg_model = linear_model.LinearRegression()
poly_reg_model.fit(poly_x_vals, y_vals)
y_poly_hats = poly_reg_model.predict(poly_x_vals)

print(f"Poly coef: {poly_reg_model.coef_}, Intercept: {poly_reg_model.intercept_}")
print("poly MSE: ")
print(mean_squared_error(y_vals, y_poly_hats))
print("poly MAE: ")
print(mean_absolute_error(y_vals, y_poly_hats))
print("poly R2: ")
print(r2_score(y_vals, y_poly_hats))

# Add model data to data dict
for filename in good_files_list:
    chan_vals = [data_list_dict[filename][f"{predictor_key} {chan_name}"]
                  for chan_name in chan_names_list]
    chan_time_list = [list(vals) for vals in zip(*chan_vals)]
    data_files_dict[filename]["Poly Fit Force (kN)"] = \
        poly_reg_model.predict(poly.fit_transform(
            np.array(chan_time_list)))

##
# FFNN fit
##

print("Fitting ffnn...")

ffnn = MLPRegressor(
    (linear_input_dim,) * 3,
    activation='relu', solver='adam',
          max_iter=800, verbose=False, random_state=12345).fit(
    x_vals, y_vals)

ffnn_y_hat = ffnn.predict(x_vals)

print("ffnn MSE: ")
print(mean_squared_error(y_vals, ffnn_y_hat))
print("ffnn MAE: ")
print(mean_absolute_error(y_vals, ffnn_y_hat))
print("ffnn R2: ")
print(r2_score(y_vals, ffnn_y_hat))

## Add model data to data dict
for filename in good_files_list:
    # break file data into windows
    chan_vals = [data_list_dict[filename][f"{predictor_key} {chan_name}"]
                  for chan_name in chan_names_list]
    chan_time_list = [list(vals) for vals in zip(*chan_vals)]

    data_files_dict[filename]["FFNN Fit Force (kN)"] = \
      ffnn.predict(chan_time_list)


##
# Poly FFNN fit
##
print("Fitting Poly FFNN...")

poly_ffnn = MLPRegressor((poly_input_dim,) * 3, activation='relu', solver='adam',
                max_iter=800, verbose=False, random_state=54321).fit(
        poly_x_vals, y_vals)

poly_ffnn_y_hat = poly_ffnn.predict(poly_x_vals)

print("poly ffnn MSE: ")
print(mean_squared_error(y_vals, poly_ffnn_y_hat))
print("poly ffnn MAE: ")
print(mean_absolute_error(y_vals, poly_ffnn_y_hat))
print("poly ffnn R2: ")
print(r2_score(y_vals, poly_ffnn_y_hat))

## Add model data to data dict
for filename in good_files_list:
    # break file data into windows
    chan_vals = [data_list_dict[filename][f"{predictor_key} {chan_name}"]
                  for chan_name in chan_names_list]
    chan_time_list = [list(vals) 
                      for vals in zip(*chan_vals)]
    poly_chan_time_list = poly.fit_transform(chan_time_list)
    
    data_files_dict[filename]["Poly FFNN Fit Force (kN)"] = \
      poly_ffnn.predict(poly_chan_time_list)

that_time = time.time()
print("Models fit in {0} sec; plotting experiments".format(that_time - this_time),
      end='\n', flush=True)
this_time = time.time()

## Add filtered force to data_files dict
for filename in good_files_list:
    data_files_dict[filename][f"{force_key} Force (kN)"] = \
      data_list_dict[filename][f"{force_key} Force (kN)"]

##
# Numerically estimate transfer function via cross-spectral density and psd
# Hbar = Pyx / Pxx; Do for poly ffnn
# Estimate linear system between output of ffnn and chosen regression target
##

psd_freqs, Pxx = signal.welch(poly_ffnn_y_hat, fs=sample_freq)
ypsd_freqs, Pyy = signal.welch(y_vals, fs=sample_freq)
csd_freqs, Pyx = signal.csd(y_vals, poly_ffnn_y_hat, fs=sample_freq)

# Get psd's for each channel 
chan_psd_freqs, chan_psd = signal.welch(x_vals, axis=0, fs=sample_freq)

#Hbar = np.divide(np.real(Pyx), Pxx)
Hbar = np.divide(Pyx, Pxx)

#fig0, axes0 = plt.subplots(2, 2)
fig0 = plt.figure()

gs = fig0.add_gridspec(2,2)
axes0 = []
axes0.append(fig0.add_subplot(gs[0, 0])) # Pyy and Pxx, top left
axes0.append(fig0.add_subplot(gs[1, 0])) # Pyx, bottom left
axes0.append(fig0.add_subplot(gs[:, 1])) # Hbar, right side

# Cross spectral density
axes0[0].plot(chan_psd_freqs, np.abs(chan_psd[:, 0]), 
    label="Chan A", linestyle=":", color="#1F0322")
axes0[0].plot(chan_psd_freqs, np.abs(chan_psd[:, 1]), 
    label="Chan B", linestyle=":", color="#8A1C7C")
axes0[0].plot(chan_psd_freqs, np.abs(chan_psd[:, 2]), 
    label="Chan C", linestyle=":", color="#DA4167")
axes0[0].plot(chan_psd_freqs, np.abs(chan_psd[:, 3]), 
    label="Chan D", linestyle=":", color="#F0BCD4")
axes0[0].plot(ypsd_freqs, np.abs(Pyy), 
    color="magenta", label="Pyy, Target Force")
axes0[0].plot(psd_freqs, np.abs(Pxx), 
    color="#3EA06D", label="Pxx, Poly FFNN Output")

axes0[0].set_title("Pyy and Pxx")
axes0[0].set_yscale("log")
axes0[0].set_ylim( 1e-6, 1e2)
axes0[0].legend(ncol=2, loc="upper right")
axes0[0].set_ylabel("Spectral Density")
axes0[0].set_xlabel("Frequency (Hz)")

#axes0[0,1].plot(psd_freqs, np.abs(Pxx), color="blue", label="Magnitude")
#axes0[0,1].legend()
#axes01_2 = axes0[0,1].twinx()
#axes01_2.plot(psd_freqs, np.angle(Pxx), color="orange", label="Phase")
#axes01_2.legend()
#axes0[0,1].set_title("Pxx")
#axes0[0,1].set_yscale("log")

axes0[1].plot(csd_freqs, np.abs(Pyx), color="purple", label="Magnitude")
axes0[1].legend(loc="upper center")
axes10_2 = axes0[1].twinx()
axes10_2.plot(csd_freqs, np.angle(Pyx), color="orange", label="Phase")
axes10_2.legend(loc="upper right")
axes10_2.set_ylabel("Phase (Rad.)")
axes0[1].set_title("Pyx: Sensor and Target Cross Spectrum")
axes0[1].set_yscale("log")
axes0[1].set_ylabel("Spectral Density")
axes0[1].set_xlabel("Frequency (Hz)")

axes0[2].plot(psd_freqs, np.abs(Hbar), color="purple", label="Magnitude")
axes0[2].legend(loc="upper center")
axes11_2 = axes0[2].twinx()
axes11_2.plot(psd_freqs, np.angle(Hbar), color="orange", label="Phase")
axes11_2.legend(loc="upper right")
axes11_2.set_ylabel("Phase (Rad.)")
axes0[2].set_title("Hbar = Pyx / Pxx. Sensor to Target TF")
axes0[2].set_yscale("log")
axes0[2].set_ylabel("Power Spectral Density Magnitude")
axes0[2].set_xlabel("Frequency (Hz)")

plt.suptitle("Finding Transfer Function between Sensor and Target")

# Plot spectrograms, general properties
spec_win_len = 80
muh_window = signal.get_window("hann", spec_win_len)
print(f"muh_window: {muh_window}")
dec_factor = 8

windy = Windowizer(muh_window, 0.8)
    
plot_keys = ["Force (kN)", "Lin Est Force (kN)", "Poly Fit Force (kN)", "FFNN Fit Force (kN)", "Poly FFNN Fit Force (kN)", f"{force_key} Force (kN)"]
plot_colors_labels = [("dimgrey", "Force Data"), ("#7E58A0", "Lin. Reg"), ("#4883D6", "Poly. Reg."), ("#78BFE0", "Lin. FFNN"), ("#3EA06D", "Poly. FFNN"), ("magenta", "Filt. <10 Hz")]
plot_colors_labels_dict = {k: cl for k,cl in zip(plot_keys, plot_colors_labels)}
plot_list = [plot_keys[0], plot_keys[5], plot_keys[4]]

def calculate_cont_t_stat(means1, means2, dev1, dev2, num_samples1, num_samples2):
  t_stat_list = []
  for m1, m2, d1, d2 in zip(means1, means2, dev1, dev2):
    t_stat_list.append(
        (m1 - m2) / np.sqrt(d1 * d1 / num_samples1 + d2 * d2 / num_samples2))
  two_tails_value = 2.626  # Approximate 99% chance dist diff for about 100 dof
  #two_tails_value = 2.581 # Approximate 99% chance distributions are different for about 940 degrees of freedom 
  #two_tails_value = 3.291 # Approximate 99.9% chance distributions are different for about 940 degrees of freedom 
  t_reject_null = [float(np.abs(t_stat) > two_tails_value) for t_stat in t_stat_list]
  return t_stat_list, t_reject_null


# Spectrogram Plot 1
# who cares about all of these (don't plot, too many)
for idx, filename in enumerate([]):#good_files_list):

    # plot time data
    # Raw force from lcm (noisy)
    spec_df = 1.0 / (data_files_dict[filename]["Time (s)"][1] - data_files_dict[filename]["Time (s)"][0])
    spec_df = spec_df / dec_factor
    print(f"{filename} has fs = {spec_df} after {dec_factor}X decimation")
    spec_fig, spec_ax = plt.subplots(len(plot_list),1)
    for idx, key in enumerate(plot_list):
      decimated = signal.decimate(data_files_dict[filename][key], dec_factor)
      f, t, Sxx = signal.spectrogram(decimated, spec_df, window=muh_window, nperseg=spec_win_len, mode="magnitude")

      spec_ax[idx].pcolormesh(t, f, Sxx, shading="gouraud")
      spec_ax[idx].set_title(plot_colors_labels_dict[key][1])
      plt.suptitle(f"{filename}")

# Spectrogram Plot 2
# Sort data from files by wear and concatenate to make data pool
new_list_dict = {p: [] for p in plot_list}
mod_list_dict = {p: [] for p in plot_list}
worn_list_dict = {p: [] for p in plot_list}
the_data = {p: [] for p in plot_list}
big_data_dict = {p: [] for p in plot_list}
for key in plot_list:
  for filename in good_files_list:
    #decimated = signal.decimate(data_files_dict[filename]["Force (kN)"], dec_factor)
    the_data[key] = signal.decimate(data_files_dict[filename][key], dec_factor)
    big_data_dict[key].extend(the_data[key])
    if "New" in filename:
      new_list_dict[key].extend(the_data[key])
      print(f"new: {filename}")
    elif "Mod" in filename:
      mod_list_dict[key].extend(the_data[key])
      print(f"mod: {filename}")
    else:
      worn_list_dict[key].extend(the_data[key])
      print(f"worn: {filename}")
    
# Normalize data to 0 mean, std 1 in time domain

for key in plot_list:
  print(f"Average Force by Wear for {key}")
  print(f"New: {sum(new_list_dict[key])/len(new_list_dict[key])} kN, N={len(new_list_dict[key])}")
  print(f"Mod: {sum(mod_list_dict[key])/len(mod_list_dict[key])} kN, N={len(mod_list_dict[key])}")
  print(f"Worn: {sum(worn_list_dict[key])/len(worn_list_dict[key])} kN, N={len(mod_list_dict[key])}")

wear_fig, wear_ax = plt.subplots(len(plot_list),1, sharex=True)
wear_colors = ["lightblue", "orange", "maroon"]
for idx, key in enumerate(plot_list):

  big_data_mean = np.mean(big_data_dict[key], axis=0)
  big_data_std  = np.std(big_data_dict[key], axis=0)
  
  means_dict = {"New": [], "Mod.": [], "Worn": []}
  devs_dict  = {"New": [], "Mod.": [], "Worn": []}
  freqs = []

  for wear_idx, (name, wear_list) in enumerate([("New", new_list_dict[key]), ("Mod.", mod_list_dict[key]), ("Worn", worn_list_dict[key])]):
    effs, welches = signal.welch(wear_list, 400, muh_window)
    welches = welches / max(welches) # Normalize to largest value

 
    # Split wear list into windows, normalize, compute ffts, average and find std dev
    windowed_data = windy.windowize(wear_list, [name]*len(wear_list))
    feature_data = np.array(windowed_data[0])
    print(f"{key} {name} has shape {feature_data.shape}")
    #feature_data -= big_data_mean
    #feature_data /= big_data_std
    fft_doer = FFTMag(after="SQRT")
    fft_data = fft_doer.transform(feature_data)
    freqs = np.linspace(0,200/dec_factor,fft_data.shape[1])
    means = np.mean(fft_data, axis=0)
    deviations = np.std(fft_data, axis=0)
    means_dict[name] = means
    devs_dict[name] = deviations

    #wear_ax[idx].plot(effs, welches, label=name, linewidth=4.0, color=wear_colors[wear_idx])
    wear_ax[idx].plot(freqs, means, label=name, linewidth=2.0, color=wear_colors[wear_idx])
    wear_ax[idx].plot(freqs, means+deviations, label="_"+name, linewidth=1.5, 
        color=wear_colors[wear_idx], linestyle=':')
    wear_ax[idx].plot(freqs, means-deviations, label="_"+name, linewidth=1.5, 
        color=wear_colors[wear_idx], linestyle=':')
    wear_ax[idx].legend()
    wear_ax[idx].set_title(key)
    wear_ax[idx].set_ylabel(r"kN/$\sqrt{Hz}$")

  t_stat, t_reject_null =  calculate_cont_t_stat(means_dict["New"], means_dict["Worn"], 
    devs_dict["New"], devs_dict["Worn"], 
    len(new_list_dict[key]), len(worn_list_dict[key]))
  print(f"Tstat for {key}: {t_stat}")
  print(f"Treject for {key}: {t_reject_null}")
  wear_ax[idx].bar(freqs, np.array(t_reject_null)*1000, width=freqs[1], color="honeydew")
  wear_ax[idx].set_ylim(0.01,40)
  wear_ax[idx].set_yscale("log")

wear_ax[len(plot_list)-1].set_xlabel("Frequency (Hz)")
wear_fig.suptitle(f"Sq. Root of Fourier Spectra Magnitude by Wear\nSignificant, p<0.01 diffs. for New and Worn highlighted.")

# Spectrogram Plot 3
# Sort file data by material type to form data pool
coal_list_dict = {p: [] for p in plot_list}
concrete_list_dict = {p: [] for p in plot_list}
for plot_item in plot_list:
  for filename in good_files_list:
    first_idx = np.argmin([abs(x-0.25) for x in data_files_dict[filename]["Time (s)"]])
    second_idx = np.argmin([abs(x-1.25) for x in data_files_dict[filename]["Time (s)"]])
    first_chunk = data_files_dict[filename][plot_item][first_idx:second_idx]
    first_chunk = scipy.signal.decimate(first_chunk, dec_factor)
    concrete_list_dict[plot_item].extend(first_chunk)

    third_idx = np.argmin([abs(x-1.75) for x in data_files_dict[filename]["Time (s)"]])
    fourth_idx = np.argmin([abs(x-3.25) for x in data_files_dict[filename]["Time (s)"]])
    second_chunk = data_files_dict[filename][plot_item][third_idx:fourth_idx]
    second_chunk = scipy.signal.decimate(second_chunk, dec_factor)
    coal_list_dict[plot_item].extend(second_chunk)

    fifth_idx = np.argmin([abs(x-3.75) for x in data_files_dict[filename]["Time (s)"]])
    sixth_idx = np.argmin([abs(x-4.75) for x in data_files_dict[filename]["Time (s)"]])
    third_chunk = data_files_dict[filename][plot_item][fifth_idx:sixth_idx]
    third_chunk = scipy.signal.decimate(third_chunk, dec_factor)
    concrete_list_dict[plot_item].extend(third_chunk)

print("Averages by Material")
print(f"Coal: {sum(coal_list_dict[plot_list[2]])/len(coal_list_dict[plot_list[2]])} kN, N={len(coal_list_dict[plot_list[2]])}")
print(f"Concrete: {sum(concrete_list_dict[plot_list[2]])/len(concrete_list_dict[plot_list[2]])} kN, N={len(coal_list_dict[plot_list[2]])}")

mat_fig, mat_ax = plt.subplots(len(plot_list),1, sharex=True)
material_colors = ["black", "grey"]
for idx, plot_item in enumerate(plot_list):

  means_dict = {"Coal": [], "Concrete": []}
  devs_dict  = {"Coal": [], "Concrete": []}
  freqs = []

  for mat_idx, (name, mat_list) in enumerate([("Coal", coal_list_dict[plot_item]), ("Concrete", concrete_list_dict[plot_item])]):

    effs, welches = signal.welch(mat_list, 400, muh_window, scaling="density")

    # Split wear list into windows, normalize, compute ffts, average and find std dev
    windowed_data = windy.windowize(mat_list, [name]*len(mat_list))
    feature_data = np.array(windowed_data[0])
    print(f"{plot_item} {name} has shape {feature_data.shape}")

    fft_doer = FFTMag(after="SQRT")
    fft_data = fft_doer.transform(feature_data)
    freqs = np.linspace(0,200/dec_factor,fft_data.shape[1])
    means = np.mean(fft_data, axis=0)
    deviations = np.std(fft_data, axis=0)
    means_dict[name] = means
    devs_dict[name] = deviations


    #mat_ax[idx].plot(effs, welches, label=name, linewidth=4.0, color=material_colors[mat_idx])
    mat_ax[idx].plot(freqs, means, label=name, linewidth=2.0, color=material_colors[mat_idx])
    mat_ax[idx].plot(freqs, means+deviations, label="_"+name, linewidth=1.5, 
        color=material_colors[mat_idx], linestyle=':')
    mat_ax[idx].plot(freqs, means-deviations, label="_"+name, linewidth=1.5, 
        color=material_colors[mat_idx], linestyle=':')
    mat_ax[idx].legend()
    mat_ax[idx].set_title(plot_item)
    mat_ax[idx].set_ylabel(r"kN/$\sqrt{Hz}$")

  t_stat, t_reject_null =  calculate_cont_t_stat(means_dict["Coal"], means_dict["Concrete"], 
    devs_dict["Coal"], devs_dict["Concrete"], 
    len(coal_list_dict[plot_item]), len(concrete_list_dict[plot_item]))
  print(f"Tstat for {plot_item}: {t_stat}")
  print(f"Treject for {plot_item}: {t_reject_null}")
  mat_ax[idx].bar(freqs, np.array(t_reject_null)*3000, width=freqs[1], color="honeydew")
  mat_ax[idx].set_ylim(0.01,40)
  mat_ax[idx].set_yscale("log")

mat_fig.suptitle(f"Sq. Root of Fourier Spectra Magnitude by Material\nSignificant, p<0.01 diffs. highlighted.")
mat_ax[len(plot_list)-1].set_xlabel("Frequency (Hz)")



plt.show(block=False)
input("Press Enter to close")

