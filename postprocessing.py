import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.nn.functional as F
import torch
from numpy.random import rand


### ANOMALY DETECTION FOR AUTOENCODERS (UNIVARIATE) ###
# Get the predicted dataset at inference time on the data retrieved from the DATACELLAR API
def get_predicted_dataset(test, reconstruction):
  # For DATACELLAR data
  scaler = MinMaxScaler(feature_range = (0,1))
  test['generation_kwh'] = scaler.fit_transform(test[['generation_kwh']])
  test['reconstruction'] = reconstruction
  test['abs_loss'] = np.abs(test.generation_kwh - test.reconstruction)
  test['rel_loss'] = np.abs((test['reconstruction']-test['generation_kwh'])/test['reconstruction'])
  return test

def get_transformer_dataset(y_test, forecast, df_test, train_window):
  # For DATACELLAR data
  predicted_test = pd.DataFrame(y_test, columns = ['generation_kwh'])
  predicted_test['reconstruction'] = forecast
  predicted_test['abs_loss'] = np.abs(predicted_test['generation_kwh'] - predicted_test['reconstruction'])
  predicted_test['rel_loss'] = np.abs((predicted_test['reconstruction']-predicted_test['generation_kwh'])/predicted_test['reconstruction'])
  return predicted_test

# Get the predicted dataset at test time on the dataset used also for training
def get_transformer_dataset_big(dataset, forecast, train_window):
  # This will be used both for the test set and for the validation set, which is going to be used as reference for thresholds in some cases
  scaler = MinMaxScaler(feature_range=(0,1))
  dfs_dict_1 = {}
  for country_code, gdf in dataset.groupby("country_code"):
      gdf[['solar_generation_actual']]=scaler.fit_transform(gdf[['solar_generation_actual']])
      dfs_dict_1[country_code] = gdf[train_window:]
  predicted_df = pd.concat(dfs_dict_1.values())
  predicted_df['forecast'] = forecast
  predicted_df['anomaly_score'] = (predicted_df.solar_generation_actual - predicted_df.forecast)**2
  predicted_df['abs_loss'] = np.abs(predicted_df.solar_generation_actual - predicted_df.forecast)
  predicted_df['rel_loss'] = np.abs((predicted_df['forecast']-predicted_df['solar_generation_actual'])/predicted_df['forecast'])
  return predicted_df

def get_predicted_synthetic_transformer_dataset_big(test, forecast, train_window):
    scaler = MinMaxScaler(feature_range = (0,1))
    dict_test = {}
    for code, gdf in test.groupby('country_code'):
      gdf[['new_solar_generation_actual']] = scaler.fit_transform(gdf[['new_solar_generation_actual']])
      dict_test[code] = gdf[train_window:]
    predicted_df_test = pd.concat(dict_test.values())
    predicted_df_test['forecast'] = forecast
    predicted_df_test['anomaly_score'] = (predicted_df_test.new_solar_generation_actual - predicted_df_test.forecast)**2
    predicted_df_test['abs_loss'] = np.abs(predicted_df_test.new_solar_generation_actual - predicted_df_test.forecast)
    predicted_df_test['rel_loss'] = np.abs((predicted_df_test['forecast']-predicted_df_test['new_solar_generation_actual'])/predicted_df_test['forecast'])
    return predicted_df_test

def get_predicted_dataset_big(test, reconstruction):
    scaler = MinMaxScaler(feature_range = (0,1))
    dict_test = {}
    for code, gdf in test.groupby('country_code'):
      gdf[['solar_generation_actual']] = scaler.fit_transform(gdf[['solar_generation_actual']])
      dict_test[code] = gdf
    predicted_df_test = pd.concat(dict_test.values())
    predicted_df_test['reconstruction'] = reconstruction
    predicted_df_test['anomaly_score'] = (predicted_df_test.solar_generation_actual - predicted_df_test.reconstruction)**2
    predicted_df_test['abs_loss'] = np.abs(predicted_df_test.solar_generation_actual - predicted_df_test.reconstruction)
    predicted_df_test['rel_loss'] = np.abs((predicted_df_test['reconstruction']-predicted_df_test['solar_generation_actual'])/predicted_df_test['reconstruction'])
    return predicted_df_test

def get_predicted_synthetic_dataset_big(test, reconstruction):
    scaler = MinMaxScaler(feature_range = (0,1))
    dict_test = {}
    for code, gdf in test.groupby('country_code'):
      gdf[['new_solar_generation_actual']] = scaler.fit_transform(gdf[['new_solar_generation_actual']])
      dict_test[code] = gdf
    predicted_df_test = pd.concat(dict_test.values())
    predicted_df_test['reconstruction'] = reconstruction
    predicted_df_test['anomaly_score'] = (predicted_df_test.new_solar_generation_actual - predicted_df_test.reconstruction)**2
    predicted_df_test['abs_loss'] = np.abs(predicted_df_test.new_solar_generation_actual - predicted_df_test.reconstruction)
    predicted_df_test['rel_loss'] = np.abs((predicted_df_test['reconstruction']-predicted_df_test['new_solar_generation_actual'])/predicted_df_test['reconstruction'])
    return predicted_df_test

# Define the threshold to use -> either absolute loss or anomaly score based
def threshold_abs_loss(val, percentile, predicted_df):
  val_mae_loss = val['abs_loss'].values
  threshold = (np.percentile(val_mae_loss, percentile)) 
  predicted_df['threshold'] = threshold
  predicted_df['predicted_anomaly'] = predicted_df['abs_loss'] > predicted_df['threshold']
  return predicted_df

def threshold_anom_score_perc(predicted_df_val, predicted_df_test, percentile):
   anom_val_score = predicted_df_val.anomaly_score.values
   threshold = np.percentile(anom_val_score, percentile)
   predicted_df_test['threshold'] = threshold
   predicted_df_test['predicted_anomaly'] = predicted_df_test.anomaly_score > predicted_df_test.threshold
   return predicted_df_test


def anomaly_detection(predicted_df_val, predicted_df_test, method_nr, percentile, weight_overall = 0.5, k = 1.5):
  if method_nr == 0:
    predicted_df = threshold_abs_loss(predicted_df_val, percentile, predicted_df_test)
  elif method_nr == 1:
     predicted_df = threshold_anom_score_perc(predicted_df_val, predicted_df_test, percentile)
  predicted_df['predicted_anomaly']=predicted_df['predicted_anomaly'].replace(False,0)
  predicted_df['predicted_anomaly']=predicted_df['predicted_anomaly'].replace(True,1)
  return predicted_df


# Define methods to perform synthetic injection of anomalies into the time series
def select_region(length, contamination, period, seed):
    # given the total length, generate periods that will be transformed

    np.random.seed(int(seed))
    n = int(length*contamination)
    period = max(period, 10)
    m = int(n/period)
    region = []

    for i in range(m):
        s = int(length/m)*i
        e = int(length/m)*(i+1)
        r = np.random.choice(np.arange(s, e-period), size=1)
        region.append([int(r), int(r+period)])

    return region

def get_context_region(data, y_true, contamination, period, seed, anomaly_amplitude_factor):
  region = select_region(len(data), contamination, period, seed)
  for r in region:
    # Compute mean and std over the region
    local_mean = np.mean(data[r[0]:r[1]])
    local_std = np.std(data[r[0]:r[1]])
    # Scale the amplitude of the anomalies
    anomaly_amplitude = anomaly_amplitude_factor * local_std
    # Generate anomalies
    anomaly = local_mean + np.random.uniform(-anomaly_amplitude, anomaly_amplitude, r[1] - r[0])
    data[r[0]:r[1]] = anomaly
    y_true[r[0]:r[1]] = 1
  return data, y_true

def add_point_outlier(data, label, seed=5, outlierRatio=0.01):
    n = int(len(data) * outlierRatio)
    index = np.random.choice(len(data), n, replace=False)
    for i in index:
      outlier = data.iloc[i] + 1.5 * np.std(data)
      if outlier < 1:
        data.iloc[i] = outlier
      else:
        data.iloc[i] = 1

    label[index] = 1

    return data, label
  
def flat_region(data, label, contamination, period, seed=5):
    # Replace with a flat region to the original data.
    region = select_region(len(data), contamination, period, seed)
    for r in region:
        data[r[0]:r[1]] = data[r[0]]
        label[r[0]:r[1]] = 1

    return data, label

def flip_segment(data, label, contamination, period, seed=5):
    # Flip a specific segment of the original time series.
    region = select_region(len(data), contamination, period, seed)
    for r in region:
        data[r[0]:r[1]] = np.flipud(data[r[0]:r[1]])
        label[r[0]:r[1]] = 1

    return data, label

def generate_anomalous_dataset(data, contamination, period, anomaly_amplitude_factor):
  """
  data: input time series
  contamination: percentage of anomalies to add
  period: length of section of time series affected by anomaly
  seed: random seed for replication purposes
  anomaly_amplitude_factor: to determine the amplitude of contextual anomalies
  """
  timeseries = data.solar_generation_actual
  series = timeseries.copy()
  y_true = np.zeros(len(series))
  # Call the first method to generate synthetic anomalies
  # N.B.: we are going to turn 3% of the data points in anomalies, 1% contextual, 1% flattened region, 1% flip region
  # N.B.: do not give to the anomaly injection methods the same seed, otherwise they will just substitute one another
  new_contamination = contamination / 3
  series, y_true = get_context_region(series, y_true, new_contamination, period, 5, anomaly_amplitude_factor)
  series, y_true = flat_region(series, y_true, new_contamination, period, 42)
  series, y_true = flip_segment(series, y_true, new_contamination * 0.8, period, 99)
  series, y_true = add_point_outlier(series, y_true, 76, new_contamination * 0.2)
  data['new_solar_generation_actual'] = series
  data['synthetic_anomaly'] = y_true
  return data