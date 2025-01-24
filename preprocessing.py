# In this file we can find all the code related to the preprocessing step over the timeseries data
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch

### IMPUTING MISSING DATES ###
def impute_missing_dates(dataframe):
  """
  Take first and last timestamp available. Create a new index starting from these two values, making sure that the index is 
  sampled with 1 hour jump. Use ffill to impute the missing values for the dates newly created.
  """
  dataframe = dataframe.set_index(['datetime'])
  start_ts = min(dataframe.index)
  end_ts = max(dataframe.index)
  new_index = pd.date_range(start_ts, end=end_ts, freq="1H")
  new_df = dataframe.reindex(new_index, method = "ffill")
  return new_df

def impute_missing_prod(dataframe):
  """ 
  For each country present in the dataframe we need to impute the missing values.
  Imputation by replicating the pattern of the previous 24 hours, given the periodicity of the series.
  Note: if more than 24 values are missing, the pattern cannot be replicated, so we need to perform iterations to fill 24 values at a time.
  """
  iterations_n_by_country = {}
  for code, gdf in dataframe.groupby(['country_code']):
      # Save the country code
      country_code = code
      if gdf.solar_generation_actual.isna().sum() > 0 :
          # Define a boolean mask to identify the NaNs
          is_na = gdf['solar_generation_actual'].isna()
          # Find groups of consecutive NaNs, by assigning a uid to each group (every time is_na changes from False to True or viceversa)
          gdf['na_group'] = (is_na != is_na.shift()).cumsum() * is_na
          # Compute the length of each group to find the longest
          na_lengths = gdf.groupby('na_group').size()
          longest_na_group = na_lengths[na_lengths.index != 0].idxmax()  # Exclude group 0, which is not NaN
          # Get the length of the longest NaN sequence and the positions
          longest_na_length = na_lengths[longest_na_group]
          # Remove temporary column
          gdf.drop(columns=['na_group'], inplace=True)
          iterations = int(np.ceil(longest_na_length / 24))
          iterations_n_by_country[country_code] = iterations

  for k, v in iterations_n_by_country.items():
    gdf = dataframe[dataframe.country_code == k]
    for i in range(v):
        dataframe.solar_generation_actual = dataframe.solar_generation_actual.fillna(dataframe.solar_generation_actual.shift(24))
  return dataframe

### SPLIT THE DATASET IN TRAIN, VAL, TEST ###
def split_big(dataframe):
  df_train = dataframe.iloc[:int(0.6 * len(dataframe)), :]
  df_val = dataframe.iloc[int(0.6 * len(dataframe)):int(0.8 * len(dataframe)), :]
  df_test = dataframe.iloc[int(0.8 * len(dataframe)):, :]
  return df_train, df_val, df_test

# Generate sequences at inference time on the data retrieved from the DATACELLAR API
def create_sequences(dataframe, time_steps, stride = 1):
    scaler = MinMaxScaler(feature_range = (0,1))
    sequences = []
    X = scaler.fit_transform(dataframe)
    for i in range(0, len(dataframe) - time_steps + 1, stride):
        #end of sequence
        end_idx = i + time_steps
        slice = X[i: (i + time_steps), :]
        sequences.append(slice)
    return np.stack(sequences)

def create_transformer_sequences(dataframe, time_steps, stride = 1):
    scaler = MinMaxScaler(feature_range = (0,1))
    dataframe = scaler.fit_transform(dataframe)
    sequences = []
    y_true = []
    for i in range(0, len(dataframe) - time_steps, stride):
        #end of sequence
        end_idx = i + time_steps

        slice = dataframe[i: (i + time_steps), :]
        y = dataframe[end_idx, 0]
        sequences.append(slice)
        y_true.append(y)
    return np.stack(sequences), np.stack(y_true)

# Generated training sequences to use in the model. Valid also for testing, where stride = time_steps
def create_sequences_big(dataframe, time_steps, stride = 1):
    scaler = MinMaxScaler(feature_range = (0,1))
    sequences = []
    for code, gdf in dataframe.groupby('country_code'):
      production = gdf[['solar_generation_actual']]
      X = scaler.fit_transform(production)
      for i in range(0, len(production) - time_steps + 1, stride):
          #end of sequence
          end_idx = i + time_steps
          if end_idx > len(production) - 1:
            break
          #if end_idx <= len(dataframe)+1:
           #   slice = X[i: (i + time_steps -1), :]
          slice = X[i: (i + time_steps), :]
          sequences.append(slice)
    return np.stack(sequences)

def create_synthetic_sequences_big(dataframe, time_steps, stride = 1):
    scaler = MinMaxScaler(feature_range = (0,1))
    sequences = []
    for code, gdf in dataframe.groupby('country_code'):
      production = gdf[['new_solar_generation_actual']]
      X = scaler.fit_transform(production)
      for i in range(0, len(production) - time_steps + 1, stride):
          #end of sequence
          end_idx = i + time_steps
          #if end_idx <= len(dataframe)+1:
           #   slice = X[i: (i + time_steps -1), :]
          slice = X[i: (i + time_steps), :]
          sequences.append(slice)
    return np.stack(sequences)

def create_test_sequences_big(dataframe, time_steps, stride = 1):
    scaler = MinMaxScaler(feature_range = (0,1))
    sequences = []
    for code, gdf in dataframe.groupby('country_code'):
      production = gdf[['solar_generation_actual']]
      X = scaler.fit_transform(production)
      for i in range(0, len(production) - time_steps + 1, stride):
          #end of sequence
          end_idx = i + time_steps
          #if end_idx <= len(dataframe)+1:
           #   slice = X[i: (i + time_steps -1), :]
          slice = X[i: (i + time_steps), :]
          sequences.append(slice)
    return np.stack(sequences)

def create_transformer_sequences_big(dataframe, time_steps, stride = 1):
    scaler = MinMaxScaler(feature_range = (0,1))
    sequences = []
    y_true = []
    for code, gdf in dataframe.groupby('country_code'):
      production = gdf[['solar_generation_actual']]
      X = scaler.fit_transform(production)
      for i in range(0, len(production) - time_steps, stride):
          #end of sequence
          end_idx = i + time_steps

          slice = X[i: (i + time_steps), :]
          y = X[end_idx, 0]
          sequences.append(slice)
          y_true.append(y)
    return np.stack(sequences), np.stack(y_true)

def create_synthetic_transformer_sequences_big(dataframe, time_steps, stride = 1):
    scaler = MinMaxScaler(feature_range = (0,1))
    sequences = []
    y_true = []
    for code, gdf in dataframe.groupby('country_code'):
      production = gdf[['new_solar_generation_actual']]
      X = scaler.fit_transform(production)
      for i in range(0, len(production) - time_steps, stride):
          #end of sequence
          end_idx = i + time_steps

          slice = X[i: (i + time_steps), :]
          y = X[end_idx, 0]
          sequences.append(slice)
          y_true.append(y)
    return np.stack(sequences), np.stack(y_true)




