from preprocessing import create_transformer_sequences_big, impute_missing_prod, split_big, create_test_sequences_big, create_synthetic_sequences_big, create_synthetic_transformer_sequences_big
import pandas as pd
import torch
import torch.utils.data as data_utils
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from postprocessing import get_predicted_synthetic_transformer_dataset_big, get_predicted_dataset_big, anomaly_detection, get_transformer_dataset_big, generate_anomalous_dataset , get_predicted_synthetic_dataset_big
import torch.utils.data as data_utils
import parser_file as pars
import warnings
warnings.filterwarnings('ignore')

args = pars.parse_arguments()

model_type = args.model_type

if model_type == "linear_ae":
    from linear_ae import *
elif model_type == "conv_ae":
    from convolutional_ae import *
elif model_type == "lstm_ae":
    from lstm_ae import *
elif model_type == "transformer":
    from transformer import *
elif model_type == "lstm":
    from lstm import *


if torch.cuda.is_available():
    device =  torch.device('cuda')
else:
    device = torch.device('cpu')

#### Open the dataset ####
# Original dataset
dataset = args.train_dataset
production_df = pd.read_csv(dataset)
production_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
# There are some countries with no values for solar generation, so we can drop them out
no_countries = ['DELU', 'HR', 'HU', 'PL']
only_prod_df = production_df[production_df.country_code.isin(no_countries) == False]

only_prod_df['utc_timestamp'] = pd.to_datetime(only_prod_df.utc_timestamp)
# Proceed with imputing the missing values
# NOTE: given that the time series have a periodic nature (production at zero during the night, then it increases and finally decreases), it makes sense to try to impute the values
# by considering replicating the previous 24 hours
final_prod_df = impute_missing_prod(only_prod_df)



### PREPROCESSING ###
# Split the dataset into train, validation and test
dfs_train, dfs_val, dfs_test = split_big(final_prod_df)
train = dfs_train.reset_index(drop = True)
val = dfs_val.reset_index(drop = True)
test = dfs_test.reset_index(drop = True)

### TRAINING THE MODEL ###
# For training we are going to create an input dataset consisting of overlapping windows of 72 measurements (3 days)
train_window = args.train_window
if model_type == "transformer" or model_type == "lstm":
    X_t, y_t = create_transformer_sequences_big(dfs_train, train_window)
    X_v, y_v = create_transformer_sequences_big(dfs_val, train_window)
    X_te, y_te = create_transformer_sequences_big(dfs_test, train_window)
else:
    X_t = create_test_sequences_big(dfs_train, train_window)
    X_v = create_test_sequences_big(dfs_val, train_window, train_window)
    X_te = create_test_sequences_big(dfs_test, train_window, train_window)


BATCH_SIZE =  args.batch_size
N_EPOCHS = args.epochs
hidden_size = args.hidden_size

# batch_size, window_length, n_features = X_train.shape
batch, window_len, n_channels = X_t.shape

w_size = X_t.shape[1] * X_t.shape[2]
z_size = int(w_size * hidden_size) 

if model_type == "conv_ae" or model_type == "lstm_ae" :
    #Credo di dover cambiare X_train.shape[0], w_size, X_train.shape[2] con X_train.shape[0], X_train.shape[1], X_train.shape[2]
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_v).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_te).float()) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
elif model_type == "transformer" or model_type == "lstm":
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_v).float(), torch.from_numpy(y_v).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
else:
    val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_v).float().reshape(([X_v.shape[0], w_size]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_te).float().reshape(([X_te.shape[0],w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

if model_type == "lstm_ae" or model_type == "conv_ae" or model_type == "lstm":
    z_size = 32

d_model = 64
dim_ff = 256
n_layer = 3
n_head = 4
# Create the model and send it on the gpu device
if model_type == "lstm_ae":
    model = LstmAE(n_channels, z_size, train_window)
elif model_type == "conv_ae":
    model = ConvAE(n_channels, z_size) 
elif model_type == "linear_ae":
    model = LinearAE(w_size, z_size)
elif model_type == "transformer":
    model = Transformer(n_channels, d_model, dim_ff, n_layer, train_window, n_head)
elif model_type == "lstm":
    model = LstmModel(n_channels, z_size, 1)

model = model.to(device)
print(model)


### TESTING THE MODEL ON THE TRAINING DATASET ###
# Recover checkpoint
checkpoint_dir = args.checkpoint_dir
checkpoint = torch.load(checkpoint_dir, map_location = 'cpu')
if model_type == "transformer" or model_type == "lstm":
    model.load_state_dict(checkpoint)
else:
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])


results, w = testing(model, test_loader, device)
# Reconstruction of validation set
results_v, w_v = testing(model, val_loader, device)


res_dir = args.res_dir

reconstruction_test = np.concatenate([torch.stack(w[:-1]).flatten().detach().cpu().numpy(), w[-1].flatten().detach().cpu().numpy()])
reconstruction_val = np.concatenate([torch.stack(w_v[:-1]).flatten().detach().cpu().numpy(), w_v[-1].flatten().detach().cpu().numpy()])
print(len(reconstruction_test))
print(len(reconstruction_val))
dim_val = X_v.shape[0] * X_v.shape[1]
dim_test = X_te.shape[0] * X_te.shape[1]
if model_type == "transformer" or model_type == "lstm":
    predicted_df_val = get_transformer_dataset_big(val, reconstruction_val, train_window)
    predicted_df_test = get_transformer_dataset_big(test, reconstruction_test, train_window)
else:
    predicted_df_val = get_predicted_dataset_big(val, reconstruction_val)
    predicted_df_test = get_predicted_dataset_big(test, reconstruction_test)

threshold_method = args.threshold
percentile = args.percentile

print("Method: ", threshold_method)

predicted_df_test = anomaly_detection(predicted_df_val, predicted_df_test, threshold_method, percentile)

### Parallel with synthetically generated anomalies on the data
if args.synthetic_generation:
    contamination = args.contamination
    period = args.period
    anom_amplitude_factor = args.anom_amplitude_factor
    synthetic_df = generate_anomalous_dataset(predicted_df_test, contamination, period, anom_amplitude_factor)
    synthetic_val = generate_anomalous_dataset(predicted_df_val, contamination, period, anom_amplitude_factor)
    if model_type == "transformer" or model_type == "lstm":
        X_s, y_s = create_synthetic_transformer_sequences_big(synthetic_df, train_window)
        synth_val, synth_y_val = create_synthetic_transformer_sequences_big(synthetic_val, train_window)
    else:
        X_s = create_synthetic_sequences_big(synthetic_df, train_window, train_window)
        synth_val = create_synthetic_sequences_big(synthetic_val, train_window, train_window)

    if model_type == "conv_ae" or model_type == "lstm_ae":
        synth_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_s).float()) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        synth_val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(synth_val).float()) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    elif model_type == "transformer" or model_type == "lstm":
        synth_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_s).float(), torch.from_numpy(y_s).float()) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        synth_val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(synth_val).float(), torch.from_numpy(synth_y_val).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    else:
        synth_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_s).float().view(([X_s.shape[0], w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        synth_val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(synth_val).float().view(([synth_val.shape[0], w_size]))) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    
    res_s, w_s = testing(model, synth_loader, device)
    r_s = np.concatenate([torch.stack(w_s[:-1]).flatten().detach().cpu().numpy(), w_s[-1].flatten().detach().cpu().numpy()])
    res_val_s, w_val_s = testing(model, synth_val_loader, device)
    r_val_s = np.concatenate([torch.stack(w_val_s[:-1]).flatten().detach().cpu().numpy(), w_val_s[-1].flatten().detach().cpu().numpy()])
    dim_s = X_s.shape[0] * X_s.shape[1]
    if model_type == "transformer" or model_type == "lstm":
        df_s = get_predicted_synthetic_transformer_dataset_big(synthetic_df, r_s, train_window)
        df_val_s = get_predicted_synthetic_transformer_dataset_big(synthetic_val, r_val_s, train_window)
    else:
        df_s = get_predicted_synthetic_dataset_big(synthetic_df, r_s)
        df_val_s = get_predicted_synthetic_dataset_big(synthetic_val, r_val_s)

    preds_s = anomaly_detection(df_val_s, df_s, threshold_method, percentile)

    print(classification_report(preds_s.synthetic_anomaly, preds_s.predicted_anomaly))
    print(roc_auc_score(preds_s.synthetic_anomaly, preds_s.predicted_anomaly))
    anomalized_df = preds_s[preds_s.synthetic_anomaly == 1]
    print(classification_report(anomalized_df.synthetic_anomaly, anomalized_df.predicted_anomaly))