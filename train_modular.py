from preprocessing import create_transformer_sequences_big, impute_missing_prod, split_big, create_sequences_big
import pandas as pd
import torch
import torch.utils.data as data_utils
from postprocessing import *
import torch.utils.data as data_utils
import parser_file
import warnings
warnings.filterwarnings('ignore')



### PREPROCESSING ON THE DATASET ###
def prepare_dataset(dataset, model_type, train_window, BATCH_SIZE, hidden_size):
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
    dfs_train, dfs_val, dfs_test = split_big(final_prod_df)
    train = dfs_train.reset_index(drop = True)
    val = dfs_val.reset_index(drop = True)
    test = dfs_test.reset_index(drop = True)
    if model_type == "transformer" or model_type == "lstm":
        X_t, y_t = create_transformer_sequences_big(dfs_train, train_window)
        X_v, y_v = create_transformer_sequences_big(dfs_val, train_window)
        X_te, y_te = create_transformer_sequences_big(dfs_test, train_window)
    else:
        X_t = create_sequences_big(dfs_train, train_window)
        X_v = create_sequences_big(dfs_val, train_window)
        X_te = create_sequences_big(dfs_test, train_window, train_window)
    batch, window_len, n_channels = X_t.shape

    w_size = X_t.shape[1] * X_t.shape[2]
    z_size = int(w_size * hidden_size) 

    if model_type == "conv_ae" or model_type == "lstm_ae" :
        train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_t).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
        val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_v).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    elif model_type == "transformer" or model_type == "lstm":
        train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_t).float(), torch.from_numpy(y_t).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
        val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_v).float(), torch.from_numpy(y_v).float()), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    else:
        train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_t).float().reshape(([X_t.shape[0], w_size])), torch.from_numpy(X_t).float().reshape(([X_t.shape[0], w_size]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
        val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(torch.from_numpy(X_v).float().view(([X_v.shape[0], w_size])), torch.from_numpy(X_v).float().view(([X_v.shape[0], w_size]))), batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    return train_loader, val_loader, w_size, z_size, n_channels

def train(model_type, N_EPOCHS, train_window, train_loader, val_loader, w_size, z_size, n_channels, checkpoint_path):
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

    print(device)
    model = model.to(device) 
    print(model)

    # Start training
    history = training(N_EPOCHS, model, train_loader, val_loader, device)
    print(history)
    
    
    if model_type == "lstm_ae" or model_type == "linear_ae" or model_type == "conv_ae":
        torch.save({
                'encoder': model.encoder.state_dict(),
                'decoder': model.decoder.state_dict()
                }, checkpoint_path) # the path should be set in the run.job file
    else:
        torch.save(model.state_dict(), checkpoint_path)

if __name__ == "__main__":
    args = args = parser_file.parse_arguments()

    model_type = args.model_type
    dataset = args.train_dataset
    train_window = args.train_window
    BATCH_SIZE =  args.batch_size
    N_EPOCHS = args.epochs
    hidden_size = args.hidden_size
    checkpoint_path = args.save_checkpoint_dir

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

    train_loader, val_loader, w_size, z_size, n_channels = prepare_dataset(dataset, model_type, train_window, BATCH_SIZE, hidden_size)
    train(model_type, N_EPOCHS, train_window, train_loader, val_loader, w_size, z_size, n_channels, checkpoint_path)

    
