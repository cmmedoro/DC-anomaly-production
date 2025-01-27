import torch
import parser_file as pars
import onnx

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

BATCH_SIZE =  args.batch_size
N_EPOCHS = args.epochs
hidden_size = args.hidden_size
train_window = args.train_window
# Create a dummy dataset to give to the onnx model just as a placeholder for the input shapes
X_train = torch.rand(BATCH_SIZE, train_window, 1)
batch, window, n_channels = X_train
w_size = X_train.shape[1] * X_train.shape[2]
z_size = int(w_size * hidden_size) 

if model_type == "linear_ae":
    X_train = torch.rand(BATCH_SIZE, z_size)
else:
    X_train = torch.rand(BATCH_SIZE, train_window, 1)

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

# Recover checkpoint
checkpoint_dir = args.checkpoint_dir
checkpoint = torch.load(checkpoint_dir)
if model_type == "transformer" or model_type == "lstm":
    model.load_state_dict(checkpoint)
else:
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])

model.eval()
onnx_file = args.onnx_checkpoint
# Export model
torch.onnx.export(
    model,                           # model
    X_train.to(device),                         # dummy input
    onnx_file,                       # Output file name
    export_params=True,              # Export weights
    opset_version=11,                # ONNX version (>=11)
    do_constant_folding=True,        # Optimize constants during export
    input_names=["input"],           # Input tensor name
    output_names=["output"],         # Output tensor name
    dynamic_axes={                   # Specify dynamic axes (ex. variable batch size)
        "input": {0: "batch_size"},  # Input 0 axis is dynamic
        "output": {0: "batch_size"}  # Output 0 axis is dynamic
    }
)

#Verify model

# Carica il modello ONNX
onnx_model = onnx.load("model.onnx")

# Verifica il modello
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))
"""
#Verify model
import onnx
# Carica il modello ONNX
onnx_model = onnx.load("model.onnx")

# Verifica il modello
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))

import os
import torch
from lib import model as model_lib
from lib import utils
from lib.miner import MVAusgrid
import torch.utils.data as data

def export_model_to_onnx(checkpoint_path, onnx_filename, data_root, batch_size=1):
    # Load the trained model from the checkpoint
    model = model_lib.MPVForecaster.load_from_checkpoint(checkpoint_path)

    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor with the correct shape
    # You can adjust the shape based on the model's expected input dimensions.
    # For MATNet, the input might be a tuple of tensors (x1, x2, x3)
    batch_size = 1
    seq_len_in = model.hparams.model_kwargs['n_steps_in']
    seq_len_out = model.hparams.model_kwargs['n_steps_out']
    pv_features = model.hparams.model_kwargs['pv_features']
    hw_features = model.hparams.model_kwargs['hw_features']
    fw_features = model.hparams.model_kwargs['fw_features']

    x1 = torch.randn(batch_size, seq_len_in, pv_features)  # dummy PV production input
    x2 = torch.randn(batch_size, seq_len_in, hw_features)  # dummy historical weather input
    x3 = torch.randn(batch_size, seq_len_out, fw_features)  # dummy forecast weather input
    # Define the input tuple
    dummy_input = (x1, x2, x3)


    # Load a sample from the MVAusgrid dataset
    dataset = MVAusgrid(root=data_root, train=False, win_length=24, step=24, time_horizon=24)  # Test dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    test_set = MVAusgrid(root="Data", train=False, plants=None, max_kwp=True, win_length=24, step=24,
                               time_horizon=24, normalize='min-max', scaler=None, eps=1e-5,
                               swx_on=True, fwx_on=True, hour_on=True, day_on=True, month_on=True,
                               plant=None)
    
    # self, root="./Data", train=True, plants=None, max_kwp=True, win_length=336, step=24, time_horizon=24,
                #  normalize='min-max', scaler=None, eps=1e-5, pv_on=True, swx_on=True, fwx_on=True, hour_on=True,
                #  day_on=True, month_on=True, plant=None

    # Create a PyTorch DataLoader for the test set
    test_loader = data.DataLoader(test_set, batch_size=len(test_set), shuffle=False, drop_last=False, num_workers=0)

    # Get one batch of data
    for batch in dataloader:
        (pv_input, hw_input, fw_input), _ = batch
        break  # We only need the first batch

    # Define the input tuple
    real_input = (pv_input, hw_input, fw_input)



    # Export the model to ONNX format
    torch.onnx.export(
        model,                          # the model being exported
        dummy_input,                    # model input (or a tuple of inputs)
        onnx_filename,                  # the file path where the model will be saved
        export_params=True,             # store the trained parameter weights
        opset_version=14,               # the ONNX version to use
        input_names=['pv_input', 'hw_input', 'fw_input'],  # input names
        output_names=['output'],        # output names
        dynamic_axes={
            'pv_input': {0: 'batch_size'},    # variable batch size for pv input
            'hw_input': {0: 'batch_size'},    # variable batch size for historical weather input
            'fw_input': {0: 'batch_size'},    # variable batch size for forecast weather input
            'output': {0: 'batch_size'}       # variable batch size for output
        }
    )
    print(f"Model successfully exported to {onnx_filename}")


def test_model(checkpoint_path,data_root):

    model = model_lib.MPVForecaster.load_from_checkpoint(checkpoint_path)
    model.eval()

    batch_size = 1
    seq_len_in = model.hparams.model_kwargs['n_steps_in']
    seq_len_out = model.hparams.model_kwargs['n_steps_out']
    pv_features = model.hparams.model_kwargs['pv_features']
    hw_features = model.hparams.model_kwargs['hw_features']
    fw_features = model.hparams.model_kwargs['fw_features']

    x1 = torch.randn(batch_size, seq_len_in, pv_features)  # dummy PV production input
    x2 = torch.randn(batch_size, seq_len_in, hw_features)  # dummy historical weather input
    x3 = torch.randn(batch_size, seq_len_out, fw_features)  # dummy forecast weather input
    # Define the input tuple
    dummy_input = (x1, x2, x3)

    print("DUMMY INPUT")
    print(x1.shape, x2.shape, x3.shape)

    output = model(x1, x2, x3)
    print(output.shape)

    # Load a sample from the MVAusgrid dataset
    dataset = MVAusgrid(root=data_root, train=False, win_length=24, step=24, time_horizon=24)  # Test dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Get one batch of data
    for batch in dataloader:
        (pv_input, hw_input, fw_input), _ = batch
        break  # We only need the first batch
  

    print("REAL INPUT")
    # Define the input tuple
    real_input = (pv_input, hw_input, fw_input)
    print(pv_input.shape, hw_input.shape, fw_input.shape)

    output = model(pv_input, hw_input, fw_input)
    print(output.shape)

    return 

if __name__ == "__main__":
    checkpoint_path = "./saved_models/MATNet/MHAMPVNet_adaptive-concat_hour_on_day_on_month_on_wx_history_wx_forecast_pv_forecast_24_24_24/lightning_logs/version_735634/checkpoints/last.ckpt"
    onnx_filename = "production-short-term.onnx"
    data_root = "./Data"
    export_model_to_onnx(checkpoint_path, onnx_filename, data_root=data_root)
    test_model(checkpoint_path,data_root)


"""