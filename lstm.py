import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#from utils_ae import *
#device = get_default_device()


class LstmModel(nn.Module):
  def __init__(self, in_size, latent_size, out_size): 
    super().__init__()
    """
    in_size: number of features in input
    latent_size: size of the latent space of the lstm
    Ex. in_size = 5, latent_size = 50
    """
    self.lstm = nn.LSTM(input_size=in_size, hidden_size=latent_size, num_layers=1, batch_first=True, dropout = 0.2
            # input and output tensors are provided as (batch, seq_len, feature(size))
        )
    self.dropout = nn.Dropout(0.2)
    self.relu = nn.ReLU()
    self.fc = nn.Linear(latent_size, out_size) #for swat: instead of 1 put in_size

  def forward(self, w):
    z, (h_n, c_n) = self.lstm(w)
    forecast = z[:, -1, :]
    forecast = self.relu(forecast)
    forecast = self.dropout(forecast)
    output = self.fc(forecast)
    return output
  

def training(epochs, model, train_loader, val_loader, device, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters())
    criterion = nn.MSELoss().to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for X_batch, y_batch in train_loader:
          X_batch = X_batch.to(device)
          y_batch = y_batch.to(device) 

          z = model(X_batch)
          loss = criterion(z.squeeze(), y_batch)
          train_loss.append(loss)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        result_train = torch.stack(train_loss).mean()

        model.eval()
        batch_loss = []
        for X_batch, y_batch in val_loader:
          X_batch = X_batch.to(device) 
          y_batch = y_batch.to(device) 
          with torch.no_grad():
            z = model(X_batch)
            loss = criterion(z.squeeze(), y_batch)
          batch_loss.append(loss)

        result = torch.stack(batch_loss).mean()

        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(epoch, result_train, result))


        history.append(result_train)
    return history
    
def testing(model, test_loader, device):
    results=[]
    forecast = []
    criterion = nn.MSELoss().to(device) 
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device) 
            y_batch = y_batch.to(device) 
            w=model(X_batch)
            
            results.append(torch.mean((y_batch.unsqueeze(1)-w)**2,axis=1)) 
            forecast.append(w)
    return results, forecast

def testing_substitution(model, test, train_window, threshold, device):
   """
   Idea: instead of testing in the traditional way, at inference time we evaluate for each forecasted data point whether it can
   be associated to an anomaly or not. If we would predict an anomaly, we then proceed by substituting the meter_reading value
   with the predictions just made.
   """
   predictions = []
   forecasts = []
   scaler = MinMaxScaler(feature_range=(0,1))
   for building_id, gdf in test.groupby("building_id"):
      gdf[['meter_reading']] = scaler.fit_transform(gdf[['meter_reading']])
      building_data = np.array(gdf[['meter_reading']]).astype(float) 
      for i in range(len(building_data)):
        # find the end of this sequence
        end_ix = i + train_window
        # check if we are beyond the dataset length for this building
        if end_ix > len(building_data)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = building_data[i:end_ix, :], building_data[end_ix, 0]
        gt_y = torch.from_numpy(np.array(seq_y)).float()
        n_feat = seq_x.shape[1]
        window = seq_x.reshape(-1, train_window, n_feat)
        window1 = torch.from_numpy(window).float().to(device)
        next_ts = model(window1)
        if np.abs(next_ts.item() - gt_y) >= threshold * next_ts.item():
           # This means I predict an anomaly
           predictions.append(1)
           # Need to substitute
           building_data[end_ix] = next_ts.item()
        else:
           predictions.append(0)  
        forecasts.append(next_ts.item())
   return predictions
               
   
