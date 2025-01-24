import torch
import torch.nn as nn

#from utils_ae import *
#device = get_default_device()

class Encoder(nn.Module):
  def __init__(self, in_size, latent_size): 
    super().__init__()
    self.lstm = nn.LSTM(input_size=in_size, hidden_size=latent_size, num_layers=1, batch_first=True, dropout = 0.2
            # input and output tensors are provided as (batch, seq_len, feature(size))
        )
    self.dropout = nn.Dropout(0.2)
  def forward(self, w):
    z, (h_n, c_n) = self.lstm(w)
    h_n = self.dropout(h_n)
    return h_n
    
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size, train_window): 
    super().__init__()
    self.latent_size = latent_size
    self.window = train_window
    out_size = 1
    self.lstm = nn.LSTM(input_size=latent_size, hidden_size=latent_size, num_layers=1, batch_first=True, dropout = 0.2
            # input and output tensors are provided as (batch, seq_len, feature(size))
        )
    self.dropout = nn.Dropout(0.2)
    self.output_layer = nn.Linear(latent_size, out_size)
        
  def forward(self, z):
    batch = z.size()[1]
    z = z.squeeze()
    input = z.repeat(1, self.window)
    input = input.reshape((batch, self.window, self.latent_size))
    w, (h_n, c_n) = self.lstm(input)
    w = self.dropout(w)
    out = self.output_layer(w)
    return out
    
class LstmAE(nn.Module):
  def __init__(self, input_dim, latent_size, train_window): 
    super().__init__()
    self.encoder = Encoder(input_dim, latent_size)
    self.decoder = Decoder(latent_size, input_dim, train_window)
  
  def training_step(self, batch, criterion, n):
    z = self.encoder(batch)
    w = self.decoder(z)
    loss = criterion(w, batch)
    return loss

  def validation_step(self, batch, criterion, n):
    with torch.no_grad():
        z = self.encoder(batch)
        w = self.decoder(z)
        loss = criterion(w, batch)
    return loss
    
  def epoch_end(self, epoch, result, result_train):
    print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(epoch, result_train, result))
    
def evaluate(model, val_loader, criterion, device, n):
    batch_loss = []
    for [batch] in val_loader:
       batch = batch.to(device) 
       loss = model.validation_step(batch, criterion, n) 
       batch_loss.append(loss)

    epoch_loss = torch.stack(batch_loss).mean()
    return epoch_loss


def training(epochs, model, train_loader, val_loader, device, opt_func=torch.optim.Adam): 
    history = []
    optimizer = opt_func(list(model.encoder.parameters())+list(model.decoder.parameters()))
    criterion = nn.L1Loss().to(device)
    for epoch in range(epochs):
        train_loss = []
        for [batch] in train_loader:
            batch = batch.to(device) 
            optimizer.zero_grad()

            loss = model.training_step(batch, criterion, epoch+1)
            train_loss.append(loss)
            loss.backward()
            optimizer.step()
        result_train = torch.stack(train_loss).mean()    
            
        result= evaluate(model, val_loader, criterion, device, epoch+1)
        model.epoch_end(epoch, result, result_train)
        res = result_train.item()
        history.append((res, result.item()))
    return history 
    
def testing(model, test_loader, device):
    results=[]
    reconstruction = []
    criterion = nn.MSELoss().to(device)
    with torch.no_grad():
        for [batch] in test_loader: 
            batch = batch.to(device) 
            w=model.decoder(model.encoder(batch))
            batch_s = batch.reshape(-1, batch.size()[1] * batch.size()[2])
            w_s = w.reshape(-1, w.size()[1] * w.size()[2])
            results.append(torch.mean((batch_s-w_s)**2,axis=1))
            reconstruction.append(w)
    return results, reconstruction
