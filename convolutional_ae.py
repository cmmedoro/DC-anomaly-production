import torch
import torch.nn as nn


class Encoder(nn.Module):
  def __init__(self, n_features, latent_size):
    super().__init__()
    self.conv1 = nn.Conv1d(in_channels=n_features, out_channels= latent_size, kernel_size=7, padding=3, stride=2)
    self.conv2 = nn.Conv1d(in_channels=latent_size, out_channels= latent_size//2, kernel_size=7, padding=3, stride=2)
    self.conv3 = nn.Conv1d(in_channels=latent_size//2, out_channels= latent_size//4, kernel_size=7, padding=3, stride=2)
    self.relu = nn.ReLU(True)
    self.dropout = nn.Dropout(p=0.2)
  def forward(self, w):
    out = self.conv1(w.permute(0, 2, 1)) 
    out = self.relu(out)
    out = self.dropout(out)
    out = self.conv2(out)
    out = self.relu(out)
    out = self.conv3(out)
    z = self.relu(out)
    return z
    
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size):
    super().__init__()
    self.conv1 = nn.ConvTranspose1d(latent_size//4, latent_size//2, 7, 2, 3, 1) 
    self.conv3 = nn.ConvTranspose1d(latent_size//2, latent_size, 7, 2, 3, 1)
    self.conv4 = nn.ConvTranspose1d(latent_size, 1, 7, 2, 3, 1) 
    self.relu = nn.ReLU(True)
    self.dropout = nn.Dropout(p=0.2)
    self.sigmoid = nn.Sigmoid()
        
  def forward(self, z):
    out = self.conv1(z)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.conv3(out)
    out = self.relu(out)
    out = self.conv4(out) 
    w = self.sigmoid(out)
    return w.permute(0, 2, 1)
    
class ConvAE(nn.Module):
  def __init__(self, input_dim, latent_size): 
    super().__init__()
    self.encoder = Encoder(input_dim, latent_size)
    self.decoder = Decoder(latent_size, input_dim)
  
  def training_step(self, batch, criterion, n):
    z = self.encoder(batch)
    w = self.decoder(z)
    batch_n = batch[:, :, 0].unsqueeze(-1)
    loss = criterion(w, batch_n)
    return loss

  def validation_step(self, batch, criterion, n):
    with torch.no_grad():
        z = self.encoder(batch)
        w = self.decoder(z)
        batch_n = batch[:, :, 0].unsqueeze(-1)
        loss = criterion(w, batch_n)
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
    criterion = nn.MSELoss().to(device)
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
            
        result = evaluate(model, val_loader, criterion, device, epoch+1) 
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
            batch_s = batch[:, :, 0]
            batch_s = batch_s.reshape(batch.size()[0], batch.size()[1], 1)
            w_s = w
            results.append(torch.mean((batch_s-w_s)**2,axis=1))
            reconstruction.append(w)
    return results, reconstruction
