import torch
import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self, in_size, latent_size):
    super().__init__()
    self.linear1 = nn.Linear(in_size, int(in_size/2))
    self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
    self.linear3 = nn.Linear(int(in_size/4), latent_size)
    self.relu = nn.ReLU(True)
        
  def forward(self, w):
    out = self.linear1(w) 
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    z = self.relu(out)
    return z
    
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size):
    super().__init__()
    self.linear1 = nn.Linear(latent_size, int(out_size/4))
    self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
    self.linear3 = nn.Linear(int(out_size/2), out_size)
    self.relu = nn.ReLU(True)
    self.sigmoid = nn.Sigmoid()
        
  def forward(self, z):
    out = self.linear1(z)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    w = self.sigmoid(out)
    return w
    
class LinearAE(nn.Module):
  def __init__(self, w_size, z_size):
    super().__init__()
    self.z_size = 9
    self.encoder = Encoder(w_size, self.z_size)
    self.decoder = Decoder(self.z_size, w_size) 

  def forward(self, x):
    out = self.encoder(x)
    out = self.decoder(out)
    return out

    

def training(epochs, model, train_loader, val_loader, device, opt_func=torch.optim.Adam): 
    history = []
    optimizer = opt_func(list(model.encoder.parameters())+list(model.decoder.parameters()))
    criterion = nn.MSELoss().to(device)
    for epoch in range(epochs):
        train_loss = []
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device) 
            y_batch = y_batch.to(device) 

            optimizer.zero_grad() 

            #Train AE
            w = model(X_batch)
            loss = criterion(w, y_batch)
            train_loss.append(loss)
            loss.backward()
            optimizer.step()           
        result_train = torch.stack(train_loss).mean().item()
        model.eval()
        val_loss = []
        for X_batch, y_batch in val_loader:
          X_batch = X_batch.to(device) 
          y_batch = y_batch.to(device) 
          with torch.no_grad():
            w = model(X_batch)
            loss = criterion(w, y_batch)
          val_loss.append(loss)

        result_val = torch.stack(val_loss).mean().item()

        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(epoch, result_train, result_val))
        history.append((result_train, result_val))
    return history
    
def testing(model, test_loader, device):
    results=[]
    reconstruction = []
    with torch.no_grad():
        for [batch] in test_loader:
            batch = batch.to(device)
            w=model.decoder(model.encoder(batch))
            results.append(torch.mean((batch-w)**2,axis=1))
            reconstruction.append(w)
    return results, reconstruction

