import torch
import torch.nn as nn

# define custom dataset
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, target, features, seq_len = 60):
        self.dataframe = dataframe
        self.target = target
        self.seq_len = seq_len
        self.x = torch.tensor(dataframe[features].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[target].values, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if idx >= self.seq_len - 1:
            idx_start = idx - self.seq_len + 1
            x = self.x[idx_start:idx+1,:]
        else:
            padding = self.x[0].repeat(self.seq_len - idx - 1, 1)
            x = self.x[0:(idx+1),:]
            x = torch.cat((padding, x), dim=0)

        return x, self.y[idx]

# define model
class ShallowRegressionLSTM(nn.Module):
    def __init__(self,n_features, n_hidden, n_layers=1, device='cpu'):
        super(ShallowRegressionLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_size = n_features, 
            hidden_size=n_hidden, 
            num_layers=n_layers, 
            batch_first=True)
        self.linear = nn.Linear(in_features = self.n_hidden, out_features=1)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.n_layers, batch_size, self.n_hidden).requires_grad_().to(self.device)
        c0 = torch.zeros(self.n_layers, batch_size, self.n_hidden).requires_grad_().to(self.device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten() #first dim of hn is the layer dimension
        return out

# define training function and testing function
def train(model, loader, loss, optimizer, device):
    model.train()
    train_loss = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        batch_loss = loss(y_pred, y)
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss.item()

    avg_loss = train_loss / len(loader)
    print(f"Train loss: {avg_loss:.4f}")
    
def test(model, loader, loss, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            batch_loss = loss(y_pred, y)
            test_loss += batch_loss.item()

    avg_loss = test_loss / len(loader)
    print(f"Test loss: {avg_loss:.4f}")