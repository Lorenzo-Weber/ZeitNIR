import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from cnnClf import CnnCLF
from spectraNet import SpectraNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('data/dados_nir.csv')
X = data.iloc[:, 10:].values  
y = data.iloc[:, :10].values

imputer = SimpleImputer(strategy='median')
y_imputed = imputer.fit_transform(y)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y_imputed)

X_train_f, X_test, y_train_f, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_f, y_train_f, test_size=0.1, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
dataset_val = TensorDataset(X_val_tensor, y_val_tensor)

dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)

model = CnnCLF().to(device)
opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()

def r2_score(y_true, y_pred):
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2.item() 

# epochs = 65
# for epoch in range(epochs):
#     model.train()
#     run_loss = 0.0
#     run_r2 = 0.0

#     for inputs, labels in dataloader_train:
#         inputs, labels = inputs.to(device), labels.to(device)

#         opt.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         opt.step()

#         run_loss += loss.item()  
#         run_r2 += r2_score(labels, outputs)  

#     avg_loss = run_loss / len(dataloader_train)
#     avg_r2 = run_r2 / len(dataloader_train)

#     print(f"Epoch [{epoch+1}/{epochs}], Loss (MSE): {avg_loss:.4f}, R² Score: {avg_r2:.4f}")

spectra = SpectraNet().to(device)
opt = optim.Adam(spectra.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()

epochs = 65
for epoch in range(epochs):
    spectra.train()
    run_loss = 0.0
    run_r2 = 0.0

    for inputs, labels in dataloader_train:
        inputs, labels = inputs.to(device), labels.to(device)

        opt.zero_grad()
        outputs = spectra(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()

        run_loss += loss.item()  
        run_r2 += r2_score(labels, outputs)  

    avg_loss = run_loss / len(dataloader_train)
    avg_r2 = run_r2 / len(dataloader_train)

    print(f"Epoch [{epoch+1}/{epochs}], Loss (MSE): {avg_loss:.4f}, R² Score: {avg_r2:.4f}")

