import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from cnnClf import CnnCLF
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
dataset_val = TensorDataset(X_val_tensor, y_val_tensor)
dataset_test = TensorDataset(X_test_tensor, y_test_tensor)

dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

model = CnnCLF()
opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()

epochs = 50

for epoch in range(epochs):
    model.train()
    run_loss = 0.0

    for inputs, labels in dataloader_train:
        opt.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        opt.step()
        run_loss += loss

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {run_loss / len(dataloader_train):.4f}")


# TODO: 
# - Testar o modelo (alem da loss)
# - RMSE Loss and R2 Loss
# - Tensorgraphs (mostrar as configs com diferentes learning rates)
# - README.md
# - Grid Search CV para mostrar diferentes redes (umas 3 - 4 configs)
# - Fine tune com arquiteturas do Hands On (talvez a xception)