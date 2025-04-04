import torch.nn as nn
import torch.nn.functional as F

class CnnCLF(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=4)
        self.b1 = nn.BatchNorm1d(32)

        self.c2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=4)
        self.b2 = nn.BatchNorm1d(32)

        self.m2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.c3 = nn.Conv1d(in_channels= 32, out_channels=32, kernel_size=4)
        self.b3 = nn.BatchNorm1d(32)

        self.c4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8)
        self.b4 = nn.BatchNorm1d(64)

        self.m3 = nn.MaxPool1d(kernel_size=4, stride=2)

        self.f3 = nn.Flatten()

        self.l5 = nn.Linear(in_features=4352, out_features=4352)
        self.b5 = nn.BatchNorm1d(4352)
        self.d5 = nn.Dropout(0.4)

        self.l6 = nn.Linear(in_features=4352, out_features=2048)
        self.b6 = nn.BatchNorm1d(2048)
        self.d6 = nn.Dropout(0.4)

        self.l7 = nn.Linear(in_features=2048, out_features=512)
        self.b7 = nn.BatchNorm1d(512)

        self.out = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))

        x = self.m2(x)

        x = F.relu(self.b3(self.c3(x)))
        x = F.relu(self.b4(self.c4(x)))

        x = self.m3(x)

        x = self.f3(x)

        x = F.relu(self.b5(self.l5(x)))
        x = self.d5(x)

        x = F.relu(self.b6(self.l6(x)))
        x = self.d6(x)

        x = F.relu(self.b7(self.l7(x)))

        x = self.out(x)
        return x