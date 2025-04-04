import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SpectraNet(nn.Module):
    def __init__(self, input_size = None):
        super(SpectraNet, self).__init__()

        self.c1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=8)
        self.b1 = nn.BatchNorm1d(16)
        self.m1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.c2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16)
        self.b2 = nn.BatchNorm1d(num_features=32)
        self.m2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.f1 = nn.Flatten()
        self.d1 = nn.Dropout(0.4)
        self.l1 = nn.Linear(4512, 1024)
        self.b3 = nn.BatchNorm1d(1024)

        self.l2 = nn.Linear(1024, 256)
        self.b4 = nn.BatchNorm1d(256)
        self.d2 = nn.Dropout(0.4)
        
        self.out = nn.Linear(256, 10) 

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.elu(self.b1(self.c1(x)))
        x = self.m1(x)
        x = F.elu(self.b2(self.c2(x)))
        x = self.m2(x)
        x = self.f1(x)
        x = self.d1(x)
        x = F.elu(self.b3(self.l1(x)))
        x = F.elu(self.b4(self.l2(x)))
        x = self.d2(x)
        return self.out(x)