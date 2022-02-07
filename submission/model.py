import torch
import torch.nn as nn

#########################################
#       Improve this basic model!       #
#########################################


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(in_features=12 * 128 * 128, out_features=256)
        self.layer2 = nn.Linear(in_features=256, out_features=256)
        self.layer3 = nn.Linear(in_features=256, out_features=24 * 64 * 64)

    def forward(self, features):
        x = features.view(-1, 12 * 128 * 128) / 1024.0
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))

        return x.view(-1, 24, 64, 64) * 1024.0
