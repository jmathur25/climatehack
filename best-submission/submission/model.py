import torch
import torch.nn as nn

from fastai.vision.all import *

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


class CenterCrop(nn.Module):
    def __init__(self, size=(64, 64)):
        super().__init__()

    def forward(self, x):
        return x[:, :, 32:96, 32:96]


class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=24, weights=None):
        super(UNet, self).__init__()
        self.model = create_unet_model(
            arch=models.resnet50,
            img_size=(128, 128),
            n_out=in_channels,
            pretrained=True,
            n_in=out_channels,
            self_attention=True,
        )

        if weights:
            self.load_state_dict(torch.load(weights))
        self.model.layers.add_module("CenterCrop", CenterCrop())

    def forward(self, x):
        return self.model(x)


class EnsembleNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=24, weights=None):
        super(EnsembleNet, self).__init__()
        self.model = create_unet_model(
            arch=models.resnet50,
            img_size=(128, 128),
            n_out=in_channels,
            pretrained=True,
            n_in=out_channels,
            self_attention=True,
        )

        self.models = models

        assert len(weights) == len(self.models)

        for i, model in enumerate(self.models):
            model.layers.add_module("CenterCrop", CenterCrop())
            self.add_module(f"model_{i}", model)

        if weights:
            self.load_state_dict(torch.load(weights))
        self.model.layers.add_module("CenterCrop", CenterCrop())

    def forward(self, x):

        outputs = []
        for model in self.models:
            outputs.append(model(x))

        return torch.stack(outputs, dim=1).mean(dim=1)
