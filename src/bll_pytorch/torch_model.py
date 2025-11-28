import torch
from torch import nn


class JointModel(nn.Module):
    def __init__(self, architecture):
        super(JointModel, self).__init__()
        self.hidden = nn.Sequential(
            *(layer(**specs) for layer, specs in architecture[:-1])
        )
        self.last_layer = architecture[-1][0](
            **architecture[-1][1]
        )  # Last layer without activation

        self.inputs = architecture[0][1]["in_features"]
        self.outputs = architecture[-1][1]["out_features"]
        self.feature_dim = architecture[-3][1]["out_features"]

    def forward(self, x):
        features = self.hidden(x)
        output = self.last_layer(features)
        return features, output


class sin_activation(nn.Module):
    def __init__(self):
        super(sin_activation, self).__init__()

    def forward(self, x):
        return torch.sin(x)
