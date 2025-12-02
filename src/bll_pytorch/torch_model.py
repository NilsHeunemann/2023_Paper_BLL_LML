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

        self.inputs: int = architecture[0][1]["in_features"]
        self.outputs: int = architecture[-1][1]["out_features"]
        self.feature_dim: int = architecture[-3][1]["out_features"]
        self.architecture = architecture

    def forward(self, x):
        features = self.hidden(x)
        output = self.last_layer(features)
        return features, output

    def get_save_dict(self):
        state_dict = {
            "architecture": self.architecture,
            "model_state_dict": self.state_dict(),
        }
        return state_dict
    
    @classmethod
    def from_dict(cls, save_dict):
        model = cls(save_dict["architecture"])
        model.load_state_dict(save_dict["model_state_dict"])
        return model


class sin_activation(nn.Module):
    def __init__(self):
        super(sin_activation, self).__init__()

    def forward(self, x):
        return torch.sin(x)
