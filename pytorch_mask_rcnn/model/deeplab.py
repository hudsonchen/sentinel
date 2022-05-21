import torch
from torch import nn


class deeplab(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.original_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)

        self.original_model.aux_classifier[4] = torch.nn.Conv2d(256, n_class, kernel_size=(1, 1), stride=(1, 1))
        self.original_model.classifier[4] = torch.nn.Conv2d(256, n_class, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        output = self.original_model(x)['out']
        return output
