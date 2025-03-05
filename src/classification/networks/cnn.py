# Third party imports
import torch
import torch.nn as nn

# My imports
from .tools import freeze_layers, unfreeze_layers


class FMRINet(nn.Module):
    def __init__(self, f1=8, froi=246, f2=16, d=2, p=0.2, n_classes=7):
        """
        :param number_of_classes: number of classes to classify
        :param f1: number of temporal filters
        :param f2: number of pointwise filters f2 = f1 * D
        :param d: number of spatial filters to learn within each temporal convolution
        :param kernel_length: length of temporal convolution in first layer
        :param p: dropout rate
        """
        super(FMRINet, self).__init__()
        self.num_classes = n_classes

        self.layer1_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=f1, kernel_size=(1, 4)),
            nn.BatchNorm2d(num_features=f1, momentum=0.01, affine=True, eps=1e-3),
            nn.Conv2d(in_channels=f1, out_channels=f1 * d, groups=f1, kernel_size=(froi, 1)),
            nn.BatchNorm2d(num_features=f1 * d, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 3)),
            nn.Dropout2d(p=p),
        )

        self.layer1_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=f1, kernel_size=(1, 6), padding=(0, 1)),
            nn.BatchNorm2d(num_features=f1, momentum=0.01, affine=True, eps=1e-3),
            nn.Conv2d(in_channels=f1, out_channels=f1 * d, groups=f1, kernel_size=(froi, 1)),
            nn.BatchNorm2d(num_features=f1 * d, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 3)),
            nn.Dropout2d(p=p),
        )

        self.layer1_3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=f1, kernel_size=(1, 10), padding=(0, 3)),
            nn.BatchNorm2d(num_features=f1, momentum=0.01, affine=True, eps=1e-3),
            nn.Conv2d(in_channels=f1, out_channels=f1 * d, groups=f1, kernel_size=(froi, 1)),
            nn.BatchNorm2d(num_features=f1 * d, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 3)),
            nn.Dropout2d(p=p),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=f1 * d * 3, out_channels=f1 * d * 3, groups=f1 * d, kernel_size=(1, 4), bias=False),
            nn.Conv2d(in_channels=f1 * d * 3, out_channels=f2, kernel_size=(1, 1), stride=(1, 1), bias=False,
                      padding=(0, 0)),
            nn.BatchNorm2d(num_features=f2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 3)),
            nn.Dropout2d(p=p),
        )

        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=self.num_classes),
            nn.Softmax(dim=1),
        )

        self.number_of_layers = 5

    def forward(self, x):
        x_1 = self.layer1_1(x)
        x_2 = self.layer1_2(x)
        x_3 = self.layer1_3(x)
        x = torch.cat((x_1, x_2, x_3), dim=1)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def freeze_feature_layers(self):
        freeze_layers(self)

    def unfreeze_feature_layers(self):
        unfreeze_layers(self)


if __name__ == '__main__':
    pass
