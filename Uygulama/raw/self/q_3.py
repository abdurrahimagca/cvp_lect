import torch 
import torch.nn as nn

layer = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
)

img = torch.randn(3,25,25)
out = layer(img)
print(out.shape)

