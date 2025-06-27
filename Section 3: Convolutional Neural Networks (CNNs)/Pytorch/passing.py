import torch
import torch.nn as nn

model = nn.Sequential(
    #fillout below
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

# Generate a random input of shape (a,b)
x = torch.randn(1, 3)

# TODO: Pass input through the model
output = model(x)

print("Output shape:", output.shape)