import torch
import torch.nn as nn

# Hint: First layer input 3 → hidden 5, then hidden 5 → output 2
model = nn.Sequential(
    #fillout below
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

print(model)