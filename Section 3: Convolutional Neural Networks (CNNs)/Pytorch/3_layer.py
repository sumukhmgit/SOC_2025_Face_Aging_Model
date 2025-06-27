import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define Linear layers
        self.layer1 = nn.Linear(12, 8)
        self.layer2 = nn.Linear(8, 5)
        self.layer3 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = FeedForwardNN()
print(model)

x=torch.randn(2, 12)
output = model.forward(x)
print(output)