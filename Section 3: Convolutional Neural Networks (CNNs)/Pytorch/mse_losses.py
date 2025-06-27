import torch
import torch.nn as nn

#get some random numbers

y_true =torch.randn(5, 3)
y_pred =torch.randn(5, 3)


#write function here
loss_function=nn.MSELoss()
loss=loss_function(y_pred, y_true)

print("MSE Loss:", loss.item())