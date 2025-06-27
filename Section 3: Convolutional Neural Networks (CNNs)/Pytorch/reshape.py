import torch

#take some random integers
a=2
b=3
c=4
#every variable can be assumed random
x = torch.randn(a, b, c)

# TODO: Flatten to shape (a1,a2)
viewed = x.view(a, b*c)

# TODO: Permute to shape (b1,b2)
permuted = x.permute(2,1,0)

print("Viewed shape:", viewed.shape)
print("Permuted shape:", permuted.shape)