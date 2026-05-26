import torch
x=torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
L = (x**3).sum()                # L = x_0^3 + x_1^3 + x_2^2
L.backward()

print(x.grad)                   # tensor([12, 27, 48])

# note that partial L / partial x_i = 3x_i^2, so [3(4), 3(9), 3(16)] is 
# expected.

# So this means the gradient is calculated at the original values of x

# note that it MUST be reduced to a scalar. 

