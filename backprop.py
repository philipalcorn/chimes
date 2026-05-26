# First we must know the chain rule:
# d/dx f(g) = f'(g) * g'(x) , usually using partials for multi-dimensions

# Then we must know the computational graph
# at each node an operation is applied with some inputs 
# Each node can store a "local gradients" which are easy to caluclate. 
# Usually at the very end of the graph a scalar loss function is defined,
# and we need the gradient of L wrt all inputs. These partial gradients 
# Can be used with the chain rule to easily "back propogate" through the 
# graph to get the final gradient. 

# Forward pass - Compute loss
# ----> computing local gradients as we go

# Backward pass - Compute gradient of loss wrt parameters using chain rule

# Perform linear regression to minimize loss 
# 1. fwd pass. 2. local gradients 3. final gradient with bkwd pass 
# Note we don't need to know derivatives releative to inputs we're not 
# concerned with 


import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True) # with weight we are interested in gradient

y_hat = w*x 
loss = (y_hat - y)**2

print(loss)
loss.backward() # does the backprop and gets gradient
print(w.grad) # This is the first gradient after the first fwd and backward pass 

# now we update the weights 
# and do the next fwd / backward pass  

