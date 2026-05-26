# Autograd is pytorch's automatic differentiation engine 
# It makes neural networks possible by computing gradients automatically

# PyTorch records the operations you do on tensors and can compute derivatives
# by working through them backwards, known as reverse-mide automatic 
# differentiation.  It does this by building a dynamic computation graph 
# as the code runs, tracking how each tensor is produced from other tensors.

# When .backward() is called it traverses this graph in reverse, applying the 
# chain rule at each step. 


##### Tracking gradients #####
# only happens  if 'requires_grad==True' :

import torch
x=torch.tensor([2.0, 3.0], requires_grad=True)
y = x**2
z = y.sum()

z.backward()
print(x.grad) #tensor ([4., 6., dz/dx = 2x)

##### Computation Graph #####
# Each tensor that resuls from an operation stores a reference to a
# 'grad_fn', which is the function that knows how to compute 
# the gradient of that operation
x=torch.tensor(2.0, requires_grad=True)
y = x**2
print(y.grad_fn)

##### Leaf Tensors vs. Intermediate Tensors #####
# Leaf tensors are at the edge of the graph, created directly.
# after .backward(), gradients are accumulated only into leaf tensors' 
# .grad attribute by default. Intermediate tensors don't retain their 
# gradients without specifying that behavior. 
x = torch.tensor(2.0, requires_grad=True)
y = x**2
z = y**2
y.retain_grad()
z.backward()
print(x.grad)
print(y.grad)

#####  How Backward Works #####
# backward() computes gradients of a scalar with resspect to graph leaves
# if you call it on a non-scalar tensor, you must supply a gradient argument 
# (that is, the vector for the vector-Jacobian product)
# under the hood it is calculating vector-Jacobian products 
# rather than full Jacobians, which is what makes it so efficient.
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y=x**2
# y is non-scalar, so we provide the upstream gradient
y.backward(torch.tensor([1.0,1.0,1.0]))
print(x.grad)

##### Gradient Accumulation ######
# Gradients accumulate, that is, calling .backward() adds to existing 
# .grad values rather than replacing them. This is why training loops zero
# gradients each iteration:
"""
for inputs, targets in dataloader:
    optimizer.zero_grad()           # Clears old gradients
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()                 # accumulates new gradients
    optimizer.step()                # updates the parameters
"""


##### Disabling Autograd #####
# If gradients are not needed, disable them to save memory and computation
"""
with torch.no_grad():
    predicitons = model(inputs)     # no graph built
"""

##### grad Function Alternatives ######
# instead of .backward, gradients can be computed functionally with
# torch.autograd.grad(), which returns the gradients directly without
# populating .grad
# This is handy when you need gradients as intermediate values, such 
# as when you need higher order derivatives or gradient penalties
x = torch.tensor(2.0, requires_grad=True)
y = x**3
grads = torch.autograd.grad(y,x)
print(grads)

##### higher order Derivatives 
# by setting grate_graph=True, the backwards pass itself becomes 
# a differentiable operation, letting you compute second and higher derivatives
x = torch.tensor(2.0, requires_grad=True)
y = x**3
grad1=torch.autograd.grad(y,x,create_graph=True)[0] # 3x^2 = 12
grad2=torch.autograd.grad(grad1, x)[0]              # 6x = 12
print(grad1)
print(grad2)

##### Custom Autograd Functions ######
# Useful when you need custom forward/backwards behavior, such as 
# a non-differentiable operation to approximate, or a hand-optimized gradient
# use torch.autograd.Function subclass:
"""
class MyReLU(roch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        return grad_input

relu = MyReLU.apply
"""

# the ctx (stands for context) 
# object carries state data from forward to backward
# save_for_backward is the correct way to stash tensors you'll need

###### Miscellaneous Tips ######
# In place operations, those ending with _, can break autograd
# if they overwrite values needed for hte backward pass. PyTorch should 
# raise an error when this happens

# Operations like ReLU at zero use subgradient conventions. Pytorch will 
# pick a reasonable value rather than failing.

# grad is None until .backward

# Be default the graph is freed after .backward. If it is necesssary 
# to backpropogate twice, use retain_graph=True.










