# Gradient descent - used to optimize model with autograd 

# manual prediction -- now using pytorch model
# manual gradient computation -- now using autograd
# manual loss computation -- now use Pytorch Loss 
# manual parameter optimization -- now use PyTorch Optimizer

import torch 

# f = w * x             function = weight * input
# f = 2*x 
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32) 
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32) 


# initial weight
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) 

# Model prediction -- manually
def forward(x):
    return w*x

# loss  -- manually (in this case mean squared error for linear regression)
def loss(y, y_predicted):
   return ((y_predicted-y)**2).mean() 

# gradient --- automatic
print(f'Prediction before training: f(5) = {forward(5):.3f}')

learning_rate = 0.01
n_iter=100

# training loop
for epoch in range(n_iter):
    # prediction = fwd pass 
    y_predicted = forward(X)

    #loss 
    l = loss(Y, y_predicted)

    # gradients
    # dw = gradient(X, Y, y_predicted) -- old manual
    l.backward() # -- new automatic gradient dl/dw 
    
    # update weights -- note we don't track this in the autograd graph
    with torch.no_grad():
        w -= learning_rate * w.grad # w.grad is that specific gradient dl/dw
        
    # empty the gradients for the new loop
    w.grad.zero_() # modify in place 

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w={w:.3f}, loss = {l:.8f}')


print(f'Prediction after training: f(5) = {forward(5):.3f}')
