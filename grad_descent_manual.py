# Gradient descent - used to optimize model with autograd 

# manual prediction -- now using pytorch model
# manual gradient computation -- now using autograd
# manual loss computation -- now use Pytorch Loss 
# manual parameter optimization -- now use PyTorch Optimizer

import numpy as np

# f = w * x             function = weight * input
# f = 2*x 
X = np.array([1, 2, 3, 4], dtype=np.float32) 
Y = np.array([2, 4, 6, 8], dtype=np.float32) 

w = 0 # initial weight

# Model prediction -- manually
def forward(x):
    return w*x

# loss  -- manually (in this case mean squared error for linear regression)
def loss(y, y_predicted):
   return ((y_predicted-y)**2).mean() 

# gradient --- manually
# mean squared error MSE J = 1/N * (w*x-y)**2
# dJ/dw = 1/N 2x (wx-y) numerically computed derivative
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()
    # implements dJ/dw
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
    dw = gradient(X, Y, y_predicted)
    
    # update weights
    # take a step in the negative gradient at a step size of lr
    w -= learning_rate *dw 

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w={w:.3f}, loss = {l:.8f}')


print(f'Prediction after training: f(5) = {forward(5):.3f}')
