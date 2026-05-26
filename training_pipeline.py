# Here will will automate  the loss and parameter updates

# 1) design model (input, output size, forward pass)
# 2) construct loss and optimizer 
# 3) training loop
#   ---> foward pass: compute prediction 
#   ---> backward pass: get gradient 
#   ---> update weights
#   ---> iterate 


import torch 
import torch.nn as nn
# f = w * x             function = weight * input
# f = 2*x 
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32) 
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32) 

#the pytorch model already knows what the weights are 
X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
print (n_samples, n_features)
# 4 samples and 1 feature per sample
input_size = n_features
output_size = n_features

# this is the automatic model
#model = nn.Linear(input_size, output_size)

#This is how we would get a custom model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
    

model = LinearRegression(input_size, output_size)
# Model prediction -- automatic

# loss  -- automatic

# gradient --- automatic
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

learning_rate = 0.01
n_iter=10000

# automatic loss 
loss = nn.MSELoss() # returns a callable function f(actual, pred)
# stochastic gradient descent 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

# training loop
for epoch in range(n_iter):
    # prediction = fwd pass 
    y_predicted = model(X)

    #loss 
    l = loss(Y, y_predicted)

    # gradients
    # dw = gradient(X, Y, y_predicted) -- old manual
    l.backward() # -- new automatic gradient dl/dw 
    
    # update weights -- automatic
    optimizer.step() # note the weights are passed in on initialization
        
    # empty the gradients for the new loop
    optimizer.zero_grad()
     
    if epoch % 10 == 0:
        # need to unpack to print
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w={w[0][0].item():.3f}, loss = {l:.8f}')


print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

