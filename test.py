import torch

# Everything in pytorch is a tensor, basically an any dimensional vector

x = torch.empty(3) 
# (The argument here is the number of elements in each dimenstion)
#print(x)
x = torch.rand(3, 3)
#print(x)
x = torch.zeros(3, 3, 5)
#print(x)
x = torch.ones(3, 3, 5, 4)
#print(x)

# Can also add data types with dtype parameter:
x = torch.ones(3, 3, dtype=torch.float16)
print(x)
x = torch.ones(3, 3, dtype=torch.int16)
print(x)

# Values can be manually added as well
