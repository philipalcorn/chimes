import torch

# Constants 
r_cut_in = 0.1 # 0.98 in paper
r_cut_out = 5.0  # Angstroms


A = 1
f0 = 0.75
d = r_cut_out * (1-f0)

lam = 1.4


# Functions
#Penalty function
def fp(r, r_cut_out, d, A): 
    return  A * (r_cut_out + d - r).pow(3) if (r < r_cut_out+d) else  0

def fs(r, r_cut_out, d) :
    theta = torch.pi * (r-d)/(r_cut_out - d) + torch.pi/2
    return 0.5 + 0.5 * torch.sin(theta)

def r_to_x(r, lam):
    return torch.exp(-r/lam)

def x_to_s(x, x_avg, x_diff)
    return (x - x_avg)/x_diff 

# Intermediate values for s_transformation
x_cut_in = r_to_x(r_cut_in) 
x_cut_out = r_to_x(r_cut_out)  
x_avg = 0.5 * abs(x_cut_out - x_cut_in)
x_diff = 0.5 * abs(x_cut_out + x_cut_in)

# init system
c1 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
c2 = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
d_ij = (c1 - c2).pow(2).sum(-1).sqrt() # Distance 
d_ij.requires_grad_(True)


# Transformation to s to map to [-1, 1]
x_ij = r_to_x(d_ij, lam)
s_ij = x_to_s(x_ij, lam)



'''
U = 0.04 * -0.4 / d_ij
U.backward()
print(d_ij.grad)
'''

# do chebyshev math summation and backwards here
# Energy = fp(r) + fs(r)SUM_0^20(c_n T_n(s(r)) 
#s = 


# stop tacking grad
d_ij = d_ij.detach()



# check for accuracy using torch.allclose()
