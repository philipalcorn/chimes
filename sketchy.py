import numpy as np
import torch
import torch.nn as nn
from numpy.polynomial import chebyshev as C

torch.set_default_dtype(torch.double)


class Energy(nn.Module): 
    #TODO Need to change the default inputs to r and add the r_to_x for x_avg 
    # and stuff
    def __init__(self, n, lam, x_cut_min, x_cut_max, f0, A, init=None):
        super().__init__()
        self.x_diff = 0.5 * abs(x_cut_min + x_cut_max)
        self.x_avg = 0.5 * abs(x_cut_min - x_cut_max)
        self.d = r_cut_max * (1-f0)
    
    # transformation of input radius r to intermediate value s 
    # and its derivatives, required for chebyshev polynomials 
    def r_to_s(self, r):                                    
        x = torch.exp(-r/self.lam)
        s = (x - x_avg) / x_diff
        ds = -exp(-r/self.lam) / (lam * x_diff)
        dds = exp(-r/self.lam) / (lam * lam * x_diff)
        return s, ds, dds # s(r), ds/dr(r), d2s/dr2(r)
    
    # Smoothing function and its derivatives wrt r
    def f_s(self, r):                                             
        if r > self.r_cut_max: 
            return 0
        if r < self.d:
            return 1
        theta = torch.pi* (r-self.d)/(self.r_cut_max-self.d) + torch.pi/2
        fs = (0.5 + 0.5*torch.sin(theta), # fs(r)
        dfs = torch.pi/(2*(self.r_cut_max-self.d)) * torch.cos(theta), # dfs/dr (r) 
        ddfs = -torch.pi**2/(2*(self.r_cut_max-self.d)**2) * torch.sin(theta)) #d2fs/dr2 (r) 
        return fs, dfs, ddfs

    # Penalty function and its derivatives wrt r
    def f_p(self, r):                                             
        u = self.rc + self.d - r
        fp = self.A*u**3
        dfp = -3*self.A*u**2
        ddfp = 6*self.A*u
        return fp, dfp, ddfp 

    # Generate a vector of chebyshev polynomials evaluated at s, 
    # Along with a similar vector for the first and second derivative
    # this will be used once per radius in a system, so once in a two
    # body system, three times in a three body system, six times in a four 
    # body system, et cetera
    def chebyshev(s, n):
        T   = torch.empty(n, dtype=s.dtype, device=s.device)
        dT  = torch.empty(n, dtype=s.dtype, device=s.device)
        ddT = torch.empty(n, dtype=s.dtype, device=s.device)
        U   = torch.empty(n, dtype=s.dtype, device=s.device)

        # Set initial values
        T[0],  dT[0], ddT[0], U[0] =  1.0, 0.0, 0.0, 1.0
        if n > 1:
            T[1], dT[1], ddT[1], U[1] = s, 1.0, 1.0, 2*s 

        # Build remainder of vectors 
        for k in range(2, n): 
            T[k]   = 2*s*T[k-1]   - T[k-2]
            U[k]   = 2*s*U[k-1]   - U[k-2]
            dT[k]  = k * U[k-1] 
            # need to handle edge cases for T''
            # TODO Might need to replace the == for a tolerance
            # such as if (abs(s) < 1e6 or something
            if (s==1):
                ddT[k] = (k*k*k*k - k*k) / 3
            elif (s==-1):
                ddT[k] = pow(-1, k) * (k*k*k*k - k*k) / 3
            else:
                ddT[k] = k * ((k+1) * T[k] - U[k]) / (s**2 - 1)
        return T, dT, ddT

    def forward(self, r):
        # DO STUFF - get all the energies basically 
        return #E, dE, ddE


#def _demo():


if __name__ == "__main__":
    return
