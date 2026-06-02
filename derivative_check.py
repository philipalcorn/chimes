import torch
import math
torch.autograd.set_detect_anomaly(True, check_nan=False)
torch.set_default_dtype(torch.float32)


def main():
    # Constants 
    n2 = 20   # orders 0..20  (must match chebyshev(s, 20) output length)
    n3 = 10
    n4 = 4
    r_cut_in = 0.1 # 0.98 in paper
    r_cut_out = 5.0  # Angstroms
    A = 1
    f0 = 0.75
    d = r_cut_out * (1-f0)
    lam = 1.4

    # Intermediate values for s_transformation
    x_cut_in = r_to_x(r_cut_in, lam) 
    x_cut_out = r_to_x(r_cut_out, lam)  
    x_avg = 0.5 * abs(x_cut_out - x_cut_in)
    x_diff = 0.5 * abs(x_cut_out + x_cut_in)

    # init system
    c1 = torch.tensor([0.0, 0.0, 0.0])
    c2 = torch.tensor([2.0, 2.0, 2.0])
    d_ij = (c1 - c2).pow(2).sum(-1).sqrt() # Distance 
    d_ij.requires_grad_(True)
        
    # Test Coefficient Matrices
    C2 = torch.ones(n2, requires_grad=True)                      # 2-body: (n2)
    C3 = torch.ones(n3, n3, n3, requires_grad=True)                # 3-body: (n3)
    C4 = torch.ones(n4, n4, n4, n4, n4, n4, requires_grad=True)       # 4-body: (n4)


    # do chebyshev math summation and backwards here
    E, dE, ddE, dEdC, dFdC = e2b(n2, C2, d_ij, r_cut_in, r_cut_out, d,A, lam, x_avg, x_diff)


    # Needed to use this more complicated form instead of .backward
    # to get the second derivative
    g1 = torch.autograd.grad(E, d_ij,create_graph=True)[0]
    g2 = torch.autograd.grad(g1, d_ij,create_graph=True)[0]
    gc1 = torch.autograd.grad(E, C2,create_graph=True)[0]
    gc2 = torch.autograd.grad(g1, C2,create_graph=True)[0]
    
    print("2 Body systems")
    print("Energy: ", E.item()) # forwards
    #print("dE  match:", torch.allclose(g1, dE),  "| autograd", g1.item(), "analytic", dE.item())
    #print("ddE  match:", torch.allclose(g2, ddE),  "| autograd", g2.item(), "analytic", ddE.item())
    print("dE  match:", torch.allclose(g1, dE))
    print("ddE  match:", torch.allclose(g2, ddE))
    print("dEdC  match:", torch.allclose(gc1, dEdC))
    print("dFdC  match:", torch.allclose(gc2, dFdC))


    # --- 3 body ---
    r1 = torch.tensor(1.3, requires_grad=True)
    r2 = torch.tensor(1.7, requires_grad=True)
    r3 = torch.tensor(2.1, requires_grad=True)
    rs = [r1, r2, r3]

    E, dE, ddE, dEdC, dFdC = e3b(n3, C3, r1, r2, r3, r_cut_in, r_cut_out, d, lam, x_avg, x_diff)

    # gradient vector: dE/dr_i
    g1 = torch.autograd.grad(E, rs, create_graph=True)   # tuple of 3 scalars
    g1 = torch.stack(g1)

    # Hessian diagonal: d2E/dr_i2  — grad each g1[i] back against r_i only
    g2 = torch.stack([
        torch.autograd.grad(g1[i], rs[i], create_graph=True)[0]
        for i in range(3)
    ])

    gc1 = torch.autograd.grad(E, C3,create_graph=True)[0]
    gc2 = torch.stack([torch.autograd.grad(g1[0], C3, create_graph=True, retain_graph=True)[0],
                       torch.autograd.grad(g1[1], C3, create_graph=True, retain_graph=True)[0],
                       torch.autograd.grad(g1[2], C3, create_graph=True, retain_graph=True)[0]])
    print("3 Body systems")
    print("Energy:", E.item())
    #print("dE  match:",  torch.allclose(g1, dE),  "\n  autograd", g1.detach().tolist(),  "\n  analytic", dE.detach().tolist())
    #print("ddE match:",  torch.allclose(g2, ddE), "\n  autograd", g2.detach().tolist(),  "\n  analytic", ddE.detach().tolist())
    print("dE  match:",  torch.allclose(g1, dE))
    print("ddE match:",  torch.allclose(g2, ddE))
    print("dEdC  match:", torch.allclose(gc1, dEdC))
    print("dFdC match:", torch.allclose(gc2, dFdC))



    # --- 4 body (6 edges) ---
    rs = [torch.tensor(v, requires_grad=True) for v in (1.3, 1.5, 1.7, 1.9, 2.1, 2.3)]

    E, dE, ddE, dEdC, dFdC = e4b(n4, C4, *rs, r_cut_in, r_cut_out, d, lam, x_avg, x_diff)

    g1 = torch.stack(torch.autograd.grad(E, rs, create_graph=True))
    g2 = torch.stack([
        torch.autograd.grad(g1[i], rs[i], create_graph=True)[0]
        for i in range(6)
    ])

    gc1 = torch.autograd.grad(E, C4,create_graph=True)[0]
    gc2 = torch.stack([torch.autograd.grad(g1[0], C4, create_graph=True, retain_graph=True)[0],
                       torch.autograd.grad(g1[1], C4, create_graph=True, retain_graph=True)[0],
                       torch.autograd.grad(g1[2], C4, create_graph=True, retain_graph=True)[0],
                       torch.autograd.grad(g1[3], C4, create_graph=True, retain_graph=True)[0],
                       torch.autograd.grad(g1[4], C4, create_graph=True, retain_graph=True)[0],
                       torch.autograd.grad(g1[5], C4, create_graph=True, retain_graph=True)[0]])
    print("4 Body systems")
    print("Energy:", E.item())
    #print("dE  match:",  torch.allclose(g1, dE),  "\n  autograd", g1.detach().tolist(),  "\n  analytic", dE.detach().tolist())
    #print("ddE match:",  torch.allclose(g2, ddE), "\n  autograd", g2.detach().tolist(),  "\n  analytic", ddE.detach().tolist())
    print("dE  match:",  torch.allclose(g1, dE))
    print("ddE match:",  torch.allclose(g2, ddE))
    print("dEdC  match:", torch.allclose(gc1, dEdC))
    print("dFdC match:", torch.allclose(gc2, dFdC))


# Penalty function + Derivatives
def f_p(r, r_cut_out, d, A):                                             
    u = r_cut_out + d - r
    fp = A*u**3
    dfp = -3*A*u**2
    ddfp = 6*A*u
    return fp, dfp, ddfp


# Smoothing function + Derivatives
def f_s(r, r_cut_out, d) :
    if r > r_cut_out: 
        return 0
    if r < d:
        return 1
    theta = torch.pi* (r-d)/(r_cut_out-d) + torch.pi/2
    fs = (0.5 + 0.5*torch.sin(theta)) # fs(r)
    dfs = torch.pi/(2*(r_cut_out-d)) * torch.cos(theta) # dfs/dr (r) 
    ddfs = -torch.pi**2/(2*(r_cut_out-d)**2) * torch.sin(theta) #d2fs/dr2 (r) 
    return fs, dfs, ddfs


# Transformation from r to intermediate x 
# SCALAR ONLY, used for x_avg and x_diff
def r_to_x(r, lam): 
    return math.exp(-r/lam)


#Transformation from r to s
def r_to_s(r, lam, x_avg, x_diff ):                                    
    x = torch.exp(-r/lam)
    s = (x - x_avg) / x_diff
    ds = -torch.exp(-r/lam) / (lam * x_diff)
    dds = torch.exp(-r/lam) / (lam * lam * x_diff)
    return s, ds, dds # s(r), ds/dr(r), d2s/dr2(r)


# Generate a vector of chebyshev polynomials evaluated at s, 
# Along with a similar vector for the first and second derivative
# this will be used once per radius in a system, so once in a two
# body system, three times in a three body system, six times in a four 
# body system, et cetera
def chebyshev(s, n):
    o = torch.tensor(1.0, dtype=s.dtype, device=s.device)
    z = torch.tensor(0.0, dtype=s.dtype, device=s.device)

    T = [o, s]
    dT = [z, o]
    ddT = [z, z]
    U = [o, 2*s]

    # Build remainder of vectors 
    for k in range(2, n): 
        T.append(2*s*T[k-1]   - T[k-2])
        U.append(2*s*U[k-1]   - U[k-2])
        dT.append(k * U[k-1])
        # there are techncially some edge cases here 
        # But that should never happen with floats
        ddT.append(k * ((k+1) * T[k] - U[k]) / (s**2 - 1))
    return torch.stack(T[:n]), torch.stack(dT[:n]), torch.stack(ddT[:n])


# Calculates 2-body energy from a known radius
# Energy = fp(r) + fs(r)SUM_0^20(c_n T_n(s(r)) 
def e2b(n, C, r, r_cut_in, r_cut_out, d, A, lam, x_avg, x_diff):
    # s(r) transformation  
    s, ds, dds = r_to_s(r, lam, x_avg, x_diff) 
    #f_smooth(r)
    fs, dfs, ddfs = f_s(r, r_cut_out, d)
    #f_penalty(r)
    fp, dfp, ddfp = f_p(r, r_cut_out, d, A)
    #T(s) 
    T, dT, ddT = chebyshev(s, n)
    
    # E(r), d/dr E(r), d2/dr2 E(r)
    E = fp + fs * (T @ C)
    dE = dfp + dfs * (T @ C) + fs * ds * (dT @ C)
    ddE = (ddfp+ddfs*(T@C)+ 2*dfs*ds* (dT @ C)
           + fs*((ddT@C)*ds.pow(2) + (dT@C)*dds))
    dEdC = torch.mul(T,fs)   # Should return a tensor of all of them 
    dFdC = torch.mul(T,dfs) + torch.mul(dT, fs*ds)   # Should return a tensor of all of them 
    return E, dE, ddE, dEdC, dFdC

# Here C is the coefficient tensor, in this case n*n*n
def e3b(n, C, r1, r2, r3, r_cut_in, r_cut_out, d, lam, x_avg, x_diff):
    # s(r) transformation  
    s1, ds1, dds1 = r_to_s(r1, lam, x_avg, x_diff) 
    s2, ds2, dds2 = r_to_s(r2, lam, x_avg, x_diff) 
    s3, ds3, dds3 = r_to_s(r3, lam, x_avg, x_diff) 

    #f_smooth(r)
    fs1, dfs1, ddfs1 = f_s(r1, r_cut_out, d)
    fs2, dfs2, ddfs2 = f_s(r2, r_cut_out, d)
    fs3, dfs3, ddfs3 = f_s(r3, r_cut_out, d)

    #T(s) 
    T1, dT1, ddT1 = chebyshev(s1, n)
    T2, dT2, ddT2 = chebyshev(s2, n)
    T3, dT3, ddT3 = chebyshev(s3, n)
    
    # E(r), d/dr E(r), d2/dr2 E(r)
    E = fs1*fs2*fs3 * torch.einsum('abc, a, b, c->', C, T1, T2, T3)
    dE1 = (fs1*fs2*fs3*ds1*torch.einsum('abc, a, b, c->', C, dT1, T2, T3) 
           + dfs1*fs2*fs3*torch.einsum('abc, a, b, c->', C, T1, T2, T3))

    dE2 = (fs1*fs2*fs3*ds2*torch.einsum('abc, a, b, c->', C, T1, dT2, T3) 
           + fs1*dfs2*fs3*torch.einsum('abc, a, b, c->', C, T1, T2, T3))

    dE3 = (fs1*fs2*fs3*ds3*torch.einsum('abc, a, b, c->', C, T1, T2, dT3) 
           + fs1*fs2*dfs3*torch.einsum('abc, a, b, c->', C, T1, T2, T3))


    ddE1 = (fs1*fs2*fs3 *
                (ds1.pow(2) * torch.einsum('abc, a, b, c->', C, ddT1, T2, T3)
                 + dds1 * torch.einsum('abc, a, b, c->', C, dT1, T2, T3))
            + 2*dfs1*fs2*fs3 * ds1 * torch.einsum('abc, a, b, c->', C, dT1, T2, T3)
            + ddfs1*fs2*fs3 * torch.einsum('abc, a, b, c->', C, T1, T2, T3))

    ddE2 = (fs1*fs2*fs3 *
                (ds2.pow(2) * torch.einsum('abc, a, b, c->', C, T1, ddT2, T3)
                + dds2 * torch.einsum('abc, a, b, c->', C, T1, dT2, T3))
            + 2*fs1*dfs2*fs3 * ds2 * torch.einsum('abc, a, b, c->', C, T1, dT2, T3)
            + fs1*ddfs2*fs3 * torch.einsum('abc, a, b, c->', C, T1, T2, T3))

    ddE3 = (fs1*fs2*fs3 *
                (ds3.pow(2) * torch.einsum('abc, a, b, c->', C, T1, T2, ddT3)
                + dds3 * torch.einsum('abc, a, b, c->', C, T1, T2, dT3))
            + 2*fs1*fs2*dfs3 * ds3 * torch.einsum('abc, a, b, c->', C, T1, T2, dT3)
            + fs1*fs2*ddfs3 * torch.einsum('abc, a, b, c->', C, T1, T2, T3))

    dEdC = fs1*fs2*fs3 * torch.einsum('a,b,c->abc', T1, T2, T3)

    dFdC1 = (fs1*fs2*fs3*ds1 * torch.einsum('a,b,c->abc', dT1, T2, T3) +
            dfs1*fs2*fs3 * torch.einsum('a,b,c->abc', T1, T2, T3))
    dFdC2 = (fs1*fs2*fs3*ds2*torch.einsum('a,b,c->abc', T1, dT2, T3)
            + fs1*dfs2*fs3*torch.einsum('a,b,c->abc', T1, T2, T3))
    dFdC3 = (fs1*fs2*fs3*ds3*torch.einsum('a,b,c->abc', T1, T2, dT3)
             + fs1*fs2*dfs3*torch.einsum('a,b,c->abc', T1, T2, T3))

    dFdC = torch.stack([dFdC1, dFdC2, dFdC3])   # shape (3, n, n, n)

    return E, torch.stack([dE1, dE2, dE3]), torch.stack([ddE1, ddE2, ddE3]), dEdC, dFdC

# Here C is the coefficient tensor, in this case n*n*n*n*n*n
def e4b(n, C, r1, r2, r3, r4, r5, r6, r_cut_in, r_cut_out, d, lam, x_avg, x_diff):
    # s(r) transformation  
    s1, ds1, dds1 = r_to_s(r1, lam, x_avg, x_diff) 
    s2, ds2, dds2 = r_to_s(r2, lam, x_avg, x_diff) 
    s3, ds3, dds3 = r_to_s(r3, lam, x_avg, x_diff) 
    s4, ds4, dds4 = r_to_s(r4, lam, x_avg, x_diff) 
    s5, ds5, dds5 = r_to_s(r5, lam, x_avg, x_diff) 
    s6, ds6, dds6 = r_to_s(r6, lam, x_avg, x_diff) 
    #f_smooth(r)
    fs1, dfs1, ddfs1 = f_s(r1, r_cut_out, d)
    fs2, dfs2, ddfs2 = f_s(r2, r_cut_out, d)
    fs3, dfs3, ddfs3 = f_s(r3, r_cut_out, d)
    fs4, dfs4, ddfs4 = f_s(r4, r_cut_out, d)
    fs5, dfs5, ddfs5 = f_s(r5, r_cut_out, d)
    fs6, dfs6, ddfs6 = f_s(r6, r_cut_out, d)
    #T(s) 
    T1, dT1, ddT1 = chebyshev(s1, n)
    T2, dT2, ddT2 = chebyshev(s2, n)
    T3, dT3, ddT3 = chebyshev(s3, n)
    T4, dT4, ddT4 = chebyshev(s4, n)
    T5, dT5, ddT5 = chebyshev(s5, n)
    T6, dT6, ddT6 = chebyshev(s6, n)
    
    # E(r), d/dr E(r), d2/dr2 E(r)
    E = fs1*fs2*fs3*fs4*fs5*fs6 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, T6)
    dE1 = (fs1*fs2*fs3*fs4*fs5*fs6*ds1*torch.einsum('abcdef, a, b, c, d, e, f->', C, dT1, T2, T3, T4, T5, T6) 
           + dfs1*fs2*fs3*fs4*fs5*fs6*torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, T6))
    
    dE2 = (fs1*fs2*fs3*fs4*fs5*fs6*ds2*torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, dT2, T3, T4, T5, T6) 
           + fs1*dfs2*fs3*fs4*fs5*fs6*torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, T6))

    dE3 = (fs1*fs2*fs3*fs4*fs5*fs6*ds3*torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, dT3, T4, T5, T6) 
           + fs1*fs2*dfs3*fs4*fs5*fs6*torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, T6))

    dE4 = (fs1*fs2*fs3*fs4*fs5*fs6*ds4*torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, dT4, T5, T6) 
           + fs1*fs2*fs3*dfs4*fs5*fs6*torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, T6))

    dE5 = (fs1*fs2*fs3*fs4*fs5*fs6*ds5*torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, dT5, T6) 
           + fs1*fs2*fs3*fs4*dfs5*fs6*torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, T6))

    dE6 = (fs1*fs2*fs3*fs4*fs5*fs6*ds6*torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, dT6) 
           + fs1*fs2*fs3*fs4*fs5*dfs6*torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, T6))

    ddE1 = (fs1*fs2*fs3*fs4*fs5*fs6 *
                (ds1.pow(2) * torch.einsum('abcdef, a, b, c, d, e, f->', C, ddT1, T2, T3, T4, T5, T6)
                 + dds1 * torch.einsum('abcdef, a, b, c, d, e, f->', C, dT1, T2, T3, T4, T5, T6))
            + 2*dfs1*fs2*fs3*fs4*fs5*fs6 * ds1 * torch.einsum('abcdef, a, b, c, d, e, f->', C, dT1, T2, T3, T4, T5, T6)
            + ddfs1*fs2*fs3*fs4*fs5*fs6 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, T6))

    ddE2 = (fs1*fs2*fs3*fs4*fs5*fs6 *
                (ds2.pow(2) * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, ddT2, T3, T4, T5, T6)
                 + dds2 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, dT2, T3, T4, T5, T6))
            + 2*fs1*dfs2*fs3*fs4*fs5*fs6 * ds2 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, dT2, T3, T4, T5, T6)
            + fs1*ddfs2*fs3*fs4*fs5*fs6 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, T6))

    ddE3 = (fs1*fs2*fs3*fs4*fs5*fs6 *
                (ds3.pow(2) * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, ddT3, T4, T5, T6)
                 + dds3 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, dT3, T4, T5, T6))
            + 2*fs1*fs2*dfs3*fs4*fs5*fs6 * ds3 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, dT3, T4, T5, T6)
            + fs1*fs2*ddfs3*fs4*fs5*fs6 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, T6))

    ddE4 = (fs1*fs2*fs3*fs4*fs5*fs6 *
                (ds4.pow(2) * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, ddT4, T5, T6)
                 + dds4 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, dT4, T5, T6))
            + 2*fs1*fs2*fs3*dfs4*fs5*fs6 * ds4 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, dT4, T5, T6)
            + fs1*fs2*fs3*ddfs4*fs5*fs6 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, T6))

    ddE5 = (fs1*fs2*fs3*fs4*fs5*fs6 *
                (ds5.pow(2) * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, ddT5, T6)
                 + dds5 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, dT5, T6))
            + 2*fs1*fs2*fs3*fs4*dfs5*fs6 * ds5 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, dT5, T6)
            + fs1*fs2*fs3*fs4*ddfs5*fs6 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, T6))

    ddE6 = (fs1*fs2*fs3*fs4*fs5*fs6 *
                (ds6.pow(2) * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, ddT6)
                 + dds6 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, dT6))
            + 2*fs1*fs2*fs3*fs4*fs5*dfs6 * ds6 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, dT6)
            + fs1*fs2*fs3*fs4*fs5*ddfs6 * torch.einsum('abcdef, a, b, c, d, e, f->', C, T1, T2, T3, T4, T5, T6))

    dEdC = fs1*fs2*fs3*fs4*fs5*fs6 * torch.einsum('a,b,c,d,e,f->abcdef', T1, T2, T3, T4, T5, T6)

    fsp = fs1*fs2*fs3*fs4*fs5*fs6   # full cutoff product, reused

    dFdC1 = (fsp*ds1*torch.einsum('a,b,c,d,e,f->abcdef', dT1, T2, T3, T4, T5, T6)
             + dfs1*fs2*fs3*fs4*fs5*fs6*torch.einsum('a,b,c,d,e,f->abcdef', T1, T2, T3, T4, T5, T6))

    dFdC2 = (fsp*ds2*torch.einsum('a,b,c,d,e,f->abcdef', T1, dT2, T3, T4, T5, T6)
             + fs1*dfs2*fs3*fs4*fs5*fs6*torch.einsum('a,b,c,d,e,f->abcdef', T1, T2, T3, T4, T5, T6))

    dFdC3 = (fsp*ds3*torch.einsum('a,b,c,d,e,f->abcdef', T1, T2, dT3, T4, T5, T6)
             + fs1*fs2*dfs3*fs4*fs5*fs6*torch.einsum('a,b,c,d,e,f->abcdef', T1, T2, T3, T4, T5, T6))

    dFdC4 = (fsp*ds4*torch.einsum('a,b,c,d,e,f->abcdef', T1, T2, T3, dT4, T5, T6)
             + fs1*fs2*fs3*dfs4*fs5*fs6*torch.einsum('a,b,c,d,e,f->abcdef', T1, T2, T3, T4, T5, T6))

    dFdC5 = (fsp*ds5*torch.einsum('a,b,c,d,e,f->abcdef', T1, T2, T3, T4, dT5, T6)
             + fs1*fs2*fs3*fs4*dfs5*fs6*torch.einsum('a,b,c,d,e,f->abcdef', T1, T2, T3, T4, T5, T6))

    dFdC6 = (fsp*ds6*torch.einsum('a,b,c,d,e,f->abcdef', T1, T2, T3, T4, T5, dT6)
             + fs1*fs2*fs3*fs4*fs5*dfs6*torch.einsum('a,b,c,d,e,f->abcdef', T1, T2, T3, T4, T5, T6))

    dFdC = torch.stack([dFdC1, dFdC2, dFdC3, dFdC4, dFdC5, dFdC6])   # (6, n,n,n,n,n,n)

    return (E,
            torch.stack([dE1, dE2, dE3, dE4, dE5, dE6]),
            torch.stack([ddE1, ddE2, ddE3, ddE4, ddE5, ddE6]),
            dEdC, dFdC)

if (__name__=="__main__"):
    main()



