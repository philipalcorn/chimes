"""
four body case
This is to determine the index of a flattened coeffient matrix with 
permutation invariance with respect to the original indices i, j, k
(that is, c[0][0][1] and c[1][0][0] map to the same place a[n]

Each set of coefficient is dependent on the specifc atom types present. 
As an example, given atoms C, H, O, = 0,1,2, a four bodied system 
would have a set of coefficients for E=0000, 0001, 0002, 0011, et cetera,
also with permutation invariance.

To find a specific coefficient c[a], we use the formula 
c[a] = N_c * m + n

N_c is the number of coefficients per element combination
m is speficic element pair offset 
n is the specific coefficient offset


The total number of coefficients per element combination E is given as 
comb(O+k-1, k) where O is the maximum chebyshev polynomial order,
and k is the total number of atom pairs. 

For a four body system with O=4, there are comb(9,6) = 84 coefficients per 
Element combination.

for both m and n, the sorted coefficient index [i,j,k,l,m,n] needs to be 
converted to a strictly increasing index by adding [0, 1,2,3,4,5], for however
long that index is.

With the transformed coefficients, the specific m and n are given as the sum
of combinations comb(c_1, 1) + comb(c_1, 1) + ... + comb(c_k, k),1


This will start with a four body example.
"""

import math
def main():
    ORDER_4B = 4        # Max Chebyshev order for four bodies
    K_4B = 6            # Number of indices for four body coefficient c_abcdef
    N_COEFF_4B = math.comb(ORDER_4B+K_4B -1, K_4B) # Number of coefficnets per element combination

    C = [0,1,4,4,3,2]   # Chebyshev coefficient - permutation invariant
    E = [1, 1, 0, 4]    # Specific elements in system - permutation invariant

    # goal - to get the flattened index that mapps to a 1d array c[a] 
    # a = N_coeff * m + n

    m = comb_sum(E)     # specific element "index" 
    p = N_COEFF_4B * m # Actual index taking into account number of coefficients per element pairA
    n = comb_sum(C)     # mapped index for the specifc cheby coefficient we want 
    index =  p + n
    print(index)

def transform_indices(a):
    b = sorted(a)
    for  i, v in enumerate(b):
        b[i] += i 
    return b

def comb_sum(a):
    sum = 0
    b = transform_indices(a)
    for i, v in enumerate(b):
        sum += math.comb(v, i)
    return sum

if(__name__=="__main__"):
    main()







