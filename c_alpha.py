# run with python c_alpha.py 2> dump.txt,
# I would then remove the dump file after viewing

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
import sys
import os
# capture the test array output only with 
# python c_alpha.py 3> dump.txt
from itertools import combinations_with_replacement, permutations

def main():
    order = [20, 10, 4]     # Max Chebyshev order for four bodies, affects number of coefficients
    n_bodies = [2, 3, 4]    # numeber of bodies in the system
    n_atom_types=[10, 6, 4] # Number of atom types under consideration
    #n_coefficients could probably also be precomputed here

    for i in range(len(order)):
        print(f"Testing {n_bodies[i]}_body systems with {n_atom_types[i]} different types of atoms.")
        n_coefficients = math.comb(order[i] + math.comb(n_bodies[i], 2) - 1, math.comb(n_bodies[i], 2))
        test_all_indices(n_atom_types[i], n_bodies[i], order[i], n_coefficients)
    # goal - to get the flattened index that mapps to a 1d array c[a] 
    # a = N_coeff * m + n

"""
The lexicographic ranking of a number requires that each 
rank is strictly increasing- this function transforms 
the indices to ensre that all sets are strictly increasing 
by doing the following operation :

  [c1, c2, c3, c4, c5 ... cn] <-- Original Indices
+ [ 0,  1,  2,  3,  4 ... n ] <-- transformation
= output 

Note that this transformation assumes the original 
indices are *non-decreasing*. This is garunteed by the 
fact that the indices are sorted. 

The output is garunteed to be strictly *increasing*.

"""
def transform_indices(a):
    b = sorted(a)
    for  i, v in enumerate(b):
        b[i] += i 
    return b

"""
This function performs the ranking of a set of indices 
to a unique integer rank. Every non-negative number
has a unique representation in this form
"""
def comb_sum(a):
    sum = 0
    b = transform_indices(a)
    for i, v in enumerate(b):
        sum += math.comb(v, i+1)
    return sum

"""
This function calculates the index of a set of coeffictients. 
Each set of coefficitions is dependent on the specific element combination
given as a tuple such as (0, 0, 1). Note that the coefficitons are permutation
invariant.

Each Element combination has a set of Chebyshev coefficients associated with
it, which are also permutation invariant. A simple example of how the
indexing looks for a three body system with a max
chebyshev order of 2 is as followed:

[e000c000, e000c001, e000c002, e000c011, e000c012, e000c111, e000c112, e000c122, e000c222, 
e001c000, e001c001, e001c002, e001c011, e001c012, e001c111, e001c112, e001c122, e001c222]
"""
def calculate_index(e_tuple, c_tuple, n_coefficients):
    m = comb_sum(e_tuple)
    n = comb_sum(c_tuple)
    idx = n_coefficients * m + n
    return idx



""" 
This builds a test array strictly for the four body case
"""
def build_test_array(n_atom_types, n_bodies, max_order, n_coefficients):
    n_pairs = math.comb(n_bodies, 2)
    elem_combos  = list(combinations_with_replacement(range(n_atom_types), n_bodies))
    coeff_combos = list(combinations_with_replacement(range(max_order), n_pairs))

    max_m = comb_sum(list(elem_combos[-1]))
    total = n_coefficients * (max_m + 1)
    arr = [None] * total

    for e_tuple in elem_combos:
        m = comb_sum(list(e_tuple))
        e_str = "E" + "".join(str(x) for x in e_tuple)
        for c_tuple in coeff_combos:
            n = comb_sum(list(c_tuple))
            c_str = "C" + "".join(str(x) for x in c_tuple)
            arr[n_coefficients * m + n] = e_str + c_str

    return arr

"""
Tests every index given by calculate_index and compares the resulting 
value with the test index, again strictly for the four body case
"""
def test_all_indices(n_atom_types, n_bodies, max_order, n_coefficients):
    n_pairs = math.comb(n_bodies, 2)
    elem_combos = list(combinations_with_replacement(range(n_atom_types), n_bodies))
    coeff_combos = list(combinations_with_replacement(range(max_order), n_pairs))
    arr = build_test_array(n_atom_types, n_bodies, max_order, n_coefficients)
    passed = 0
    failed = 0
    for e_tuple in elem_combos:
        print(f"Current Element Combination: {e_tuple}")
        for e_perm in set(permutations(e_tuple)):
            for c_tuple in coeff_combos:
                for c_perm in set(permutations(c_tuple)):
                    idx = calculate_index(e_perm, c_perm, n_coefficients)
                    actual = arr[idx]
                    expected = "E" + "".join(str(x) for x in e_tuple) + \
                               "C" + "".join(str(x) for x in c_tuple)
                    if actual == expected:
                        passed += 1
                    else:
                        failed += 1
                        print(f"FAIL: E={e_perm} C={c_perm} idx={idx} got={actual} expected={expected}")
    print(f"Passed: {passed}/{passed+failed}")

if(__name__=="__main__"):
    main()







