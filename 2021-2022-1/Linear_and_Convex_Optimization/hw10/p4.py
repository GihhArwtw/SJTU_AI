# -*- coding: utf-8 -*-

import numpy as np

# The following function takes in y and z and returns the optimal 
#     solution and its corresponding Lagrange multiplier.
# We regrad the absolute value of number no smaller than infty as
#     "infinity".

def AffineConstrained_l2Norm(y,z,infty = 5000000):

    u = [-infty]
    w = [-infty]
    for i in range(len(y)):
        if (y[i]==1):
            u.append(z[i])
        else:
            w.append(-z[i])
    u.sort()                       # Sort u.
    u.append(+infty)
    w.sort()                       # Sort w.
    w.append(+infty)
    lambd = 0.                     # Lagrange multiplier
    
    k = 0
    l = 0
    while (k<len(u)-1) and (l<len(w)-1):  # infty is u[p+1] and w[m+1]
        lambd = 0.
        for i in range(len(u)-k-2):
            lambd += u[k+1+i]
        for j in range(l):
            lambd += w[j+1]
        lambd /= (len(u)-k+l-2)
        if (lambd >= u[k]) and (lambd <= u[k+1]) and (lambd >= w[l]) and (lambd <= w[l+1]):
            break
        if (k+1==len(u)):
            l += 1
            continue
        if (l+1==len(w)):
            k += 1
            continue
        if (u[k+1]<w[l+1]):
            k += 1
        else:
            l += 1
            
    x = np.array(np.array(z) - lambd*np.array(y))
    for i in range(len(x)):
        if(x[i]<0):
            x[i] = 0
    return [x,lambd]


y = np.array([1,1,-1])
z = np.array([1,2,1])

x, lambd = AffineConstrained_l2Norm(y,z)

print(x)
print(lambd)