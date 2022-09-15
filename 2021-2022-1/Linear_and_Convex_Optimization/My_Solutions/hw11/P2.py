import numpy as np
import proj_gd as gd
import utils
import matplotlib.pyplot as plt
import matplotlib.patches as mp

a = np.array([1,2,2], dtype=float)
A = np.array([1,1,1]).reshape([1,-1])
b = np.array([1])

def f(x):
	return np.sum(np.exp(a*x))

def fp(x):
	return a*np.exp(a*x)

# Implement the projection operator proj onto the affine space Ax = b
#   START OF YOUR CODE
precal = A.T @ np.linalg.inv(A@A.T)   # which will be used in all iterations, precomputed
def proj(x):
	return x + precal @ (b - A@x)
#   END OF YOUR CODE

x0 = np.array([0,0,0], dtype=float)
stepsize = 0.1
x_traces, _ = gd.proj_gd(fp, proj, x0, stepsize=stepsize, tol=1e-8)

f_value = f(x_traces[-1])

print()
print('number of iterations:', len(x_traces)-1)
print('solution:', x_traces[-1])
print('value:', f_value)
