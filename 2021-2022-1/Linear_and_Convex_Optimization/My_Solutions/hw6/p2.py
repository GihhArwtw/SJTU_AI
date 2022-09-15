import numpy as np
import gd

X = np.array([[2,0],[0,1],[0,0]])
y = np.array([3,2,2])

def f(w):
    return (X@w-y).T@(X@w-y)

def fp(w):
    return 2*(X@w-y)@X

def f_2d(w1, w2):
    return (2*w1-3)**2 + (w2-2)**2 + 4

w0 = np.array([1.0, 1.0])
stepsize = 0.1

w_traces = gd.gd_const_ss(fp, w0, stepsize=stepsize) #,maxiter=1000)

print(f'stepsize={stepsize}, number of iterations={len(w_traces)-1}')
print('w_k =',w_traces[-1])
print('f(w_k) =',f(w_traces[-1]),end="\n\n")
#for i in range(0,1000,50):
    #    print(fp(x_traces[i]),end="")

w_solution = np.linalg.solve(X.T@X,X.T@y)
print('The solution calculated using numpy.linalg.solve:')
print('w_k =',w_solution)
print('f(w_k) =',f(w_solution))