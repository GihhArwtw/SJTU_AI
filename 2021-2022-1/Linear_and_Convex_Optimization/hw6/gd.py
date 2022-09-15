import numpy as np

def gd_const_ss(fp, x0, stepsize, tol=1e-5, maxiter=100000):
    """
	fp: function that takes an input x and returns the derivative of f at x
	x0: initial point in gradient descent
	stepsize: constant step size used in gradient descent
	tol: toleracne parameter in the stopping crieterion. Gradient descent stops 
	     when the 2-norm of the gradient is smaller than tol
	maxiter: maximum number of iterations in gradient descent.

	This function should return a list of the sequence of approximate solutions
	x_k produced by each iteration
	"""
    x_traces = [np.array(x0)]
    x = np.array(x0)
    
	#   START OF YOUR CODE
    
    i = 0
    while ((np.linalg.norm(fp(x)) >= tol) and (i<maxiter)):
        x = x - stepsize * fp(x)
        x_traces = x_traces + [x]
        i = i + 1
    
	#	END OF YOUR CODE

    return x_traces 
