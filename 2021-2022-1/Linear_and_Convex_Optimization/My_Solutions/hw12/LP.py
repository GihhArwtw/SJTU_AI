import newton as nt
import numpy as np

def centering_step(c, A, b, x0, t):
	"""
	c, A, b: parameters in LP

		min   c^T x
		s.t.  Ax = b
		      x >= 0

	x0: feasible initial point for constrained Newton's method
	t:  penalty parameter in barrier method

	This function returns the central point x^*(t) 
	"""
	#   START OF YOUR CODE
		
	from newton import newton_eq as nteq
	x = x0
    
	def obj(x):
		if np.min(x)<0:
			return float("inf")
		return c.T @ x - np.log(x).sum() * 1. / t
    
	def obj_p(x):
		return c - (1./x) / t
    
	def obj_pp(x):
		fpp = np.array( list(1./t/(x[i]**2) for i in range(len(x))) )
		return np.diag(fpp)
    
	return nteq(obj, obj_p, obj_pp, x0, A, b)[-1]

	#	END OF YOUR CODE


def barrier(c, A, b, x0, tol=1e-8, t0=1, rho=10):
	"""
	c, A, b: parameters in LP
	
		min   c^T x
		s.t.  Ax = b
		      x >= 0
		     
	x0:  feasible initial point for the barrier method
	tol: tolerance parameter for the suboptimality gap. The algorithm stops when

	         f(x) - f^* <= tol

	t0:  initial penalty parameter in barrier method
	rho: factor by which the penalty parameter is increased in each centering step

	This function should return a list of the iterates
	"""	
	t = t0
	x = np.array(x0)
	x_traces = [np.array(x0)]

	#   START OF YOUR CODE

	t_limit = x.shape[0] * 1. / tol
		
	while (t < t_limit):
		x = centering_step(c,A,b,x,t)
		x_traces.append(x)
		t *= rho
        
	#	END OF YOUR CODE

	return x_traces