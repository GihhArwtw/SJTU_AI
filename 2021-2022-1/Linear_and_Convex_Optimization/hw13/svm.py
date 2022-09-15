import numpy as np
import matplotlib.pyplot as plt
import proj_gd as gd

def svm(X,y):
	"""
	X: n x m matrix, X[i,:] is the m-D feature vector of the i-th sample
	y: n-D vector, y[i] is the label of the i-th sample, with values +1 or -1

	This function returns the primal and dual optimal solutions w^*, b^*, mu^*
	"""
	Xy = X * y
	Q = Xy @ Xy.T

	def fp(mu):
		# f(mu) = 0.5 * mu.T@Q@mu - np.sum(mu)
		return Q@mu - np.ones_like(mu)

	def proj(mu):      # projected gradient descent
		infty = 500000
		# START OF YOUR CODE
		u = [-infty]
		w = [-infty]
		for i in range(len(y)):
			if (y[i]==1):
				u.append(mu[i])
			else:
				w.append(-mu[i])
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
            
		x = np.array(np.array(mu) - lambd*np.array(y))
		for i in range(len(x)):
			if(x[i]<0):
				x[i] = 0
		return x
		# END OF YOUR CODE

	mu0 = np.zeros_like(y)
	mu_traces, _ = gd.proj_gd(fp, proj, mu0, stepsize=0.1, tol=1e-8)
	mu = mu_traces[-1]

	# recover the primal optimal solution from dual optimal solution
	# START OF YOUR CODE
	w = np.sum(mu * Xy, axis = 0)
	b = 0
	for i in range(len(mu)):
		if mu[i]>0:
			b = y[i] - X[i] @ w
			break
    
	# END OF YOUR CODE

	return w, b, mu
