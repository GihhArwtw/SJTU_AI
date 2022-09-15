import numpy as np
import gd
import utils

def Prob01(gamma=0.1, stepsize=1):
    # The following code until [END] is mainly the code given by Professor and TA.
    
    #gamma = 0.1
    Q = np.diag([gamma, 1])

    def f(x):
	    return 0.5 * x.T@Q@x

    def fp(x):
        return Q@x 

    def f_2d(x1, x2):
        return 0.5 * gamma * x1**2 + 0.5 * x2**2

    x0 = np.array([1.0, 1.0])

    #stepsize = 1


    x_traces = gd.gd_const_ss(fp, x0, stepsize=stepsize) #,maxiter=1000)

    print(f'gamma={gamma}, stepsize={stepsize}, number of iterations={len(x_traces)-1}')
    print('x_k =',x_traces[-1])
    #for i in range(0,1000,50):
        #    print(fp(x_traces[i]),end="")

    utils.plot_traces_2d(f_2d, x_traces, f'gd_traces_gamma{gamma}_ss{stepsize}.svg')
    utils.plot(f, x_traces, f'gd_f_gamma{gamma}_ss{stepsize}.svg')
    
    # [END]


# Problem 1(c)
# Prob01(stepsize=2.2)   # Need to alter line 23 to "...stepsize, maxiter=1000)"
# Prob01(stepsize=1)
# Prob01(stepsize=0.1)
# Prob01(stepsize=0.01)

# Problem 1(d)
gammas = [1, 0.1, 0.01, 0.001]
for gamma in gammas:
    Prob01(gamma)