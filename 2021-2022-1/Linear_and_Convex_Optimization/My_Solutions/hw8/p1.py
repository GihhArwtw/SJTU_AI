import numpy as np
import newton
import utils
import matplotlib.pyplot as plt


def f(x):
	return f_2d(x[0], x[1])

def fp(x):
	#   START OF YOUR CODE
	fp1 = np.exp(x[0]+3*x[1]-0.1) + np.exp(x[0]-3*x[1]-0.1) - np.exp(-x[0]-0.1)
	fp2 = 3*np.exp(x[0]+3*x[1]-0.1) - 3*np.exp(x[0]-3*x[1]-0.1)
	return np.array([fp1,fp2])
	#	END OF YOUR CODE

def fpp(x):
	#   START OF YOUR CODE
	f11 = np.exp(x[0]+3*x[1]-0.1) + np.exp(x[0]-3*x[1]-0.1) + np.exp(-x[0]-0.1)
	f12 = 3*np.exp(x[0]+3*x[1]-0.1) - 3*np.exp(x[0]-3*x[1]-0.1)
	f22 = 9*np.exp(x[0]+3*x[1]-0.1) + 9*np.exp(x[0]-3*x[1]-0.1)
	return np.array([[f11,f12],[f12,f22]])
	#	END OF YOUR CODE

def f_2d(x1, x2):
	return np.exp(x1+3*x2-0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1-0.1)

# use the value you find in HW7
f_opt = 2 * np.sqrt(2) * np.exp(-0.1)

def gap(x):
	return f(x) - f_opt

c = 'a'
for x0 in [np.array([-1.5,1.0]), np.array([1.5,1.0])]:
    
    #### Newton
    x_traces = newton.newton(fp, fpp, x0)
    f_value= f(x_traces[-1])


    print()
    print("Newton's method with initial point =",x0)
    print('  number of iterations:', len(x_traces)-1)
    print('  solution:', x_traces[-1])
    print('  value:', f_value)

    utils.plot_traces_2d(f_2d, x_traces, f'1({c})_nt_traces.svg')
    utils.plot(gap, x_traces, f'1({c})_nt_gap.svg')
    c = chr(ord(c)+1)