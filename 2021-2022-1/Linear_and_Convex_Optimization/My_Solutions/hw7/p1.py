import numpy as np
import gd
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

def f_2d(x1, x2):
	return np.exp(x1+3*x2-0.1) + np.exp(x1-3*x2-0.1) + np.exp(-x1-0.1)

# use the value you find in part (a) 
f_opt = 2 * np.sqrt(2) * np.exp(-0.1)

def gap(x):
	return f(x) - f_opt


# Problem 1(b)
x0 = np.array([-1.5,1.0]) # initial point

#### Armijo
x_traces_armijo, stepsize_traces, num_iter_inner = gd.gd_armijo(f, fp, x0, alpha=0.1, beta=0.7)
f_value_armijo = f(x_traces_armijo[-1])

print()
print('gradient descent with Armijo')
print('  number of iterations in outer loop:', len(x_traces_armijo)-1)
print('  total number of iterations in inner loop:', num_iter_inner)
print('  solution:', x_traces_armijo[-1])
print('  value:', f_value_armijo)

utils.plot_traces_2d(f_2d, x_traces_armijo, 'gd_traces_armijo.svg')
utils.plot(gap, x_traces_armijo, 'gd_error_armijo.svg')

fig = plt.figure(figsize=(3.5,2.5))
plt.plot(stepsize_traces, '-o', color='#637ca1')
plt.xlabel('iteration (k)')
plt.ylabel('stepsize')
plt.tight_layout(pad=0.1)
fig.savefig('gd_armijo_ss.svg')

# Problem 1(c)
#### constant stepsize
for const_ss in [0.1, 0.01]:
	x_traces_css = gd.gd_const_ss(fp, x0, stepsize=const_ss)
	f_value_css= f(x_traces_css[-1])
	print()
	print('gradient descent with constant stepsize ', const_ss)
	print('  number of iterations:', len(x_traces_css)-1)
	print('  solution:', x_traces_css[-1])
	print('  value:', f_value_css)

	utils.plot_traces_2d(f_2d, x_traces_css, f'gd_traces_css{const_ss}.svg')
	utils.plot(gap, x_traces_css, f'gd_error_css{const_ss}.svg')
  

# Problem 1(d)  
x0 = np.array([1.5,1.0]) # initial point

#### Armijo
x_traces_armijo, stepsize_traces, num_iter_inner = gd.gd_armijo(f, fp, x0, alpha=0.1, beta=0.7)
f_value_armijo = f(x_traces_armijo[-1])

print()
print('gradient descent with Armijo')
print('  number of iterations in outer loop:', len(x_traces_armijo)-1)
print('  total number of iterations in inner loop:', num_iter_inner)
print('  solution:', x_traces_armijo[-1])
print('  value:', f_value_armijo)

utils.plot_traces_2d(f_2d, x_traces_armijo, 'o_gd_traces_armijo.svg')
utils.plot(gap, x_traces_armijo, 'o_gd_error_armijo.svg')

fig = plt.figure(figsize=(3.5,2.5))
plt.plot(stepsize_traces, '-o', color='#637ca1')
plt.xlabel('iteration (k)')
plt.ylabel('stepsize')
plt.tight_layout(pad=0.1)
fig.savefig('o_gd_armijo_ss.svg')

#### constant stepsize
for const_ss in [0.005, 0.1, 0.01]:
	x_traces_css = gd.gd_const_ss(fp, x0, stepsize=const_ss)
	f_value_css= f(x_traces_css[-1])
	print()
	print('gradient descent with constant stepsize ', const_ss)
	print('  number of iterations:', len(x_traces_css)-1)
	print('  solution:', x_traces_css[-1])
	print('  value:', f_value_css)

	if (const_ss == 0.005):
		utils.plot_traces_2d(f_2d, x_traces_css, f'o_gd_traces_css{const_ss}.svg')
		utils.plot(gap, x_traces_css, f'o_gd_error_css{const_ss}.svg')