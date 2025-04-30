
import matplotlib.pyplot as plt
import numpy as np

"""
x = np.array(range(10))
y = np.exp(x)
z = np.exp(-x)
w = np.exp(x)/1.25

fig = plt.figure(figsize=(10,5))

ax1 = fig.add_subplot(111)
ax1.bar(x, y, label='e^x', color='#637ca1', alpha=.7)
ax1.plot(x, w, label='(e^x)/1.25', color='#7d548b', linewidth=3, alpha=1.)
ax1.set_ylabel('Y')
ax1.set_xlabel('X')
# ax1.set_yscale('log')
ax1.set_ylim(0,10000)
ax1.legend(loc=2)

ax2 = ax1.twinx()
ax2.plot(x,z,label='e^(-x)',color='#cc4e63', marker='o', linewidth=1, alpha=.8)
ax2.set_ylabel('Z')
ax2.set_ylim(0,1.3)
ax2.legend(loc=1)

plt.show()
plt.savefig('a.svg')
"""

fig = plt.figure(figsize=(10,5))

y = [0,
0.0,
0.0,
0.0,
0.30844800000000006,
0.3698006400000002,	
0.43020797760000024,	
0.45144142646400026,
0.46425593286432015,	
0.4694096791328546,	
0.472018544004403,	
0.4731270329324718,	
0.47386877297884755,	
0.4745700399701813,
0.47497561583128023,
0.4752173643973898	
]
z = [0.06707238791024638,
0.4739234961345665,	
0.47535659071557324]


plt.plot(np.array(range(16)), y, label='value iteration', color='#637ca1', linewidth=3, alpha=1.)
plt.plot(np.array(range(3)), z, label='policy iteration', color='#cc4e63', linewidth=3, alpha=1.)

# plt.yscale('log')
# plt.ylim(0,1)
plt.ylabel('Utility Estimate')
plt.legend(loc=0)
plt.title('STATE: (2,0)')

# plt.show()
plt.savefig('converge-6.pdf')