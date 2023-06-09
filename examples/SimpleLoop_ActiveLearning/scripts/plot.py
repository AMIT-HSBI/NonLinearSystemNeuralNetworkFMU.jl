import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("../data/sims/simpleLoop_1000/data/eq_18.csv", delimiter=',', skip_header=1, names=['s', 'r', 'y'])


xs = data['s']
ys = data['r']
zs = data['y']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs, ys, zs, marker='o')

ax.set_xlabel('s')
ax.set_ylabel('r')
ax.set_zlabel('y')

#plt.show()
plt.savefig('foo.png')
