import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X1 = X2 = np.linspace(-10, 10)

Y = 15 * X2 - 2 * X1

fig = plt.figure()
ax = Axes3D(fig)

ax.plot(xs=np.sqrt(X1), ys=X2, zs=Y)
plt.show()
