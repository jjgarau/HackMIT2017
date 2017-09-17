import matplotlib
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
plt.figure()

#plt.imshow(aux)
#aux = np.random.rand(Z1.shape[0],Z1.shape[1])

plt.imshow(Z1, alpha=0.7, cmap='YlOrBr')
plt.show()
