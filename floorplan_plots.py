###########################################################################
# IMPORT
###########################################################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from scipy.misc import imread                   # Cargo imread de scipy.misc
import matplotlib.mlab as mlab
from PIL import Image

###########################################################################
# DATA
###########################################################################

# Grafo (nodos en orden)
nodos_conec = [[1],[2,3]] # Conectar cada nodo como minimo consigo mismo
nodos_x = np.array([75,80,35,55])
nodos_y = np.array([20,185,185,235])

# Camino optimo
nodos_opt = [0,1,3]

###########################################################################
# PLOT FONDO (MAPA)
###########################################################################
imagen_superficial = Image.open('33_4b.png')
# Creo una figura
plt.figure()
# Muestro la imagen en pantalla
plt.imshow(imagen_superficial)

###########################################################################
# PLOT GRAFO
###########################################################################
N = len(nodos_conec) # Numero total de nodos

# Aristas
for ii in range(0,N):
    x_ii = np.array(nodos_x[ii])
    y_ii = np.array(nodos_y[ii])
    conexiones = np.array(len(nodos_conec[ii]))
    aristas_x = [x_ii*np.ones(conexiones),nodos_x[np.array(nodos_conec[ii])]]
    aristas_y = [y_ii*np.ones(conexiones),nodos_y[np.array(nodos_conec[ii])]]
    plt.plot(aristas_x,aristas_y,'k')

# Aristas camino optimo
for ii in nodos_opt[0:-1]:
    x_ii = np.array(nodos_x[ii])
    y_ii = np.array(nodos_y[ii])
    xfin = np.array(nodos_x[ii+1])
    yfin = np.array(nodos_y[ii+1])
    aristas_x = [x_ii, xfin]
    aristas_y = [y_ii, yfin]
    plt.plot(aristas_x,aristas_y,'r')

# Nodos
plt.plot(nodos_x,nodos_y,'or')

#plt.plot(0,0,'ok') #<-- plot a black point at the origin
plt.axis('equal')  #<-- set the axes to the same scale
#plt.xlim([-1,5]) #<-- set the x axis limits
#plt.ylim([-1,5]) #<-- set the y axis limits
plt.grid(b=True, which='major') #<-- plot grid lines


###########################################################################
# PLOT HOTSPOTS
###########################################################################
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
sizex = imagen_superficial.size[0]
sizey = imagen_superficial.size[1]

# Factors
xfactor = 78.5
yfactor = 79.6
meansize = (sizex+sizey)/2
mufactor = 2000

x = np.arange(-sizex/xfactor, sizex/xfactor, delta)
y = np.arange(-sizey/yfactor, sizey/yfactor, delta)
X, Y = np.meshgrid(x, y)

Z1 = mlab.bivariate_normal(X, Y, meansize/mufactor, meansize/mufactor, 0, 0)


plt.imshow(Z1, alpha=0.7, cmap='YlOrBr')

plt.show()
