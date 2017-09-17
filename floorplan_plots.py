###########################################################################
# IMPORT LIBRARIES AND MAP
###########################################################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.misc import imread                   # Cargo imread de scipy.misc
import matplotlib.mlab as mlab
from PIL import Image

imagen_superficial = Image.open('33_4b2.png')
size_img = imagen_superficial.size

###########################################################################
# DATA
###########################################################################

# Grafo (nodos en orden)
nodos_conec = [[1],[2,3],[2],[4],[4]] # Conectar cada nodo como minimo consigo mismo
nodos_x = np.array([75,80,35,55,50])
nodos_y = np.array([20,185,185,235,300])

N = len(nodos_conec) # Numero total de nodos

# Camino optimo
nodos_opt = [0,1,3]

###########################################################################
# HOTSPOTS DISTRIBUTION
###########################################################################
Nh = 3 # Number of hotspots per floor
D = [1,3,2] # Density of people connected to each hotspot
mindis = 0.1 # Minimum distance between hotspots factor
h0 = np.random.randint(0,N)
xh0 = nodos_x[h0]
yh0 = nodos_y[h0]

h1 = np.random.randint(0,N)
xh1 = nodos_x[h1]
yh1 = nodos_y[h1]
l01 = ((xh1-xh0)**2+(yh0-yh1)**2)**0.5
while (l01 < (size_img[1]*size_img[0])**0.5*mindis):
    h1 = np.random.randint(0,N)
    xh1 = nodos_x[h1]
    yh1 = nodos_y[h1]
    l01 = ((xh1-xh0)**2+(yh0-yh1)**2)**0.5


h2 = np.random.randint(0,N)
xh2 = nodos_x[h2]
yh2 = nodos_y[h2]
l02 = ((xh2-xh0)**2+(yh0-yh2)**2)**0.5
l12 = ((xh2-xh1)**2+(yh2-yh1)**2)**0.5
while (l02 < (size_img[1]*size_img[0])**0.5*mindis) or (l12 < (size_img[1]*size_img[0])**0.5*mindis):
    h2 = np.random.randint(0,N)
    xh2 = nodos_x[h2]
    yh2 = nodos_y[h2]
    l02 = ((xh2-xh0)**2+(yh0-yh2)**2)**0.5
    l12 = ((xh2-xh1)**2+(yh2-yh1)**2)**0.5


xh = [xh0,xh1,xh2]
yh = [yh0,yh1,yh2]

###########################################################################
# PLOT FONDO (MAPA)
###########################################################################
# Creo una figura
plt.figure()
# Muestro la imagen en pantalla
plt.imshow(imagen_superficial)

###########################################################################
# PLOT GRAFO
###########################################################################

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

###########################################################################
# PLOT HOTSPOTS
###########################################################################
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

for jj in range(0,Nh-1):
    delta = 0.025
    sizex = imagen_superficial.size[0]
    sizey = imagen_superficial.size[1]

    if D[jj] == 1:
        colorr = 'Greens'
    elif D[jj] == 2:
        colorr = 'Blues'
    else:
        colorr = 'Reds'

    # Factors
    xfactor = 80
    yfactor = 80
    meansize = (sizex+sizey)/2
    mufactor = 400

    x = np.arange(-sizex/xfactor, sizex/xfactor, delta)
    y = np.arange(-sizey/yfactor, sizey/yfactor, delta)
    X, Y = np.meshgrid(x, y)

    asd = 40
    Z1 = mlab.bivariate_normal(X, Y, meansize/mufactor, meansize/mufactor, (xh[jj]-sizex/2)/asd, (yh[jj]-sizey/2)/asd)

    plt.imshow(Z1, alpha=0.5, cmap=colorr)

print(xh)
print(yh)


#plt.plot(xh,yh,'og') #<-- plot a black point at the origin
plt.axis('equal')  #<-- set the axes to the same scale
#plt.xlim([-1,5]) #<-- set the x axis limits
#plt.ylim([-1,5]) #<-- set the y axis limits
plt.grid(b=True, which='major') #<-- plot grid lines
plt.show()
