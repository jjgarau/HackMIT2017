###########################################################################
# IMPORT LIBRARIES AND MAP
###########################################################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.misc import imread                   # Cargo imread de scipy.misc
import matplotlib.mlab as mlab
from PIL import Image
import graph
import nodes

filepath = '/Users/alvarogomezinesta/Documents/Anno_5_TFG/HackMIT/33_4all.png'
ns = nodes.NodeSet(filepath)

#imagen_superficial = Image.open('33_4.png')
imagen_superficial = ns.map_
size_img = imagen_superficial.shape

###########################################################################
# DATA
###########################################################################

# Grafo (nodos en orden)
#nodos_conec = [[1],[2,3],[2],[4],[4]] # Conectar cada nodo como minimo consigo mismo

nodos_x = np.array(ns.nodes)[:, 1]
nodos_y = np.array(ns.nodes)[:, 0]

N = len(ns.nodes) # Numero total de nodos

# adjAndDist = ....
# gr = graph.Graph(adjAndDist)
# orig = 0
# dest = 1
# path, dista = gr.get_dijkstra_path(orig, dest)
# print(path)
# print(dista)

# Camino optimo
nodos_opt = [0,1,3]

###########################################################################
# HOTSPOTS DISTRIBUTION
###########################################################################
Nh = 3 # Number of hotspots per floor
D = [3,2.5,2] # Density of people connected to each hotspot
mindis = 0.2 # Minimum distance between hotspots factor
h0 = np.random.randint(0,N)
xh0 = nodos_x[h0]
yh0 = nodos_y[h0]

h1 = np.random.randint(0,N)
xh1 = nodos_x[h1]
yh1 = nodos_y[h1]
l01 = ((xh1-xh0)**2+(yh0-yh1)**2)**0.5
print(l01,size_img)
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

# ###########################################################################
# # PLOT FONDO (MAPA)
# ###########################################################################
# Creo una figura
plt.figure()
plt.set_cmap('gray_r')
# Muestro la imagen en pantalla
plt.imshow(imagen_superficial)
#
# ###########################################################################
# # PLOT GRAFO
# ###########################################################################
#
# # Aristas
# for ii in range(0,N):
#     x_ii = np.array(nodos_x[ii])
#     y_ii = np.array(nodos_y[ii])
#     conexiones = np.array(len(nodos_conec[ii]))
#     aristas_x = [x_ii*np.ones(conexiones),nodos_x[np.array(nodos_conec[ii])]]
#     aristas_y = [y_ii*np.ones(conexiones),nodos_y[np.array(nodos_conec[ii])]]
#     plt.plot(aristas_x,aristas_y,'k')
#
# # Aristas camino optimo
# for ii in nodos_opt[0:-1]:
#     x_ii = np.array(nodos_x[ii])
#     y_ii = np.array(nodos_y[ii])
#     xfin = np.array(nodos_x[ii+1])
#     yfin = np.array(nodos_y[ii+1])
#     aristas_x = [x_ii, xfin]
#     aristas_y = [y_ii, yfin]
#     plt.plot(aristas_x,aristas_y,'r')
#
# # Nodos
plt.plot(nodos_x,nodos_y,'or')
#
# ###########################################################################
# # PLOT HOTSPOTS
# ###########################################################################
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 1
factorsigma = 10

sizex = imagen_superficial.shape[1]
sizey = imagen_superficial.shape[0]
Z1 = np.zeros((sizey,sizex))
for jj in range(0,Nh):

     x = np.arange(0, sizex, delta)
     y = np.arange(0, sizey, delta)
     sigma = sizex/factorsigma
     X, Y = np.meshgrid(x, y)

     Z1 = Z1+(mlab.bivariate_normal(X, Y, sigma, sigma, xh[jj], yh[jj]))*D[jj]

plt.imshow(Z1, alpha=0.5, cmap='Blues')
print(xh)
print(yh)


#plt.plot(xh,yh,'og') #<-- plot a black point at the origin
#plt.axis('equal')  #<-- set the axes to the same scale
#plt.xlim([-1,5]) #<-- set the x axis limits
#plt.ylim([-1,5]) #<-- set the y axis limits
plt.grid(b=True, which='major') #<-- plot grid lines
plt.show()
