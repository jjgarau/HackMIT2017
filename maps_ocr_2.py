import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import corner_harris, corner_peaks
from skimage.morphology import dilation, opening
import scipy
import torch
from sklearn.cluster import KMeans

base_uri = '/Users/juanjogarau/Documents/MIT/HackMIT2017/png_maps'
img_path = os.path.join(base_uri, '1_0.png')

img = Image.open(img_path).convert('L')

pix = np.array(img) * 1. / 255.
pix = (pix < 0.7).astype(int)

corners = np.array(corner_peaks(corner_harris(pix), min_distance=1))

plt.figure()
plt.set_cmap('gray_r')
plt.imshow(pix)
plt.scatter(corners[:,1], corners[:,0], s=1)

aux = np.zeros(pix.shape)
for corner in corners:
    aux[corner[0], corner[1]] = 1

selem_mat = np.ones((12,12))

for _ in range(5):
    aux = dilation(aux)
plt.figure()
plt.set_cmap('gray_r')
plt.imshow(aux)

for _ in range(1):
    aux = opening(aux, selem=selem_mat)
    
for _ in range(8):
    aux = dilation(aux)
    
points = []
for i in range(len(aux)):
    for j in range(len(aux[i])):
        if aux[i, j] > 0.5:
            points.append([i, j])

inertias = []
inertias_d = []
n_c_min = 10
n_c_max = 70
for i in range(n_c_min, n_c_max):
    print(i)
    kmeans = KMeans(n_clusters=i).fit(np.array(points))
    inertias.append(kmeans.inertia_)
    if i!=10:
        inertias_d.append((kmeans.inertia_ - inertias[-2])/inertias[-1])

plt.figure()
plt.plot(np.arange(n_c_min, n_c_max), np.array(inertias))

plt.figure()
plt.plot(np.arange(n_c_min+1, n_c_max), np.array(inertias_d))

n_c_o = np.argmin(inertias_d) + n_c_min + 1
print("n_c_o: " + str(n_c_o))

kmeans = KMeans(n_clusters=n_c_o).fit(np.array(points))
centroids = kmeans.cluster_centers_
 
plt.figure()
plt.set_cmap('gray_r')
plt.imshow(aux)
plt.plot(centroids[:, 1], centroids[:, 0], 'ro')

plt.figure()
plt.imshow(pix)
plt.plot(centroids[:, 1], centroids[:, 0], 'ro')