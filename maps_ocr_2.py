import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import corner_harris, corner_peaks
from skimage.morphology import dilation, opening
import scipy

base_uri = '/Users/juanjogarau/Documents/MIT/HackMIT2017/png_maps'
img_path = os.path.join(base_uri, '33_4.png')

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
plt.figure()
plt.set_cmap('gray_r')
plt.imshow(aux)