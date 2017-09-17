import os
import numpy as np
import mpld3
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import probabilistic_hough_line

base_uri = r'/Users/miguelperezsanchis/Downloads/hackMIT/png_maps'
img_path = os.path.join(base_uri, '1_0.png')

img = Image.open(img_path).convert('L')

pix = np.array(img) * 1. / 255.

pix = (pix < 0.5).astype(int)

print(pix.shape)

lines_xy = probabilistic_hough_line(pix, theta=np.linspace(start=-np.pi/2., stop=np.pi/2., num=13))

print(lines_xy)

lines_xy_flat = []
for line in lines_xy:
    lines_xy_flat += line
    lines_xy_flat.append(None)

fix, axes = plt.subplots(1, 2)

axes[0].imshow(img, cmap=plt.cm.gray)
axes[0].set_title('Input image')

axes[1].plot(lines_xy_flat)
axes[1].set_title('Hough transform')
axes[1].set_xlabel('Angle (degree)')
axes[1].set_ylabel('Distance (pixel)')

plt.tight_layout()
mpld3.show(open_browser=False, ip='0.0.0.0', port=8000)

