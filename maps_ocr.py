import os
import numpy as np
import mpld3
import matplotlib.pyplot as plt
from PIL import Image
from skimage import feature as f
from skimage import morphology as morph
from skimage.transform import hough_line, hough_line_peaks

base_uri = r'/Users/miguelperezsanchis/Downloads/hackMIT/png_maps'
img_path = os.path.join(base_uri, '4_2_cut.png')

img = Image.open(img_path).convert('L')

pix = np.array(img) * 1. / 255.

# pix = f.canny(pix)

pix = (pix < 0.5).astype(int)

# ske = np.zeros(pix.shape)
# for i in range(100):
#     print(i)
#     ske = np.logical_or(ske, np.logical_and(pix, np.logical_not(morph.binary_opening(pix))))
#     pix = morph.binary_erosion(pix)
#
# plt.plot(ske)
# mpld3.show(open_browser=False, ip='0.0.0.0', port=8000)


# for _ in range(2):
#     pix = morph.binary_dilation(pix)
#
# for _ in range(1):
#     pix = morph.binary_opening(pix)
#
for _ in range(20):
    pix = morph.binary_dilation(pix)

print(pix.shape)

h, theta, d = hough_line(pix, theta=np.linspace(start=-np.pi/2., stop=np.pi/2, num=13))

print(h.shape, theta.shape, d.shape)

# plt.hist(out.flatten())
# mpld3.show(open_browser=False, ip='0.0.0.0', port=8000)

# lines = np.argwhere(h > 800.)

fig, ax = plt.subplots()

for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=20, num_peaks=12, threshold=0.3*h.max())):
    y0 = (dist - 0 * np.cos(angle)) / (np.sin(angle) + 0.0001)
    y1 = (dist - pix.shape[1] * np.cos(angle)) / (np.sin(angle) + 0.001)
    ax.plot((0, pix.shape[1]), (y0, y1), '-r')
ax.set_xlim((0, pix.shape[1]))
ax.set_ylim((pix.shape[0], 0))
ax.set_axis_off()
ax.set_title('Detected lines')
ax.imshow(pix, cmap=plt.cm.gray)
ax.set_title('Input image')

plt.tight_layout()
mpld3.show(open_browser=False, ip='0.0.0.0', port=8000)

