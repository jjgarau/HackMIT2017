import os
import numpy as np
import mpld3
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import hough_line, hough_line_peaks

base_uri = r'/Users/miguelperezsanchis/Downloads/hackMIT/png_maps'
img_path = os.path.join(base_uri, '1_0.png')

img = Image.open(img_path).convert('L')

pix = np.array(img) * 1. / 255.

pix = (pix < 0.5).astype(int)

print(pix.shape)

out, angles, d = hough_line(pix, theta=np.linspace(start=-np.pi/2., stop=np.pi/2., num=13))

out, angles, d = hough_line_peaks(out, angles, d)

print(angles)
print(d)

# fig, ax = plt.subplots()
# ax.set_title("Floor 1 of bldg 1", size=20)
#
#
# scatter = ax.imshow(pix, origin='lower')
# tooltip = mpld3.plugins.PointHTMLTooltip(scatter, labels=titles)
# mpld3.plugins.connect(fig, tooltip)

plt.imshow(
    out
)
plt.tight_layout()
mpld3.show(open_browser=False, ip='0.0.0.0', port=8000)

