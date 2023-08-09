from typing import List, Optional, Union, Dict, Tuple
import json
import argparse
from pathlib import Path

import numpy as np
from scipy import signal as sg
from scipy.ndimage import maximum_filter
from PIL import Image
import matplotlib.pyplot as plt


image = Image.open(r"C:\Users\ouriel\Downloads\kernel_7.png")
image = image.convert('RGB')

image = np.array(image)
print(image.shape)
green_kernel = image[:, :, 1]


green_kernel = green_kernel - np.mean(green_kernel)
green_kernel = green_kernel.astype(np.float32)
green_kernel /= 255
green_kernel -= np.mean(green_kernel)
green_kernel /= np.max(green_kernel)
print(np.sum(green_kernel))
# save the kernel
np.save(r"kernels\kernel_7.npy", green_kernel)



plt.imshow(green_kernel)
plt.show()
print()