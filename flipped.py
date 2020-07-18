# example to show: How does the augmented data look like
import matplotlib.pyplot as plt

import cv2
import numpy as np

originalImage = cv2.imread('/home/workspace/CarND-Behavioral-Cloning-P3/center_2016_12_01_13_31_14_194.jpg')
image_original = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)

'''plt.imshow(image_original)
plt.title("image_original")
plt.show()'''

image_flipped = np.fliplr(image_original)
plt.imshow(image_flipped)
plt.title("image_flipped")
plt.show()