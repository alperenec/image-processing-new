import cv2
import numpy as np
from matplotlib import pyplot as plt

foto_gri = cv2.imread("baboon.bmp",0)
cv2.imshow("baboon_gri",foto_gri)
cv2.waitKey()

h = np.zeros(256, np.int32)
row, col = foto_gri.shape[0], foto_gri.shape[1]

for i in range(0, row):
  for j in range(0, col):
    h[foto_gri[i, j]] += 1
plt.figure()
plt.plot(h)
plt.show()