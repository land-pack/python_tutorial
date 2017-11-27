import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../sample_data/image/digits.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size

cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

x = np.array(cells)

train = x[:, :50].reshape(-1, 400).astype(np.float32) # Size = (2500, 400)
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]

np.savez('../../sample_data/npz/knn_data.npz', train=train,train_labels=train_labels)


