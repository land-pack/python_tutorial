# docs from `http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_knn/py_knn_understanding/py_knn_understanding.html#knn-understanding`
# but it's already update some function name in opencv 3.x
# below code run on opencv 3.x


import cv2
import numpy as np
import matplotlib.pyplot as plt


# Feature set containing (x, y) values of 25 known/training data

trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)

# Labels each one either Red or Blue with numbers 0 and 1
responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)

# Take Red families and plot them
red = trainData[responses.ravel() == 0]
plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')

# Take Blue families and plot them
blue = trainData[responses.ravel() == 1]
plt.scatter(blue[:, 0], blue[:,1], 90, 'b', 's')


# For one comer
# newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)
# For many comer
newcomer = np.random.randint(0, 100, (10, 2)).astype(np.float32)
plt.scatter(newcomer[:, 0], newcomer[:,1], 80, 'g', 'o')
#knn = cv2.ml.KNearest()
knn = cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE, responses)
#ret, results, neighbours, dist = knn.find_nearest(newcomer, 3)
ret, result, neighbours, dist = knn.findNearest(newcomer, 3)

print 'result: ', result
print 'neighbours: ', neighbours
print 'distance: ', dist 

plt.show()
