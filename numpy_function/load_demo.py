import numpy as np

with np.load('../../sample_data/npz/knn_data.npz') as data:
    print data.files
    train = data['train']
    train_labels = data['train_labels']

