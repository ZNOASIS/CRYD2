import numpy as np
from imblearn.over_sampling import RandomOverSampler

def upsampling(data, labels):
    N, C, T, V, M = data.shape
    data = data.reshape(N, -1)
    ros = RandomOverSampler(random_state=0)
    X_resampled, Y_resampled = ros.fit_resample(data, labels)
    X_resampled = X_resampled.reshape(X_resampled.shape[0], C, T, V, M)
    return X_resampled, Y_resampled

data = np.load()
labels = np.load()

up_data, up_labels = upsampling(data, labels)

np.save('up_data.npy', up_data)
np.save('up_labels.npy', up_labels)
