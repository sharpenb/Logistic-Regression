import numpy as np
import os


def extract_gesture(index, directory):
    gesture = []
    directory += "/gesture" + str(index)
    try:
        for file_name in os.listdir(directory):
            if file_name.endswith(".txt"):
                file = open(directory + "/" + file_name, "r")
                x_array = []
                y_array = []
                z_array = []
                for line in file:
                    s = line.split()
                    x_array.append(float(s[0]))
                    y_array.append(float(s[1]))
                    z_array.append(float(s[2]))
                file.close()
                gesture.append(np.concatenate(([x_array], [y_array], [z_array]), axis=0))
    except:
        print("\nThe directory does not exist")

    return gesture


def create_dataset(gestures, shuffle=False):
    n_samples = 0
    n_features = 0
    for gesture in gestures:
        n_samples += len(gesture)
        for sample in gesture:
            if n_features < sample.shape[1]:
                n_features = sample.shape[1]
    X = np.zeros((n_samples, 3 * n_features))
    y = np.zeros(n_samples, dtype=int)
    i = 0
    for index, gesture in enumerate(gestures):
        for sample in gesture:
            X[i, :sample.shape[1]] = sample[0, :]
            X[i, n_features:n_features+sample.shape[1]] = sample[1, :]
            X[i, 2*n_features:2*n_features+sample.shape[1]] = sample[2, :]
            y[i] = int(index)
            i += 1
    if shuffle:
        p = np.random.permutation(n_samples)
        X = X[p, :]
        y = y[p]
    return X, y


