from logistic_regression import LogisticRegression
from uWaveGestureHandler import *
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np

directory = "../Datasets/uWaveGestureLibrary"

n_samples = 4481
n_training = int((50/100) * n_samples)
n_validation = int((20/100) * n_samples)

# Generate synthetic dataset
# X, y = make_blobs(n_samples=n_samples, random_state=None, centers=4, n_features=2)
# print(X.shape, y.shape)
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.xlabel('First feature')
# plt.ylabel('Second feature')
# plt.show()

# Extract uWaveGesture dataset
gesture_1 = extract_gesture(1, directory)
gesture_2 = extract_gesture(2, directory)
gesture_3 = extract_gesture(3, directory)
gesture_4 = extract_gesture(4, directory)
gesture_5 = extract_gesture(5, directory)
gesture_6 = extract_gesture(6, directory)
gesture_7 = extract_gesture(7, directory)
gesture_8 = extract_gesture(8, directory)
X, y = create_dataset([gesture_1, gesture_2, gesture_3, gesture_4, gesture_5, gesture_6, gesture_7, gesture_8], shuffle=True)
print(X.shape, y.shape)

# Build logistic regression model
log_reg = LogisticRegression(regularization_factor=.001)

# Training
X_train = X[:n_training]
y_train = y[:n_training]
X_val = X[n_training:n_training + n_validation]
y_val = y[n_training:n_training + n_validation]
training_accuracy, training_cross_entropy, validation_accuracy, validation_cross_entropy = log_reg.train_early_stopping(X_train, y_train, X_val, y_val, n_early_stopping=100, epochs=2000)

# Test & Visualization
plt.plot(training_accuracy, 'b', label="Training accuracy")
plt.plot(validation_accuracy, 'r', label="Test accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

plt.plot(training_cross_entropy, 'b', label="Training cross entropy")
plt.plot(validation_cross_entropy, 'r', label="Test cross entropy")
plt.xlabel('Epoch')
plt.ylabel('Cross entropy')
plt.show()

X_test = X[n_training+n_validation:]
y_test = y[n_training+n_validation:]
print(log_reg.regularized_cross_entropy(X_test, y_test), log_reg.accuracy(X_test, y_test))
confusion_map = log_reg.confusion_map(X_test, y_test)
plt.imshow(confusion_map, cmap='Reds', interpolation='nearest')
plt.show()

training_accuracies = []
training_cross_entropies = []
test_accuracies = []
test_cross_entropies = []
kf = KFold(n_splits=4)
for train_index, test_index in kf.split(X):
    log_reg = LogisticRegression(regularization_factor=.001)
    training_accuracy, training_cross_entropy = log_reg.train(X[train_index], y[train_index],  epochs=1200)
    training_accuracies.append(log_reg.accuracy(X[train_index], y[train_index]))
    training_cross_entropies.append(log_reg.regularized_cross_entropy(X[train_index], y[train_index]))
    test_accuracies.append(log_reg.accuracy(X[test_index], y[test_index]))
    test_cross_entropies.append(log_reg.regularized_cross_entropy(X[test_index], y[test_index]))

plt.plot(training_accuracies, 'b', label="Training accuracies")
plt.axhline(y=np.mean(training_accuracies), color='b', linestyle='--')
plt.plot(test_accuracies, 'g', label="Test accuracies")
plt.axhline(y=np.mean(test_accuracies), color='g', linestyle='--')
plt.xlabel('K-fold')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0, 1)
plt.show()

plt.plot(training_cross_entropies, 'b', label="Training cross entropies")
plt.axhline(y=np.mean(training_cross_entropies), color='b', linestyle='--')
plt.plot(test_cross_entropies, 'g', label="Test cross entropies")
plt.axhline(y=np.mean(test_cross_entropies), color='g', linestyle='--')
plt.xlabel('K-fold')
plt.ylabel('Cross entropy')
plt.legend()
plt.show()

