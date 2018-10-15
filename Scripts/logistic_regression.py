import numpy as np


class LogisticRegression:

    def __init__(self, regularization_factor=0):
        self.W = None
        self.b = None
        self.regularization_factor = regularization_factor

    def softmax(self, s):
        exp_s = np.exp(s - np.max(s, axis=1, keepdims=True))
        softmax = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        return softmax

    def train(self, X_train, y_train, epochs=1000, learning_rate=.1, std=.001, W=None, b=None):
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        if W is None:
            self.W =  np.random.normal(loc=0.0, scale=std, size=(n_features, n_classes))
        else:
            self.W = W
        if b is None:
            self.b = np.zeros((1, n_classes))
        else:
            self.b = b
        training_accuracy = []
        training_cross_entropy = []

        for e in range(epochs):
            grad_W, grad_b = self.gradients(X_train, y_train)
            self.W -= learning_rate * grad_W
            self.b -= learning_rate * grad_b

            training_accuracy.append(self.accuracy(X_train, y_train))
            training_cross_entropy.append(self.regularized_cross_entropy(X_train, y_train))
            print(e, self.regularized_cross_entropy(X_train, y_train), self.accuracy(X_train, y_train))
        return training_accuracy, training_cross_entropy

    def train_early_stopping(self, X_train, y_train, X_val, y_val, n_early_stopping=100, epochs=2000, learning_rate=.1, std=.001, W=None, b=None):
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))
        if W is None:
            self.W =  np.random.normal(loc=0.0, scale=std, size=(n_features, n_classes))
        else:
            self.W = W
        if b is None:
            self.b = np.zeros((1, n_classes))
        else:
            self.b = b
        training_accuracy = [0]
        training_cross_entropy = []
        validation_accuracy = [0]
        validation_cross_entropy = []
        n_decrease_accuracy = 0

        for e in range(epochs):
            grad_W, grad_b = self.gradients(X_train, y_train)
            self.W -= learning_rate * grad_W
            self.b -= learning_rate * grad_b

            training_accuracy.append(self.accuracy(X_train, y_train))
            training_cross_entropy.append(self.regularized_cross_entropy(X_train, y_train))
            validation_accuracy.append(self.accuracy(X_val, y_val))
            validation_cross_entropy.append(self.regularized_cross_entropy(X_val, y_val))
            if validation_accuracy[-1] > validation_accuracy[-2]:
                n_decrease_accuracy = 0
            else:
                n_decrease_accuracy += 1
                if n_early_stopping <= n_decrease_accuracy:
                    break

            print(e, self.regularized_cross_entropy(X_train, y_train), self.accuracy(X_train, y_train))
        return training_accuracy, training_cross_entropy, validation_accuracy, validation_cross_entropy

    def gradients(self, X, y):
        n_samples = X.shape[0]
        p = self.softmax(np.dot(X, self.W) + self.b)
        p[np.arange(n_samples), y] -= 1
        p /= n_samples

        grad_W = X.T.dot(p) + self.regularization_factor * self.W
        grad_b = np.sum(p, axis=0, keepdims=True)

        return grad_W, grad_b

    def regularized_cross_entropy(self, X, y):
        n_samples = X.shape[0]
        p = self.softmax(np.dot(X, self.W) + self.b)

        cross_entropy = np.sum(-np.log(p[np.arange(n_samples), y])) / n_samples
        regularization = 1./2. * self.regularization_factor * np.sum(self.W * self.W)

        return cross_entropy + regularization

    def predict(self, X):
        p = self.softmax(np.dot(X, self.W) + self.b)
        y_pred = np.argmax(p, axis=1)

        return y_pred

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def confusion_map(self, X, y):
        n_classes = len(np.unique(y))
        confusion_map = np.zeros((n_classes, n_classes))
        y_pred = self.predict(X)
        for i in range(len(y)):
            if y[i] != y_pred[i]:
                confusion_map[y[i]][y_pred[i]] += 1
                confusion_map[y_pred[i]][y[i]] += 1
        return confusion_map
