import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist


def plot_data(X, y):
    fig = plt.figure()
    fig.subplots_adjust(hspace=.5)
    X = X.reshape((X.shape[0], 28, -1))
    for i in range(9):
        ax = fig.add_subplot(330 + 1 + i)
        ax.imshow(X[i], cmap=plt.get_cmap('gray'))
        ax.set_title(f"{y[i]},{y[i]}",fontsize=10)
        ax.set_axis_off()
    
    plt.show()

def plot_result(X_test: np.ndarray, y_test: np.ndarray, y_hat: np.ndarray):
    X_test = X_test.reshape((X_test.shape[0], 28, -1))
    indexes = np.random.randint(X_test.shape[0], size=9)

    fig = plt.figure("Predictions for some input samples")
    fig.subplots_adjust(hspace=.5)
    for i, sample_indx in enumerate(indexes):
        ax = fig.add_subplot(330 + 1 + i)
        ax.imshow(X_test[sample_indx], cmap=plt.get_cmap('gray'))
        ax.set_title(f"Label: {y_test[sample_indx]}, Predicted: {y_hat[sample_indx]}",fontsize=10)
        ax.set_axis_off()
    
    plt.show()

def plot_cost_hist(cost_hist):
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
    ax1.plot(cost_hist[:500])
    ax2.plot(500 + np.arange(len(cost_hist[500:])), cost_hist[500:])
    ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
    ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
    ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
    plt.show()

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train, y_train = shuffle(X_train, y_train, random_state=0)

    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))

    return X_train, y_train, X_test, y_test

def relu(Z: np.ndarray):
    return np.maximum(Z, 0)

def relu_derivative(Z: np.ndarray):
    return Z > 0

def softmax(Z: np.ndarray):
    return np.exp(Z) / sum(np.exp(Z))