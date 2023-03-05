import math
import numpy as np

from utils import relu, relu_derivative, softmax

class Model:
    def zscore_normalize_features(self, X: np.ndarray):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)

        X_norm = np.divide(X - mu, sigma, out=np.zeros(X.shape, dtype=float), where=sigma!=0)
        return (X_norm, mu, sigma)

    def preprocess_data(self, X_train: np.ndarray):
        X_train, mu, sigma = self.zscore_normalize_features(X_train)
        X_train = X_train.transpose()

        return X_train, mu, sigma

    def one_hot(self, Y: np.ndarray):
        Y_one_hot = np.zeros((Y.size, Y.max() + 1))
        Y_one_hot[np.arange(Y.size), Y] = 1
        # for indx, y in enumerate(Y_one_hot):
        #     y[Y[indx]] = 1

        return Y_one_hot.T

    def get_predictions(self, A3: np.ndarray):
        return np.argmax(A3, axis=0)

    def compute_accuracy(self, predictions: np.ndarray, Y: np.ndarray):
        return np.sum(predictions == Y) / Y.size
    
    def init_params(self, layer_units):
        self.W1 = np.random.rand(layer_units[1], layer_units[0]) - 0.5
        self.b1 = np.random.rand(layer_units[1], 1) - 0.5
        self.W2 = np.random.rand(layer_units[2], layer_units[1]) - 0.5
        self.b2 = np.random.rand(layer_units[2], 1) - 0.5
        self.W3 = np.random.rand(layer_units[3], layer_units[2]) - 0.5
        self.b3 = np.random.rand(layer_units[3], 1) - 0.5

    # Forward propagation for m samples
    # X: 784 x m (784 features x m samples)
    # W1: 25 x 784
    # b1: 25 x 1
    # W2: 15 x 25
    # b2: 15 x 1
    # W3: 10 x 15
    # b2: 10 x 1
    def forward_prop(self, X: np.ndarray):
        Z1 = np.matmul(self.W1, X) + self.b1 # Z1: 25 x m
        A1 = relu(Z1) # A1: 25 x m
        Z2 = np.matmul(self.W2, A1) + self.b2 # Z2: 15 x m
        A2 = relu(Z2) # A2: 15 x m
        Z3 = np.matmul(self.W3, A2) + self.b3 # Z3: 10 x m
        A3 = softmax(Z3) # A3: 10 x m

        return Z1, A1, Z2, A2, Z3, A3

    def backward_prop(self, Y_one_hot: np.ndarray, Z1: np.ndarray, A1: np.ndarray, Z2: np.ndarray, A2: np.ndarray, A3: np.ndarray):
        m = self.X_train.shape[1] # number of training samples
        
        dj_dZ3 = A3 - Y_one_hot # 10 x m
        dj_dW3 = np.matmul(dj_dZ3, A2.T) # 10 x 15 (sum of dj_dW3 of all samples)
        dj_dW3 /= m # 10 x 15 (average value of dj_dW3 - gradient for W3)
        dj_db3 = dj_dZ3 # 10 x m
        dj_db3 = np.sum(dj_db3, axis=1, keepdims=True) / m # 10 x 1

        dj_dZ2 = np.matmul(self.W3.T, dj_dZ3) * relu_derivative(Z2) # 15 x m
        dj_dW2 = np.matmul(dj_dZ2, A1.T) # 15 x 25
        dj_dW2 /= m
        dj_db2 = dj_dZ2 # 15 x m
        dj_db2 = np.sum(dj_db2, axis=1, keepdims=True) / m # 15 x 1

        dj_dZ1 = np.matmul(self.W2.T, dj_dZ2) * relu_derivative(Z1) # 25 x m
        dj_dW1 = np.matmul(dj_dZ1, self.X_train.T) # 25 x 784
        dj_dW1 /= m
        dj_db1 = dj_dZ1 # 25 x m
        dj_db1 = np.sum(dj_db1, axis=1, keepdims=True) / m # 25 x 1

        return dj_dW1, dj_db1, dj_dW2, dj_db2, dj_dW3, dj_db3

    def update_params(
            self,
            dj_dW1: np.ndarray, dj_db1: np.ndarray, dj_dW2: np.ndarray, dj_db2: np.ndarray, dj_dW3: np.ndarray, dj_db3: np.ndarray,
    ):
        self.W1 -= self.alpha * dj_dW1
        self.b1 -= self.alpha * dj_db1
        self.W2 -= self.alpha * dj_dW2
        self.b2 -= self.alpha * dj_db2
        self.W3 -= self.alpha * dj_dW3
        self.b3 -= self.alpha * dj_db3

    # A3: 10 x m
    # Y: m x 1
    def compute_cost(self, A3: np.ndarray, Y: np.ndarray):
        probs = A3[Y.T, np.arange(Y.size)][0]
        cost = -np.sum(np.log(probs))

        return cost


    def gradient_descent(self):
        Y_one_hot = self.one_hot(self.y_train)
        Y_2D = self.y_train.reshape(-1 , 1)
        cost_hist = []
        for i in range(self.iterations):
            Z1, A1, Z2, A2, _, A3 = self.forward_prop(self.X_train)
            dj_dW1, dj_db1, dj_dW2, dj_db2, dj_dW3, dj_db3 = self.backward_prop(
                Y_one_hot, Z1, A1, Z2, A2, A3
            )
            self.update_params(
                dj_dW1, dj_db1, dj_dW2, dj_db2, dj_dW3, dj_db3
            )

            cost = self.compute_cost(A3, Y_2D)
            cost_hist.append(cost)

            if i% math.ceil(self.iterations/50) == 0:
                print("=======")
                print("Iteration: ", i)
                print("Cost: ", cost)
                print("Accuracy: ", self.compute_accuracy(self.get_predictions(A3), self.y_train))

        self.cost_hist = cost_hist

    def fit(self, X: np.ndarray, Y: np.ndarray, alpha: float, iterations: int):
        self.X_train, self.mu, self.sigma = self.preprocess_data(X)
        self.y_train = Y
        self.alpha = alpha
        self.iterations = iterations
        layer_units = [self.X_train.shape[0], 25, 15, 10]

        self.init_params(layer_units)
        self.gradient_descent()

        print(
            f"Total {np.sum([self.W1.size, self.b1.size, self.W2.size, self.b2.size, self.W3.size, self.b3.size])} params trained!"
        )
    
    def predict(self, X: np.ndarray):
        X_norm = np.divide(X - self.mu, self.sigma, out=np.zeros(X.shape, dtype=float), where=self.sigma!=0)
        X_norm = X_norm.transpose()
        _, _, _, _, _, A3 = self.forward_prop(X_norm)

        return self.get_predictions(A3)
    
    def clean_up_training_data(self):
        self.X_train = []
        self.y_train = []
