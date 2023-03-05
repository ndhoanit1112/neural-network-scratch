from datetime import datetime
import pickle
import numpy as np

from model import Model
from utils import load_data, plot_cost_hist
np.set_printoptions(precision=4)
np.seterr(divide='ignore', invalid='ignore')


def main():
    X_train, y_train, _, _ = load_data()
    
    alpha = 0.1
    iterations = 1500
    model = Model()
    model.fit(X_train, y_train, alpha, iterations)

    model.clean_up_training_data()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    pickle.dump(model, open(f"trained_models/model_{now}", 'wb'))

    plot_cost_hist(model.cost_hist)

if __name__ == "__main__":
    main()
