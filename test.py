import glob
import os
import pickle
import numpy as np

from utils import load_data, plot_result


def get_latest_model():
    list_of_files = glob.glob('trained_models/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading model: {latest_file}")
    model = pickle.load(open(latest_file, 'rb'))

    return model

def main():
    _, _, X_test, y_test = load_data()
    
    model = get_latest_model()
    y_hat = model.predict(X_test)
    accuracy = np.sum(y_hat == y_test) / y_test.size
    print(f"Accuracy on test set: {accuracy:.0%}")

    plot_result(X_test, y_test, y_hat)

if __name__ == "__main__":
    main()
