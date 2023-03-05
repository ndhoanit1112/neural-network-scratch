# Handwritten Digit Recognition with a Neural Network

This is a neural network built to recognize handwritten digits. The network is built using only NumPy library and no pre-built machine learning libraries like TensorFlow or PyTorch.

## Requirements

This project requires the following packages to be installed:

- NumPy
- Keras
- Matplotlib

These packages can be installed using pip:

```properties
pip install numpy keras matplotlib
```

## Dataset

The model was trained on the MNIST dataset, which contains 60,000 training images and 10,000 testing images of handwritten digits. The images are grayscale and have a resolution of 28x28 pixels. The dataset is imported using the keras library.

## Model Architecture

The neural network consists of an input layer with 784 neurons (28x28 pixels), two hidden layers with 25 and 15 neurons respectively, and an output layer with 10 neurons corresponding to the 10 possible digit classes (0-9). The activation function used in the hidden layers is ReLU, and the output layer uses Softmax. The loss function used is cross-entropy loss.

## Files

- `model.py`: contains the implementation of the neural network model.
- `train.py`: loads the training data, trains the model, and saves the trained model to a file.
- `test.py`: loads the saved model, tests it on the test data, and prints the accuracy achieved.

## Usage

Train the neural network:

```
python train.py
```

This will train the neural network on the MNIST dataset and save the model to disk.

Test the neural network:

```
python test.py
```

This will load the trained model from disk and test it on a set of images from the MNIST dataset. The output will show the predicted label for some images and the accuracy of the model on the test set.

## Results

After training for 1500 epochs, the model achieved an accuracy of 91% on the testing set. Some predictions on test data:

![image](https://user-images.githubusercontent.com/32103386/222948458-b6e6f0b2-40b6-4940-8ca9-7b45108934f5.png)

Values of Cost function during training process:

![image](https://user-images.githubusercontent.com/32103386/222948580-c8604b5c-18f9-4d24-bf01-bd4bb046d06a.png)

## Customization

You can customize the neural network architecture by modifying the `model.py` file. This file defines the layers and parameters of the neural network. You can also modify the hyperparameters of the training process in the `train.py` file.

## References

- [Derivative of the Softmax Function and the Categorical Cross-Entropy Loss](https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)
- [Deriving the Backpropagation Equations from Scratch (Part 1)](https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-1-343b300c585a)
- [Deriving the Backpropagation Equations from Scratch (Part 2)](https://towardsdatascience.com/deriving-the-backpropagation-equations-from-scratch-part-2-693d4162e779)
