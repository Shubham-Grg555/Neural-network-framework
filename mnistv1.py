### ONLY used tensor flow for getting dataset, processing images etc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from dense import Dense
from activation import Tanh, ReLU
from losses import mse, mse_gradient

from keras.datasets import mnist

mnist = tf.keras.datasets.mnist
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()


def pre_process_data(x, y : int, amount_of_images : int):
    """Normalises the training data (x) to be between 0 and 1 and the correct
    result of the data to a one-hot encoded value (y).

    Args:
        x (numpy array of floats): A numpy array where each value represents a
        pixel value, used to differentiate the written number pixels (white)
        from the background (black).
        y (int): The correct number written corresponding to the hand written
        training data x. 
        amount_of_images (int): Number of images being used to train the neural
        network (maximum of 60,000).

    Returns:
        numpy array and one hot encoded value: A numpy array that has been
        normalised to show float values between 0 and 1, but otherwise the
        same.
        One-hot encoded value of the correct result of the corresponding
        training images.
    """
    x = x.astype(np.float32) / 255.0
    x = x.reshape((x.shape[0], 28 * 28, 1))

    # One-hot encode y (e.g., digit 5 → [0,0,0,0,0,1,0,0,0,0])
    y = tf.keras.utils.to_categorical(y, num_classes=10)

    y = y.reshape((y.shape[0], 10, 1))

    return x[:amount_of_images], y[:amount_of_images]

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

epochs = 3
learning_rate = 0.01

network = [
    # Dense(28 * 28, 40),
    # Tanh(),
    # Dense(40, 10),
    # Tanh()

    Dense(28 * 28, 40),
    Tanh(),
    Dense(40, 10),
    Tanh()

    # Dense(28 * 28, 128),
    # ReLU(),
    # Dense(128, 64),
    # ReLU(),
    # Dense(64, 10),
    # ReLU()
]

loss_history = []

def train(x_train, y_train, network, epochs, learning_rate):
    """Trains the network and corrects it by using the mean squared error to
    calculate how incorrect it was. Then gets the gradient of the mean squared
    error to use as a starting gradient for back propagation with the adam
    optimiser. Finally completes the error formula by dividing it by the total
    number of training images.

    Args:
        x_train (numpy array): Training images represnted in a normalised numpy
        array between 0 and 1.
        y_train (one-hot encoded values): Correspond correct result of the
        training images.
        network (Neural network): The custom neural network made e.g number of
        neurons, activation functions etc.
        epochs (int): Number of epochs used to train the network, meaning
        total number of passes done on the whole data set.
        learning_rate (float): Paramter associated with how fast the network
        learns at.
    """
    for i in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            output = predict(network, x)

            error += mse(y, output)

            grad = mse_gradient(y, output)       

            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
            
            error /= len(x_train)
        loss_history.append(error)

amount_of_images = 60000

x_batch, y_batch = pre_process_data(x_train_full, y_train_full,
                                    amount_of_images)

train(x_batch, y_batch, network, epochs, learning_rate)

plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0,0.00006)
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)
plt.show()

number_of_testing_images = 10000
correct = 0

x_test, y_test = pre_process_data(x_test, y_test, number_of_testing_images)

# Tests the network and outputs it's accuracy at the end.
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    if np.argmax(output) == np.argmax(y):
        correct += 1

print(f"Accuracy: {correct}/{number_of_testing_images} = {correct / number_of_testing_images:.2%}")
