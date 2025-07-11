### ONLY used tensor flow for getting dataset, processing images etc
import tensorflow as tf
import numpy as np

from dense import Dense
from activation import ReLU, Softmax, Tanh
from losses import binary_cross_entropy, binary_cross_entropy_gradient

from keras.datasets import mnist

mnist = tf.keras.datasets.mnist
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

def pre_process_data(x, y, start_of_batch, end_of_batch):
    """Normalises the training data (x) to be between 0 and 1 and the correct
    result of the data to a one-hot encoded value (y).

    Args:
        x (numpy array of floats): A numpy array where each value represents a
        pixel value, used to differentiate the written number pixels (white)
        from the background (black).
        y (int): The correct number written corresponding to the hand written
        training data x. 
        start_of_batch (int): Index of training images to start taking from for
        batch normalisation.
        end_of_batch (int): Index of training images to stop taking from for
        batch normalisation

    Returns:
        numpy array and one hot encoded value: A numpy array that has been
        normalised to show float values between 0 and 1, but otherwise the
        same.
        One-hot encoded value where it turns the correct result into an easier
        format to say the number the ai thought was most likely e.g
        digit 3 → [0,0,0,1,0,0,0,0,0,0] as first digit represents 0, next digit
        represents 1 etc.
    """
    x = x.astype(np.float32) / 255.0
    x = x.reshape((x.shape[0], 28 * 28, 1))

    # One-hot encode y (e.g., digit 3 → [0,0,0,1,0,0,0,0,0,0])
    y = tf.keras.utils.to_categorical(y, num_classes=10)

    y = y.reshape((y.shape[0], 10, 1))

    return x[start_of_batch:end_of_batch], y[start_of_batch:end_of_batch]

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

epochs = 3
learning_rate = 0.001
network = [
    Dense(28 * 28, 40),
    ReLU(),
    Dense(40, 10),
    Softmax()
]

def train(x_train, y_train, network, epochs, learning_rate):
    """Trains the network and corrects it by using the binary cross entropy to
    calculate how incorrect it was. Then gets the gradient of the binary cross
    entropy to use as a starting gradient for back propagation with the adam
    optimiser. Finally completes the error formula by dividing it by the total
    number of training images.

    Args:
        x_train (numpy array): Training images represnted in a normalised numpy
        array between 0 and 1.
        y_train (one-hot encoded values): Correspond correct result of the
        training images.
        network (Neural network): The neural network made e.g number of
        neurons, activation functions etc.
        epochs (int): Number of epochs used to train the network. An epoch
        means number of passes down through the whole data set e.g training
        with with 10 images, 3 passes = 3 epoch so used the 10 images 3 times.
        learning_rate (float): How fast the network learns (not too high so it
        can reach the minimum point without overshooting, but not small so it
        takes a really long time to get to the minimum point).
    """
    for i in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            
            output = predict(network, x)

            test += binary_cross_entropy(y, output)

            test_grad = binary_cross_entropy_gradient(y, output)

            print(f"testing binary cross entropy: {test}")
            print(f"testing THE GRADIENT OF binary cross entropy: {test_grad}")            

            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
            
            error /= len(x_train)



start_of_batch = 0
end_of_batch = 60000

x_batch, y_batch = pre_process_data(x_train_full, y_train_full,
                                    start_of_batch, end_of_batch)

train(x_batch, y_batch, network, epochs, learning_rate)

# for i in range(60):
#     x_batch, y_batch = pre_process_data(x_train_full, y_train_full,
#                                         start_of_batch, end_of_batch)
#     start_of_batch += 1000
#     end_of_batch += 1000

#     train(x_batch, y_batch, network, epochs, learning_rate)

#train(x_train, y_train, network, epochs, learning_rate)

number_of_testing_images = 10000
correct = 0

x_test, y_test = pre_process_data(x_test, y_test, start_of_batch = 0,
                                  end_of_batch = 10000)
for x, y in zip(x_test, y_test):
    output = predict(network, x)

    # CAN't print images as not in correct 784 total pixel format
    # is just some numbers I beleive idk check again e.g final x and y just 10000
    #print(f"X test: {x}")
    #print(f"Y test: {y_test}")

    if np.argmax(output) == np.argmax(y):
        correct += 1

   # print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))

print(f"Accuracy: {correct}/{number_of_testing_images} = {correct / number_of_testing_images:.2%}")