### ONLY used tensor flow for getting dataset, processing images etc
# ALL neural network componenets is form custom made one
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from dense import Dense
from activation import ReLU, Softmax, Tanh
from losses import mse, mse_gradient, binary_cross_entropy, binary_cross_entropy_gradient

from keras.datasets import mnist

mnist = tf.keras.datasets.mnist

number_of_training_images = 10000
number_of_testing_images = 10000

(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

### IMPO BELOW

# def pre_process_data(x,y, amount):
#     x = np.reshape(x.shape[0], 28 * 28, 1)
#     x = tf.keras.utils.normalize(x, axis = 1)
#     #x = np.reshape(x, (x.shape[0], 28 * 28, 1))
#     #y = np_utils.to_categorical(y)
#     y = np.reshape(y, (y.shape[0], 10, 1))
#     return x[:amount], y[:amount]

def pre_process_data(x, y, start_of_batch, end_of_batch):

    x = x.astype(np.float32) / 255.0
    x = x.reshape((x.shape[0], 28 * 28, 1))

    # One-hot encode y (e.g., digit 3 → [0,0,0,1,0,0,0,0,0,0])
    y = tf.keras.utils.to_categorical(y, num_classes=10)

    y = y.reshape((y.shape[0], 10, 1))

    return x[start_of_batch:end_of_batch], y[start_of_batch:end_of_batch]


# x_train = tf.keras.utils.normalize(x_train, axis = 1)
# x_test = tf.keras.utils.normalize(x_test, axis = 1)

#print(f"printing x_train: {x_train}")

### PERHAPS IMPO BELOW

#x_train, y_train = pre_process_data(x_train, y_train, number_of_training_images)
#x_test, y_test = pre_process_data(x_test, y_test, number_of_testing_images)

# print(f"printing x_train: {x_train}")
# plt.imshow(x_train[0], cmap = 'binary')
# plt.show()

# x_train = np.reshape(x_train[0], 1)
# y_train = np.reshape(y_train[0], 1)
# x_test = np.reshape(x_test[0], 1)
# y_test = np.reshape(y_test[0], 1)



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
    for i in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            
            output = predict(network, x)

           # print(f"Want network thinks I believe: {np.argmax(output)}")
           # print(f"Actual value: {np.argmax(y_train)}")

            #print(f"Second input type test???? {input} \n input shape test {np.shape(input)}")

            error += mse(y, output)

            grad = mse_gradient(y, output)

            #softmax here?
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

correct = 0

# To fix, for some reason only run loops once
# print(f"x_test: {x_test}")
# print(f"x_test: {x_test.all}")
# print(f"y_test: {y_test}")

# plt.imshow(x_test[0], cmap = 'binary')
# plt.show() 

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
    # reason can't print explained above
    # plt.imshow(x_test[0], cmap = 'binary')
    # plt.show() 
    
    #temp = np.argmax(output)

    #print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))