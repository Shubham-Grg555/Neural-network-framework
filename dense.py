from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size : int, output_size : int):
        """Initalise the weights and biases randomly in an array with the correct
        dimensions to allow for an easy application of the output formula

        Args:
            input_size (int): Number of neuron in the input.
            output_size (int): Number of neurons in the output
        """
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

        self.prev_mean_gradient = 0
        self.prev_squared_gradient_avg = 0
        self.timestep = 1

    def forward(self, input):
        """Returns output value via the formula Y = W * X + B

        Args:
            input (_type_): _description_

        Returns:
            float: Multiplies all the weights with all the corrseponding
            inputs, then adds the corresponding bias value, then returns it.
        """
        self.input = input
        # return np.dot(self.weights, self.input) + self.bias

        # print(f"Weights dimension {self.weights.shape}")
        # print(f"Input dimension {input.shape}")

        return np.dot(self.weights, input) + self.bias

    def backward(self, output_gradient, learning_rate):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-08

        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        mean_gradient = (
            beta1 * self.prev_mean_gradient + ((1 - beta1) * weights_gradient)
        )
        squared_gradient_avg = (
            beta2 * self.prev_squared_gradient_avg + 
            ((1 - beta2) * weights_gradient ** 2)
        )

        bias_corrected_mean = mean_gradient / (1 - beta1 ** self.timestep)
        bias_corrected_variance = (
            squared_gradient_avg / (1 - beta2 ** self.timestep)
        )

        self.timestep += 1
        # Seems like actually updating the weights and not weights gradient e.g imp thing to see
        # normal is learning+rate * ewights_gradient
        # Instead we use RMSprop to do extra changes the weight gradient

        ### Do extra research to make sure etc
        self.weights -= (
            learning_rate * (bias_corrected_mean / (np.sqrt(bias_corrected_variance + epsilon)))
        )

        #self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=1, keepdims=True)
        return input_gradient
