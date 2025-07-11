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

        # variables needed for the adam optimiser
        self.prev_mean_gradient = 0
        self.prev_gradient_variance = 0
        # self.prev_mean_bias = 0
        # self.prev_bias_variance = 0

        self.gradient_mean = np.zeros_like(self.weights)
        self.gradient_variance = np.zeros_like(self.weights)

        self.bias_mean = np.zeros_like(self.bias)
        self.bias_variance = np.zeros_like(self.bias)

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
        return np.dot(self.weights, input) + self.bias

    def adamOptimiser(self, weights_gradient, output_gradient, learning_rate):
        """Applies the Adam optimiser algorthim to update the weights and
        biases of the network.

        Args:
            weights_gradient (numpy array): Gradient of the weights in the
            current layer to determine how much of each weight should be
            adjusted to minimise the loss.
            output_gradient (numpy array): Gradient of the loss with respect to
            the output of the current layer, coming from the next layer in
            backpropagation.
            learning_rate (float): Learning rate for speed of paramter
            updates.
        """
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-08

        # Adam optimiser on weights
        mean_gradient = (
            beta1 * self.prev_mean_gradient + ((1 - beta1) * weights_gradient)
        )
        gradient_variance = (
            beta2 * self.prev_gradient_variance + 
            ((1 - beta2) * weights_gradient ** 2)
        )

        mean_gradient_hat = mean_gradient / (1 - beta1 ** self.timestep)
        gradient_variance_hat = (
            gradient_variance / (1 - beta2 ** self.timestep)
        )

        self.weights -= (
            learning_rate *
            (mean_gradient_hat / (np.sqrt(gradient_variance_hat + epsilon)))
        )

        self.prev_mean_gradient = mean_gradient
        self.prev_gradient_variance = gradient_variance

        # Adam optimiser on bias
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)
        mean_bias = beta1 * self.prev_mean_bias + (1 - beta1) * bias_gradient
        bias_variance = (
            beta2 * self.prev_bias_variance +
            (1 - beta2) * (bias_gradient ** 2)
        )

        bias_mean_hat = mean_bias / (1 - beta1 ** self.timestep)
        bias_variance_hat = bias_variance / (1 - beta2 ** self.timestep)
        self.timestep += 1

        self.prev_mean_bias = mean_bias
        self.prev_bias_variance = bias_variance

        self.bias -= (
            learning_rate * 
            (bias_mean_hat / (np.sqrt(bias_variance_hat) + epsilon))
        )

    def backward(self, output_gradient, learning_rate):
        """_summary_

        Args:
            output_gradient (numpy array): Gradient of the loss with respect to
            the output of the current layer, coming from the next layer in
            backpropagation.
            learning_rate (float): Learning rate for speed of paramter
            updates.

        Returns:
            _type_: _description_
        """
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-08

        # Adam optimiser on weights
        # mean_gradient = (
        #     beta1 * self.prev_mean_gradient + ((1 - beta1) * weights_gradient)
        # )
        # gradient_variance = (
        #     beta2 * self.prev_gradient_variance + 
        #     ((1 - beta2) * weights_gradient ** 2)
        # )

        self.gradient_mean = (
            beta1 * self.gradient_mean + ((1 - beta1) * weights_gradient)
        )
        self.gradient_variance = (
            beta2 * self.gradient_variance + 
            ((1 - beta2) * weights_gradient ** 2)
        )

        gradient_mean_hat = self.gradient_mean / (1 - beta1 ** self.timestep)
        gradient_variance_hat = (
            self.gradient_variance / (1 - beta2 ** self.timestep)
        )

        self.weights -= (
            learning_rate *
            (gradient_mean_hat / (np.sqrt(gradient_variance_hat + epsilon)))
        )

        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)
        self.bias_mean = beta1 * self.bias_mean + (1 - beta1) * bias_gradient
        self.bias_variance = (
            beta2 * self.bias_variance +
            (1 - beta2) * (bias_gradient ** 2)
        )

        bias_mean_hat = self.bias_mean / (1 - beta1 ** self.timestep)
        bias_variance_hat = self.bias_variance / (1 - beta2 ** self.timestep)

        self.bias -= (
            learning_rate * 
            (bias_mean_hat / (np.sqrt(bias_variance_hat) + epsilon))
        )

        self.timestep += 1

        return input_gradient

# OLD one:
    # def backward(self, output_gradient, learning_rate):
    #     beta1 = 0.9
    #     beta2 = 0.999
    #     epsilon = 1e-08

    #     weights_gradient = np.dot(output_gradient, self.input.T)
    #     input_gradient = np.dot(self.weights.T, output_gradient)

    #     mean_gradient = (
    #         beta1 * self.prev_mean_gradient + ((1 - beta1) * weights_gradient)
    #     )
    #     squared_gradient_avg = (
    #         beta2 * self.prev_gradient_variance + 
    #         ((1 - beta2) * weights_gradient ** 2)
    #     )

    #     bias_corrected_mean = mean_gradient / (1 - beta1 ** self.timestep)
    #     bias_corrected_variance = (
    #         squared_gradient_avg / (1 - beta2 ** self.timestep)
    #     )

    #     self.timestep += 1
    #     # Seems like actually updating the weights and not weights gradient e.g imp thing to see
    #     # normal is learning+rate * ewights_gradient
    #     # Instead we use RMSprop to do extra changes the weight gradient

    #     ### Do extra research to make sure etc
    #     self.weights -= (
    #         learning_rate * (bias_corrected_mean / (np.sqrt(bias_corrected_variance + epsilon)))
    #     )

    #     #self.weights -= learning_rate * weights_gradient
    #     self.bias -= learning_rate * np.sum(output_gradient, axis=1, keepdims=True)
    #     return input_gradient