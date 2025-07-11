from layer import Layer
import numpy as np

"""Takes input neurons and applies an activation function to every
individual input neuron.
"""
class Activation(Layer):
    def __init__(self, activation, activation_gradient):
        self.activation = activation
        self.activation_gradient = activation_gradient

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
        
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_gradient(self.input))

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)
        
        def tanh_gradient(x):
            return 1 - np.tanh(x) ** 2
        
        super().__init__(tanh, tanh_gradient)

class ReLU(Activation):
    def __init__(self):
        def ReLU(x):
            return np.maximum(0, x)
        
        def ReLU_gradient(x):
            return np.where(x > 0, 1, 0)
        
        super().__init__(ReLU, ReLU_gradient)
    
class Softmax(Layer):
    def forward(self, input):
        
        exp_input = np.exp(input) #np.exp(input - np.max(input))
        self.output = exp_input / np.sum(exp_input, keepdims=True)
        print(f"Self output test: {self.output}, \n shape test: {np.shape(self.output)}")
        return self.output
        
        
        #print(f"Input test {input} \n Exponential of input {np.exp(input)} \n input sum exponential {np.sum(np.exp(input))}")

        #input_exp = np.exp(input)

        #self.output = input_exp / np.sum(input_exp)
        #return self.output
        #self.output = input_exp / np.sum(input_exp)

    def backward(self, output_gradient, learning_rate):
        # Two methods to try:
        #def softmax_jacobian(s):
        # s is the softmax output vector
        #s = s.reshape(-1, 1)
        #return np.diagflat(s) - np.dot(s, s.T)

    #     exp_z = np.exp(z - np.max(z))  # for numerical stability
    #     return exp_z / np.sum(exp_z)

    # def cross_entropy_derivative(softmax_output, true_label):
    #     return softmax_output - true_label  # assuming one-hot encoded label


        # temp = 1 - self.output
        # print(f"Rtet: {temp}\ \n Tersting self.output {self.output}")
        # print(f"Testing gradient: {np.dot(self.output, temp)}")
        # return np.dot(self.output, temp)
    


        #n = np.size(self.output)
        #return np.dot((np.identity(self.output) - self.output.T) * self.output, output_gradient)
        return