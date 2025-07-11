class Layer:
    """Interface or abstract thing yap
    """
        
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        """Takes in the input and then returns the corresponding output.

        Args:
            input (_type_): input being fed into the layer
        """
        pass

    def backward(self, input, output_gradient : float, learning_rate : float):
        """Takes in the derivative of the error, with respect to the output,
        then updates the needed trainable programs and then returns the
        derivative of the error with respect to the input of the layer.
        Solutions can then be optimised with the learning rate.

        Args:
            input (_type_): input being fed into the layer
            output_gradient (float): Value of the derivative of the error
            with repsect to the output of the layer
            learning_rate (float): Learning rate which can be used for
            optimising the neural network e.g gradient descent
        """
        pass