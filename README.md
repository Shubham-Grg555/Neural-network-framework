# Simple neural network framework
Simple framework where you are able to create your own custom neural networks to solve problems like mnist digit classification,
built using only python and NumPy. It currently contains the basics needed to train and test a neural network, aswell as a mnist
digit classification solution made.

### How to install
```bash
git clone https://github.com/Shubham-Grg555/Neural-network-framework
cd neuralnetpy
python -m venv .venv
pip install numpy
```

No additional dependencies required beyond Python ≥3.7 and NumPy, however if you want to use the mnist solution made, you either
need to download the [mnist data base](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) or download tensor flow to access
the data base.

## Current features:
- Dense layers (fully connected)
- Activation functions (tanh)
- Loss functions (MSE)
- Optimizers (Adam)
- Forward and backward propagation
- Training loop and testing logic

## To do:
- Finish / fix Adam optimser, ReLU optimiser, softmax and binary cross entroy.
- Add ability to create a convolutional neural network.
- Create solutions for more problems like cat and dog classification.

## License and credits
Watched this [video](https://www.youtube.com/watch?v=pauPCy_s0Ok) made by ["The Independent Code"](https://www.youtube.com/@independentcode) 
to help understand the theory and also used some of the code to make the solution

This project is licensed under the [MIT License](LICENSE).
