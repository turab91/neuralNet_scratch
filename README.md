# Neural Network From Scratch
Implementation of neural network using python and numpy.

# turabnet
It contains implementation of different layers, loss function, optimization function, batch data iterator and train function.
All the functions are implemented in python. numpy is used for vectorized implementation.
The functions will perform faster if we use data in batches.

# Train Dataset
- AND.ipynb contains a simple model to approximate a AND gate. Since outputs of AND gate are linearly separable
we can model it using a linear model.
- XOR.ipynb contains a simple model to approximate a XOR gate. Sine output of XOR gate are non-linear, we had to 
use non-linear function in the model.
- moon.ipynb contains a simple model to approximate the moon dataset from scikit-learn. The data set is non-linear
and more complex than the XOR gate. hence we need to use more neurons and preferably more layers.

# comment on learning rate and number of hidden units
- learning rate is a hyper parameter. we have seen that choosing good learning rate can help optimize the model
faster ie. using less iterations.
- choosing more hidden units means the model has more parameters and hence higher degree of freedom to approximate 
the given dataset. But we have to be mindful of overfitting and computational cost.

