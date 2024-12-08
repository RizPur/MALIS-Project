import numpy as np

class Perceptron:
    '''
    perceptron algorithm class
    __init__() initialize the model
    train() trains the model
    predict() predict the class for a new sample
    '''

    def __init__(self, alpha):
        '''
        INPUT :
        - alpha : is a float number bigger than 0 
        '''

        if alpha <= 0:
            raise Exception("Sorry, no numbers below or equal to zero. Start again!")
            
        # empty initialization of X and y
        self.X = []
        self.y = []
        # k is the parameter of the algorithm representing the number of neighborhoods
        self.alpha = alpha
        

    def train(self, X, y, epochs=100):
        '''
        INPUT :
        - X : is a 2D NxD numpy array containing the input features
        - y : is a 1D numpy array of size N containing the labels for the corresponding rows of X
        - epochs : number of times to iterate over the dataset
        '''
        # Initialize weights with zeros
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Perceptron training loop
        for _ in range(epochs):
            for idx, x_i in enumerate(X):
                # Predict the label
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = np.where(linear_output >= 0, 1, -1)

                # Update weights and bias if prediction is incorrect
                if y_pred != y[idx]:
                    self.weights += self.alpha * y[idx] * x_i   # Here's the SGD expression of perceptron problems
                    self.bias += self.alpha * y[idx]

    def predict(self, X_new):
        '''
        INPUT :
        - X_new : is a MxD numpy array containing the features of new samples whose labels have to be predicted
        OUTPUT :
        - y_hat : is a 1D numpy array of size M containing the predicted labels for the X_new samples
        '''
        linear_output = np.dot(X_new, self.weights) + self.bias
        y_hat = np.where(linear_output >= 0, 1, -1)
        return y_hat
