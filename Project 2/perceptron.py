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
        
    def train(self,X,y):
        '''
        INPUT :
        - X : is a 2D NxD numpy array containing the input features
        - y : is a 1D Nx1 numpy array containing the labels for the corrisponding row of X
        '''        
       
    def predict(self,X_new):
        '''
        INPUT :
        - X_new : is a MxD numpy array containing the features of new samples whose label has to be predicted
        A
        OUTPUT :
        - y_hat : is a Mx1 numpy array containing the predicted labels for the X_new samples
        ''' 
            

        return y_hat
    
