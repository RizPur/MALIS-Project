import numpy as np
from sklearn.model_selection import train_test_split
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
        

    def train(self, X, y, epochs=100, flag = 0):
        '''
        INPUT :
        - X : is a 2D NxD numpy array containing the input features
        - y : is a 1D numpy array of size N containing the labels for the corresponding rows of X
        - epochs : number of times to iterate over the dataset
        '''
        # training fail
        if flag >= 5:
            print("training failed")
            return 0
        # Initialize weights with zeros
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
       
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

        # Perceptron training loop
        for _ in range(epochs):
            for idx, x_i in enumerate(X_train):
                # Predict the label
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = np.where(linear_output >= 0, 1, -1)

                # Update weights and bias if prediction is incorrect
                if y_pred != y_train[idx]:
                    self.weights += self.alpha * y_train[idx] * x_i   # Here's the SGD expression of perceptron problems
                    self.bias += self.alpha * y_train[idx]
        
        y_val_pred = self.predict(X_val)
        val_accuracy = np.mean(y_val_pred == y_val)
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
        if val_accuracy < 0.8:
            print(f"Model Reject. Retrain")
            flag += 1
            return self.train(X, y, epochs=100, flag = flag)
        return 1
        

    def predict(self, X_new):
        '''
        INPUT :
        - X_new : is a MxD numpy array containing the features of new samples whose labels have to be predicted
        OUTPUT :
        - y_hat : is a 1D numpy array of size M containing the predicted labels for the X_new samples
        '''
        linear_output = np.dot(X_new, self.weights) + self.bias
        y_hat = np.where(linear_output >= 0, 1, -1)
        #print(self.weights)
        return y_hat
    
    def accuracy(self, y_pred, y):
        '''
        OUTPUT :
        - accuracy: compare two outputs, calculate their accuracy
        '''
        val_accuracy = np.mean(y_pred == y)
        print(f"Testing Accuracy: {val_accuracy * 100:.2f}%")
        return val_accuracy
    
    
    def precision_recall_f1(self, y_pred, y):
        '''
        Compute precision, recall, and F1-score.
        
        We assume:
        - Positive class: 1
        - Negative class: -1
        '''
        # True Positives (TP): predicted 1, actual 1
        TP = np.sum((y_pred == 1) & (y == 1))
        
        # False Positive (FP): predicted 1, actual -1
        FP = np.sum((y_pred == 1) & (y == -1))
        
        # False Negatives (FN): predicted -1, actual 1
        FN = np.sum((y_pred == -1) & (y == 1))
        
        # Precision: TP / (TP + FP), need to handle the case if TP+FP=0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        
        # Recall: TP / (TP + FN), need to handle if TP+FN=0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        # F1-score: 2*(precision*recall)/(precision+recall), handle if sum=0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-score: {f1:.2f}")
        
        return precision, recall, f1