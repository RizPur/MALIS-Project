from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np

import numpy as np
from sklearn.metrics import f1_score
from perceptron import Perceptron  # or however you import your Perceptron

def create_binary_labels(y, digit):
    """
    Convert an array of multi-class labels (0..9) into binary labels
    +1 if label == digit, -1 otherwise.
    """
    return np.where(y == digit, 1, -1)

def combined_predict(models, X, f1s):
    """
    Combine predictions for the multi-class task.
    
    models: dict of {class_k: trained Perceptron model_k}
    X: feature matrix to predict
    f1s: dict of {class_k: F1-score} used for tie-breaking

    Returns:
    - y_pred: array of shape (len(X),) with final digit predictions
    """
    n_samples = len(X)
    fill_value = -1
    y_pred = np.full(n_samples, fill_value)

    for k, model_k in models.items():
        y_pred_k = model_k.predict(X)
        # If y_pred_k[i] == 1, means "class k" claims this sample
        # If y_pred[i] is -1, that sample was unassigned
        # If there's a conflict, we use f1-based tie-break
        for i in range(n_samples):
            if y_pred_k[i] == 1:
                if y_pred[i] == -1:
                    y_pred[i] = k
                else:
                    # conflict
                    current_class = y_pred[i]
                    if f1s[k] > f1s[current_class]:
                        y_pred[i] = k
    return y_pred

def train_one_vs_rest(X_train, y_train, X_val, y_val, 
                      weight_samples, alpha, epochs):
    """
    Train Perceptrons for each digit (0..9) with the given sample weights.
    This is basically your 'Perceptron_train' logic, but adapted to be
    self-contained.

    Returns:
      - models: dict {digit: trained Perceptron}
      - error: combined weighted error
      - y_pred (train): final multi-class predictions on X_train
      - f1s: dict {digit: f1_score}
    """
    n_samples = len(y_train)
    n_classes = 10
    models = {}
    f1s = {}
    # We'll do a final combined prediction on X_train
    fill_value = -1
    y_pred_combined = np.full(n_samples, fill_value)

    # Train a Perceptron for each digit
    for digit in range(n_classes):
        # Binarize
        y_train_k = create_binary_labels(y_train, digit)
        y_val_k = create_binary_labels(y_val, digit)

        # Create and train
        model_k = Perceptron(alpha)
        model_k.train(X_train, X_val, y_train_k, y_val_k, 
                      weight_samples, epochs=epochs)
        
        # Evaluate on X_train
        y_pred_k = model_k.predict(X_train)
        # store model
        models[digit] = model_k

        # compute F1 for digit on train set
        precision, recall, f1_val = model_k.precision_recall_f1(y_pred_k, y_train_k)
        f1s[digit] = f1_val
    
    # Combine multi-class predictions for X_train
    for i in range(n_samples):
        # check which digits claim sample i
        candidates = []
        for digit in range(n_classes):
            y_pred_d = models[digit].predict(X_train[i].reshape(1,-1))[0]
            if y_pred_d == 1:
                candidates.append(digit)
        
        if len(candidates) == 1:
            y_pred_combined[i] = candidates[0]
        elif len(candidates) > 1:
            # tie-break using F1
            best_dig = max(candidates, key=lambda c: f1s[c])
            y_pred_combined[i] = best_dig
        # else remains -1 if no digit claimed it

    # Weighted error
    misclassified = (y_pred_combined != y_train)
    error = np.sum(weight_samples[misclassified]) / np.sum(weight_samples)

    return models, error, y_pred_combined, f1s

def run_samme_once(X_train, y_train, X_val, y_val, alpha, epoch, T):
    """
    Train a multi-class SAMME with T boosting rounds, using Perceptron as base,
    then return y_val_pred (predictions on the validation set).
    """

    # 1) Possibly do your "binary_y_train" approach here, or rely on global variables
    #    For each digit k, create y_train_k, y_val_k
    # 2) Initialize sample weights
    n_samples = len(y_train)
    weight_samples = np.ones(n_samples) / n_samples
    n_classes = 10
    Weak_classifier_perceptrons = {}
    weight_Weak_classifier = {}
    F1s = {}

    for t in range(T):
        # (A) Train
        Models_temp, Error, y_Pred, f1s = Perceptron_train(alpha, epoch, weight_samples, X_train, y_train, X_val, y_val) 
        # We might need to pass X_train, y_train here, or rely on global scope

        # (B) If Error >= 0.5, skip
        if Error >= 0.5:
            print("Classifier has high error; skipping this round.\n")
            continue

        # (C) Save
        F1s[t] = f1s
        Weak_classifier_perceptrons[t] = Models_temp
        weight_Weak_classifier[t] = np.log((1 - Error) / Error) + np.log(n_classes - 1)

        # (D) Update sample weights
        for i in range(n_samples):
            correct_class = y_train[i]
            # class_predictions[t] might be y_Pred[t], but you need to ensure
            # you are retrieving the relevant prediction for sample i 
            misclassified = (y_Pred[t] != correct_class)
            # Or adapt your code to ensure y_Pred is shaped (n_samples,) for each round
            if misclassified:
                weight_samples[i] *= np.exp(weight_Weak_classifier[t])
        weight_samples /= np.sum(weight_samples)
    
    # 3) Once T rounds are done, predict on X_val
    y_val_pred, _, _ = SAMME_predict_optimized(
        Weak_classifier_perceptrons,
        X_val,
        y_val,
        weight_Weak_classifier,
        F1s
    )
    return y_val_pred

def SAMME_predict_optimized(Weak_classifier_perceptrons, X_valid, y_valid, weight_Weak_classifier, F1s, n_classes=10):
    """
    Perform SAMME prediction by combining weighted votes from weak classifiers efficiently.

    """
    n_samples = X_valid.shape[0]
    votes = np.zeros((n_samples, n_classes))  # Accumulate votes for each class

    for t in range(len(Weak_classifier_perceptrons)):
        # Predict for all samples at once using the weak classifiers
        predictions_t = Combined_predict(Weak_classifier_perceptrons[t], X_valid, y_valid, F1s[t])
        valid_indices = predictions_t != -1  # Only count valid predictions

        # Update votes for valid predictions
        votes[valid_indices, predictions_t[valid_indices]] += weight_Weak_classifier[t]

    # Final prediction: the class with the most votes
    y_pred = np.argmax(votes, axis=1)

    # Evaluate final model
    accuracy = np.mean(y_pred == y_valid)
    f1 = f1_score(y_valid, y_pred, average='macro')
    print(f"Final SAMME Accuracy: {accuracy * 100:.2f}%")
    print(f"Final SAMME F1-score: {f1:.2f}")
    
    return y_pred, accuracy, f1


def Perceptron_train(alpha, epoch, weight_samples):
    '''
    Train Perceptron models for each class (One-vs-All) with weighted samples.

    INPUT:
    - alpha: Learning rate for the Perceptron
    - epoch: Number of epochs to train each weak classifier
    - weight_samples: Current weights for the training samples

    OUTPUT:
    - models: Dictionary of trained Perceptron models for each class
    - error: Combined weighted error across all classifiers
    - y_Pred: Predicted labels on the training data
    '''
    accuracy = {}
    f1s = {}
    models = {}

    # Iterate training over each digit (0 to 9)
    for k in range(10):
        print(f"Training Model for Class {k}")
        
        # Initialize Perceptron model
        model_k = pc.Perceptron(alpha)
        
        # Train the Perceptron on weighted samples
        model_k.train(X_train, X_valid, binary_y_train[k], binary_y_valid[k], weight_samples, epochs=epoch)
        
        # Evaluate model on test set
        y_pred = model_k.predict(X_test)
        accuracy[k] = model_k.accuracy(y_pred, binary_y_test[k])
        
        # Calculate F1 score
        precision, recall, f1 = model_k.precision_recall_f1(y_pred, binary_y_test[k])
        f1s[k] = f1  # Store F1 score
        models[k] = model_k  # Store trained model
        
        print(f"Class {k}: Accuracy = {accuracy[k]:.2f}, F1 = {f1:.2f}")
        print("~" * 20)

    # Calculate average accuracy and F1-score across all binary classifiers
    avg_accuracy = np.mean(list(accuracy.values()))
    avg_f1 = np.mean(list(f1s.values()))

    print("End performance:")
    print(f"Average Accuracy: {avg_accuracy * 100:.2f}%")
    print(f"Average F1-score: {avg_f1:.2f}")

    # Combine predictions for the multi-class task
    y_Testintrain = np.copy(y_train)  # Ground truth labels for training set
    fill_value = -1  # Placeholder for unclassified samples
    y_Pred = np.full_like(y_train, fill_value)  # Initialize predictions
    
    # Update predictions based on the reliability of models (using F1 scores)
    for k in range(10):
        y_pred_k = models[k].predict(X_train)
        y_Pred[(y_pred_k == 1) & (y_Pred == fill_value)] = k  # Assign if unclassified
        # Update predictions for conflicts based on F1 scores
        conflict_indices = np.where((y_pred_k == 1) & (y_Pred != fill_value))[0]
        for idx in conflict_indices:
            current_class = y_Pred[idx]  # Existing predicted class
            if f1s[k] > f1s[current_class]:  # Compare F1 scores
                y_Pred[idx] = k  # Update to more reliable prediction

    # Calculate combined model accuracy and F1-score
    combined_accuracy = Accuracy(y_Pred, y_Testintrain)
    combined_f1 = f1_score(y_Testintrain, y_Pred, average='macro')

    # Calculate the combined error for SAMME
    # Error is defined as the weighted sum of misclassified samples
    error = np.sum(weight_samples[y_Pred != y_Testintrain]) / np.sum(weight_samples)

    print(f"\nCombined Accuracy: {combined_accuracy * 100:.2f}%")
    print(f"Combined F1-score: {combined_f1:.2f}")
    print(f"Combined Weighted Error: {error:.4f}\n")
    
    return models, error, y_Pred, f1s
