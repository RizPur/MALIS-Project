## AdaBoosting

1. Train a “weak learner perceptron". Compute how many samples it misclassifies and sum up their weights (the “error”).
1. Give that weak learner a “classifier weight” αtαt​. If error is low, αtαt​ is higher, meaning that classifier is more trusted. If error is high, αtαt​ is smaller.
1. Update sample weights: Increase the weight for misclassified samples so that the next weak learner must focus more on them.
1. Repeat for T rounds. Combine the T learners’ predictions in a weighted vote to get a final decision.

## Sequential Logic

1. Initialize sample weights (uniformly).
1. For each boosting round t∈{1..T}t∈{1..T}:

1. Call Perceptron_train(...) to train 10 Perceptrons (one per digit).
1. Compute weighted error of that round’s combined multi-class predictions on the training set.
1. If error < 0.5, compute αtαt​. Otherwise skip.
Update sample weights for misclassified samples.


## Notes for Pan

1. No global references to X_train, X_valid, y_train, or binary_y_train[k].
1. Yes pass (X_train_fold, y_train_fold, X_val_fold, y_val_fold) into Perceptron_train(...) inside each fold.
1. Inside Perceptron_train, we do the 0..9 classification for that fold. Return the combined predictions and error with respect to that fold’s training data.