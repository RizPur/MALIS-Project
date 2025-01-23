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