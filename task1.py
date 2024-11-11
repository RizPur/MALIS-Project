import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('iris_binary.csv')

# Since length is enough to distinguish the variety, we truncate the data to only have length and its associated variety
data = data[['petal.length', 'variety']]

# Map 'variety' to 0 and 1
data['variety'] = data['variety'].map({'Setosa': 0, 'Versicolor': 1})

# Check missing values
print(data.isnull().sum())

# Visualize the data
plt.scatter(data['petal.length'], data['variety'], color='blue', marker='o')
plt.xlabel('Petal Length')
plt.ylabel('Variety')
plt.title('Petal Length vs Variety')
plt.savefig('plots/binary/petal_length_vs_variety.png')
plt.show()

# loss function
def loss_function(m, b, data):
    total_error = 0
    n = len(data)
    for i in range(n):
        x = data.iloc[i]['petal.length']
        y = data.iloc[i]['variety']
        total_error += (y - (m * x + b)) ** 2
    return total_error / n

# gradient descent
def gradient_descent(m_now, b_now, data, learning_rate):
    m_gradient = 0
    b_gradient = 0
    n = len(data)
    for i in range(n):
        x = data.iloc[i]['petal.length']
        y = data.iloc[i]['variety']
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    m_next = m_now - learning_rate * m_gradient
    b_next = b_now - learning_rate * b_gradient
    return m_next, b_next

# Initialize parameters
m = 0
b = 0
learning_rate = 0.02
epochs = 300

loss_history = []

# Now we do gradient descent for however many epochs
for epoch in range(epochs):
    m, b = gradient_descent(m, b, data, learning_rate)
    loss = loss_function(m, b, data)
    loss_history.append(loss)
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

print(f'Final good parameters: m = {m}, b = {b}')

# Plot loss history to see how small the lost function got over time
plt.figure()
plt.plot(range(epochs), loss_history)
plt.xlabel('Time')
plt.ylabel('Loss')
plt.title('Loss / Time')
plt.savefig('plots/binary/loss_over_epochs.png')
plt.show()

# Make predictions
def predict(m, b, x):
    return m * x + b

data['predicted'] = data['petal.length'].apply(lambda x: predict(m, b, x))

# Apply threshold to classify
def apply_threshold(y_pred, threshold=0.5):
    return [1 if y >= threshold else 0 for y in y_pred]

data['predicted_class'] = apply_threshold(data['predicted'])

# Evaluate the model
y_true = data['variety']
y_pred = data['predicted_class']

accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Plot the results
plt.figure()
plt.scatter(data['petal.length'], data['variety'], color='black', label='Actual')
plt.plot(data['petal.length'], data['predicted'], color='red', label='Predicted')
plt.xlabel('Petal Length')
plt.ylabel('Variety')
plt.title('Linear Regression Model')
plt.legend()
plt.savefig('plots/binary/linear_regression_model.png')
# plt.show()
