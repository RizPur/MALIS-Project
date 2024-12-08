import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('score.csv')

# Plot initial data points
plt.scatter(data.StudyHours, data.Scores)
plt.savefig('scatter_plot.png')

def loss_function(m, b, points): # Not interested in the loss but how to minimize the loss 
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].StudyHours
        y = points.iloc[i].Scores
        total_error += (y - (m * x + b))**2
    total_error = total_error / float(len(points))
    return total_error

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)
    for i in range(n):
        x = points.iloc[i].StudyHours  # Access StudyHours
        y = points.iloc[i].Scores      # Access Scores

        # Compute gradients
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    # Update parameters
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m, b

# Initialize parameters
m = 0
b = 0
L = 0.001  # Learning rate
epochs = 300

# Perform gradient descent
for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}, Loss: {loss_function(m, b, data)}")  # Track loss to monitor convergence
    m, b = gradient_descent(m, b, data, L)

# Display final parameters
print("Final parameters:", m, b)

# Plot the results
plt.scatter(data.StudyHours, data.Scores, color="black")
plt.plot(list(range(0, 10)), [m * x + b for x in range(0, 10)], color="red")
plt.savefig('gradient_descent_plot.png')
