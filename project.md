# MALIS Project 1
### Written by: Qizhi Pan and Joel Brown
### Date: November 4, 2024

## Task 1 - Binary Classifier  
###  Problem Definition 
We want to design a model that will predict between a Setosa and a Versicolor using data on their petals' and stalks' length and width. 

The initial data set `iris.csv` contained data on 3 flowers but we have reformatted it to have just the first 2 flowers for this task and renamed it `iris_binary.csv`

### ScatterPlot

In our main.py, the first thing we did was create a function to show us a scatterplot of the data. Here we could see the data could be seperated by a higher dimensional plane for all 4 input data sets put against each other.

![Pairplot](/plots/binary/pairplot.png)

#### Structure of the Scatterplot

For a dataset with 4 features (e.g., `sepal.length`, `sepal.width`, `petal.length`, and `petal.width`), the pairplot grid will look like this:

|                   | `sepal.length` | `sepal.width`  | `petal.length`  | `petal.width`  |
|-------------------|----------------|----------------|-----------------|----------------|
| **`sepal.length`** | Histogram      | Scatter Plot   | Scatter Plot    | Scatter Plot   |
| **`sepal.width`**  | Scatter Plot   | Histogram      | Scatter Plot    | Scatter Plot   |
| **`petal.length`** | Scatter Plot   | Scatter Plot   | Histogram       | Scatter Plot   |
| **`petal.width`**  | Scatter Plot   | Scatter Plot   | Scatter Plot    | Histogram      |

#### Explanation:
- **Off-diagonal cells** (scatter plots): Show pairwise relationships between different features.  
  - For example, the cell at row `sepal.length` and column `sepal.width` shows a scatter plot of `sepal.length` vs. `sepal.width`.
- **Diagonal cells** (histograms or KDE plots): Show the distribution of individual features.
  - For instance, the diagonal cell for `sepal.length` shows a histogram of `sepal.length` values across all samples.

This layout provides a **comprehensive overview** of all pairwise relationships in the dataset, allowing for easy pattern recognition and class distinction when used with a `hue` parameter.

## Task 2 - Multi Variable Classifier 

