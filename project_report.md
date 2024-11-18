# MALIS Project 1

Written by: Qizhi Pan and Joel Brown
Date: November 4, 2024
## 1. Introduction
### 1.1 Overview
In this project, we completed a binary classification task using linear regression based on iris dataset and further, completed a ternary classification task. The accuracy for task 1 can be 100%, and for task 2 can be up to 96%. 
### 1.2 ChatGPT Usage
We used ChatGPT to accelarate our coding, complete and revise our code. We originally proposed a solution to the ternary classification problem and implemented it using ChatGPT.
## 2. Dataset Description and Analysis
The Iris dataset consists of 150 samples, each representing a flower of one of three species: `Setosa`, `Versicolor`, and `Virginica`. The dataset contains 4 numerical features:`sepal.length`, `sepal.width`, `petal.length`, and `petal.width`, all of them are in centimeters.

The dataset is balanced, with 50 samples per class. Each sample is labeled with one of three classes corresponding to the flower species.

To better handle the classification task with linear regression, we need to understand the data.

### 2.1 Plot 4 Features in 2 Categories
 Considering the binary classifier in **task 1**, here we list the scatterplots and histograms for 4 features in selected 2 categories:

![Pairplot](/plots/binary/pairplot.png)


Here's the definition for each pairplot grid:

|                   | `sepal.length` | `sepal.width`  | `petal.length`  | `petal.width`  |
|-------------------|----------------|----------------|-----------------|----------------|
| **`sepal.length`** | Histogram      | Scatter Plot   | Scatter Plot    | Scatter Plot   |
| **`sepal.width`**  | Scatter Plot   | Histogram      | Scatter Plot    | Scatter Plot   |
| **`petal.length`** | Scatter Plot   | Scatter Plot   | Histogram       | Scatter Plot   |
| **`petal.width`**  | Scatter Plot   | Scatter Plot   | Scatter Plot    | Histogram      |

This layout provides a **comprehensive overview** of all pairwise relationships in the dataset.

For the categories we choose(`Setosa`, `Versicolor`), we can see that the distributions of `petal.length` or `petal.width` has distinct boundary between 2 categories and other features are more or less overlapped. 

Theoreticlly, either the `petal.length` or `petal.width` itself should be enough to differentiate between these 2 categories. In **task 1**, we decided to only use `petal.length` to create a regression model.


### 2.2 Plot 4 Features in 3 Categories
Considering the ternary classifier in **task 2**, here we plot the 4 features in 3 categories with the same grid definition as above:
![Pairplot](/plots/binary/pairplot_tri.png)
In this new condition, choosing only one feature to set up the model is not a good idea, because every feature overlaps with each other to some extent. But the `petal.length` and `petal.width` together show a beautiful linear relationship with more feasible boundaries to distinguish each other.

Thus, in task 2, we decide to us both `petal.length` and `petal.width` features to create our regression model.
## 3. Experiments
### 3.1 Task 1 - Binary Classifier  
### 3.2 Task 2 - Ternary Classifier  
## 4. Discussion
###  Problem Definition 
We want to design a model that will predict between a Setosa and a Versicolor using data on their petals' and stalks' length and width. 

The initial data set `iris.csv` contained data on 3 flowers but we have reformatted it to have just the first 2 flowers for this task and renamed it `iris_binary.csv`

### ScatterPlot




## Task 2 - Multi Variable Classifier 

