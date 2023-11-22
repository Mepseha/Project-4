# Project4
Project 4


## Authors

Starla Halliday, Yemesrach Gebremikael, Robert Takan and Mosa

November, 2023

## Overview

This repository contains code for predicting genetic disorders using machine learning models, specifically neural networks and logistic regression. The dataset used for training and testing is sourced from test.csv.

## Data Manipulation:

The code begins by reading the test.csv file into a Pandas DataFrame (GeneticDisorder_df).
Unnecessary columns are removed, NaN values are identified, and rows with missing values are dropped.
The cleaned data is then saved to a new CSV file (genetic_disorder.csv).

##Data Encoding and Splitting:

Categorical columns are identified, and one-hot encoding is applied to convert categorical variables into a numerical format.
The data is then split into training and testing sets using the train_test_split method. 
Numerical data is scaled using StandardScaler.

## Neural Network Model

A simple neural network model is defined using TensorFlow's Keras.
The model is compiled with binary cross-entropy loss and Adam optimizer. Training is done on the scaled training data.

## Logistic Regression Model:

A logistic regression model is created and fitted using the training data.
The model is then used to make predictions on both the training and testing sets.
Confusion matrices and classification reports are generated for both sets, providing a detailed evaluation of the model's performance.

### Feature Coefficients:

The coefficients of the logistic regression model are extracted and displayed, highlighting the strength and direction of the relationship between each feature and the predicted outcome.

### Feature Importance Plot:

The top features influencing the predicted outcome are identified and visualized in a horizontal bar plot, showing their respective coefficients' magnitudes.

### Confusion Matrix Heatmap:

A heatmap is created to visually represent the confusion matrix for the testing data, aiding in the assessment of the model's performance.
