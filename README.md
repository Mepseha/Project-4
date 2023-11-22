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

## Neural Network Model:

A simple neural network model is defined using TensorFlow's Keras.
The model is compiled with binary cross-entropy loss and Adam optimizer. Training is done on the scaled training data.

## Logistic Regression Model:

A logistic regression model is created and fitted using the training data.
The model is then used to make predictions on both the training and testing sets.
Confusion matrices and classification reports are generated for both sets, providing a detailed evaluation of the model's performance.

### Feature Coefficients:

The coefficients of the logistic regression model that focuses on Patient Age and Blood cell count (mcL) are extracted and displayed, highlighting the strength and direction of the relationship between each feature and the predicted outcome. Coefficients indicate the strength and direction (positive or negative) of the relationship between each feature and the predicted outcome. A negative coefficient suggests an inverse relationship between the feature and the predicted outcome. As "Patient Age" and "Blood cell count (mcL)" have negative coefficients, an increase in these values is associated with a decrease in the predicted outcome (lower likelihood of a Genetic Disorder).  Magnitude (Absolute Coefficient) of the coefficient indicates the strength of the relationship. Larger absolute values indicate a stronger influence on the predicted outcome.

0     

Patient Age     

Coefficient: -1.286599 

Absolute Coefficient: 1.286599

1

Blood cell count (mcL)  

Coefficient: -0.875418 

Absolute Coefficient: 0.875418
           

### Feature Importance Plot:

The top features influencing the predicted outcome are identified and visualized in a horizontal bar plot, showing their respective coefficients' magnitudes.

This graph helps users identify and prioritize the most influential features for the logistic regression model. "Coefficient Magnitude" is a measure of how much a feature contributes to the model's predictions, considering both the positive and negative impacts. Larger magnitudes indicate features that have a more substantial influence on the predicted outcome.


### Confusion Matrix Heatmap:

A heatmap is created to visually represent the confusion matrix for the testing data, aiding in the assessment of the model's performance. The confusion matrix heatmap allows users to visually assess how well the Logistic Regression model is performing in predicting the "Genetic Disorder" by comparing the predicted and actual classes. The goal is to have higher counts along the diagonal (from the top left to the bottom right), indicating correct predictions, and lower counts in off-diagonal cells, indicating misclassifications. The color intensity provides a quick visual summary of the distribution of predictions.




