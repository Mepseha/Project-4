# Project4
Project 4


## Authors

Starla Halliday, Yemesrach Gebremikael, Robert Takan and Mosa

November, 2023

## Overview

This repository contains code for predicting genetic disorders using machine learning models, specifically neural networks and logistic regression. The dataset used for training and testing is sourced from train.csv.

## Logistics Regression Model (All Features)
### Data Manipulation
- Read Data: Loaded the data from the 'train.csv' file into a Pandas DataFrame.
- Clean Data: Unnecessary columns are removed, and missing values are handled by dropping rows with NaN values. Duplicate rows are also identified and removed. Additionally, specific string values (e.g., '-') are replaced with more appropriate ones (e.g., 'Not applicable').
- Data Encoding: Performed one-hot encoding on categorical columns.
- Target Creation: A binary target column is created, indicating the presence (1) or absence (0) of a genetic disorder. The original columns used for this binary classification are then dropped..
- Feature Selection: Used all available features for training using a standard scaler to scale the features.
### Logistic Regression Model
- Split Data: Divided the data into training and testing sets using train_test_split.
- Handle Imbalanced Classes: Applied Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance.
- Standardization: Standardized the features using Standard Scaler.
- Model Training: Trained a logistic regression model with a random state of 9.
- Cross-Validation: Utilized cross-validation to assess model performance.
- Confusion Matrices and Classification Reports: Generated and analyzed confusion matrices and classification reports for both training and testing data.
- Balanced Accuracy: Calculated and printed the balanced accuracy score.
- Prediction on Test Data: Generated predictions on the test dataset and displayed results.
- Confusion Matrix Heatmap: Visualized confusion matrices using a heatmap.
- Feature Importance: Examined feature importance based on logistic regression coefficients.
- Model Comparison: Compared logistic regression and random forest models.
### Random Forest Classifier 
- Random Forest Classifer was also used on the dataset to evaluate model performance.

![image](https://github.com/Mepseha/Project-4/assets/133922704/5d5e87b8-ccb0-434b-bc85-28c65f3f41d1)

## Logistics Regression Model (Subset of Features)
### Data Manipulation
- Read Data: Loaded the data from the 'train.csv' file into a Pandas DataFrame.
- Clean Data: Unnecessary columns are removed, and missing values are handled by dropping rows with NaN values. Duplicate rows are also identified and removed. Additionally, specific string values (e.g., '-') are replaced with more appropriate ones (e.g., 'Not applicable').
- Data Encoding: Performed one-hot encoding on categorical columns.
- Target Creation: Created a binary target column indicating the presence or absence of a genetic disorder.
- Feature Selection: 
### Logistic Regression Model
- Split Data: Divided the data into training and testing sets using train_test_split.
- Model Training: Trained a logistic regression model.
- Confusion Matrices and Classification Reports: Generated and analyzed confusion matrices and classification reports for both training and testing data.
- Balanced Accuracy: Calculated and printed the balanced accuracy score.
- Prediction on Test Data: Generated predictions on the test dataset and displayed results.
- Confusion Matrix Heatmap: Visualized confusion matrices using a heatmap.
- Feature Importance: The make_blobs function from the sklearn.datasets module is used in the provided code to generate synthetic data for testing the logistic regression model.

![image](https://github.com/Mepseha/Project-4/assets/133922704/967b3f8e-06fc-4e8d-8329-fa52f9e16610)

## Logistics Regression Model (Neural Networks)
### Data Preprocessing
- Read Data: Loaded the data from the 'genetic_disorder.csv' file into a Pandas DataFrame.
- Feature Scaling: Scaled numerical data using StandardScaler.
- Target and Feature Selection: Separated the target variable and selected features.
### Neural Network Model
- Model Definition: Defined a neural network with three layers (input, hidden, and output).
- Model Compilation: Compiled the model using binary cross-entropy loss and the Adam optimizer.
- Model Training: Trained the model on the training data.
- Model Evaluation: Evaluated the model on the test data and displayed loss and accuracy.

## Model Comparison
Logistic Regression vs. Random Forest: Compared logistic regression and random forest models using various metrics, including balanced accuracy, classification reports, and confusion matrices.
In summary, logistic regression models were applied to predict the presence or absence of genetic disorders using various sets of features. The models were trained, evaluated, and compared based on their performance metrics. Additional exploration and model tuning may be needed for further improvements.

## Conclusion 

The classification reports, in the Logistics Regression (All Features) effort, present the evaluation metrics for both the Logistic Regression and Random Forest models. In the primary dataset of 1,899 samples, Logistic Regression achieved an accuracy of 51%, demonstrating balanced precision and recall for both classes (0 and 1). The F1-score, a harmonic mean of precision and recall, is 0.51 for both classes, indicating a moderate balance between precision and recall. On the other hand, the Random Forest model achieved an accuracy of 50%, with a slightly higher precision and recall for class 0 compared to class 1. The F1-score for class 0 is 0.56, indicating a better balance between precision and recall for this class, while class 1 has an F1-score of 0.42. In a smaller dataset for the Logistics Regression (Subset Features) effort of 25 samples, the model performed exceptionally well with an accuracy of 100%, demonstrating perfect precision, recall, and F1-score for both classes. These results suggest that the models may perform differently on the datasets of varying sizes and characteristics, emphasizing the importance of understanding the specific context and requirements of the application. Further model tuning and exploration are recommended for achieving optimal performance.

## Reference 

Of genomes and genetics - HackerEarth ML. (n.d.). Kaggle: Your Machine Learning and Data Science Community. https://www.kaggle.com/datasets/imsparsh/of-genomes-and-genetics-hackerearth-ml/data
