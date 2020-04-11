# Handling Imbalanced Data with Basic Classifier Models


## Introduction
In this project, I examined how a Logistic Regression and a Decision Tree Classifier model can be optimised if the underlying data is imbalanced. The Python code uses `scikit-learn`'s classifier models (`LogisticRegression` and `DecisionTreeClassifier` and `undersample` method. I am also using `imblearn` for the more sophisticated `smote` oversampling method. 
<br>

This GitHub repo contains the code for my [blogpost on Medium](https://medium.com/datadriveninvestor/handling-imbalanced-data-with-basic-classifier-models-5ce3d61874f1).
<br>

## Problem
The term _“imbalanced data”_ is usually applied to cases where the different values of the target variable are not equally represented in the data. Technically speaking most data is imbalanced to some degree and that’s fine, but a significant difference in target variable groups can make classifier models unreliable. 
<br>

I am concentrating on two different problems when it comes to training models with imbalanced data: 
- we want to maximize the classifier's accuracy;
- we want to maximize the classifier's balanced accuracy. 
<br>

These require different mthods, depending on the data structure, and the classifier model we are training. 

## Data
For my project, I am using two sets of simulated data. For both of them, I calculated the optimal threshold depending on whther we want to categorise the observations to maximise accuracy or balanced accuracy. 
<br>

In `Data_1`, the 


![](charts/data_1_optimal_with_histograms.png)

# Methods


## Results
My results are finalized in the table below: 

For additional details, please refer to my blogpost, link in the Introduction section. 

## Files
The repo contains the following files: 
- `imbalanced_classifier.ipynb`: A Jupyter notebook containing the majority of the work and visualisations. 
- `imbalanced_classifier_functions.py`: Contains functions and useful methods written for this project. 
- `charts` folder: exported charts from the Jupyter notebook. 
