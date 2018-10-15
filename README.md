# Logistic Regression - uWaveGestureLibrary

### Methods
This project proposes an implementation of a Multi-Class Logistic Regression for any labeled datasets. These are some features fo the implementation:
- Cost function: cross entropy
- Regularization: L2 regularization
- Optimization: gradient descent
- Stopping criteria: early stopping
- Learning visualization: loss, accuracy, cross validation and confusion map

### Dataset
To illustrate the results of this method, the dataset [uWaveGestureLibrary](http://timeseriesclassification.com/description.php?Dataset=UWaveGestureLibraryAll) is used. This dataset comprised 8 different types of time series representing gestures. A time serie is composed of positions (x,y,z) at each time step.

### Run the project

The following command run the project:
```
python Scripts/main.py
```
 The packages needed are *numpy* and *matplolib*. If you desire genererate synthetic blob datasets, *sklearn* is also needed.

### Reports & Figures


The figures in the folder Figure and the slides logistic-regression-uwavegesture.pdf give an overview of the results of the Multi-Class Logistic Regression
