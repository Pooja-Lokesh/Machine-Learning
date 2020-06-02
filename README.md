# Machine-Learning

In basic terms, ML is the process of training a piece of software, called a model, to make useful predictions using a data set. 
This predictive model can then serve up predictions about previously unseen data.

## Contents of the repository

### Regression:
    - Simple linear regression
    - Multiple linear regression
    - Polynomial regression
    - Decision tree regression
    - Support vector regression
    - Random forest regression
    
### Classification:
    - Logistic regression
    - Kernel SVM
    - K-Nearest Neighbors
    - Naive Bayes
    - Decision tree classification
    - Support vector machine
    - Random forest classification
    
### Clustering:
    - K-Means clustering
    - Hierarchical clustering

## Reducing loss

Reducing loss can be done by computing gradient . Gradient is the derivative of the loss function
with respect to the weight of the parameter. 
By computing the derivative, we get to know how much the loss changes for a give example, so we take small steps repeatedly in the direction that minimizes the loss. This is called gradient decent.
Learning rate tells us how large of a step we should take towards the negative gradient. 
If the learning rate is small, then it takes small steps and requires a lot of time to reach local minimum.
If the learning rate is large, the steps towards the local minimum is very large , and chances of over shooting the local minimum is high.

### Weight initialization :

When there is only one local minimum, the initialization of the weights doesnt matter as eventually it
reaches the local minimum.
Where as , when the function has more than one local minima , the initilization matters.

stochastic gradient decent : one example at a time
mini-batch gradinet decent : batches of 10 to 1000 at a time

## Generalization

The larger the training set, the better model we will be able to compute.
The larger the testing set, the better confidence we will be able to have in the trained model.

Validation set: this is the third set of data we get from the sample data set.
We tweak the model trained on training data, to make changes to the patterns learnt by the model, before we test it on the test set.

## Handling non-linear data

The non-linear data cant be separated on the number of features available, cause the features are limited.
Therefore we add one more feature which is the cross product of all the other features and increase the number of features. This way we increase the dimentionality and we will be able to separate the data points by a plane linearly.
These new features are called feature crosses

example: x, y are the features given ; x*y is the feature cross

Sometimes adding these feature crosses won't add any value, so we need to consider only the meaningful feature crosses.

## Regularization 

Regularization is done to avoid overfitting in a model.
This can be done by reducing the model's complexity.

### L0 Regularization:
    - Making all the weights zero.
    - But there are chances of losing the coefficients which are neccessary.
    
### L1 Regularization:
    - Penalize sum of abs(weights)
    - Encourage sparsity

### L2 Regularization:
    - new function to minimize is => minimize(loss(Data|model) + complexity(model))

