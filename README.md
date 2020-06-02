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
    
### Clustering :
    - K-Means clustering
    - Hierarchical clustering


## REDUCING LOSS :

reducing of loss can be done by computing gradient . Gradient is the derivative of the loss function
with respect to the weight of the parameter. 
By computing the derivative of the loss function by weights and biases, we get to know how much the 
loss changes for a give example, so we take small steps repeatedly in the direction that minimizes 
the loss. 
This is called gradient decent.
Learning rate tells us how large of  a step we should atke towards the negative gradient. If the learning 
rate is small, then it takes small steps and requires a lot of time to reach local minimum. Where as,
when we take a large learning rate , the steps to wards the local minimum is very large , and chances of 
over shooting the local minimum is high

### weight initialization :

when there is only one local minima, the initialization of the weights doesnt matter as eventually it
reaches the local minimum.
Where as , when the function has more than one local minima , the initilization matters.

stochastic gradient decent : one example at a time
mini-batch gradinet decent : batches of 10 to 1000 at a time


## GENERALIZATION :

Okham's razor principle : the less complex a model is , the better it is in prediction

This is the reason why we need a test set separately to test if the model can predict the new samples
accurately which has been trained on training set. The training set and test set are obtained from 
the same sample data, by splitting the data set into two.

The larger the training set, the better model we will be able to compute
The larger the testing set, the better confidence we will be able to have in the trained model

Validation set: this is the third set of data we get from the sample data set.
We tweak the model trained on training data, to make changes to the patterns learnt by the model, before
we test it on the test set.


## REPRESENTATION:

Extracting usefull features form the raw data is called feature engineering.
example: mapping the string values into a numeric vector by using one hot encoding.
one-hot-encoding is done in order to be able to multiply the feature with the numeric weights in 
the model.

Cleaning the data includes:
1) Scaling feature values
2) handling outliers
3) removing bad features which dont add values
4) removing duplicate values
5) adding missing values


## SOLVING NON-LINEAR PROBLEMS:

In this case , the non-linear data cant be separated on the number of features available, cause the features
are limited.
Therefore we add one more feature which is the cross product of all the other features and increase 
the number of features. This way we increase the dimentionality and we will be able to separate the data
points by a plane linearly.
These new features are called feature crosses

example: x, y are the features given ; x*y is the feature cross

Sometimes adding these feature crosses won't add any value, so we need to consider only the meaningful
feature crosses or should not consider them at all. 


## REGULARIZATION:

Regularization is done to avoid overfitting in a model
This can be done by reducing the model's complexity

example: L2 regularization
new function to minimize is => minimize(loss(Data|model) + complexity(model))

reducing model size and memory usage:
Caveat: Sparse feature crosses may significantly increase feature space
Possible issues:
Model size (RAM) may become huge
"Noise" coefficients (causes overfitting)

L1 Regularization
Would like to penalize L0 norm of weights
    Non-convex optimization; NP-hard
    we have to be very carefull cayse we might lose the coefficients which are neccessary
Relax to L1 regularization:
    Penalize sum of abs(weights)
    Convex problem
    Encourage sparsity unlike L2


CLASSIFICATION :

Precision: (True Positives) / (All Positive Predictions)
When model said "positive" class, was it right?
Intuition: Did the model cry "wolf" too often?
Recall: (True Positives) / (All Actual Positives)
Out of all the possible positives, how many did the model correctly identify?
Intuition: Did it miss any wolves?
=> when there is one specific classification

we still wanan get to know all the possible classification thershhold so we compute ROC for that
ROC : reciever operating characteristics curve
AUC: "Area under the ROC Curve"
Interpretation:
If we pick a random positive and a random negative, what's the probability my model ranks them in the correct order?
Intuition: gives an aggregate measure of performance aggregated across all possible classification thresholds

