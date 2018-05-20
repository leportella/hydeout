---
layout: post
title: "Machine Learning Models - My Cheat List"
categories:
  - cheatlist
tags:
  - en
  - data science
  - machine learning
  - nanodegree
  - supervised models 
  - cheatlist
last_modified_at: 2018-05-01T14:25:52-05:00
---


# Summary

* [Supervised Models](#supervised)

# Machine Learning Models Cheat List


<h2 id='supervised'>Supervised Models</h2>

This is a small revision on advantages and disadvantages of each model, based on 
suggested models of Udacity's Nanodegree in Machine Learning Engineer.

### Logistic Regression

**Advantages**:

* Don't have to worry about features being correlated
* You can easily update your model to take in new data (unlike Decision Trees or SVM)

**Disadvantages**: 

* Deals bad with outliers
* Must have lots of incomes for each class
* Presence of multicollinearity

**Real use**:

* Some obese people get gastric bypass surgery to lose weight, and some of them die as a result of the surgery. Benotti et al. (2014) wanted to know whether they could predict who was at a higher risk of dying from one particular kind of surgery, Roux-en-Y gastric bypass surgery. They obtained records on 81,751 patients who had had Roux-en-Y surgery, of which 123 died within 30 days. They did multiple logistic regression, with alive vs. dead after 30 days as the dependent variable, and 6 demographic variables (gender, age, race, body mass index, insurance type, and employment status) and 30 health variables (blood pressure, diabetes, tobacco use, etc.) as the independent variables. Manually choosing the variables to add to their logistic model, they identified six that contribute to risk of dying from Roux-en-Y surgery: body mass index, age, gender, pulmonary hypertension, congestive heart failure, and liver disease. —> from http://www.biostathandbook.com/multiplelogistic.html

### Decision Tree

**Advantages**:

* Easy to understand and interpret (for some people)
* Easy to use - Doesn’t need data normalisation, dummy variables, etc 
* Can handle multi-output models
* Easily handle feature interactions
* Don't have to worry about outliers

**Disadvantages**:

* It can be easily overfitted
* Stability —> small changes in data can lead to completely different trees
* If a class dominates, it can easily be biased
* Don't support online learning --> you should rebuilt the tree when new data comes

**Real case**:

* Used several Decision Tree configurations trying to predict lookahead and pathology —> http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.44.6872&rep=rep1&type=pdf

### Ensemble Methods

**Advantages**:

* Harder to overfit
* Usually better perfomance than a single model

**Disadvantages**:

* Scaling —> usually it trains several models, which can have a bad performance with larger datasets
* Hard to implement in real time platform
* Complexity increases
* Boosting delivers poor probability estimates (https://arxiv.org/ftp/arxiv/papers/1207/1207.1403.pdf)

**Real case**:

* AdaBoosting was used to detect baseball players game analysis (https://www.uni-obuda.hu/journal/Markoski_Ivankovic_Ratgeber_Pecev_Glusac_57.pdf)


### K-Nearest Neighbors

**Advantages**:

* Little training time
* Works well with multiclass datasets 
* Good for highly unusual data

**Disadvantages**:

* Need to determine value of k (distance)
* Neighbors-based methods are known as non-generalizing machine learning methods, since they simply “remember” all of its training data
* The accuracy of KNN can be severely degraded with high-dimension data because there is little difference between the nearest and farthest neighbor.

### Gaussian Naive Bayes 

**Advantages**:

* Need less training data tran models like logistic regression
* Highly scalable
* Not sensitive to irrelevant features
* Returns the degree of certanty of the answer
* Good when you need something fast and that perfoms well

**Disavantages**:

* Can't learn interactions between features e.g., it can’t learn that although you love movies with Brad Pitt and Tom Cruise, you hate movies where they’re together).

**Real case**:

* Emotion recognition with Gaussian Naive Bayes (https://ieeexplore.ieee.org/abstract/document/1044578/)

### SVM

**Advantages**:

* High accuracy
* Nice theoretical guarantees regarding overfitting
* Especially popular in text classification problems

**Disavantages**:

* Memory-intensive
* Hard to interpret
* Complicated to run and tune

### Stochastic Gradient Descent

**Advantages**:

* Efficiency
* Ease implementation

**Disavantages:** 

* A lot of hyperparameters to tune
* Sensitive to feature scaling

## General References

* [Choosing a machine learning classifier](http://blog.echen.me/2011/04/27/choosing-a-machine-learning-classifier/)
* [1](https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/#pros-and-cons-of-knn)
* [Sklearn documentation on Neighbors](http://scikit-learn.org/stable/modules/neighbors.html#neighbors)
* [3](http://people.revoledu.com/kardi/tutorial/KNN/Strength%20and%20Weakness.htm)
* [Sklearn documentation on Stochatic Gradient Descent](http://scikit-learn.org/stable/modules/sgd.html)
* [Sklearn documentation on Ensemble Methods](http://scikit-learn.org/stable/modules/ensemble.html)
* [Logistic Regression Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
* [Logistic Regression for machine learning](https://machinelearningmastery.com/logistic-regression-for-machine-learning/)
* [What are the advantages of logistic regression](https://www.quora.com/What-are-the-advantages-of-logistic-regression)
* [The disadvantages of Logistic Regression](https://classroom.synonym.com/disadvantages-logistic-regression-8574447.html)
