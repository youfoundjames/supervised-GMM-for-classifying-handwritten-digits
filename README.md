
# Supervised learning implementation of Gaussian mixture models to classify handwritten digits

## Overview

This is a somewhat unorthodox supervised learning implementation of Gaussian mixture models which, among other things, classifies the MNIST dataset with 100% accuracy. It is optimized for several metrics (namely Bayesian information criterion (BIC), F1 score, and accuracy, as in harmonic mean of precision/recall). It can also accurately classify various sklearn datasets; included in this implementation is a classifier on the sklearn moons dataset.

## Datasets used

- Scikit-learn MNIST dataset
    - Returns 1,797 samples (8x8 matrices) of handwritten digits 0-9. (We compress the samples using principal component analysis.)
- Scikit-learn moons dataset
    - Takes parameters n_samples, noise, random_state. Passing integer values to random_state allows reproduction of exact same dataset
    - Returns X, ndarray of 2-dimensional datapoints; y, ndarray of class membership of each datapoint


## Methodology

- We train a k-component GMM separately on each class of the dataset, using the labels included in the MNIST dataset (which we compress with PCA, keeping d=29 principal components). Using a fixed number of Gaussian components (i.e. multivariate Gaussian distributions) per GMM for each class in the dataset, we use the BIC method to find the optimal number of components k=15. (If computational cost is a concern, the GMM trained on each class could have a different and lesser number of components, which would involve computing a BIC for each component.)

- We classify the MNIST dataset using our "combined GMM" as follows: for each sample in the dataset, we compute the log likelihood of the sample under each GMM. The “class” of the GMM corresponding to the highest log likelihood is considered the predicted class for that sample.

- We also compare the predictive performance of our classifier to that of other classification models, namely gradient boosted decision trees and random decision forests. Our model performs better on these datasets, though it has also been optimized more effortfully.

## Results

The supervised model classifies the MNIST dataset with 100% accuracy. Here are the means of its predicted classes and its confusion matrix:

(means)

(confusion matrix)

We compare its predictive performance to other classifiers in the attached report. In the report, we also graph and assess its predictive performance on sklearn "toy" datasets with different levels of noise added to the data.

## Dependencies & environment


- Python 3.12.3
- numpy 1.26.4
- pandas 2.2.2
- scikit-learn 1.5.0
- (Optional) matplotlib 3.9.0 (for data visualization)
- (Optional) seaborn 0.13.2 (for enhanced data visualization)

Created in Anaconda. Jupyter notebook and standalone .py included.


## Instructions to run

Download and run the attached Jupyter notebook, or the standalone .py code

