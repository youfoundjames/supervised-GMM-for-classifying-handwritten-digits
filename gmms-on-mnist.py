import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

MNIST = load_digits()
X, y = MNIST.data, MNIST.target

# normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.astype(np.float64))

# perform PCA to reduce dimensionality
delta = 0.99 #change this to keep enough principal components to account for (100 * delta)% of variance
pca = PCA(n_components=delta, random_state=42)  #
X_pca = pca.fit_transform(X)
print("Number of principal components kept: ", pca.n_components_)

n_components=15 # this is the number of components of our GMM.

# train a GMM without using class labels (unsupervised, density estimator)
gmmUnsupervised = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42).fit(X_pca, None)

# plot GMM means for each class
def plot_gmm_means(gmm, classes):
    plt.figure(figsize=(12, 6))
    for i in range(len(classes)):
        # find indices of samples belonging to the current class
        indices = np.where(y == classes[i])[0]
        
        # extract the PCA-transformed features of the current class
        X_class = pca.transform(X[indices])
        
        # plot GMM means
        means = gmm.means_[i]
        plt.subplot(2, 5, i + 1)
        mean_image = pca.inverse_transform(means).reshape(8, 8)  # reshape using inverse PCA transform
        plt.imshow(mean_image, cmap='gray')
        plt.title('Class {}'.format(classes[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print('Means of classes of unsupervised model:')
plot_gmm_means(gmmUnsupervised, np.unique(y))

n_components_range = range(1, 64)
bic_scores = []

# get the unique class labels
classes = np.unique(y)

# initialize a list to store the GMMs for each class
gmms = []

# for all k in {numbers of components we wish to test}
for n_components in n_components_range:
    bic_scores_class = []
    # train a separate GMM for each class
    for c in classes:
        # extract the samples of the current class
        X_class = X_pca[y == c]
        
        # train a GMM on the samples of the current class
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42).fit(X_class)
        
        # add the trained GMM to the list
        gmms.append(gmm)
        
        # calculate its BIC; it gets n, k, and the likelihood function of the model from "gmm"
        bic_scores_class.append(gmm.bic(X_class))
    
    # calculate average BIC score for this number of components
    bic_scores.append(np.mean(bic_scores_class))

# store the results in a DataFrame
df = pd.DataFrame({
    'Number of Components': n_components_range,
    'BIC Score': bic_scores
})

# find/print the model with the lowest BIC score
optimal_model = df[df['BIC Score'] == df['BIC Score'].min()]
optimal_number_of_components = optimal_model['Number of Components'].values[0]

print("Optimal number of components according to BIC score:", optimal_model['Number of Components'].values[0])
print("BIC score:", optimal_model['BIC Score'].values[0])

# plot BIC scores
plt.figure(figsize=(10, 6))
plt.plot(df['Number of Components'], df['BIC Score'], marker='o', color='b', linestyle='-')
plt.title('BIC Scores for Different Numbers of Components')
plt.xlabel('Number of Components')
plt.ylabel('BIC Score')
plt.grid(True)
plt.show()

# get the unique class labels
classes = np.unique(y)

# initialize a list to store the GMMs for each class
gmms = []

# train a separate GMM for each class
for c in classes:
    # extract the samples of the current class
    X_class = X_pca[y == c]
    
    # train a GMM on the samples of the current class
    gmm = GaussianMixture(n_components=50, reg_covar=1e-2, covariance_type='full', random_state=42).fit(X_class)
    
    # Add the trained GMM to the list
    gmms.append(gmm)

# function to plot GMM means for each class
def plot_gmm_means(gmms, classes):
    plt.figure(figsize=(12, 6))
    for i in range(len(classes)):
        # Plot GMM means
        means = gmms[i].means_
        plt.subplot(2, 5, i + 1)
        mean_image = pca.inverse_transform(means[i]).reshape(8, 8)  # Reshape to 8x8
        plt.imshow(mean_image, cmap='gray')
        plt.title('Class {}'.format(classes[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print('Means of classes of supervised model:')
plot_gmm_means(gmms, classes)

# initialize an empty array to store the predictions
predictions = []

# for each sample, (we will predict the class label)
for sample in X_pca:
    
    # compute the log likelihood of the sample under each GMM
    log_likelihoods = [gmm.score_samples(sample.reshape(1, -1)) for gmm in gmms]
    
    # find the class with the highest log likelihood
    pred = np.argmax(log_likelihoods)
    
    # add the predicted class label to the list
    predictions.append(pred)

# compute the confusion matrix
cm = confusion_matrix(y, predictions)

# plot the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.show()

# predict the class labels for all samples
predictions = gmmUnsupervised.predict(X_pca)

# compute and plot the confusion matrix
cm = confusion_matrix(y, predictions)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.show()

# initialize a figure for the plots
fig, axs = plt.subplots(4, 5, figsize=(10, 8))

# generate and plot 20 samples
for i in range(20):
    # choose a random GMM
    gmm = np.random.choice(gmms)
    
    # generate a sample from the GMM
    sample, _ = gmm.sample()
    
    # transform the sample back into the original space
    sample = pca.inverse_transform(sample)
    
    # reshape the sample into an 8x8 image
    image = sample.reshape(8, 8)
    
    # plot the image
    axs[i // 5, i % 5].imshow(image, cmap='gray')
    axs[i // 5, i % 5].axis('off')

plt.tight_layout()
plt.show()

# generate 20 samples from the GMM
samples, _ = gmmUnsupervised.sample(20)

# transform the samples back to the original space
samples = pca.inverse_transform(samples)

# reshape the samples to 8x8 images
samples = samples.reshape(-1, 8, 8)

# plot the samples in 4 rows of 5
plt.figure(figsize=(10, 8))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(samples[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Comparing our model to other classifiers...

from sklearn.ensemble import GradientBoostingClassifier

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# train a gradient boosted tree classifier
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)

# predict the labels for the test set
y_pred_gb = gb.predict(X_test)

# compute the confusion matrix for the Gradient Boosting Classifier
cm_gb = confusion_matrix(y_test, y_pred_gb)
plt.figure(figsize=(10, 10))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.show()

from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# train the model
clf.fit(X_train, y_train)

# make predictions on the test set
y_pred = clf.predict(X_test)

# compute and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.show()