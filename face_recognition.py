print(__doc__)

import os
from gzip import GzipFile

import numpy as np
import pylab as pl

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

################################################################################
# Download the data, if not already on disk

url = "https://downloads.sourceforge.net/project/scikit-learn/data/lfw_preprocessed.tar.gz"
archive_name = "lfw_preprocessed.tar.gz"
folder_name = "lfw_preprocessed"

if not os.path.exists(folder_name):
    if not os.path.exists(archive_name):
        import urllib
        print("Downloading data, please Wait (58.8MB)...")
        print(url)
        opener = urllib.request.urlopen(url)
        open(archive_name, 'wb').write(opener.read())
        print

    import tarfile
    print("Decompressiong the archive: " + archive_name)
    tarfile.open(archive_name, "r:gz").extractall()
    print

################################################################################
# Load dataset in memory

faces_filename = os.path.join(folder_name, "faces.npy.gz")
filenames_filename = os.path.join(folder_name, "face_filenames.txt")

faces = np.load(GzipFile(faces_filename))
face_filenames = [l.strip() for l in open(filenames_filename).readlines()]

# normalize each picture by centering brightness
faces -= faces.mean(axis=1)[:, np.newaxis]


################################################################################
# Index category names into integers suitable for scikit-learn

# Here we convert file names in integer class indices that are suitable to be
# used as a target for training a classifier. Note the use of an array with
# unique entries to store the relation between class index and name,
# aka 'Look Up Table' (LUT).
# Also, note the use of 'searchsorted' to convert an array in a set of
# integers given a second array to use as a LUT.
categories = np.array([f.rsplit('_', 1)[0] for f in face_filenames])

# A unique integer per category
category_names = np.unique(categories)

# Turn the categories in their corresponding integer label
target = np.searchsorted(category_names, categories)

# Subsample the dataset to restrict to the most frequent categories
selected_target = np.argsort(np.bincount(target))[-5:]

# If you are using a numpy version >= 1.4, this can be done with 'np.in1d'
mask = np.array([item in selected_target for item in target])

X = faces[mask]
y = target[mask]

n_samples, n_features = X.shape

print("Dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)

split = n_samples * 3 / 4

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

################################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces" % n_components)
pca = PCA(n_components=n_components, whiten=True, svd_solver='randomized').fit(X_train)

eigenfaces = pca.components_.T.reshape((n_components, 64, 64))

# project the input data on the eigenfaces orthonormal basis
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


################################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
param_grid = {
 'C': [1, 5, 10, 50, 100],
 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
}
clf = GridSearchCV(SVC(kernel='rbf'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("Best estimator found by grid search:")
print(clf.best_estimator_)


################################################################################
# Quantitative evaluation of the model quality on the test set

y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, labels=selected_target,
                            target_names=category_names[selected_target]))

print(confusion_matrix(y_test, y_pred, labels=selected_target))


################################################################################
# Qualitative evaluation of the predictions using matplotlib

n_row = 4
n_col = 5

pl.figure(figsize=(2 * n_col, 2.3 * n_row))
pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.95, hspace=.15)
for i in range(n_row * n_col):
    pl.subplot(n_row, n_col, i + 1)
    pl.imshow(X_test[i].reshape((64, 64)), cmap=pl.cm.gray)
    pl.title('pred: %s\ntrue: %s' % (category_names[y_pred[i]],
                                     category_names[y_test[i]]), size=12)
    pl.xticks(())
    pl.yticks(())

pl.show()

# TODO: plot the top eigenfaces and the singular values absolute values
