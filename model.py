import pandas as pd
from collections import deque
from sklearn.svm import SVC
import numpy as np
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

def SVM_clf(X_train, X_test, y_train, y_test, X, y):
    clf = SVC(kernel='rbf',tol=0.01, gamma=20)#调参
    clf.fit(X_train, y_train.ravel())#训练
    prediction = clf.predict(X_test)
    #tn, fp, fn, tp = confusion_matrix(y_test.ravel(), prediction).ravel()
    score = accuracy_score(y_test, prediction)
    five_fold_score = cross_val_score(clf, X, y.ravel(), cv=5, scoring='accuracy')
    #print("tn, fp, fn, tp:", tn, fp, fn, tp)
    print("acc = {:.2%}".format(score))
    print("Accuracy: %0.2f (+/- %0.2f)" % (five_fold_score.mean(), five_fold_score.std() * 2))
    print(classification_report(y_test.ravel(), prediction))

def PCA_KNN(X_train, X_test, y_train, y_test, X, y):
    pca = PCA()
    pca_fit = pca.fit(X_train)
    x_train_pca = pca_fit.transform(X_train)
    x_test_pca = pca_fit.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train_pca, y_train.ravel())
    y_predict = knn.predict(x_test_pca)
    score = knn.score(x_test_pca, y_test.ravel(), sample_weight=None)
    five_fold_score = cross_val_score(knn, X, y.ravel(), cv=5, scoring='accuracy')
    print("acc = {:.2%}".format(score))
    print("Accuracy: %0.2f (+/- %0.2f)" % (five_fold_score.mean(), five_fold_score.std() * 2))
    print(classification_report(y_test.ravel(), y_predict))



if __name__ == "__main__":
    gao = pd.read_csv('data/G1.csv')
    wang = pd.read_csv('data/W1.csv')
    li = pd.read_csv('data/L1.csv')
    yan = pd.read_csv('data/Y1.csv')

    data = pd.concat([wang, yan, gao, li],axis=0)
    data = shuffle(data)
    
    X = np.array(data[['p','x','y','z']])
    y = np.array(data[['label']])
    sacler = StandardScaler()
    sacler = sacler.fit(X)
    X = sacler.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=2020,
                                                        shuffle=False)

    SVM_clf(X_train, X_test, y_train, y_test, X, y)
    PCA_KNN(X_train, X_test, y_train, y_test, X, y)

'''
SVM Classification
acc = 92.32%
Accuracy: 0.95 (+/- 0.03)
              precision    recall  f1-score   support
         gao       0.93      0.90      0.92       175
          li       0.88      0.98      0.93       235
        wang       0.96      0.88      0.92       180
         yan       0.95      0.92      0.93       178
    accuracy                           0.92       768
   macro avg       0.93      0.92      0.92       768
weighted avg       0.93      0.92      0.92       768
KNN(K=3) Classification
acc = 91.80%
Accuracy: 0.94 (+/- 0.02)
              precision    recall  f1-score   support
         gao       0.90      0.94      0.92       175
          li       0.95      0.94      0.94       235
        wang       0.91      0.87      0.89       180
         yan       0.91      0.92      0.91       178
    accuracy                           0.92       768
   macro avg       0.92      0.92      0.92       768
weighted avg       0.92      0.92      0.92       768
'''