import pandas as pd
import numpy as np

import re
import os

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics  import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#CLASSIFICATION METHOD
def Classifications(X_train_word_features, test_features, y_train, y_test, method_name):
    #Naive Bayes
    clf = MultinomialNB().fit(X_train_word_features, y_train)
    predicted = clf.predict(test_features)

    #SVM
    svm_classifier = svm.SVC(kernel='linear', C = 1.0)
    clf2=svm_classifier.fit(X_train_word_features, y_train)
    predicted2=clf2.predict(test_features)
    
    print("\nAccuracies by using ",method_name)
    print("Accuracy with Naive Bayes Classifier: ", accuracy_score(y_test,predicted))
    print("Accuracy with SVM: ", accuracy_score(y_test,predicted2))


    return

###WEIGHTING METHODS
#TF*IDF
def TfidfWeighting (x_train, x_test):
    word_vectorizer=TfidfVectorizer(
    stopwords.words('turkish'),
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{2,}',  #vectorize 2-character words or more
    ngram_range=(1, 1),
    max_features=30000)

    word_vectorizer.fit(x_train)
    X_train_word_features = word_vectorizer.transform(x_train)
    test_features = word_vectorizer.transform(x_test)

    return X_train_word_features, test_features

#TF
def TfWeighting (x_train, x_test):
    tf = TfidfVectorizer(use_idf=False, smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
    tf.fit(x_train)
    X_train_word_features = tf.transform(x_train)
    test_features = tf.transform(x_test)

    return X_train_word_features, test_features

#BW
def BinaryWeighting (x_train, x_test):
    count_vec = CountVectorizer(stopwords.words('turkish'), analyzer='word', 
                            ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, binary=True)
    X_train_word_features = count_vec.fit_transform(x_train).toarray()
    test_features = count_vec.transform(x_test).toarray()

    return X_train_word_features, test_features




















