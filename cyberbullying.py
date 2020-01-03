import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import pickle 
import time

import nltk
from nltk import Text
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import word_tokenize  
from nltk.tokenize import sent_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics  import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from methods import Classifications, TfidfWeighting, TfWeighting, BinaryWeighting 


#READING DATASET
folder = 'C:\\Users\\imge\\Desktop\\Final Project\\data'
labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()
for f in ('test', 'train'):    
    for l in ('pos', 'neg'):
        path = os.path.join(folder, f, l)
        for file in os.listdir (path) :
            with open(os.path.join(path, file),'r') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]],ignore_index=True)
            
df.columns = ['Comments', 'Positivity']

comment=df.Comments
positivity=df.Positivity

#DATA PREPROCESSING
comments_list = []
turkish_stop_words = stopwords.words('turkish')
for comment in df.Comments:
    comment=comment.lower()
    turkish_letters = '[^a-zçğışöü]'
    comment = re.sub(turkish_letters, ' ', comment)
    comment = re.sub(r'[0-9]+', '', comment)
    comment = comment.split()
    comment = [word for word in comment if not word in turkish_stop_words]
    comment = " ".join(comment)
    comments_list.append(comment)


x_train, x_test, y_train, y_test = train_test_split(
         comments_list, positivity, test_size=0.10, random_state=42)

#TF WEIGHTING
train_features1, test_features1=TfWeighting(x_train, x_test)
Classifications(train_features1, test_features1, y_train, y_test, "TF (Term Frequency)")

#TF*IDF WEIGHTING
train_features2, test_features2=TfidfWeighting(x_train, x_test)
Classifications(train_features2, test_features2, y_train, y_test, "TF*IDF (Term Frequency*Inverse Document Frequency")

#BINARY WEIGHTING
train_features3, test_features3=BinaryWeighting(x_train, x_test)
Classifications(train_features3, test_features3, y_train, y_test, "BW (Binary Weighting)")


