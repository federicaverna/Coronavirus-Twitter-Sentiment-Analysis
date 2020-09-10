import TextElaboration as elab
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import metrics, model_selection
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score

# ----- PARAMETRI DI CONFIGURAZIONE -----
num_fold=10
table="training_set" #tabella contenente il training set
train_file="k_fold/training_set" #nome del file csv su cui salvare il training set
name_classifier = 'NB' # NB or SVM
num_features=1298
# ---------------------------------------

elab.createTrainCsv(train_file,table)
train_set = pd.read_csv(train_file+'.csv')
print("----DISTRIBUZIONE DATASET----")
print(train_set['tag'].value_counts())

data=train_set['text']
tag=train_set['tag']

#Pipeline Classifier
if(name_classifier=="SVM"):
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('fselect', SelectKBest(chi2, k=num_features)),
        ('clf', svm.SVC()),
    ])
else:
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('fselect', SelectKBest(chi2, k=num_features)),
        ('clf', MultinomialNB()),
    ])


train=np.asarray(data) 
# Data cleaning and stemming
expanded_train=elab.expandAbbreviations(train)
cleaned_train=elab.cleaning(expanded_train)
stemmed_train=elab.stemming(cleaned_train)
"""
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score(average='micro')),
           'recall' : make_scorer(recall_score(average='micro')), 
           'f1_score' : make_scorer(f1_score(average='micro'))}
"""

scoring = {'recall0': make_scorer(f1_score, average = 'micro', labels = [0],zero_division=True),
       'recall1': make_scorer(f1_score, average = 'micro', labels = [1], zero_division=True),
       'recall2': make_scorer(f1_score, average = 'micro', labels = [2], zero_division=True)}

results = cross_validate(estimator=text_clf,
                                          X=stemmed_train,
                                          y=tag,
                                          cv=10,
                                          scoring=scoring
                                          )
print(np.mean(results['test_recall0']))
print(np.mean(results['test_recall1']))
print(np.mean(results['test_recall2']))

text_clf.fit(stemmed_train, tag)

#SALVATAGGIO MODELLO
joblib.dump(text_clf, 'modello/SVM_fs_1905.pkl', compress = 1)
print('THE CLASSIFIER HAS BEEN SAVED.') 

