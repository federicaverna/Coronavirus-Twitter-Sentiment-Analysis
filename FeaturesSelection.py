import TextElaboration as elab
import pandas as pd
import matplotlib.pyplot as plt
import TextElaboration as elab
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
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import chi2
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import f_classif


# ----- PARAMETRI DI CONFIGURAZIONE -----
num_fold=10
table="training_set" #tabella contenente il training set
train_file="k_fold/training_set" #nome del file csv su cui salvare il training set
#name_classifier = 'SVM' # NB or SVM

# ---------------------------------------

elab.createTrainCsv(train_file,table)
train_set = pd.read_csv(train_file+'.csv')


data=train_set['text']
tag=train_set['tag']

#X_train, X_test, y_train, y_test = train_test_split(data, tag, test_size=0.3, random_state=1)
train=np.asarray(data) 

# Data cleaning and stemming
expanded_train=elab.expandAbbreviations(train)
cleaned_train=elab.cleaning(expanded_train)
stemmed_train=elab.stemming(cleaned_train)

for num_features in range(500,2200,1):
    print(num_features)
    #Pipeline Classifier
    text_svm = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('fselect', SelectKBest(chi2, k=num_features)),
        ('clf', svm.SVC()),
        ])
  
    text_nb = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('fselect', SelectKBest(chi2, k=num_features)),
        ('clf', MultinomialNB()),
        ])
    
    scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score,average='micro',labels=tag,zero_division=True),
           'recall' : make_scorer(recall_score,average='micro',labels=tag,zero_division=True), 
           'f1_score' : make_scorer(f1_score,average='micro',labels=tag,zero_division=True)}


    results_svm = cross_validate(estimator=text_svm,
                                          X=stemmed_train,
                                          y=tag,
                                          cv=10,
                                          scoring=scoring
                                          )

    results_nb = cross_validate(estimator=text_nb,
                                          X=stemmed_train,
                                          y=tag,
                                          cv=10,
                                          scoring=scoring
                                          )
    
    with open('NB_acc_train_c.txt', 'a') as f:
        f.write("%s\n" % np.mean(results_nb['test_accuracy']))
    with open('NB_prec_train_c.txt', 'a') as f:
        f.write("%s\n" % np.mean(results_nb['test_precision']))
    with open('NB_rec_train_c.txt', 'a') as f:
        f.write("%s\n" % np.mean(results_nb['test_recall']))
    with open('NB_f_train_c.txt', 'a') as f:
        f.write("%s\n" % np.mean(results_nb['test_f1_score']))
    
    with open('SVM_acc_train_c.txt', 'a') as f:
        f.write("%s\n" % np.mean(results_svm['test_accuracy']))
    with open('SVM_prec_train_c.txt', 'a') as f:
        f.write("%s\n" % np.mean(results_svm['test_precision']))
    with open('SVM_rec_train_c.txt', 'a') as f:
        f.write("%s\n" % np.mean(results_svm['test_recall']))
    with open('SVM_f_train_c.txt', 'a') as f:
        f.write("%s\n" % np.mean(results_svm['test_f1_score']))


