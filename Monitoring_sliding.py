#import TwintScrape as ts
from sklearn.externals import joblib
import TextElaboration as elab
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
import mysql.connector
import random

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
import csv

# ----- PARAMETRI DI CONFIGURAZIONE -----
dataset_table="monitoring"
date_since="2020-06-03"

# ---------------------------------------

mydb = mysql.connector.connect(
	host="localhost",
	user="root",
	passwd="fede95",
	database="tweet_db"
)


# --------------- CLASSIFICAZIONE TRAMITE MODELLO ALLENATO----------------
text_clf = joblib.load('modello_sliding_window/SVM_fs_1905.pkl')
text_increm = joblib.load('modello_incrementale/SVM_fs_1905.pkl')

cursor = mydb.cursor()
cursor.execute("SELECT * FROM "+dataset_table+" WHERE date=\""+date_since+"\" ")
records = cursor.fetchall()
print("Total number of rows: ", cursor.rowcount)
test_id=[]
test_text=[]
for row in records:
    test_id.append(row[0])
    test_text.append(row[4])


# Data cleaning and stemming
expanded_train=elab.expandAbbreviations(test_text)
cleaned_train=elab.cleaning(test_text)
stemmed_train=elab.stemming(cleaned_train)
predicted = text_clf.predict(stemmed_train)
predicted_increm = text_increm.predict(stemmed_train)

mycursor = mydb.cursor()
update = "UPDATE "+dataset_table+" SET tag_pred_window= %s WHERE id= %s"
for rows in range(0,cursor.rowcount):
    val = (int(predicted[rows]), test_id[rows])
    mycursor.execute(update, val)
    mydb.commit()

mycursor_increm = mydb.cursor()
update = "UPDATE "+dataset_table+" SET tag_pred_increm= %s WHERE id= %s"
for rowss in range(0,cursor.rowcount):
    val = (int(predicted_increm[rowss]), test_id[rowss])
    mycursor_increm.execute(update, val)
    mydb.commit()

# --------------- MODIFICA DEL MODELLO A FINESTRA MOBILE ----------------
print("SLIDING WINDOW")

# Prelevo i record del giorno selezionato divisi per tag
curs_new = mydb.cursor()
curs_new.execute("SELECT * FROM "+dataset_table+" WHERE tag_man=0 and date=\""+date_since+"\" ")
new_neutr = curs_new.fetchall()

curs_new = mydb.cursor()
curs_new.execute("SELECT * FROM "+dataset_table+" WHERE tag_man=1 and date=\""+date_since+"\"  ")
new_rass = curs_new.fetchall()

curs_new = mydb.cursor()
curs_new.execute("SELECT * FROM "+dataset_table+" WHERE tag_man=2 and date=\""+date_since+"\" ")
new_all = curs_new.fetchall()

# Cerco il numero minimo
list_val=[]

list_val.append(len(new_rass))
list_val.append(len(new_neutr))
list_val.append(len(new_all))
print(list_val)

min_val=min(list_val)
print(min_val)

# Prelevo i record dal training set a finestra mobile
curs0 = mydb.cursor()
curs0.execute("select * from train_window where tag=0 order by date limit %s", (min_val,))
neutr = curs0.fetchall()

curs1 = mydb.cursor()
curs1.execute("select * from train_window where tag=1 order by date limit %s", (min_val,))
rass = curs1.fetchall()

curs2 = mydb.cursor()
curs2.execute("select * from train_window where tag=2 order by date limit %s", (min_val,))
alla = curs2.fetchall()




# Seleziono i record del training set a finestra mobile da eliminare
delete_n = random.sample(neutr, min_val)
print("ELIMINO "+str(len(delete_n))+ " ELEMENTI")

delete_r = random.sample(rass, min_val)
print("ELIMINO "+str(len(delete_r))+ " ELEMENTI")

delete_a = random.sample(alla, min_val)
print("ELIMINO "+str(len(delete_a))+ " ELEMENTI")


cur0 = mydb.cursor()
query = "DELETE FROM train_window where id=%s"
for rows in delete_n:
    cur0.execute(query, (rows[0],))
    print("Righe cancellate "+str(cur0.rowcount))
    mydb.commit()

cur1 = mydb.cursor()
for rows2 in delete_r:
    cur1.execute(query, (rows2[0],))
    print("Righe cancellate "+str(cur1.rowcount))
    mydb.commit()

cur2 = mydb.cursor()
for rows3 in delete_a:
    cur2.execute(query, (rows3[0],))
    print("Righe cancellate "+str(cur2.rowcount))
    mydb.commit()


# Inserisco i nuovi record nel training set a finestra mobile
insert_n = random.sample(new_neutr, min_val)
print("INSERISCO "+str(len(insert_n))+ " ELEMENTI")

insert_r = random.sample(new_rass, min_val)
print("INSERISCO "+str(len(insert_r))+ " ELEMENTI")

insert_a = random.sample(new_all, min_val)
print("INSERISCO "+str(len(insert_a))+ " ELEMENTI")

cur_new0 = mydb.cursor()
query = "INSERT INTO train_window (id,username,date,time,text,tag) VALUES(%s,%s,%s,%s,%s,%s)"
for rows in insert_n:
    val=(rows[0], rows[1], rows[2],rows[3],rows[4],rows[5])
    cur_new0.execute(query, val)
    print("Righe inserite "+str(cur_new0.rowcount))
    mydb.commit()

cur_new1 = mydb.cursor()
for rows2 in insert_r:
    val=(rows2[0], rows2[1], rows2[2],rows2[3],rows2[4],rows2[5])
    cur_new1.execute(query, val)
    print("Righe inserite "+str(cur_new1.rowcount))
    mydb.commit()

cur_new2 = mydb.cursor()
for rows3 in insert_a:
    val=(rows3[0], rows3[1], rows3[2],rows3[3],rows3[4],rows3[5])
    cur_new2.execute(query, val)
    print("Righe inserite "+str(cur_new2.rowcount))
    mydb.commit()

elab.createTrainCsv("modello_sliding_window/train_window","train_window")
train_set = pd.read_csv("modello_sliding_window/train_window.csv")
print("----DISTRIBUZIONE DATASET----")
print(train_set['tag'].value_counts())

data=train_set['text']
tag=train_set['tag']

#Pipeline Classifier
text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('fselect', SelectKBest(chi2, k=1905)),
        ('clf', svm.SVC()),
    ])


train=np.asarray(data) 
# Data cleaning and stemming
expanded_train=elab.expandAbbreviations(train)
cleaned_train=elab.cleaning(expanded_train)
stemmed_train=elab.stemming(cleaned_train)


results = cross_validate(estimator=text_clf,
                                          X=stemmed_train,
                                          y=tag,
                                          cv=10,
                                          )

acc=np.mean(results['test_score'])
time=np.mean(results['fit_time'])
print("accuracy "+ str(acc))
print("time "+str(time))

text_clf.fit(stemmed_train, tag)

joblib.dump(text_clf, 'modello_sliding_window/SVM_fs_1905.pkl', compress = 1)
print('THE CLASSIFIER HAS BEEN SAVED.')

with open("modello_sliding_window/perf.csv", "a", newline='',encoding="utf-8") as csv_file:  # Python 3 version  
	filewrite = csv.writer(csv_file, delimiter=',')  
	row=[]
	row.append(acc)
	row.append(time)
	filewrite.writerow(row)
	csv_file.close

# --------------- MODIFICA DEL MODELLO INCREMENTALE ----------------
print("INCREMENTALE")
cur_new0_ = mydb.cursor()
query = "INSERT INTO train_increm (id,username,date,time,text,tag) VALUES(%s,%s,%s,%s,%s,%s)"
for rows in insert_n:
    val=(rows[0], rows[1], rows[2],rows[3],rows[4],rows[5])
    cur_new0_.execute(query, val)
    print("Righe inserite "+str(cur_new0_.rowcount))
    mydb.commit()

cur_new1_ = mydb.cursor()
for rows2 in insert_r:
    val=(rows2[0], rows2[1], rows2[2],rows2[3],rows2[4],rows2[5])
    cur_new1_.execute(query, val)
    print("Righe inserite "+str(cur_new1_.rowcount))
    mydb.commit()

cur_new2_ = mydb.cursor()
for rows3 in insert_a:
    val=(rows3[0], rows3[1], rows3[2],rows3[3],rows3[4],rows3[5])
    cur_new2_.execute(query, val)
    print("Righe inserite "+str(cur_new2_.rowcount))
    mydb.commit()


elab.createTrainCsv("modello_incrementale/train_increm","train_increm")
train_set = pd.read_csv("modello_incrementale/train_increm.csv")
print("----DISTRIBUZIONE DATASET----")
print(train_set['tag'].value_counts())

data=train_set['text']
tag=train_set['tag']

#Pipeline Classifier
text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('fselect', SelectKBest(chi2, k=1905)),
        ('clf', svm.SVC()),
    ])



train=np.asarray(data) 
# Data cleaning and stemming
expanded_train=elab.expandAbbreviations(train)
cleaned_train=elab.cleaning(expanded_train)
stemmed_train=elab.stemming(cleaned_train)


results = cross_validate(estimator=text_clf,
                                          X=stemmed_train,
                                          y=tag,
                                          cv=10,
                                          )

acc=np.mean(results['test_score'])
time=np.mean(results['fit_time'])
print("accuracy "+ str(acc))
print("time "+str(time))

text_clf.fit(stemmed_train, tag)

joblib.dump(text_clf, 'modello_incrementale/SVM_fs_1905.pkl', compress = 1)
print('THE CLASSIFIER HAS BEEN SAVED.')

with open("modello_incrementale/perf.csv", "a", newline='',encoding="utf-8") as csv_file:  # Python 3 version  
	filewrite = csv.writer(csv_file, delimiter=',')  
	row=[]
	row.append(acc)
	row.append(time)
	filewrite.writerow(row)
	csv_file.close
