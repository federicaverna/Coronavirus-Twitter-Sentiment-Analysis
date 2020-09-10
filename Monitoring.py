#import TwintScrape as ts
from sklearn.externals import joblib
import TextElaboration as elab
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
import mysql.connector
import matplotlib.pyplot as plt

# ----- PARAMETRI DI CONFIGURAZIONE -----
dataset_table="monitoring"
date_since="2020-06-08"
date_until="2020-06-09"
# ---------------------------------------

mydb = mysql.connector.connect(
	host="localhost",
	user="root",
	passwd="fede95",
	database="tweet_db"
)

# Le seguenti righe sono commentate in quanto il collezionamento dei tweet è stato effettuato
# separatamente tramite lo script TwintScrape.py

#list_tweets=ts.getTweetsTwint(date_since, date_until)
#ts.storeTweets(list_tweets,dataset_table)


# --------------- CLASSIFICAZIONE TRAMITE MODELLO ALLENATO----------------
text_clf = joblib.load('modello/SVM_fs_1905.pkl')

cursor = mydb.cursor()
cursor.execute("SELECT * FROM "+dataset_table+" WHERE date>=\""+date_since+"\" and date<\""+date_until+"\" ")
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


mycursor = mydb.cursor()
update = "UPDATE "+dataset_table+" SET tag_pred_orig= %s WHERE id= %s"
for rows in range(0,cursor.rowcount):
    val = (int(predicted[rows]), test_id[rows])
    mycursor.execute(update, val)
    mydb.commit()


