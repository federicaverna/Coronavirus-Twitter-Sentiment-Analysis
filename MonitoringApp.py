#import TwintScrape as ts
from sklearn.externals import joblib
import TextElaboration as elab
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
import mysql.connector
import matplotlib.pyplot as plt

# ----- PARAMETRI DI CONFIGURAZIONE -----

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
# --------------- SETTAGGIO PARAMETRI ----------------
print("Enter the table name in which saving Tweets")
dataset_table=input()

print("Enter the date from which to start scraping in the format %Y-%m-%d")
date_since=input()

print("Enter the end date of the scraping in the format %Y-%m-%d")
date_until=input()

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

cursor = mydb.cursor()
cursor.execute("select date,tag_pred_orig as tag, count(id)as count from monitoring where date>=\""+date_since+"\" and date<\""+date_until+"\"group by date,tag_pred_orig ORDER BY date")
records = cursor.fetchall()
tag0=[]
tag1=[]
tag2=[]
for row in records:
    if(row[1]=="0"):
        tag0.append(row[2])
    elif(row[1]=="1"):
        tag1.append(row[2])
    elif(row[1]=="2"):
        tag2.append(row[2])

labels = 'Neutrali', 'Rassicuranti', 'Allarmanti'
sizes = [tag0, tag1, tag2]

values=[tag0[0],tag1[0],tag2[0]]

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct
    
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct=make_autopct(values),
        shadow=True, startangle=90)
ax1.axis('equal') 
plt.title("People perception "+date_since)
plt.show()
