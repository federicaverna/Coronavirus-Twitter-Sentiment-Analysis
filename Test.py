from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import random
import csv
import mysql.connector
from sklearn.metrics import precision_recall_fscore_support

# ----- PARAMETRI DI CONFIGURAZIONE -----
table="monitoring"
date="2020-06-03"
# ---------------------------------------

mydb = mysql.connector.connect(
	host="localhost",
	user="root",
	passwd="fede95",
	database="tweet_db"
)

cursor = mydb.cursor()
cursor.execute("select * from "+table+" where date=\""+date+"\" and tag_man is null")
records = cursor.fetchall()
print("Total number of rows: ", cursor.rowcount)

test_list = random.sample(records, 80)
mycursor = mydb.cursor()
insert = "UPDATE "+table+" set tag_man=%s where id= %s "
for rows in range(0,len(test_list)):
	print(test_list[rows][4])
	key=input()
	val=(key, test_list[rows][0])
	mycursor.execute(insert, val)
	mydb.commit()

