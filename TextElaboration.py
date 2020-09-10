import mysql.connector
import csv
import numpy as np
import pandas as pd
import re
import string
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


mydb = mysql.connector.connect(
	host="localhost",
	user="root",
	passwd="fede95",
	database="tweet_db"
)

# ----- PREPARING TRAINING SET -----
def createTrainCsv(name_file, table):
	cursor = mydb.cursor()
	cursor.execute("select * from "+table+" ")

	with open(name_file+".csv", "w", newline='',encoding="utf-8") as csv_file:  # Python 3 version    
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow([i[0] for i in cursor.description])
		csv_writer.writerows(cursor)


# ----- DATA PREPROCESSING -----

def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

# Espande le abbreviazioni
def expandAbbreviations(data):
	train=[]
	
	for txt in data:
		if(isinstance(txt, str)):
			demoji=deEmojify(txt)

		text=txt.lower()

		splitted = re.split("\s+|'+", text)
		blank=""
		
		for word in splitted:
			with open("abbreviazioni.txt", "r") as myCSVfile:
			# Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
				dataFromFile = csv.reader(myCSVfile, delimiter="=")
				for row in dataFromFile:
                # Check if selected word matches short forms[LHS] in text file.
					if word == row[0]:
                    # If match found replace it with its appropriate phrase in text file.
						word = row[1]
				myCSVfile.close()
			
			# Concatena le parole in un'unica frase
			if word.isspace()==False:		
				blank=blank+word+" "
		
		if blank.isspace()==False:
			train.append(blank)
	
	return train


# Carica stop words list
def loadStopWlist():
	stopword_list = list()
	file = open("stop_words.txt", "r") 
	line=file.readline()
	while line: 
		x = line.split()
		stopword_list.append(x[0])
		line=file.readline()
	file.close()
	return stopword_list

# Rimuove #,menzioni, caratteri speciali e numeri
def cleaning(data):
	stopw_list=loadStopWlist()

	cleaned_train=[]
	
	for txt in data:
		
		text=txt.lower()

		splitted = re.split("\s+|'+", text)
		blank=""
		
		for word in splitted:
			
			if re.match(r'^#', word):
				word=re.sub("#", "", word)
			if re.match(r'^@', word):
				word=""
			if re.match(r'^([\s\d]+)$', word):
				word=""
			#word = re.sub('[^0-9a-zA-Z]+', '', word)
			word = re.sub('[^A-Za-z0-9àèìòùé]+', ' ', word)
			
			if word=="":
				break

			for sw in stopw_list:
				if re.match(" "+sw+" ", " "+word+" "):
					word=""
					break
			
			# Concatena le parole in un'unica frase
			if word.isspace()==False:		
				blank=blank+word+" "

		if blank.isspace()==False or blank!="" or blank!=" ":
			cleaned_train.append(blank)
	
	return cleaned_train


def stemming(data):
	stemmer = SnowballStemmer("italian")
		
	stemmed_train=[]

	for text in data:
		
		splitted = re.split("\s+|'+", text)
		stemmed=[]

		for word in splitted:
			stemmed.append(stemmer.stem(word))
		
		blank=''
		for word in stemmed:
			if word.isspace()==False:
				blank=blank+word+' '
		
		if blank.isspace()==False or blank!="" or blank!=" ":
			stemmed_train.append(blank)
	return stemmed_train

