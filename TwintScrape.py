import twint 
import mysql.connector
import re
from datetime import datetime

# ----- PARAMETRI DI CONFIGURAZIONE -----
table="monitoring" #tabella contenente il training set
since="2020-06-08"
until="2020-06-09" 
# ---------------------------------------

mydb = mysql.connector.connect(
	host="localhost",
	user="root",
	passwd="fede95",
	database="tweet_db"
)

def storeTweets(tweets, table):
	cursor = mydb.cursor()
	for tweet_object in tweets:
		if(checkUsername(tweet_object.username) & checkHashtag(tweet_object.hashtags) & checkText(tweet_object.tweet)):
			
			datetime_str = tweet_object.datestamp+" "+ tweet_object.timestamp
			
			print(datetime_str)
			
			sql = "INSERT IGNORE INTO "+table+"(id, username, date, time, text) VALUES (%s, %s, %s, %s, %s)"
			val = (tweet_object.id, tweet_object.username, tweet_object.datestamp, tweet_object.timestamp, tweet_object.tweet)
			cursor.execute(sql, val)
			mydb.commit()
	cursor.close()


def checkUsername(username):
	with open('blacklist_users.txt','r') as f:
		for line in f:
			for word in line.split():
				if re.match(word, username):
						#print("salto "+username+" "+line)
					return False
			
	with open('blacklist_regexp.txt','r') as f:
		for new_line in f:
			for new_word in new_line.split():
				if re.search(new_word, username, re.IGNORECASE):
					file=open('blacklist_users.txt','a')
					file.write(username+"\n")
					file.close
					return False
			
	return True


def checkText(text):
	with open('blacklist_words.txt','r') as f:
		for line in f:
				for word in line.split():
					for word_text in text.split():
						#print(word_text+"   "+word)
						if re.search(word, word_text, re.IGNORECASE):
							return False
	return True

def checkHashtag(list_h):
	with open('blacklist_hashtag.txt','r') as f:
		for line in f:
				for word in line.split():
					for hasht in list_h:
						hasht=re.sub("#", "", hasht)
						#print(hasht+"   "+word)
						if re.match(word, hasht, re.IGNORECASE):
							#print(hasht+"   "+word)
							return False
	return True



def getTweetsTwint(since, until):
    # Configure
	c = twint.Config()
	c.Search = "#coronavirusitalia OR #restiamoincasa OR #RestiamoInCasa OR #covid_19 OR #COVID_19 OR #Covid_19 OR #COVID2019 OR #isoliamoilvirus OR #fermiamoloinsieme OR #tuttoandràbene OR #distantimavicini OR #COVID2019italia OR #quarantena OR #Restiamoincasa OR #Covid OR #restateincasa OR #forzacelafaremo OR #lockdown OR #CoronaVirusIT OR #Andratuttobene OR #StopCOVID19 OR #COVID19italia OR #coronavirus OR #insiemecelafaremo OR #celafaremo OR #iorestoacasa OR #andratuttobene OR #andràtuttobene OR #restiamoacasa OR #CoronavirusOutbreak OR #coronaviruschina OR #ChinaCoronaVirus OR #nCoV OR #coronaviruses OR #nCoV2020 OR #nCov2019 OR #covid-19 OR #COVID-19 OR #COVID19 OR #covid19 OR #CoronaVirusitaly OR #CoronaVirusItaly"
	c.Lang = "it"
	c.Since=since
	c.Until=until
	c.Hide_output=True
	c.Links="exclude"
	c.Media=False
	c.Store_object = True

	twint.run.Search(c)
	return twint.output.tweets_list



print("Scraping...")
list_tweets=getTweetsTwint(since,until)
print("Storing...")
storeTweets(list_tweets,table)









