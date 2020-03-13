import GetOldTweets3 as got
import mysql.connector

mydb = mysql.connector.connect(
	host="localhost",
	user="root",
	passwd="fede95",
	database="tweets_db"
)


def streamTweets(tweets):
	for tweet in tweets:
		if(checkUsername(tweet.username) & checkWord(tweet.text)):
			sql = "INSERT IGNORE INTO downloaded_tweets(id, date, username, hashtags, retweets,text) VALUES (%s, %s, %s, %s, %s, %s)"
			val = (tweet.id, tweet.date, tweet.username, tweet.hashtags, tweet.retweets, tweet.text)
			mycursor.execute(sql, val)
			mydb.commit()

def checkUsername(username):
	with open('blacklist_regexp.txt','r') as f:
			for line in f:
				for word in line.split():
					if(word in username):
						#print("salto "+username+" "+line)
						return False
			#print("inserisco "+ username+" "+line)
			return True

def checkWord(text):
	with open('blacklist_words.txt','r') as f:
			for line in f:
				for word in line.split():
					for word_text in text.split():
						print(word+ " "+ word_text)
						if(word is word_text):
							#print("salto "+text+" "+line)
							return False
			#print("inserisco "+ text+" "+line)
			return True

"""
def fetchOrdinaryUsers():
	cursor = mydb.cursor()
	cursor.execute("SELECT * FROM downloaded_tweets")
	result = cursor.fetchall()

	with open('words.txt','r') as f:
		for row in result:
			for line in f:
				if(line in row.username):
					print(row.username+" "+line)
					continue
				else:
					sql = "INSERT IGNORE INTO ordinary_users(id, date, username, hashtags, retweets,text) VALUES (%s, %s, %s, %s, %s, %s)"
					val = (row.id, row.date, row.username, row.hashtags, row.retweets, tweet.text)
					mycursor.execute(sql, val)
					mydb.commit()
"""
	


mycursor = mydb.cursor()
tweetCriteria = got.manager.TweetCriteria().setQuerySearch('#coronavirus OR #WuhanCoronavirus OR #CoronavirusOutbreak OR #coronaviruschina OR #coronaviruswuhan OR #ChinaCoronaVirus OR #nCoV OR #coronaviruses OR ChinaWuHan OR #nCoV2020 OR #nCov2019').setLang('it').setSince("2020-01-30").setUntil("2020-02-01").setMaxTweets(20)
got.manager.TweetManager.getTweets(tweetCriteria,streamTweets)