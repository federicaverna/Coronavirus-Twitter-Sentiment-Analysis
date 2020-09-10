import mysql.connector


# ----- PARAMETRI DI CONFIGURAZIONE -----
table="train_inc"

# ---------------------------------------


mydb = mysql.connector.connect(
	host="localhost",
	user="root",
	passwd="fede95",
	database="tweet_db"
)

mycursor = mydb.cursor()

select = "SELECT * FROM "+table+" WHERE tag is null"
insert = "UPDATE "+table+ " SET tag= %s WHERE id=%s"

mycursor.execute(select)
records = mycursor.fetchall()
print("Total number of rows: ", mycursor.rowcount)
for row in records:
    #print("Username  = ", row[1])
    #print("Date = ", row[2]) 
    print("Text  = ", row[4])
    #print("Tag  = ", row[5], "\n")
    print(type(row[0]))
    key = input()
    if(key=='n'):
        tag=0
        val = (tag, row[0])
        mycursor.execute(insert, val)
        mydb.commit()
    elif(key=='p'):
        tag=2
        val = (tag, row[0])
        mycursor.execute(insert, val)
        mydb.commit()
    elif(key=='t'):
        tag=1
        val = (tag, row[0])
        mycursor.execute(insert, val)
        mydb.commit()
    elif(key=='c'):
        tag='null'
        val = (tag, row[0])
        mycursor.execute(insert, val)
        mydb.commit()
    
