import mysql.connector
import re
# import pymysql

def executeScriptsFromFile(filename,cursor):
    fd = open(filename, 'r')
    sqlFile = fd.read()
    fd.close()

    # all SQL commands (split on ';')
    sqlCommands = sqlFile.split(';\n')
    counter = 0
    for command in sqlCommands:
        command = re.sub(r"^--.*$","",command)
        newSqlCommand = command.replace("TYPE=MyISAM","ENGINE=MyISAM")
        print(newSqlCommand)
        if ("INSERT" in newSqlCommand or "CREATE" in newSqlCommand or "UPDATE" in newSqlCommand):
            cursor.execute(newSqlCommand)
            counter+=1  
     
db = mysql.connector.connect(user='root', password='123123', host='127.0.0.1')
#db = pymysql.connect(host="localhost",
#                     user="root",
#                     passwd="123123")
cursor=db.cursor()
cursor.execute("SET autocommit=0;")
cursor.execute("drop database if exists books")
cursor.execute("create database books")
cursor.execute("use books")

filenames = ["BX-Books.sql","BX-Users.sql","BX-Book-Ratings.sql"]
       
for filename in filenames:
    executeScriptsFromFile(filename,cursor)

cursor.close()

