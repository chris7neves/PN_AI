#====================================================
#				  SQL SERVER ACCESS
#====================================================

#IMPORTS
import pyodbc

class SQL_Database:
    def __init__(self,server,database,username,password):
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.drivers  = pyodbc.drivers()

    def Connect(self,string):
        cnxn = pyodbc.connect('DRIVER='+self.drivers[0]+';SERVER='+self.server+';DATABASE='+self.database+';UID='+self.username+';PWD='+self.password)
        cursor = cnxn.cursor()
        # Sample select query
        cursor.execute(string)
        # rows = cursor.fetchall()
        while True:
            row = cursor.fetchone()
            if not row:
                break
            print(row)