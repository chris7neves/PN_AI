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

    def Query(self,string):
        cnxn = pyodbc.connect('DRIVER='+self.drivers[0]+';SERVER='+self.server+';DATABASE='+self.database+';UID='+self.username+';PWD='+self.password)
        cursor = cnxn.cursor()
        # Sample select query
        cursor.execute(string)
        rowArray = []
        while True:
            row = cursor.fetchone()
            if not row:
                break
            # print(row)
            rowArray.append(row)
        return rowArray

    def Participants(self):
        cnxn = pyodbc.connect('DRIVER=' + self.drivers[0] + ';SERVER=' + self.server + ';DATABASE=' + self.database + ';UID=' + self.username + ';PWD=' + self.password)
        cursor = cnxn.cursor()
        cursor.execute("select * from data_headers")
        rowArray = []
        while True:
            row = cursor.fetchone()
            if not row:
                break
            # print(row)
            rowArray.append(row)
        return rowArray

    def Results(self, H_AUTO):
        cnxn = pyodbc.connect('DRIVER=' + self.drivers[0] + ';SERVER=' + self.server + ';DATABASE=' + self.database + ';UID=' + self.username + ';PWD=' + self.password)
        cursor = cnxn.cursor()
        cursor.execute(f"select * from data_trials where H_AUTO_KEY = {H_AUTO}")
        rowArray = []
        while True:
            row = cursor.fetchone()
            if not row:
                break
            # print(row)
            rowArray.append(row)
        return rowArray
