import pyodbc 
conn = pyodbc.connect("Driver = {SQL Server Native Client 11.0};"
						"Server = 26.105.85.122;"
						"Database = PNLab;"
						"username = adm;"
						"password = massimo123;"
						"Trusted_Connection = yes;")

cursor = conn.cursor()
cursor.execute('SELECT * FROM data_headers')

#data_details
#data_headers
#data_trials

for row in cursor:
    print(row)