#====================================================
#				  SQL SERVER ACCESS
#====================================================

#IMPORTS
import pyodbc


class SQLDatabase:
    def __init__(self, server, database, username, password):
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.drivers = pyodbc.drivers()

    def query(self, string):
        cnxn = pyodbc.connect('DRIVER='+self.drivers[0]+';SERVER='+self.server+';DATABASE='+self.database+';UID='+self.username+';PWD='+self.password)
        cursor = cnxn.cursor()
        # Sample select query
        cursor.execute(string)
        row_array = []
        while True:
            row = cursor.fetchone()
            if not row:
                break
            # print(row)
            row_array.append(row)
        return row_array

    def participants(self):
        cnxn = pyodbc.connect('DRIVER=' + self.drivers[0] + ';SERVER=' + self.server + ';DATABASE=' + self.database + ';UID=' + self.username + ';PWD=' + self.password)
        cursor = cnxn.cursor()
        cursor.execute("select * from data_headers")
        row_array = []
        while True:
            row = cursor.fetchone()
            if not row:
                break
            # print(row)
            row_array.append(row)
        return row_array

    def results(self, H_AUTO):
        cnxn = pyodbc.connect('DRIVER=' + self.drivers[0] + ';SERVER=' + self.server + ';DATABASE=' + self.database + ';UID=' + self.username + ';PWD=' + self.password)
        cursor = cnxn.cursor()
        cursor.execute(f"select * from data_trials where H_AUTO_KEY = {H_AUTO}")
        row_array = []
        while True:
            row = cursor.fetchone()
            if not row:
                break
            # print(row)
            row_array.append(row)
        return row_array

    def eye_tracker_data(self, T_AUTO):
        cnxn = pyodbc.connect('DRIVER=' + self.drivers[0] + ';SERVER=' + self.server + ';DATABASE=' + self.database + ';UID=' + self.username + ';PWD=' + self.password)
        cursor = cnxn.cursor()
        cursor.execute(f"select * from data_details where T_AUTO_KEY = {T_AUTO}")
        row_array = []
        while True:
            row = cursor.fetchone()
            if not row:
                break
            # print(row)
            row_array.append(row)
        return row_array

    def execute_statement(self, string):
        cnxn = pyodbc.connect('DRIVER=' + self.drivers[0] + ';SERVER=' + self.server + ';DATABASE=' + self.database + ';UID=' + self.username + ';PWD=' + self.password)
        cursor = cnxn.cursor()
        cursor.execute(string)
        cnxn.commit()

    def user_input_results(self, H_AUTO_KEY):
        string = f"""select
        case when T1_C = T1_I then 1 else 0 end as T1_A,
        case when T2_C = T2_I then 1 else 0 end as T2_A,
        case when T3_C = T3_I then 1 else 0 end as T3_A,
        case when T4_C = T4_I then 1 else 0 end as T4_A,
        case when T5_C = T5_I then 1 else 0 end as T5_A,
        case when T6_C = T6_I then 1 else 0 end as T6_A,
        case when T7_C = T7_I then 1 else 0 end as T7_A,
        case when T8_C = T8_I then 1 else 0 end as T8_A,
        T1_S,
        T2_S,
        T3_S,
        T4_S,
        T5_S,
        T6_S,
        T7_S,
        T8_S
        from DATA_TRIALS
        where h_auto_key = {H_AUTO_KEY}"""
        cnxn = pyodbc.connect('DRIVER='+self.drivers[0]+';SERVER='+self.server+';DATABASE='+self.database+';UID='+self.username+';PWD='+self.password)
        cursor = cnxn.cursor()
        # Sample select query
        cursor.execute(string)
        row_array = []
        while True:
            row = cursor.fetchone()
            if not row:
                break
            # print(row)
            row_array.append(row)
        return row_array

    def tracker_data_inputs(self, H_AUTO_KEY, T_AUTO_KEY, Trial_Number):
        string = f"""select 
        stamp_order,
        x_data,
        y_data,
        time_data
        from data_details
        
        where h_auto_key = {H_AUTO_KEY}
        and t_auto_key = {T_AUTO_KEY}
        and trial_num = {Trial_Number}
        
        ORDER BY STAMP_ORDER asc
        
        OFFSET (select min(stamp_order) from data_details where time_data <> -1 and time_data <> 0 and h_auto_key = 1000081 and t_auto_key = 1000680 and trial_num = 1) ROWS
        FETCH NEXT 1200 ROWS ONLY;"""
        cnxn = pyodbc.connect('DRIVER='+self.drivers[0]+';SERVER='+self.server+';DATABASE='+self.database+';UID='+self.username+';PWD='+self.password)
        cursor = cnxn.cursor()
        # Sample select query
        cursor.execute(string)
        row_array = []
        while True:
            row = cursor.fetchone()
            if not row:
                break
            # print(row)
            row_array.append(row)
        return row_array

    def coordinate_data(self, participant_name, trial_name, trial_number):

        conditions = "x_data > 0.05 and x_data < 0.95 and y_data > 0.05 and y_data < 0.95 and time_data <> -1 and time_data <> 0"
        connection = pyodbc.connect('DRIVER=' + self.drivers[0] + ';SERVER=' + self.server + ';DATABASE=' + self.database + ';UID=' + self.username + ';PWD=' + self.password)
        cursor = connection.cursor()

        H_AUTO_KEY = cursor.execute(f"select H_AUTO_KEY from DATA_HEADERS where T_NAME = '{participant_name}'").fetchone()[0]
        T_AUTO_KEY = cursor.execute(f"select T_AUTO_KEY from DATA_TRIALS where TRIAL_NAME = '{trial_name}' and H_AUTO_KEY = {H_AUTO_KEY}").fetchone()[0]

        x_y_data = cursor.execute(f"select x_data, y_data from DATA_DETAILS where T_AUTO_KEY = {T_AUTO_KEY} and H_AUTO_KEY = {H_AUTO_KEY} and TRIAL_NUM = {trial_number} and {conditions}").fetchall()
        x_data = [xy[0] for xy in x_y_data]
        y_data = [xy[1] for xy in x_y_data]

        return x_data, y_data








