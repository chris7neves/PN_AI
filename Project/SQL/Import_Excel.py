import tkinter
import xlrd
from tkinter import Tk
from SQL import Data_Extract

from tkinter.filedialog import askopenfilename

Tk().withdraw()
filename = askopenfilename()

wb = xlrd.open_workbook(filename)

sheet = wb.sheet_by_index(0)

T_NAME = sheet.cell_value(3, 2)
T_AGE = sheet.cell_value(4, 2)
T_GENDER = sheet.cell_value(5, 2)
T_NOTES = sheet.cell_value(6, 2)

sSQL = f"insert into data_headers(T_NAME,T_AGE,T_GENDER,T_NOTES) values('{T_NAME}',{T_AGE},'{T_GENDER}','{T_NOTES}')"

P1 = Data_Extract.SQL_Database('26.105.85.122', 'PNLab', 'adm', 'massimo123')
P1.execute_statement(sSQL)

array = P1.Query('select max(h_auto_key) from data_headers')
h_auto_key = array[0][0]

for s in wb.sheets():
    if s.name != "Summary":
        #s = wb.sheet_by_index(shidx)
        TRIAL_NAME = s.cell_value(8, 1)
        #print(s.name)
        if s.cell_type(13, 3) != xlrd.empty_cell:
            T1_C = s.cell_value(12, 3)
            T2_C = s.cell_value(12, 4)
            T3_C = s.cell_value(12, 5)
            T4_C = s.cell_value(12, 6)
            T5_C = s.cell_value(12, 7)
            T6_C = s.cell_value(12, 8)
            T7_C = s.cell_value(12, 9)
            T8_C = s.cell_value(12, 10)
            T1_I = s.cell_value(13, 3)
            T2_I = s.cell_value(13, 4)
            T3_I = s.cell_value(13, 5)
            T4_I = s.cell_value(13, 6)
            T5_I = s.cell_value(13, 7)
            T6_I = s.cell_value(13, 8)
            T7_I = s.cell_value(13, 9)
            T8_I = s.cell_value(13, 10)
            T1_S = s.cell_value(14, 3)
            T2_S = s.cell_value(14, 4)
            T3_S = s.cell_value(14, 5)
            T4_S = s.cell_value(14, 6)
            T5_S = s.cell_value(14, 7)
            T6_S = s.cell_value(14, 8)
            T7_S = s.cell_value(14, 9)
            T8_S = s.cell_value(14, 10)

            sSQL = f"insert into data_trials(TRIAL_NAME," \
                   f"T1_C,T2_C,T3_C,T4_C,T5_C,T6_C,T7_C,T8_C," \
                   f"T1_I,T2_I,T3_I,T4_I,T5_I,T6_I,T7_I,T8_I," \
                   f"T1_S,T2_S,T3_S,T4_S,T5_S,T6_S,T7_S,T8_S,H_AUTO_KEY) " \
                   f"values('{TRIAL_NAME}'," \
                   f"'{T1_C}','{T2_C}','{T3_C}','{T4_C}','{T5_C}','{T6_C}','{T7_C}','{T8_C}'," \
                   f"'{T1_I}','{T2_I}','{T3_I}','{T4_I}','{T5_I}','{T6_I}','{T7_I}','{T8_I}'," \
                   f"{T1_S},{T2_S},{T3_S},{T4_S},{T5_S},{T6_S},{T7_S},{T8_S},{h_auto_key})"

            P1.execute_statement(sSQL)
            array = P1.Query('select max(t_auto_key) from data_trials')
            t_auto_key = array[0][0]

            for j in range(1, 8):
                for k in range(17, 3016):
                    X_DATA = s.cell_value(k, 12 + j*3)
                    Y_DATA = s.cell_value(k, 13 + j*3)
                    TIME_DATA = s.cell_value(k, 14 + j*3)
                    STAMP_ORDER = k - 16
                    sSQL = f"insert into data_details" \
                           f"(TRIAL_NUM,STAMP_ORDER,X_DATA,Y_DATA,TIME_DATA,H_AUTO_KEY,T_AUTO_KEY) values(" \
                           f"{j},{STAMP_ORDER},{X_DATA},{Y_DATA},{TIME_DATA},{h_auto_key},{t_auto_key})"
                    P1.execute_statement(sSQL)
                    #print(sSQL)