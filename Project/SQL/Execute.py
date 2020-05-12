from SQL import Data_Extract

P1 = Data_Extract.SQL_Database('26.105.85.122','PNLab','adm','massimo123')

P1.Connect("select * from data_headers")