from SQL import Data_Extract

P1 = Data_Extract.SQL_Database('26.105.85.122','PNLab','adm','massimo123')

array = P1.Query("select * from data_headers")      # Custom Query Call
print(array)                                        # seeing all participants
print(array[0])                                     # seeing first participant
print(array[0][1])                                  # seeing name of first participant


#So that we do not have to memorize the queries we have predefined
#Functions to perform these queries for us as follows:
array1 = P1.Participants()                          # Can use participants to return above query call
print(array1[2])                                    # Participant
print(array1[2][1])                                 # Name if Participant


print("Example 3")
print(array[2][0])
array2 = P1.Results(array[2][0])
print(array2[0])