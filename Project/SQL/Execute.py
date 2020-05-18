from SQL import Data_Extract
import IO

P1 = Data_Extract.SQL_Database(IO.os.environ["PNAI_SERVER"],IO.os.environ["PNAI_DATABASE"],IO.os.environ["PNAI_USERNAME"],IO.os.environ["PNAI_PASSWORD"])

print("================================Example 1==============================================")
array = P1.Query("select * from data_headers")      # Custom Query Call
print(array)                                        # seeing all participants
print(array[0])                                     # seeing first participant
print(array[0][1])                                  # seeing name of first participant


#So that we do not have to memorize the queries we have predefined
#Functions to perform these queries for us as follows:
#output array = [H_AUTO_KEY , NAME_OF_EXCEL_FILE , AGE, GENDER, ASD/NON_ASD]
print("================================Example 2==============================================")
array1 = P1.Participants()                          # Can use participants to return above query call
print(array1[2])                                    # Participant
print(array1[2][1])                                 # Name if Participant


print("================================Example 3==============================================")
print(array[2][0])
array2 = P1.Results(array[2][0])
print(array2[0])

print("================================Example 4==============================================")
All_Excel_Files=P1.Participants()
print(All_Excel_Files)
Trials_From_One_Excel_File = P1.Results(All_Excel_Files[0][0])  # Feed H_AUTO_KEY
print(Trials_From_One_Excel_File)
Eye_Tracker_Data_For_One_Sheet = P1.EyeTrackerData(Trials_From_One_Excel_File[0][0]) # Feed T_AUTO_KEY
print(Eye_Tracker_Data_For_One_Sheet[0]) #Eye Tracker [D_auto, Trial Number, Row Number, X, Y, Time, H_auto, T_Auto ]