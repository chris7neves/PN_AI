from SQL import Data_Extract
import filesystem

P1 = Data_Extract.SQL_Database(filesystem.os.environ["PNAI_SERVER"],
                               filesystem.os.environ["PNAI_DATABASE"],
                               filesystem.os.environ["PNAI_USERNAME"],
                               filesystem.os.environ["PNAI_PASSWORD"])

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
All_Excel_Files = P1.Participants()  # We call Participants which returns array as follows
print(All_Excel_Files)             # [ H_AUTO_KEY, NAME_OF_EXCEL_FILE, AGE, SEX, ASD/TYPICAL]

Trials_From_One_Excel_File = P1.Results(All_Excel_Files[0][0])  # We Call Results function using H_AUTO_KEY from above
print(Trials_From_One_Excel_File)  # [ T_AUTO_KEY, TRIAL_NAME, RESULT_TRIAL_1, ..etc.. ]

Eye_Tracker_Data_For_One_Sheet = P1.EyeTrackerData(Trials_From_One_Excel_File[0][0])  # Call EyeTrackerData with T_AUTO
print(Eye_Tracker_Data_For_One_Sheet[0])  # Eye Tracker [D_auto, Trial Number, Row Number, X, Y, Time, H_auto, T_Auto ]


print("================================Example 4==============================================")
All_Excel_Files = P1.Participants()  # We call Participants which returns array as follows
print(All_Excel_Files)             # [ H_AUTO_KEY, NAME_OF_EXCEL_FILE, AGE, SEX, ASD/TYPICAL]
print(len(All_Excel_Files))

response = P1.Query("""select
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
where h_auto_key = 1000081""")

print(response)
