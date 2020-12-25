import torch
from torch.utils.data import Dataset, DataLoader
from SQL import Data_Extract
import filesystem
import numpy as np

class ResponseDataset(Dataset):  # https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    def __init__(self):
        outputArray = [[0, 0]]
        # connect to the database
        SQLObject = Data_Extract.SQL_Database(filesystem.os.environ["PNAI_SERVER"],
                                       filesystem.os.environ["PNAI_DATABASE"],
                                       filesystem.os.environ["PNAI_USERNAME"],
                                       filesystem.os.environ["PNAI_PASSWORD"])

        # Make initial query to find out all excel files in Database
        All_Excel_Files = SQLObject.Participants()


        # Start Loop through the Excel Files
        index = 0
        for ExcelFile in All_Excel_Files:
            ResultsArray = []
            print(index)
            # We want to see if the subject is ASD or TYP
            if ExcelFile[4] == 'ASD':
                print("ASD")
                IsASD = [[0, 1]]
            else:
                IsASD = [[1, 0]]

            # We want to grab the Age
            Age = ExcelFile[2]

            # We want to grab the Sex
            if ExcelFile[3] == 'M':
                Sex = 1
            else:
                Sex = 0

            print(ExcelFile[1])
            if ExcelFile[1] == 'A008' or ExcelFile[1] == 'A011' or ExcelFile[1] == 'A026' or ExcelFile[1] == 'A028' or \
                    ExcelFile[1] == 'A034' or ExcelFile[1] == 'A006':
                continue

            if Age <= 10:
                continue

            # Now we will query the data regarding the subject's inputs for each trial
            Results = SQLObject.UserInputResults(ExcelFile[0])

            # Prep Data (divide times by 10)

            ResultsArray = np.insert(ResultsArray, len(ResultsArray), Age)
            ResultsArray = np.insert(ResultsArray, len(ResultsArray), Sex)

            for Set in Results:
                TotalCorrect = 0
                TotalTime = 0
                for i in range(0, 8):
                    TotalCorrect = TotalCorrect + Set[i]
                    TotalTime = TotalTime + Set[8 + i]
                CorrectAverage = TotalCorrect / 8
                TimeAverage = TotalTime / 80  # 8 * 10
                for j in range(8, 16):
                    Set[j] = Set[j] / 10.0
                #ResultsArray = np.concatenate((ResultsArray, CorrectAverage, TimeAverage), axis=0)
                ResultsArray = np.insert(ResultsArray, len(ResultsArray), CorrectAverage)
                ResultsArray = np.insert(ResultsArray, len(ResultsArray), TimeAverage)

            # Concatenate them all together
            array = np.concatenate((Results[0],
                                   Results[1],
                                   Results[2],
                                   Results[3],
                                   Results[4],
                                   Results[5],
                                   Results[6],
                                   Results[7],
                                   Results[8],
                                   Results[9],
                                   Results[10],))

            #Beging building our tensor
            InputTensor = torch.tensor([ResultsArray], dtype=torch.float)
            # OutputTensor = torch.tensor(IsASD, dtype=torch.float)
            if index == 0:
                self.Input = torch.tensor([ResultsArray], dtype=torch.float)
                outputArray = np.array(IsASD)
                # self.Output = torch.tensor(IsASD, dtype=torch.float)
            else:
                self.Input = torch.cat((self.Input, InputTensor), 0)
                #self.Output = torch.cat((self.Output, OutputTensor), 0)
                outputArray = np.append(outputArray, IsASD, axis=0)
            index = index + 1

        self.Output = torch.tensor(outputArray, dtype=torch.float)
        self.InputDimensions = self.Input.size()
        self.OutputDimensions = self.Output.size()
        print("Data Acquisition Complete")

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        input_data = self.Input[index]
        output_data = self.Output[index]
        return input_data, output_data

    def GetScatterData(self):
        outputArray = [[0, 0, 0]]
        # connect to the database
        SQLObject = Data_Extract.SQL_Database(filesystem.os.environ["PNAI_SERVER"],
                                       filesystem.os.environ["PNAI_DATABASE"],
                                       filesystem.os.environ["PNAI_USERNAME"],
                                       filesystem.os.environ["PNAI_PASSWORD"])

        # Make initial query to find out all excel files in Database
        All_Excel_Files = SQLObject.Participants()

        # Start Loop through the Excel Files
        i = 0
        totalASD = 0
        totalTYP = 0
        for ExcelFile in All_Excel_Files:
            print(i)
            # We want to see if the subject is ASD or TYP
            if ExcelFile[4] == 'ASD':
                print("ASD")
                IsASD = 1
                totalASD = totalASD + 1
            else:
                IsASD = 0
                totalTYP = totalTYP + 1

            age = ExcelFile[2]

            print(ExcelFile[1])
            if ExcelFile[1] == 'A008' or ExcelFile[1] == 'A011' or ExcelFile[1] == 'A026' or ExcelFile[1] == 'A028' or\
               ExcelFile[1] == 'A034' or ExcelFile[1] == 'A006':
                continue

            # Now we will query the data regarding the subject's inputs for each trial
            Results = SQLObject.UserInputResults(ExcelFile[0])

            # Find number of correct answers
            correctCounter = 0
            # Prep Data (divide times by 10)
            for Set in Results:
                for x in range(1, 8):
                    if Set[x] == 1:
                        correctCounter = correctCounter + 1

            if i == 0:
                outputArray = np.array([[IsASD, correctCounter, age]])
            else:
                outputArray = np.append(outputArray, [[IsASD, correctCounter, age]], axis=0)
            i = i + 1

            print(f'Total ASD: {totalASD}')
            print(f'Total TYP: {totalTYP}')
        return outputArray