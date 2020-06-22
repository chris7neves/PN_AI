import torch
from torch.utils.data import Dataset, DataLoader
from SQL import Data_Extract
import filesystem
import numpy as np

class ResponseDataset(Dataset):  # https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    def __init__(self):
        outputArray = [[0, 0]]
        # connect to the database
        SQLObject = Data_Extract.SQLDatabase(filesystem.os.environ["PNAI_SERVER"],
                                             filesystem.os.environ["PNAI_DATABASE"],
                                             filesystem.os.environ["PNAI_USERNAME"],
                                             filesystem.os.environ["PNAI_PASSWORD"])

        # Make initial query to find out all excel files in Database
        All_Excel_Files = SQLObject.participants()

        # Start Loop through the Excel Files
        i = 0
        for ExcelFile in All_Excel_Files:
            print(i)
            # We want to see if the subject is ASD or TYP
            if ExcelFile[4] == 'ASD':
                print("ASD")
                IsASD = [[0, 1]]
            else:
                IsASD = [[1, 0]]

            print(ExcelFile[1])
            if ExcelFile[1] == 'A008' or ExcelFile[1] == 'A011' or ExcelFile[1] == 'A026' or ExcelFile[1] == 'A028' or\
               ExcelFile[1] == 'A034' or ExcelFile[1] == 'A006':
                continue

            # Now we will query the data regarding the subject's inputs for each trial
            Results = SQLObject.user_input_results(ExcelFile[0])

            # Prep Data (divide times by 10)
            for Set in Results:
                for x in range(8, 16):
                    Set[x] = Set[x] / 10.0

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
            InputTensor = torch.tensor([array], dtype=torch.float)
            # OutputTensor = torch.tensor(IsASD, dtype=torch.float)
            if i == 0:
                self.Input = torch.tensor([array], dtype=torch.float)
                outputArray = np.array(IsASD)
                # self.Output = torch.tensor(IsASD, dtype=torch.float)
            else:
                self.Input = torch.cat((self.Input, InputTensor), 0)
                #self.Output = torch.cat((self.Output, OutputTensor), 0)
                outputArray = np.append(outputArray, IsASD, axis=0)
            i = i + 1
            # print(self.Input)
            # print(outputArray)
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
