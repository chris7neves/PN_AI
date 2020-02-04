import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import openpyxl


TRIAL_TYPES = ["Practice Front-Front", "Front-Front Condition 1M", "Front-Front Condition 2F",
               "Practice FrontSide-SideFront 1", "FrontSide-SideFront 1F", "FrontSide-SideFront 2M",
               "SideFront-FrontSide 3M", "SideFront-FrontSide 4F", "Practice Inverted 1", "Inverted 1M",
               "Inverted 1F"]


def generate_heat_map(excelFile, sheetNumber):

    wb = openpyxl.load_workbook(f'Data\\{excelFile}.xlsm')
    worksheet = wb[f'Sheet{sheetNumber}']
    fig, axs = plt.subplots(3, 3)
    trialNumber = 1

    for ax, in zip(axs.flatten()):
        row_start = 18
        column_start = 16 + (trialNumber - 1) * 3
        x, y = get_coordinate_arrays(worksheet, row_start, column_start)
        ax.plot(x, y, 'k.', markersize=5)
        trialNumber += 1
        if trialNumber == 9:
            break
    fig.suptitle(f"Test subject {excelFile} - {TRIAL_TYPES[sheetNumber-1]}")
    plt.show()


def myplot(x, y, s, bins=500):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def get_coordinate_arrays(worksheet, rowStart, columnStart):
    x_coordinate_array = []
    y_coordinate_array = []

    for row_cells in worksheet.iter_rows(min_row=rowStart, min_col=columnStart, max_col=columnStart + 1):
        for index, cell in enumerate(row_cells):
            try:
                float(cell.value)
            except ValueError:
                break
            if index == 0:
                x_coord = float(cell.value)
                continue
            else:
                y_coord = float(cell.value)

            if x_coord <= 0.05 or y_coord <= 0.05:
                break
            x_coordinate_array.append(x_coord)
            y_coordinate_array.append(y_coord)

    return np.array(x_coordinate_array), np.array(y_coordinate_array)


if __name__ == "__main__":
    print("-----------HEAT MAP GENERATOR POC-----------")
    excelFile = input("Enter the Excel file name from the Data folder you would like to generate scatter plots for:\n")
    sheetNumber = int(input("Enter the sheet number you would like to generate scatter plots for (1-11):\n"))
    generate_heat_map(excelFile, sheetNumber)
