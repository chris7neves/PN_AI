import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import openpyxl
import os


def generate_heat_map(excelFile, sheetNumber, trailNumber):
    row_start = 18
    column_start = 16 + (trailNumber - 1) * 3
    PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ExcelFilesPath = PATH + '\\' + 'Excel_Files'

    wb = openpyxl.load_workbook(ExcelFilesPath + '\\' + excelFile + '.xlsm')
    worksheet = wb[f'Sheet{sheetNumber}']
    x, y = get_coordinate_arrays(worksheet, row_start, column_start)

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=80)
    heatmap_extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    image_extent = [0, 1, 0, 1]

    #Image overlay
    face_image = img.imread(ExcelFilesPath + '\\FrontFront.jpg')

    plt.clf()
    plt.imshow(heatmap.T, extent=heatmap_extent, origin='lower')
    plt.imshow(face_image, extent=image_extent, origin='upper', cmap='gray', interpolation='none', alpha=0.3)
    plt.show()


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
    print("IMPORTANT NOTE: The image overlay and the respective coordinates are not accurate.")
    print("THIS IS JUST A POC")
    excelFile = input("Enter the Excel file name you would like to generate a heatmap for:\n")
    sheetNumber = int(input("Enter the sheet number you would like to generate a heatmap for (1-11):\n"))
    trailNumber = int(input("Enter the trial number you would like to generate a heatmap for (1-8):\n"))
    generate_heat_map(excelFile, sheetNumber, trailNumber)
