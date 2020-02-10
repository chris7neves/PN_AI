import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import openpyxl
import os


TRIAL_TYPES = ["Practice Front-Front", "Front-Front Condition 1M", "Front-Front Condition 2F",
               "Practice FrontSide-SideFront 1", "FrontSide-SideFront 1F", "FrontSide-SideFront 2M",
               "SideFront-FrontSide 3M", "SideFront-FrontSide 4F", "Practice Inverted 1", "Inverted 1M",
               "Inverted 1F"]


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_scatter_plot(wb, excel_file_name, sheet_number):
    worksheet = wb[f'Sheet{sheet_number}']
    fig, axs = plt.subplots(3, 3)
    trial_number = 1

    for ax in axs.flatten():
        row_start = 18
        column_start = 16 + (trial_number - 1) * 3
        x, y = get_coordinate_arrays(worksheet, row_start, column_start)
        ax.plot(x, y, 'k.', markersize=5)
        trial_number += 1
        if trial_number == 9:
            break

    fig.suptitle(f"Test subject {excel_file_name} - {TRIAL_TYPES[sheet_number - 1]}")
    plt.show()


def generate_heat_map(wb, excel_file_name, sheet_number, trial_number):
    row_start = 18
    column_start = 16 + (trial_number - 1) * 3
    worksheet = wb[f'Sheet{sheet_number}']
    x, y = get_coordinate_arrays(worksheet, row_start, column_start)

    img, extent = myplot(x, y, 5)
    plt.imshow(img, extent=extent, origin='lower', cmap="Greys")
    plt.axis('off')
    plt.savefig(f"Images\\{excel_file_name}\\{TRIAL_TYPES[sheet_number-1]}_Trial{trial_number}.png")


def myplot(x, y, s, bins=300):
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
                x_coord = round(float(cell.value), 2)
                continue
            else:
                y_coord = round(float(cell.value), 2)

            if x_coord <= 0.05 or y_coord <= 0.05:
                break
            x_coordinate_array.append(x_coord)
            y_coordinate_array.append(y_coord)

    x_coordinate_array.append(min(x_coordinate_array)-0.1)
    x_coordinate_array.append(max(x_coordinate_array)+0.1)
    y_coordinate_array.append(min(y_coordinate_array)-0.1)
    y_coordinate_array.append(max(y_coordinate_array)+0.1)
    return np.array(x_coordinate_array), np.array(y_coordinate_array)


if __name__ == "__main__":

    scatter_plot = input("Would you like to generate a table of scatter plots for all trials of a specific type? Y/N\n")

    if scatter_plot.lower() == 'y':
        print("-----------SCATTER PLOT GENERATOR-----------")

        excel_file = input("Enter the Excel file name from the Data folder you would like to generate heat maps for:\n")
        sheet_number = int(input("Enter the sheet number you would like to generate scatter plots for (1-11):\n"))
        wb = openpyxl.load_workbook(f'Data\\{excel_file}.xlsm')
        generate_scatter_plot(wb, excel_file, sheet_number)

    heat_map = input("Would you like to generate all heatmap images for a specific test subject in the 'Images' folder?"
                     "NOTE: Doesn't work on mac and 88 images are generated!! Y/N\n")

    if heat_map.lower() == 'y':
        print("-----------HEAT MAP GENERATOR-----------")
        print("Outputs grayscale heatmaps for each sheet / trial number of the selected excel file.")

        excel_file = input("Enter the Excel file name from the Data folder you would like to generate heat maps for:\n")
        wb = openpyxl.load_workbook(f'Data\\{excel_file}.xlsm')
        ensure_dir(f"Images\\{excel_file}\\")
        for sheet_number in range(1, 12):
            for trial_number in range(1, 9):
                generate_heat_map(wb, excel_file, sheet_number, trial_number)
