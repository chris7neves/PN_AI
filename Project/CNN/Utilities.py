
import os


#def to_file_np_array(np_array):


def count_files_recursive(directory):
    file_num = sum([len(files) for r, d, files in os.walk(directory)])
    return file_num

