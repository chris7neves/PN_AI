
import os
import cv2
import numpy as np
import torch
import torch.utils.data


#def to_file_np_array(np_array):


def count_files_recursive(directory):
    file_num = sum([len(files) for r, d, files in os.walk(directory)])
    return file_num


def diff_check(img1, img2, lab1=None, lab2=None):
    if lab1 is not None and lab2 is not None:
        if lab1 != lab2:
            print("The two images have different labels")
    if img1.shape != img2.shape:
        print("The two images have a different shape.")
    else:
        delta = cv2.subtract(img1, img2)
        if cv2.countNonZero(delta) != 0:
            print("The images are not the same.")

