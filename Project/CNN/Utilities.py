
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
    """
    Function that performs the element-wise comparison of 2 images in np array format and
    checks for differences.

    :param img1: First image to be compared. Should be in np array format.
    :param img2: Second image to be compared. Should be in np array format.
    :param lab1: Label of the first image. Integer type.
    :param lab2: Label of second image. Integer type.
    :return: Returns False if images are the same and True if there is a difference between the two images.
    """
    if lab1 is not None and lab2 is not None:
        if lab1 != lab2:
            print("The two images have different labels")
            return True
    if img1.shape != img2.shape:
        print("The two images have a different shape.")
        return True
    else:
        delta = cv2.subtract(img1, img2)
        if cv2.countNonZero(delta) != 0:
            print("The images are not the same.")
            return True
        else:
            return False


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()