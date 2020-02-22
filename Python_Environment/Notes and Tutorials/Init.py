# File: Fooling around

# https://medium.com/@hussnainfareed/setup-an-environment-for-machine-learning-and-deep-learning-with-anaconda-in-windows-5d7134a3db10
# -------------------------
# Libraries
import sys
import numpy

import time

from sklearn import tree

import MassCLass




# -------------------------
def hello():
    name = str(input("Enter your name: "))
    if name:
        print("Hello " + str(name))
    else:
        print("Hello World")
    return


def main():
    print("Massimo Script Begin")

    print(sys.version)  # Python Version
    print(numpy.version.version)  # Numpy Version

    # String Manipulation
    print("----------------String Manipulation Example----------------")
    message = "Hello World \n"
    my = "Massimo's world"
    new_string = message + my
    print(new_string)

    print("-----------------------------------------------------------")
    print("\n \n \n")

    # Arrays
    print("--------------------------Arrays---------------------------")

    hello = [1, 2, 3, 4, 5]  # initialization
    # --slot--0,1,2,3,4
    print(hello)

    hello[1] = 78  # Changing a value in array
    print("Change 2nd slot to %d" % (hello[1]))

    x = len(hello)  # Getting length of array
    # FORMATTERS
    # %d numbers
    # %s strings
    # %r "print this no matter what"
    print("Length of Array is %r: while the 3rd entry is %r \n" % (x, hello[2]))
    print(hello)

    hello.append(25)
    print("We now appended a 6th slot to the array with value %d \n" % (hello[5]))
    print(hello)

    hello.pop(4)
    print("The 5th slot was now removed \n")
    print(hello)

    print("-----------------------------------------------------------")
    print("\n \n \n")

    # Loops
    print("--------------------------Loops----------------------------")

    print("For loop")
    for x in hello:
        print(x)

    for x in range(1, 10, 1):  # (start, stop, increment)
        print(x, end=" ", flush=True)

    print("")
    print("While loop")
    i = 1

    while (i < 5):
        print(i, end=" ", flush=True)
        i += 1

    print("")

    print("-----------------------------------------------------------")
    print("\n \n \n")

    # Loops
    print("--------------------------Tensorflow-----------------------")

    features = [[140, 1], [130, 1], [150, 0], [170, 0]]
    labels = ["Mass", "Mass", "Bianca", "Bianca"]
    clf = tree.DecisionTreeClassifier()

    start = time.perf_counter()
    clf = clf.fit(features, labels)
    predicted_value = clf.predict([[113, 1]])
    print(predicted_value)
    end = time.perf_counter()

    # time.sleep(5.5)//how to sleep

    # print(start)
    # print(end)
    print(end - start, " seconds")

    print("-----------------------------------------------------------")
    print("\n \n \n")

    print("")

    print("--------------------------File I/O-------------------------")

    # f = open("C:\Users\Massimo\Desktop\PythonTest.txt",mode = 'r',encoding = 'utf-8')
    # f.write("my first file\n")
    # f.write("This file\n\n")
    # f.write("contains three lines\n")
    # f.close()

    # Open a file
    # r: read
    # w: write
    # x: create and write new file
    # a: append to a new file
    # r+: use for reading and writing to same file

    fo = open("foo.txt", "w")
    print("Name of the file: ", fo.name)
    print("Closed or not : ", fo.closed)
    print("Opening mode : ", fo.mode)
    fo.write("Python is a great language.\nYeah its great!!\n");
    fo.close()

    # reading from file

    f = open("foo.txt", "r")
    # filestring = f.readline() #Read one line - Therefore, once you read a line with the readline operation it will pass to the next line. So if you were to call this operation again, it would return the next line in the file, as shown.
    filestring = f.readlines()  # Store lines into array
    # filestring = f.read() #Read entire txt
    f.close()

    print(filestring[1])
    print("")

    for n in range(0, len(filestring[1]), 1):  # cycling through each character in the string
        print(filestring[1][n])

    print("-----------------------------------------------------------")
    print("\n \n \n")

    print("--------------------------Classes--------------------------")

    classobj = MassClass.Employee("Massimo", "62000")

    classobj.displayCount()

    print("-----------------------------------------------------------")
    print("\n \n \n")
    return


main()

hello()

