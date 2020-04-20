import numpy as np


def ReadFile(inputPath, DIMX, DIMY):
    file = open(inputPath, "r")
    read = file.read()
    file.close()
    temp = np.empty([DIMX, DIMY])
    for i in read:
        t = np.double(i)
        np.append(temp, t)
    return temp


if __name__ == "__main__":
    X = ReadFile("./inputX.txt")
    print(np.size(X))
    print(type(X))
    #X = np.reshape(X, (1, 81))
    print(X)
