import re
import numpy as np
import matplotlib.pyplot as plt

from Utils.DNN_Model import L_layer_model, predict, regularization


def ReadFile(inputPath):
    file = open(inputPath, "r")
    read = file.read()
    file.close()
    read = list(re.split(' |\n', read))
    return read


def DataProcess(inputString, DIMX, DIMY):
    temp = np.zeros((DIMX, DIMY))
    K = 0
    for i in range(0, DIMX):
        for j in range(0, DIMY):
            t = np.double(inputString[K])
            K = K + 1
            temp[i, j] = t

    temp = np.reshape(temp, (DIMX, DIMY))
    return temp


if __name__ == "__main__":

    inputStr = ReadFile("./inputX.txt")
    X = DataProcess(inputStr, 81, 2)
    X = X.T
    #X = regularization(X)
    inputStr = ReadFile("./inputY.txt")
    # it should be (9, 81) but i am a lazy input writer so i flipped it to fit my data
    Y = DataProcess(inputStr, 81, 9)
    Y = Y.T

    layers_dims = (2, 9, 9, 9)
    lr = 0.005
    IterNum = 90000

    param, costs = L_layer_model(X, Y, layers_dims, lr, IterNum, 1000, True)

    # plot the cost
    plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(lr))
    plt.show()

    testy = predict(X, Y, param)
    print(testy)
