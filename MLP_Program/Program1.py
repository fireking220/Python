# Scott Patterson
# 4/29/2019
# Programming HW #1

# This script constructs a Mulit Layer perceptron with one hidden layer and trains
# the network to recognize handwritten digits from the MNIST database. The script
# will train the network and then test the newly updated network against some training
# data. During this, it will compute accuracy values and construct a confusion matrix
# as it trains the network

# This script will prompt the user for information about the network, including the
# amount of hidden units, momentum value, number of training examples/test examples
# to test/train against, and the names of the csv files that will contain the
# accuracy and confusion matrix

import numpy as np
import math


def main():
    print("Enter number of hidden units: ")
    numHidden = int(input())
    print("Enter momentum value: ")
    momentum = float(input())
    print("Enter number of training inputs: ")
    numTrain = int(input())
    print("Enter number of test inputs: ")
    numTest = int(input())
    print("Enter what the accuracy training file should be called: ")
    trainFile = str(input())
    print("Enter what the accuracy test file should be called: ")
    testFile = str(input())
    print("Enter what the training confusion matrix should be called: ")
    matrixTrainFile = str(input())
    print("Enter what the testing confusion matrix should be called: ")
    matrixTestFile = str(input())

    MLP(numHidden, momentum, numTest, numTrain, trainFile, testFile, matrixTrainFile, matrixTestFile)


# Constructs, trains, and tests a 2 layer neural network
# numHidden = number of hidden units
# momentum = momentum for this MLP
# testNum = amount of inputs to test
# trainNum = amount of inputs to train
# accTrainFile = accuracy train file name
# accTestFile = accuracy test file name
# matrixTrainFile = matrix train file name
# matrixTestFile = matrix test file name
def MLP(numHidden, momentum, testNum, trainNum, accTrainFile, accTestFile, matrixTrainFile, matrixTestFile):
    # learning rate
    n = 0.1
    # inputs, labels are stored at the 0th position, hence 786 instead of 785
    inputsTrain = np.zeros((trainNum, 786))
    inputsTest = np.zeros((testNum, 786))
    # Random weights (-0.5-0.5)
    weights_ItoH = np.random.random((785, numHidden)) - 0.5
    weights_HtoO = np.random.random((numHidden + 1, 10)) - 0.5
    # previous weight update for hidden to output and input to hidden
    weightsPrev_HtoO = np.zeros((numHidden + 1, 10))
    weightsPrev_ItoH = np.zeros((785, numHidden))
    # number of correct hits
    hitTrain = 0
    hitTest = 0
    # accuracy
    accuracyTrain = 0.0
    accuracyTest = 0.0
    # confusion matrix
    conMatrixTrain = np.zeros((10, 10))
    conMatrixTest = np.zeros((10, 10))
    # counters
    i = 0
    z = 0

    # load training inputs
    with open("mnist_train.csv", 'r') as dataSet:
        for j in range(trainNum):
            line = dataSet.readline()
            # build list off of string line
            line = line.split(',')
            line[-1] = line[-1].rstrip()
            temp = int(line.pop(0))
            # map x/255 to every value in list line
            line = list(map(float, line))
            line = list(map(small, line))
            # prepend input layer bias
            line.insert(0, 1.0)
            # put back label
            line.insert(0, temp)
            # replace row in numpy input matrix
            inputsTrain[j] = line

    # load test inputs
    with open("mnist_test.csv", 'r') as dataSet:
        for j in range(testNum):
            line = dataSet.readline()
            line = line.split(',')
            line[-1] = line[-1].rstrip()
            temp = int(line.pop(0))
            line = list(map(float, line))
            line = list(map(small, line))
            line.insert(0, 1.0)
            line.insert(0, temp)
            inputsTest[j] = line

    for epoch in range(50):
        print(epoch)
        # shuffle inputs
        np.random.shuffle(inputsTrain)
        batch = 0
        sumO = np.zeros(10)
        sumH = np.zeros(numHidden)
        # integrate over training set
        for line in inputsTrain:
            # -------------------
            # forward propagation
            # -------------------
            # get the target label
            target = int(line[0])
            targets = np.empty(10)
            targets.fill(0.1)
            targets[target] = 0.9
            # build hidden layer
            hidden = buildLayer(line[1:], weights_ItoH)
            # prepend hidden layer bias
            hidden.insert(0, 1.0)
            # convert to numpy array
            hidden = np.array(hidden)
            # build output
            output = buildLayer(hidden, weights_HtoO)
            conMatrixTrain[output.index(max(output))][target] += 1
            if output.index(max(output)) == target:
                hitTrain += 1
            # -------------------
            # Back propagation
            # -------------------
            # calculate errors
            errorO = calculateOutputError(output, targets)
            sumO += errorO
            errorH = calculateHiddenError(hidden[1:], weights_HtoO[1:], errorO, numHidden)
            sumH += errorH
            # update our weights after batch passes, prevents updating al the time
            if batch >= 10:
                # -------------------
                # update weights
                # -------------------
                sumO = sumO * (1 / 10)
                sumH = sumH * (1 / 10)
                weightsChange = n * np.outer(hidden, sumO) + momentum * weightsPrev_HtoO
                weightsPrev_HtoO = weightsChange
                weights_HtoO += weightsChange

                weightsChange = n * np.outer(line[1:], sumH) + momentum * weightsPrev_ItoH
                weightsPrev_ItoH = weightsChange
                weights_ItoH += weightsChange
                i += 1
                batch = 0
                sumO = sumO * 0
                sumH = sumH * 0
            else:
                batch += 1
        # iterate over test set
        for line in inputsTest:
            # -------------------
            # forward propagation
            # -------------------
            target = int(line[0])
            targets = np.empty(10)
            targets.fill(0.1)
            targets[target] = 0.9
            # build hidden layer
            hidden = buildLayer(line[1:], weights_ItoH)
            # prepend hidden layer bias
            hidden.insert(0, 1.0)
            # convert to nump array
            hidden = np.array(hidden)
            # build output
            output = buildLayer(hidden, weights_HtoO)
            conMatrixTest[output.index(max(output))][target] += 1
            if output.index(max(output)) == target:
                hitTest += 1
            z += 1
        # -------------------
        # compute accuracy
        # -------------------
        accuracyTrain = hitTrain / trainNum * 100
        accuracyTest = hitTest / testNum * 100
        with open(accTrainFile, 'a') as data:
            data.write(str(epoch) + ", " + str(accuracyTrain) + "\n")
        with open(accTestFile, 'a') as data:
            data.write(str(epoch) + ", " + str(accuracyTest) + "\n")
        # reset counters
        i = 0
        z = 0
        hitTrain = 0
        hitTest = 0
    # write to confusion matrix
    np.set_printoptions(suppress=True)
    with open(matrixTrainFile, 'w') as matrix:
        for line in conMatrixTrain:
            line = line.tolist()
            line = list(map(int, line))
            line = str(line)
            line = line[:-1]
            line = line[1:]
            matrix.write(line + "\n")
    with open(matrixTestFile, 'w') as matrix:
        for line in conMatrixTest:
            line = line.tolist()
            line = list(map(int, line))
            line = str(line)
            line = line[:-1]
            line = line[1:]
            matrix.write(line + "\n")


# Calculate hidden error
def calculateHiddenError(layer, weights, errorO, hNum):
    ones = np.ones(hNum)
    twos = ones - layer
    threes = weights @ errorO
    return layer * twos * threes


# Calculate output error
def calculateOutputError(layer, targets):
    ones = np.ones(10)
    return layer * (ones - layer) * (targets - layer)


# Build layers
def buildLayer(data, weights):
    layer = data @ weights
    layer = layer.tolist()
    layer = list(map(sigmoid, layer))
    return layer


# Apply transformation to given x
def small(x):
    return x / 255


# Apply sigmoid to given x
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


if __name__ == "__main__":
    main()
