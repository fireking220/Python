import math
import random
import numpy as np
import matplotlib.pyplot as plot
from pointK import point


def main():
    random.seed()
    # declare variables
    k = input('How Many Clusters?: ')
    timesToRun = input('How many times to repeat?: ')
    points = []
    maxX = 0.0
    maxY = 0.0
    minX = 0.0
    minY = 0.0

    # read in points
    with open("GMM_dataset_Spring_2019.txt") as file:
        for line in file:
            temp = line.split()
            temp = list(map(float, temp))
            if maxX <= temp[0]:
                maxX = temp[0]
            if minX > temp[0]:
                minX = temp[0]
            if maxY <= temp[1]:
                maxY = temp[1]
            if minY > temp[1]:
                minY = temp[1]
            points.append(point(temp, None))

    # define bounds for random means
    maxX = math.ceil(maxX)
    maxY = math.ceil(maxY)
    minX = math.floor(minX)
    minY = math.floor(minY)

    points = np.asarray(points)

    leastSqr = np.zeros(timesToRun)
    for o in range(timesToRun):
        means = []
        # build random means
        for i in range(k):
            temp = []
            temp.append(random.uniform(minX, maxX))
            temp.append(random.uniform(minY, maxY))
            meanPoint = point(temp, i)
            means.append(meanPoint)
        means = np.asarray(means)
        sumSqrErr = np.zeros(k)

        for i in range(10):
            # ASSIGNMENT STEP
            # calculate kClass for each point by computing the L2 norm
            # and taking the kClass from the smallest L2Norm
            for myPoint in points:
                kClass = 0
                minDistance = 0
                for meanPoint in means:
                    L2Norm = np.linalg.norm(myPoint.getCord() - meanPoint.getCord())
                    if minDistance == 0 or minDistance > L2Norm:
                        minDistance = L2Norm
                        kClass = meanPoint.kClass
                myPoint.kClass = kClass

            # UPDATE STEP
            # Update the means to be the center of their given class
            for i in range(k):
                num = 0
                newMean = np.array([0, 0])
                for myPoint in points:
                    if myPoint.kClass == i:
                        newMean = newMean + myPoint.getCord()
                        num += 1
                newMean = np.true_divide(newMean, num)
                newMean = newMean.tolist()
                means[i].setCord(newMean)

        for myPoint in points:
            for i in range(k):
                if myPoint.kClass == i:
                    sumSqrErr[i] += np.linalg.norm(myPoint.getCord() - means[i].getCord())
        x = []
        y = []
        for i in range(k):
            curListX = []
            curListY = []
            for myPoint in points:
                if myPoint.kClass == i:
                    curListX.append(myPoint.x)
                    curListY.append(myPoint.y)
            x.append(curListX)
            y.append(curListY)

        plot.figure(num='Plot ' + str(o))
        for i in range(k):
            plot.scatter(x[i], y[i], s=5)
        meanX = []
        meanY = []
        for i in range(k):
            meanX.append(means[i].x)
            meanY.append(means[i].y)
        plot.scatter(meanX, meanY, s=20, c=['black'])
        leastSqr[o] = np.sum(sumSqrErr)
        plot.title('Clusters: ' + str(k) + ' Times to Repeat ' + str(timesToRun))
        plot.show()
    print("Least Squared error for " + str(k) + " clusters: " + str(np.min(leastSqr)))


if __name__ == "__main__":
    main()
