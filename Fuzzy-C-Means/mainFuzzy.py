from pointFuzzy import point
import random
import math
import numpy as np
import matplotlib.pyplot as plot


def main():
    # declare variables
    random.seed()
    k = 4 # input('How many clusters?: ')
    timesToRun = 4 # input('How many times to repeat?: ')
    m = 2
    points = []
    maxX = 0.0
    maxY = 0.0
    minX = 0.0
    minY = 0.0

    # read in points
    with open("GMM_dataset_Spring_2019.txt") as file:
        for line in file:
            grades = []
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
            # create random membership grades for point
            for _ in range(k):
                grades.append(random.uniform(0, 1))
            points.append(point(temp, grades))

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
            grades = []
            temp.append(random.uniform(minX, maxX))
            temp.append(random.uniform(minY, maxY))
            for _ in range(k):
                grades.append(0)
            meanPoint = point(temp, grades)
            means.append(meanPoint)
        means = np.asarray(means)
        sumSqrErr = np.zeros(k)

        for _ in range(5):
            # compute centroids
            for i in range(k):
                sumTop = 0
                sumBottom = 0
                for myPoint in points:
                    w = myPoint.grades[i]
                    sumBottom += np.power(w, m)
                    sumTop += np.power(w, m) * myPoint.getCord()
                newMean = np.true_divide(sumTop, sumBottom)
                newMean = newMean.tolist()
                means[i].setCord(newMean)

            # compute grades
            for myPoint in points:
                for i in range(k):
                    sumBottom = 0
                    for x in range(k):
                        a = np.linalg.norm(myPoint.getCord() - means[i].getCord())
                        b = np.linalg.norm(myPoint.getCord() - means[x].getCord())
                        sumBottom = sumBottom + np.power(a / b, 2 * (m - 1))
                    myPoint.setGrade(1 / sumBottom, i)
        for i in range(k):
            for myPoint in points:
                sumSqrErr[i] += np.linalg.norm(myPoint.getCord() - means[i].getCord()) * myPoint.getGrade(i)
        leastSqr[o] = np.sum(sumSqrErr)

        plot.figure(num='Plot ' + str(o))
        for myPoint in points:
            print(myPoint.grades)
            maxGrade = max(myPoint.grades)
            if maxGrade >= 0.5:
                plot.plot(myPoint.x, myPoint.y, marker='o', markersize=3, color="red")
            elif max(myPoint.grades) >= 0.25:
                plot.plot(myPoint.x, myPoint.y, marker='o', markersize=3, color="orange")
            else:
                plot.plot(myPoint.x, myPoint.y, marker='o', markersize=3, color="yellow")
        for i in means:
            plot.plot(i.x, i.y, marker='o', markersize=7, color="black")
        plot.title('Clusters: ' + str(k) + ' Times to Repeat ' + str(timesToRun) + ' m: ' + str(m))
        plot.show()
    print(np.min(leastSqr))


if __name__ == "__main__":
    main()
