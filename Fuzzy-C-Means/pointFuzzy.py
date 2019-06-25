import numpy as np
__X__ = 0
__Y__ = 1


class point():
    x = None
    y = None
    grades = None

    def __init__(self, cordinate, grades):
        self.x = cordinate[__X__]
        self.y = cordinate[__Y__]
        self.grades = grades.copy()

    def __str__(self):
        return "[X: " + str(self.x) + ", Y: " + str(self.y) + ", Grades: " + str(self.grades) + "]"

    def __repr__(self):
        return "[X: " + str(self.x) + ", Y: " + str(self.y) + ", Grades: " + str(self.grades) + "]"

    def setCord(self, cordinate):
        self.x = cordinate[0]
        self.y = cordinate[1]

    def setGrade(self, newGrade, pos):
        self.grades[pos] = newGrade

    def getGrade(self, i):
        return self.grades[i]

    def getCord(self):
        return np.array([self.x, self.y])
