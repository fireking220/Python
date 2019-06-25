import numpy as np


class point():
    x = 0
    y = 0
    kClass = None

    def __init__(self, cordinate, kClass):
        self.x = cordinate[0]
        self.y = cordinate[1]
        self.kClass = kClass

    def __str__(self):
        return "[X: " + str(self.x) + ", Y: " + str(self.y) + ", Kluster: " + str(self.kClass) + "]"

    def __repr__(self):
        return "[X: " + str(self.x) + ", Y: " + str(self.y) + ", Kluster: " + str(self.kClass) + "]"

    def setCord(self, cordinate):
        self.x = cordinate[0]
        self.y = cordinate[1]

    def getCord(self):
        return np.array([self.x, self.y])
