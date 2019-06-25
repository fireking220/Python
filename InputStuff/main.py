from pynput.mouse import Button, Controller
import time
import random


def main():
    mouse = Controller()
    random.seed()
        # mouse.position = (random.randint(0, 2559), random.randint(0, 1439))
        # print("Current Mouse Position is {0}".format(mouse.position))
    mouse.move(5, -5)
    time.sleep(3)


if __name__ == "__main__":
    main()
