import numpy as np


class BaselineHelper:

    @staticmethod
    def greater(a, b):
        return a > b

    @staticmethod
    def greater_equal(a, b):
        return a >= b

    Third_try = [
        np.array([0, 150, 25]) / 255.,  # lower1
        np.array([10, 255, 180]) / 255.,  # upper1
        np.array([160, 150, 25]) / 255.,  # lower2
        np.array([180, 255, 180]) / 255.,  # upper2
    ]
    # Upper and lower bounds for red pixels hues
    Second_try = [
        np.array([0, 50, 25]) / 255.,  # lower1
        np.array([10, 255, 200]) / 255.,  # upper1
        np.array([160, 50, 25]) / 255.,  # lower2
        np.array([180, 255, 200]) / 255.,  # upper2
    ]
    First_try = [
        np.array([0, 100, 20]) / 255.,  # lower1
        np.array([10, 255, 255]) / 255.,  # upper1 =
        np.array([160, 100, 20]) / 255.,  # lover2
        np.array([179, 255, 255]) / 255.,  # upper2
    ]
