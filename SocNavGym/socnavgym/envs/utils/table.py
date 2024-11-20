import cv2
import numpy as np
from socnavgym.envs.utils.object import Object
from socnavgym.envs.utils.utils import w2px, w2py


class Table(Object):
    """
    Class for tables
    """

    def __init__(self, id=None, x=None, y=None, theta=None, width=None, length=None) -> None:
        super().__init__(id, "table")
        self.width = None  # width of the table
        self.length = None  # length of the table
        self.set(id, x, y, theta, width, length)

    def set(self, id, x, y, theta, width, length):
        super().set(id, x, y, theta)
        self.width = width
        self.length = length

    def draw(self, img, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y):
        if self.color == None:
            color = (0, 75, 150)  # brown
        else: color = self.color

        assert (
            self.length != None and self.width != None
        ), "Length or breadth is None type."
        assert (
            self.x != None and self.y != None and self.orientation != None
        ), "Coordinates or orientation are None type"

        # p1, p2, p3, p4 are the coordinates of the corners of the rectangle. calculation is done so as to orient the rectangle at an angle.


        p1 = [
            w2px(
                (
                    self.x
                    + self.length / 2 * np.cos(self.orientation)
                    - self.width / 2 * np.sin(self.orientation)
                ),
                PIXEL_TO_WORLD_X,
                MAP_SIZE_X,
            ),
            w2py(
                (
                    self.y
                    + self.length / 2 * np.sin(self.orientation)
                    + self.width / 2 * np.cos(self.orientation)
                ),
                PIXEL_TO_WORLD_Y,
                MAP_SIZE_Y,
            ),
        ]

        p2 = [
            w2px(
                (
                    self.x
                    + self.length / 2 * np.cos(self.orientation)
                    + self.width / 2 * np.sin(self.orientation)
                ),
                PIXEL_TO_WORLD_X,
                MAP_SIZE_X,
            ),
            w2py(
                (
                    self.y
                    + self.length / 2 * np.sin(self.orientation)
                    - self.width / 2 * np.cos(self.orientation)
                ),
                PIXEL_TO_WORLD_Y,
                MAP_SIZE_Y,
            ),
        ]

        p3 = [
            w2px(
                (
                    self.x
                    - self.length / 2 * np.cos(self.orientation)
                    + self.width / 2 * np.sin(self.orientation)
                ),
                PIXEL_TO_WORLD_X,
                MAP_SIZE_X,
            ),
            w2py(
                (
                    self.y
                    - self.length / 2 * np.sin(self.orientation)
                    - self.width / 2 * np.cos(self.orientation)
                ),
                PIXEL_TO_WORLD_Y,
                MAP_SIZE_Y,
            ),
        ]

        p4 = [
            w2px(
                (
                    self.x
                    - self.length / 2 * np.cos(self.orientation)
                    - self.width / 2 * np.sin(self.orientation)
                ),
                PIXEL_TO_WORLD_X,
                MAP_SIZE_X,
            ),
            w2py(
                (
                    self.y
                    - self.length / 2 * np.sin(self.orientation)
                    + self.width / 2 * np.cos(self.orientation)
                ),
                PIXEL_TO_WORLD_Y,
                MAP_SIZE_Y,
            ),
        ]
        points = np.array([p1, p2, p3, p4])
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(img, [np.int32(points)], color)  # filling the rectangle made from the points with the specified color
        cv2.polylines(img, [np.int32(points)], True, (0, 0, 0), 2)  # bordering the rectangle
