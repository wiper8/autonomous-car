import math

import numpy as np
from math import isclose

def deg(x):
    return x * 180 / math.pi

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f'({self.x}, {self.y})'
    def __eq__(self, pt2):
        return self.y == pt2.y and self.x == pt2.x

class Line:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def length(self):
        return math.sqrt((self.point2.y - self.point1.y) ** 2 + (self.point2.x - self.point1.x) ** 2)

    def __eq__(self, line2):
        return self.point1 == line2.point1 and self.point2 == line2.point2

    def __str__(self):
        return f'from\n{self.point1}\nto\n{self.point2}'

    def slope(self):
        if self.point2.x == self.point1.x:
            return "Inf"
        else:
            return (self.point2.y - self.point1.y)/(self.point2.x - self.point1.x)

    def intercept(self):
        if self.slope() == "Inf":
            return None
        else:
            return self.point2.y - self.slope() * self.point2.x

    def plot(self):
        return ([self.point1.x, self.point2.x], [self.point1.y, self.point2.y])

    def orientation(self):
        if self.point2.x == self.point1.x:
            if self.point2.y == self.point1.y:
                return 0
            if self.point2.y > self.point1.y:
                return 90
            else:
                return -90
        return deg(math.atan2(self.point2.y - self.point1.y, self.point2.x - self.point1.x))

def clock_wise(line1, point1):
    line_A = Line(line1.point2, point1)
    orien = line_A.orientation() - line1.orientation()
    if orien < -180:
        orien += 360
    if orien > 180:
        orien += -360

    if orien == 0 or orien == 180:
        return 0
    if orien > 0:
        return -1
    else:
        return 1

def onSegment(line1, point1):
    if line1.point1.x == line1.point2.x:
        return line1.point1.x == point1.x and (point1.y <= max(line1.point2.y, line1.point1.y) and point1.y >= min(line1.point2.y, line1.point1.y))
    else:
        a = line1.slope()
        b = line1.intercept()
        if isclose(point1.y, a * point1.x + b, rel_tol=0.01):
            return (point1.y <= max(line1.point2.y, line1.point1.y) and point1.y >= min(line1.point2.y, line1.point1.y))
        else:
            return False

    if line1.point2.y >= point1.y and point1.y >= line1.point1.y:
        if line1.point2.x >= point1.x and point1.x >= line1.point1.x:
            return True
    return False


def intersects(line1, line2):
    clock1 = clock_wise(line1, line2.point2)
    clock2 = clock_wise(line1, line2.point1)
    clock3 = clock_wise(line2, line1.point1)
    clock4 = clock_wise(line2, line1.point2)

    if clock1 != clock2 and clock3 != clock4: return True

    if onSegment(line1, line2.point2): return True
    if onSegment(line1, line2.point1): return True
    if onSegment(line2, line1.point1): return True
    if onSegment(line2, line1.point2): return True

    return False

def stop_line_on_other(line1, line2):
    if intersects(line1, line2):
        #y=ax+b

        if line2.point2.x == line2.point1.x and line1.point2.x == line1.point1.x:
            l1 = Line(Point(line1.point1), line2.point1)
            l2 = Line(Point(line1.point1), line2.point1)
            if l1.length() < l2.length():
                return l1
            else:
                return l2

        if line2.point2.x == line2.point1.x:
            a1 = (line1.point2.y - line1.point1.y) / (line1.point2.x - line1.point1.x)
            b1 = line1.point1.y - a1 * line1.point1.x
            y = a1 * line2.point1.x + b1
            return Line(line1.point1, Point(line2.point1.x, y))

        if line1.point2.x == line1.point1.x:
            a2 = (line2.point2.y - line2.point1.y) / (line2.point2.x - line2.point1.x)
            b2 = line2.point1.y - a2 * line2.point1.x
            y = a2 * line1.point1.x + b2
            return Line(line1.point1, Point(line1.point1.x, y))

        a1 = line1.slope()
        b1 = line1.intercept()

        x = (line2.intercept() - b1) / (a1 - line2.slope())
        y = a1 * x + b1
        return Line(line1.point1, Point(x, y))

    return line1


def if_dist_betw_two_lines_at_least(line1, line2, dist):

    a = line1.slope()
    b = line1.intercept()
    if a == 0:
        projected_point = Point(line2.point2.x, line1.point2.y)
        if onSegment(line1, projected_point):
            return abs(line2.point2.y - line1.point2.y) > dist
    else:
        a2 = -1/a
        b2 = line2.point2.y - a2 * line2.point2.x
        projected_point = Point((b2 - b)/(a - a2), a * (b2 - b)/(a - a2) + b)

        if onSegment(line1, projected_point):
            return abs(line2.point2.y - a * line2.point2.x - b) / math.sqrt(1 + a ** 2) > dist
        else:
            a = line2.slope()
            b = line2.intercept()
            if a == 0:
                projected_point = Point(line1.point2.x, line2.point2.y)
                if onSegment(line2, projected_point):
                    return abs(line1.point2.y - line2.point2.y) > dist
            else:
                a2 = -1 / a
                b2 = line1.point2.y - a2 * line1.point2.x
                projected_point = Point((b2 - b)/(a - a2), a * (b2 - b)/(a - a2) + b)

                if onSegment(line2, projected_point):
                    return abs(line1.point2.y - a * line1.point2.x - b) / math.sqrt(1 + a ** 2) > dist
                else:
                    return True







