import math
from utils import *
from shapes import *
import numpy.random as random
from plotnine import *

# finish est une Line de 2 points
# walls_points est une list de n points tous reliés (en ordre)
class Track:
    def __init__(self, finish, walls_points_left, starting_line, walls_points_right):

        self.walls = [starting_line]
        self.walls_points = [starting_line.point1, starting_line.point2]

        #pour que la piste soit fermée
        assert finish.point1 == walls_points_left[-1] or finish.point1 == walls_points_right[-1]
        assert finish.point2 == walls_points_left[-1] or finish.point2 == walls_points_right[-1]

        assert starting_line.point1 == walls_points_left[0]
        assert starting_line.point2 == walls_points_right[0]

        self.finish = finish
        self.walls_points_left = walls_points_left
        self.starting_line = starting_line
        self.walls_points_right = walls_points_right

        self.centre = []
        self.centre_orien = []

        if len(self.walls_points_left) > 0:
            for i in range(len(self.walls_points_left)-1):
                self.walls.append(Line(self.walls_points_left[i], self.walls_points_left[i + 1]))

        if len(self.walls_points_right) > 0:
            for i in range(len(self.walls_points_right)-1):
                self.walls.append(Line(self.walls_points_right[i], self.walls_points_right[i + 1]))

        if len(self.walls) > 0:
            for i in range(len(self.walls)):
                self.walls_points.append(self.walls[i].point1)
                self.walls_points.append(self.walls[i].point2)

        assert len(self.walls_points_right) == len(self.walls_points_right)
        self.centre = [None] * len(self.walls_points_right)
        self.centre_orien = [None] * len(self.walls_points_right)

        for i in range(len(self.walls_points_right)):
            self.centre[i] = Point((self.walls_points_right[i].x + self.walls_points_left[i].x)/2, (self.walls_points_right[i].y + self.walls_points_left[i].y)/2)


        for i in range(len(self.walls_points_right)):
            self.centre_orien[i] = math.atan2(self.walls_points_right[i].y - self.walls_points_left[i].y, self.walls_points_right[i].x - self.walls_points_left[i].x)*180/math.pi

    def plot(self):
        plot = ggplot()
        plot += theme_bw()

        plot += geom_path(aes(x=self.finish.plot()[0], y=self.finish.plot()[1]), color="red")
        for i in range(len(self.walls)):
            plot += geom_path(aes(x=self.walls[i].plot()[0], y=self.walls[i].plot()[1]))
        #for i in range(len(self.centre)-1):
        #    plot += geom_path(aes(x=[self.centre[i].x, self.centre[i+1].x], y=[self.centre[i].y, self.centre[i+1].y]), color="green")

        rangexy = max(self.most_up()-self.most_down(), self.most_right()-self.most_left())
        limitsx = [(self.most_right() + self.most_left()) / 2 - rangexy / 2*10.23/7, ((self.most_right() + self.most_left()) / 2 + rangexy / 2*10.23/7)]
        limitsy = [(self.most_up()+self.most_down())/2 - rangexy/2, (self.most_up()+self.most_down())/2 + rangexy/2]
        breaksx = seq(limitsx[0], limitsx[1], 20)
        breaksy = seq(limitsy[0], limitsy[1], 20)

        plot += scale_x_continuous(breaks=breaksx)
        plot += scale_y_continuous(breaks=breaksy)
        plot += theme(figure_size=(10.23,7))

        plot += coord_cartesian(xlim=limitsx, ylim=limitsy)
        return plot

    def most_right(self):
        max_x = -math.inf
        for point in self.walls_points:
            max_x = max(max_x, point.x)
        return max_x
    def most_left(self):
        max_x = math.inf
        for point in self.walls_points:
            max_x = min(max_x, point.x)
        return max_x
    def most_up(self):
        max_x = -math.inf
        for point in self.walls_points:
            max_x = max(max_x, point.y)
        return max_x
    def most_down(self):
        max_x = math.inf
        for point in self.walls_points:
            max_x = min(max_x, point.y)
        return max_x


def Track_generator(straight=True):

    a = random.beta(4, 4)*15 + 5
    b = random.beta(4, 4)*15 + 5
    c = random.beta(4, 4)*15 + 5

    walls_points_left = [Point(-a, -c), Point(-a, 0)]
    walls_points_right = [Point(b, -c), Point(b, 0)]

    if straight:
        n = random.negative_binomial(18, 0.9) + 1
        max_angle = math.asin(5/(a+b))
        if n != 0:
            for i in range(n):
                #o1 = Line(walls_points[1], walls_points[0]).orientation()
                #o2 = Line(walls_points[-2], walls_points[-1]).orientation()

                dist1 = random.beta(2, 6) * 290 + 10
                theta1 = random.beta(20, 20) * math.pi

                while theta1 < max_angle or theta1 > math.pi - max_angle:
                    theta1 = random.beta(20, 20) * math.pi

                x1 = walls_points_left[-1].x + math.cos(theta1) * dist1
                y1 = walls_points_left[-1].y + math.sin(theta1) * dist1
                x2 = walls_points_right[-1].x + math.cos(theta1) * dist1
                y2 = walls_points_right[-1].y + math.sin(theta1) * dist1

                walls_points_left.append(Point(x1, y1))
                walls_points_right.append(Point(x2, y2))

        track = Track(Line(walls_points_left[-1], walls_points_right[-1]), walls_points_left, Line(Point(-a, -c), Point(b, -c)), walls_points_right)
    else:
        """
        n = random.negative_binomial(10, 0.5) + 1
        # track qui ne va pas tjrs vers le nord
        # elle ne doit pas se croiser elle-même
        # distance minimale de 5m entre deux coins de murs
        if n != 0:
            for i in range(n):

                angle_limite = Line(walls_points_left[-1], walls_points_right[-1]).orientation() / 180 * math.pi

                dist1 = random.beta(2, 6) * 290 + 10
                theta1 = random.beta(20, 20) * math.pi + angle_limite

                x1 = walls_points_left[-1].x + math.cos(theta1) * dist1
                y1 = walls_points_left[-1].y + math.sin(theta1) * dist1
                x2 = walls_points_right[-1].x + math.cos(theta1) * dist1
                y2 = walls_points_right[-1].y + math.sin(theta1) * dist1

                line1 = Line(walls_points_left[-1], Point(x1, y1))
                line2 = Line(walls_points_right[-1], Point(x2, y2))

                while (not if_dist_betw_two_lines_at_least(line1, line2, 5)) or intersects(line1, line2):
                    if intersects(line1, line2):
                        print("intersects")
                    else:
                        print("5m")
                    dist1 = random.beta(2, 6) * 290 + 10
                    theta1 = random.beta(20, 20) * math.pi + angle_limite
                    
                    x1 = walls_points_left[-1].x + math.cos(theta1) * dist1
                    y1 = walls_points_left[-1].y + math.sin(theta1) * dist1
                    x2 = walls_points_right[-1].x + math.cos(theta1) * dist1
                    y2 = walls_points_right[-1].y + math.sin(theta1) * dist1

                    line1 = Line(walls_points_left[-1], Point(x1, y1))
                    line2 = Line(walls_points_right[-1], Point(x2, y2))

                intersect_previous_wall = False
                if len(walls_points_left[:-2]) > 0:
                    for j in range(len(walls_points_left[:-3])):
                        intersect_previous_wall = intersect_previous_wall or intersects(
                            Line(walls_points_left[-2], walls_points_left[-1]),
                            Line(walls_points_left[j], walls_points_left[j + 1])) or intersects(
                            Line(walls_points_right[-2], walls_points_right[-1]),
                            Line(walls_points_right[j], walls_points_right[j + 1]))

                    if intersect_previous_wall:
                        return Track(Line(walls_points_left[-1], walls_points_right[-1]), walls_points_left,
                                     Line(Point(-a, -c), Point(b, -c)), walls_points_right)

                    else:
                        walls_points_left.append(Point(x1, y1))
                        walls_points_right.append(Point(x2, y2))
                else:
                    walls_points_left.append(Point(x1, y1))
                    walls_points_right.append(Point(x2, y2))

        track = Track(Line(walls_points_left[-1], walls_points_right[-1]), walls_points_left,
                      Line(Point(-a, -c), Point(b, -c)), walls_points_right)
        """
        n = random.negative_binomial(10, 0.5) + 1
        #track qui ne va pas tjrs vers le nord
        #elle ne doit pas se croiser elle-même
        #distance minimale de 5m entre deux coins de murs
        if n != 0:
            for i in range(n):

                angle_limite = Line(walls_points_left[-1], walls_points_right[-1]).orientation()/180*math.pi

                dist1 = random.beta(2, 6) * 290 + 10
                dist2 = random.beta(2, 6) * 290 + 10
                theta1 = random.beta(20, 20) * math.pi/2 + angle_limite
                theta2 = (random.beta(20, 20)+1) * math.pi/2 + angle_limite

                x1 = walls_points_left[-1].x + math.cos(theta1) * dist1
                y1 = walls_points_left[-1].y + math.sin(theta1) * dist1
                x2 = walls_points_right[-1].x + math.cos(theta2) * dist2
                y2 = walls_points_right[-1].y + math.sin(theta2) * dist2

                line1 = Line(walls_points_left[-1], Point(x1, y1))
                line2 = Line(walls_points_right[-1], Point(x2, y2))

                while not if_dist_betw_two_lines_at_least(line1, line2, 5) or intersects(line1, line2):

                    dist1 = random.beta(2, 6) * 290 + 10
                    dist2 = random.beta(2, 6) * 290 + 10
                    theta1 = random.beta(20, 20) * math.pi + angle_limite
                    theta2 = (random.beta(20, 20)+1) * math.pi/2 + angle_limite

                    x1 = walls_points_left[-1].x + math.cos(theta1) * dist1
                    y1 = walls_points_left[-1].y + math.sin(theta1) * dist1
                    x2 = walls_points_right[-1].x + math.cos(theta2) * dist2
                    y2 = walls_points_right[-1].y + math.sin(theta2) * dist2

                    line1 = Line(walls_points_left[-1], Point(x1, y1))
                    line2 = Line(walls_points_right[-1], Point(x2, y2))

                intersect_previous_wall = False
                for j in range(len(walls_points_left)-2):
                        intersect_previous_wall = intersect_previous_wall or intersects(line1, Line(walls_points_left[j], walls_points_left[j+1])) or intersects(line2, Line(walls_points_right[j], walls_points_right[j+1])) or intersects(line2, Line(walls_points_left[j], walls_points_left[j+1])) or intersects(line1, Line(walls_points_right[j], walls_points_right[j+1]))

                if intersect_previous_wall:
                        return Track(Line(walls_points_left[-1], walls_points_right[-1]), walls_points_left,
                                      Line(Point(-a, -c), Point(b, -c)), walls_points_right)

                else:
                        walls_points_left.append(Point(x1, y1))
                        walls_points_right.append(Point(x2, y2))

        track = Track(Line(walls_points_left[-1], walls_points_right[-1]), walls_points_left,
                      Line(Point(-a, -c), Point(b, -c)), walls_points_right)

    return track


def Track_generator_by_driving():
    walls_points_left = [Point(-a, -c), Point(-a, 0)]
    walls_points_right = [Point(b, -c), Point(b, 0)]

    if straight:
        n = random.negative_binomial(18, 0.9) + 1
        max_angle = math.asin(5 / (a + b))
        if n != 0:
            for i in range(n):
                # o1 = Line(walls_points[1], walls_points[0]).orientation()
                # o2 = Line(walls_points[-2], walls_points[-1]).orientation()

                dist1 = random.beta(2, 6) * 290 + 10
                theta1 = random.beta(20, 20) * math.pi

                while theta1 < max_angle or theta1 > math.pi - max_angle:
                    theta1 = random.beta(20, 20) * math.pi

                x1 = walls_points_left[-1].x + math.cos(theta1) * dist1
                y1 = walls_points_left[-1].y + math.sin(theta1) * dist1
                x2 = walls_points_right[-1].x + math.cos(theta1) * dist1
                y2 = walls_points_right[-1].y + math.sin(theta1) * dist1

                walls_points_left.append(Point(x1, y1))
                walls_points_right.append(Point(x2, y2))

        track = Track(Line(walls_points_left[-1], walls_points_right[-1]), walls_points_left,
                      Line(Point(-a, -c), Point(b, -c)), walls_points_right)
    else:
        """
        n = random.negative_binomial(10, 0.5) + 1
        # track qui ne va pas tjrs vers le nord
        # elle ne doit pas se croiser elle-même
        # distance minimale de 5m entre deux coins de murs
        if n != 0:
            for i in range(n):

                angle_limite = Line(walls_points_left[-1], walls_points_right[-1]).orientation() / 180 * math.pi

                dist1 = random.beta(2, 6) * 290 + 10
                theta1 = random.beta(20, 20) * math.pi + angle_limite

                x1 = walls_points_left[-1].x + math.cos(theta1) * dist1
                y1 = walls_points_left[-1].y + math.sin(theta1) * dist1
                x2 = walls_points_right[-1].x + math.cos(theta1) * dist1
                y2 = walls_points_right[-1].y + math.sin(theta1) * dist1

                line1 = Line(walls_points_left[-1], Point(x1, y1))
                line2 = Line(walls_points_right[-1], Point(x2, y2))

                while (not if_dist_betw_two_lines_at_least(line1, line2, 5)) or intersects(line1, line2):
                    if intersects(line1, line2):
                        print("intersects")
                    else:
                        print("5m")
                    dist1 = random.beta(2, 6) * 290 + 10
                    theta1 = random.beta(20, 20) * math.pi + angle_limite

                    x1 = walls_points_left[-1].x + math.cos(theta1) * dist1
                    y1 = walls_points_left[-1].y + math.sin(theta1) * dist1
                    x2 = walls_points_right[-1].x + math.cos(theta1) * dist1
                    y2 = walls_points_right[-1].y + math.sin(theta1) * dist1

                    line1 = Line(walls_points_left[-1], Point(x1, y1))
                    line2 = Line(walls_points_right[-1], Point(x2, y2))

                intersect_previous_wall = False
                if len(walls_points_left[:-2]) > 0:
                    for j in range(len(walls_points_left[:-3])):
                        intersect_previous_wall = intersect_previous_wall or intersects(
                            Line(walls_points_left[-2], walls_points_left[-1]),
                            Line(walls_points_left[j], walls_points_left[j + 1])) or intersects(
                            Line(walls_points_right[-2], walls_points_right[-1]),
                            Line(walls_points_right[j], walls_points_right[j + 1]))

                    if intersect_previous_wall:
                        return Track(Line(walls_points_left[-1], walls_points_right[-1]), walls_points_left,
                                     Line(Point(-a, -c), Point(b, -c)), walls_points_right)

                    else:
                        walls_points_left.append(Point(x1, y1))
                        walls_points_right.append(Point(x2, y2))
                else:
                    walls_points_left.append(Point(x1, y1))
                    walls_points_right.append(Point(x2, y2))

        track = Track(Line(walls_points_left[-1], walls_points_right[-1]), walls_points_left,
                      Line(Point(-a, -c), Point(b, -c)), walls_points_right)
        """
        n = random.negative_binomial(10, 0.5) + 1
        # track qui ne va pas tjrs vers le nord
        # elle ne doit pas se croiser elle-même
        # distance minimale de 5m entre deux coins de murs
        if n != 0:
            for i in range(n):

                angle_limite = Line(walls_points_left[-1], walls_points_right[-1]).orientation() / 180 * math.pi

                dist1 = random.beta(2, 6) * 290 + 10
                dist2 = random.beta(2, 6) * 290 + 10
                theta1 = random.beta(20, 20) * math.pi / 2 + angle_limite
                theta2 = (random.beta(20, 20) + 1) * math.pi / 2 + angle_limite

                x1 = walls_points_left[-1].x + math.cos(theta1) * dist1
                y1 = walls_points_left[-1].y + math.sin(theta1) * dist1
                x2 = walls_points_right[-1].x + math.cos(theta2) * dist2
                y2 = walls_points_right[-1].y + math.sin(theta2) * dist2

                line1 = Line(walls_points_left[-1], Point(x1, y1))
                line2 = Line(walls_points_right[-1], Point(x2, y2))

                while not if_dist_betw_two_lines_at_least(line1, line2, 5) or intersects(line1, line2):
                    dist1 = random.beta(2, 6) * 290 + 10
                    dist2 = random.beta(2, 6) * 290 + 10
                    theta1 = random.beta(20, 20) * math.pi + angle_limite
                    theta2 = (random.beta(20, 20) + 1) * math.pi / 2 + angle_limite

                    x1 = walls_points_left[-1].x + math.cos(theta1) * dist1
                    y1 = walls_points_left[-1].y + math.sin(theta1) * dist1
                    x2 = walls_points_right[-1].x + math.cos(theta2) * dist2
                    y2 = walls_points_right[-1].y + math.sin(theta2) * dist2

                    line1 = Line(walls_points_left[-1], Point(x1, y1))
                    line2 = Line(walls_points_right[-1], Point(x2, y2))

                intersect_previous_wall = False
                for j in range(len(walls_points_left) - 2):
                    intersect_previous_wall = intersect_previous_wall or intersects(line1, Line(walls_points_left[j],
                                                                                                walls_points_left[
                                                                                                    j + 1])) or intersects(
                        line2, Line(walls_points_right[j], walls_points_right[j + 1])) or intersects(line2, Line(
                        walls_points_left[j], walls_points_left[j + 1])) or intersects(line1,
                                                                                       Line(walls_points_right[j],
                                                                                            walls_points_right[j + 1]))

                if intersect_previous_wall:
                    return Track(Line(walls_points_left[-1], walls_points_right[-1]), walls_points_left,
                                 Line(Point(-a, -c), Point(b, -c)), walls_points_right)

                else:
                    walls_points_left.append(Point(x1, y1))
                    walls_points_right.append(Point(x2, y2))

        track = Track(Line(walls_points_left[-1], walls_points_right[-1]), walls_points_left,
                      Line(Point(-a, -c), Point(b, -c)), walls_points_right)

    return track
