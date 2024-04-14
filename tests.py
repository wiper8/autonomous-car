
import math
from tracks import *
from shapes import *
from Car import *

def test_deg():
    assert round(deg(1.4), 8) == 80.21409132
    assert deg(math.pi) == 180
    assert deg(0) == 0

def test_orientation():
    point_test1 = Point(4, 5)
    assert round(Line(point_test1, Point(6, 10)).orientation(), 8) == 68.19859051

def test_shapes():
    point_test1 = Point(4, 5)
    assert (Point(4, 5) == Point(4, 5)) == True

    assert Line(point_test1, Point(4, 10)).orientation() == 90
    assert Line(point_test1, Point(4, 3)).orientation() == -90
    assert Line(point_test1, Point(4, 5)).orientation() == 0

    assert onSegment(Line(point_test1, Point(6, 10)), Point(5.5, 8.75)) == True
    assert onSegment(Line(point_test1, Point(6, 10)), Point(8, 15)) == False

    assert intersects(Line(Point(4, 5), Point(6, 10)), Line(Point(2, 3), Point(6, 8))) == True
    assert intersects(Line(Point(4, 5), Point(6, 10)), Line(Point(2, 3), Point(6, 2))) == False

    assert intersects(Line(Point(4, 5), Point(6, 10)), Line(Point(5.5, 8.75), Point(6, 2))) == True
    assert intersects(Line(Point(4, 5), Point(6, 10)), Line(Point(8, 15), Point(6, 2))) == False
    assert intersects(Line(Point(4, 5), Point(6, 10)), Line(Point(5.5, 8.75), Point(8, 15))) == True
    assert intersects(Line(Point(4, 5), Point(6, 10)), Line(Point(6.5, 11.25), Point(8, 15))) == False

    assert intersects(Line(Point(-8.1, 1), Point(28.8, 244.4)), Line(Point(-0.75, -0.278), Point(-0.75, 4.38))) == False

    assert intersects(Line(Point(0, 0), Point(10, 0)), Line(Point(5, -5), Point(5, 10))) == True
    assert intersects(Line(Point(5, -5), Point(5, 10)), Line(Point(0, 0), Point(10, 0))) == True

    l1 = Line(Point(4, 5), Point(6, 10))
    l2 = Line(Point(40, 50), Point(60, 100))
    l3 = Line(Point(2, 3), Point(6, 8))

    assert stop_line_on_other(l1, l2) == l1
    assert stop_line_on_other(l1, l3) == Line(l1.point1, Point(4.4, 6))

def test_track():
    assert Track(Line(Point(50, 100), Point(50, 90)), [Point(0, 100), Point(50, 100)], Line(Point(0, 0), Point(10, 0)), [Point(10, 90), Point(50, 90)])

def test_car():
    auto = Car()
    pass

