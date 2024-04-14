import math

from shapes import *
from physics import *
from utils import *
from shapes import deg
from plotnine import *
from copy import copy

def cosd(x):
    return math.cos(math.radians(x))

def sind(x):
    return math.sin(math.radians(x))

def tand(x):
    return math.tan(math.radians(x))

def color(x, max_x, palette = "uni"): #palette either "uni" or "bi"
    if palette == "uni":
        assert x <= max_x
        hex_code = hex(round((1-x/max_x) * 255))[2:]
        if len(hex_code) == 1:
            hex_code = "0"+hex_code
        colors = "#ff" + hex_code * 2 + "ff"
        return colors.upper()

    if palette == "bi":
        assert abs(x) <= abs(max_x)
        if x >= 0:
            hex_code = hex(round((1-x/max_x) * 255))[2:]
            if len(hex_code) == 1:
                hex_code = "0" + hex_code
            colors = "#ff" + hex_code * 2 + "ff"

        else:
            hex_code = hex(round((1+x/max_x) * 255))[2:]
            if len(hex_code) == 1:
                hex_code = "0" + hex_code
            colors = "#" + hex_code + "ff" + hex_code + "ff"

        return colors.upper()



class Car():
    x=0
    y=0
    orien=90
    speed=0
    acc=0
    steer=0
    max_acc = 8.944
    min_acc = -10.868
    max_steer = 21.96
    max_speed = 212 * 1.61 / 3.6
    odo=0
    friction=1.3
    g=9.80665
    mass=3560 #lbs
    pSC2=0
    range_radars = 250
    max_turning_radius = 11.48/2
    cen_to_back = 1.004
    cen_to_front = 3.657
    inner_w = 1.595
    inner_l = 2.69
    outer_w = 1.955
    outer_l = cen_to_back + cen_to_front
    t_eps = outer_l / max_speed
    height = 1.241
    mass_height = 0.558 #approximation
    radars_count = 24
    radars = []
    dists = []

    def place_wheels_walls(self):
        self.coin_f_l = Point(
            self.centre.x + cosd(self.orien) * self.cen_to_front + cosd(self.orien + 90) * self.outer_w / 2,
            self.centre.y + sind(self.orien) * self.cen_to_front + sind(self.orien + 90) * self.outer_w / 2)
        self.coin_f_r = Point(
            self.centre.x + cosd(self.orien) * self.cen_to_front + cosd(self.orien - 90) * self.outer_w / 2,
            self.centre.y + sind(self.orien) * self.cen_to_front + sind(self.orien - 90) * self.outer_w / 2)
        self.coin_b_r = Point(
            self.centre.x + cosd(self.orien + 180) * self.cen_to_back + cosd(self.orien - 90) * self.outer_w / 2,
            self.centre.y + sind(self.orien + 180) * self.cen_to_back + sind(self.orien - 90) * self.outer_w / 2)
        self.coin_b_l = Point(
            self.centre.x + cosd(self.orien + 180) * self.cen_to_back + cosd(self.orien + 90) * self.outer_w / 2,
            self.centre.y + sind(self.orien + 180) * self.cen_to_back + sind(self.orien + 90) * self.outer_w / 2)

        self.wheel_f_l = Point(
            self.centre.x + cosd(self.orien) * self.inner_l + cosd(self.orien + 90) * self.inner_w / 2,
            self.centre.y + sind(self.orien) * self.inner_l + sind(self.orien + 90) * self.inner_w / 2)
        self.wheel_f_r = Point(
            self.centre.x + cosd(self.orien) * self.inner_l + cosd(self.orien - 90) * self.inner_w / 2,
            self.centre.y + sind(self.orien) * self.inner_l + sind(self.orien - 90) * self.inner_w / 2)
        self.wheel_b_r = Point(
            self.centre.x + cosd(self.orien + 180) * self.cen_to_back + cosd(self.orien - 90) * self.outer_w / 2,
            self.centre.y + sind(self.orien + 180) * self.cen_to_front + sind(self.orien - 90) * self.outer_w / 2)
        self.wheel_b_l = Point(
            self.centre.x + cosd(self.orien + 180) * self.cen_to_back + cosd(self.orien + 90) * self.outer_w / 2,
            self.centre.y + sind(self.orien + 180) * self.cen_to_front + sind(self.orien + 90) * self.outer_w / 2)

        self.outer_walls = [Line(self.coin_b_l, self.coin_f_l), Line(self.coin_b_r, self.coin_f_r),
                            Line(self.coin_f_l, self.coin_f_r), Line(self.coin_b_l, self.coin_b_r)]
        self.wheels = [Line(self.wheel_b_l, self.wheel_f_l), Line(self.wheel_b_r, self.wheel_f_r),
                       Line(self.wheel_f_l, self.wheel_f_r), Line(self.wheel_b_l, self.wheel_b_r)]

        self.radars_points = [
            Point(self.centre.x + cosd(i / self.radars_count * 360 + self.orien + 180) * self.range_radars,
                  self.centre.y + sind(i / self.radars_count * 360 + self.orien + 180) * self.range_radars) for i in
            range(self.radars_count)]
        self.radars_finish_points = self.radars_points

        self.radars_lines = [Line(self.centre, rad) for rad in self.radars_points]
        self.radars_finish_lines = self.radars_lines

    def __init__(self, x=x, y=y, orien=orien, speed=speed, range_radars=range_radars, t_eps=t_eps, radars_count=radars_count, max_acc=max_acc, max_steer = max_steer, min_acc = min_acc, max_speed = max_speed, inner_l=inner_l, inner_w=inner_w, outer_w = outer_w, cen_to_front = cen_to_front, cen_to_back = cen_to_back):
        self.centre = Point(x, y)
        self.orien = orien
        self.range_radars = range_radars
        self.t_eps = t_eps
        self.speed = speed
        self.max_speed = max_speed
        self.max_acc = max_acc
        self.min_acc = min_acc
        self.max_steer = max_steer
        self.inner_l = inner_l
        self.inner_w = inner_w
        self.outer_w = outer_w
        self.cen_to_back = cen_to_back
        self.cen_to_front = cen_to_front
        self.physics_acc = 0
        self.physics_steer = 0
        self.line = [(copy(self.centre), copy(self.orien), copy(self.speed), copy(self.physics_acc), copy(self.physics_steer))]
        self.radars_count = radars_count
        self.track = None
        self.has_finished = False
        self.has_collapsed = False
        self.is_inside = False
        self.check_points = 0
        self.outer_radius = math.sqrt((self.outer_w/2) ** 2 + max(self.cen_to_front, self.cen_to_back) ** 2)


        self.place_wheels_walls()

        self.radars_dists = [0] * self.radars_count
        self.radars_finish_dists = [0] * self.radars_count

    def physics_limit(self):
        limited_acc = min(max(self.acc, self.min_acc), self.max_acc)
        limited_steer = min(max(self.steer, -self.max_steer), self.max_steer)
        if limited_steer == 0:
            limited_steer = 0.0000000001
        turning_radius = self.inner_l / tand(limited_steer)

        #air grad
        #self.pSC2 = self.max_acc / self.max_speed ** 2
        #self.physics_acc = limited_acc - self.pSC2 * self.speed ** 2 - self.friction * self.g * self.t_eps
        #en commentaires car ça s'annule
        self.physics_acc = limited_acc - self.max_acc * (self.speed / self.max_speed) ** 2
        self.physics_acc -= self.friction * self.g * self.t_eps * sign(self.speed)
        new_speed = self.speed + (self.physics_acc - self.friction * self.g) * self.t_eps

        #si le char va trop vite, c'est le steering que je limite
        max_slipping_speed = math.sqrt(self.friction * self.g * abs(turning_radius))
        max_tonneau_speed = math.sqrt(self.g * self.outer_w * abs(turning_radius) / 2 / self.mass_height)
        if new_speed > min(max_slipping_speed, max_tonneau_speed):
            #en commentaires car c'est toujours le cas pour cet auto
            #if max_slipping_speed < max_tonneau_speed:
                self.physics_steer = sign(limited_steer) * deg(math.atan(self.friction * self.g * self.inner_l / new_speed ** 2))
            #else:
            #    self.physics_steer = sign(limited_steer) * deg(math.atan(self.g * self.inner_l * self.outer_w / new_speed ** 2/self.mass_height/2))
        else:
            self.physics_steer = limited_steer


    def move(self, acc, steer):
        self.acc = acc
        self.steer = steer
        self.physics_limit()
        distance = self.speed * self.t_eps + self.physics_acc/2 * self.t_eps ** 2
        self.speed += self.physics_acc * self.t_eps
        self.odo += distance
        turning_radius = self.inner_l / tand(self.physics_steer)

        if abs(turning_radius) > 20000:
            self.centre.x += cosd(self.orien) * distance
            self.centre.y += sind(self.orien) * distance
        else:
            turning_cen_x = self.centre.x + cosd(self.orien + 90) * turning_radius
            turning_cen_y = self.centre.y + sind(self.orien + 90) * turning_radius
            turning_cen_orien = self.orien - 90 * sign(turning_radius)
            turning_cen_new_orien = turning_cen_orien + deg(distance/turning_radius)
            self.centre.x = turning_cen_x + cosd(turning_cen_new_orien) * abs(turning_radius)
            self.centre.y = turning_cen_y + sind(turning_cen_new_orien) * abs(turning_radius)
            self.orien = turning_cen_new_orien + 90 * sign(turning_radius)

        self.place_wheels_walls()
        self.check_collapse()
        self.check_finish()
        if self.has_finished:
            print("Finished!")
        self.calculate_radar_dists()
        self.line.append((copy(self.centre), copy(self.orien), copy(self.speed), copy(self.physics_acc), copy(self.physics_steer)))

    def check_if_checkpoint_passed(self):
        assert self.track is not None

        b = self.track.centre[self.check_points + 1].y - tand(self.track.centre_orien[self.check_points+1]) * self.track.centre[self.check_points + 1].x

        cond1 = tand(self.track.centre_orien[self.check_points + 1]) * self.centre.x + b > self.centre.y

        cond2 = tand(self.track.centre_orien[self.check_points + 1]) * self.line[-2][0].x + b > self.line[-2][0].y
        if cond1 != cond2:
            #print("CHECKPOINT!")
            self.check_points += 1
        return cond1 != cond2

    def far_from_car(self, pt1, pt2, dist = None):
        if dist is None:
            dist = self.range_radars
        if pt1.x == pt2.x:
            return abs(pt2.x - self.centre.x) > dist
        else:
            a = (pt2.y-pt1.y)/(pt2.x-pt1.x)
            b = pt2.y - a*pt2.x
            return abs(self.centre.y - a * self.centre.x - b) / math.sqrt(1 + a ** 2) > dist

    def check_collapse(self):
        self.has_collapsed = False
        if self.track is not None:
            for wall in self.track.walls:
                if not self.far_from_car(wall.point1, wall.point2, self.outer_radius):
                    if intersects(wall, self.outer_walls[0]):
                        self.has_collapsed = True
                    if intersects(wall, self.outer_walls[1]):
                        self.has_collapsed = True
                    if intersects(wall, self.outer_walls[2]):
                        self.has_collapsed = True
                    if intersects(wall, self.outer_walls[3]):
                        self.has_collapsed = True

            if not self.has_collapsed:
                ray = Line(self.centre, Point(max(self.track.most_right(), self.centre.x), self.centre.y))
                intersections = 0
                for wall in self.track.walls:
                    if intersects(ray, wall):
                        intersections += 1
                if intersects(ray, self.track.finish):
                    intersections += 1
                if intersections % 2 != 0:
                    self.is_inside = True


    def check_finish(self):
        self.has_finished = False
        if self.track is not None:
            if intersects(self.track.finish, self.outer_walls[0]):
                self.has_finished = True
            if intersects(self.track.finish, self.outer_walls[1]):
                self.has_finished = True
            if intersects(self.track.finish, self.outer_walls[2]):
                self.has_finished = True
            if intersects(self.track.finish, self.outer_walls[3]):
                self.has_finished = True


    def calculate_radar_dists(self):
        if self.track is not None:
            #checker si chq radar intercepte avec un mur mais pas finish
            for wall in self.track.walls:
                if not self.far_from_car(wall.point1, wall.point2):
                    for i in range(self.radars_count):
                        if intersects(self.radars_lines[i], wall):
                            self.radars_lines[i] = stop_line_on_other(self.radars_lines[i], wall)
                            self.radars_points[i] = self.radars_lines[i].point2

            #2e set de radars conçu pour intercepter avec les walls puis la finish line.
            #self.radars_finish_lines = self.radars_lines
            #self.radars_finish_points = self.radars_points

            #for i in range(self.radars_count):
            #    if intersects(self.radars_lines[i], self.track.finish):
            #        self.radars_finish_lines[i] = stop_line_on_other(self.radars_finish_lines[i], self.track.finish)
            #        self.radars_finish_points[i] = self.radars_finish_lines[i].point2

        for i in range(self.radars_count):
            self.radars_dists[i] = self.radars_lines[i].length()
            #self.radars_finish_dists[i] = self.radars_finish_lines[i].length()

    def put_on_track(self, track):
        self.track = track
        self.check_collapse()
        self.check_finish()
        self.calculate_radar_dists()

    def most_up(self):
        return(max(self.coin_b_r.y, self.coin_f_r.y, self.coin_f_l.y, self.coin_b_l.y))
    def most_down(self):
        return(min(self.coin_b_r.y, self.coin_f_r.y, self.coin_f_l.y, self.coin_b_l.y))
    def most_right(self):
        return(max(self.coin_b_r.x, self.coin_f_r.x, self.coin_f_l.x, self.coin_b_l.x))
    def most_left(self):
        return(min(self.coin_b_r.x, self.coin_f_r.x, self.coin_f_l.x, self.coin_b_l.x))

    def plot_car(self, guide=False, radars=True, full=True):
        self.physics_limit()
        if self.track is not None:
            car_plot = self.track.plot()
        else:
            car_plot = ggplot()
            car_plot += theme_bw()

            rangexy = max(self.most_up() - self.most_down(), self.most_right() - self.most_left())
            limitsx = [(self.most_right() + self.most_left()) / 2 - rangexy / 2 * 10 / 6.843,
                       ((self.most_right() + self.most_left()) / 2 + rangexy / 2 * 10 / 6.843)]
            limitsy = [(self.most_up() + self.most_down()) / 2 - rangexy / 2,
                       (self.most_up() + self.most_down()) / 2 + rangexy / 2]
            breaksx = seq(limitsx[0], limitsx[1], 1)
            breaksy = seq(limitsy[0], limitsy[1], 1)

            car_plot += scale_x_continuous(breaks=breaksx)
            car_plot += scale_y_continuous(breaks=breaksy)
            car_plot += theme(figure_size=(10, 6.843))

            car_plot += coord_cartesian(xlim=limitsx, ylim=limitsy)

        for i in range(4):
            if i == 2:
                car_plot += geom_path(aes(x=self.outer_walls[i].plot()[0], y=self.outer_walls[i].plot()[1]), color="green")
            else:
                car_plot += geom_path(aes(x=self.outer_walls[i].plot()[0], y=self.outer_walls[i].plot()[1]),
                                      color="orange")
        if radars:
            for i in range(self.radars_count):
                car_plot += geom_path(aes(x=self.radars_lines[i].plot()[0], y=self.radars_lines[i].plot()[1]), color="blue")

        car_plot += geom_point(aes(x=self.centre.x, y=self.centre.y))

        temps_davance = 2
        if guide:
            distance = self.speed * temps_davance + self.physics_acc / 2 * (temps_davance ** 2)
            max_steer = math.atan(self.friction * self.g * self.inner_l / self.speed ** 2) * 180 / math.pi
            min_turning_radius = self.inner_l / tand(max_steer)
            for j in range(1, 10, 1):

                turning_cen1 = Point(self.centre.x + cosd(self.orien + 90) * min_turning_radius, self.centre.y + sind(self.orien + 90) * min_turning_radius)
                turning_cen_orien1 = self.orien - 90
                turning_cen_new_orien1 = turning_cen_orien1 + distance*j/10 / min_turning_radius*180/math.pi
                turning_cen2 = Point(self.centre.x + cosd(self.orien - 90) * min_turning_radius, self.centre.y + sind(self.orien - 90) * min_turning_radius)
                turning_cen_orien2 = self.orien + 90
                turning_cen_new_orien2 = turning_cen_orien2 - distance*j/10 / min_turning_radius*180/math.pi

                car_plot += geom_point(aes(x=turning_cen1.x + cosd(turning_cen_new_orien1) * abs(min_turning_radius), y=turning_cen1.y + sind(turning_cen_new_orien1) * abs(min_turning_radius)), color="red", size=0.1)
                car_plot += geom_point(aes(x=turning_cen2.x + cosd(turning_cen_new_orien2) * abs(min_turning_radius), y=turning_cen2.y + sind(turning_cen_new_orien2) * abs(min_turning_radius)), color="red", size=0.1)
        if not full:
            car_plot += coord_cartesian(xlim=[self.centre.x - 100*10/6.843, self.centre.x + 100*10/6.843], ylim=[self.centre.y - 100, self.centre.y + 100])

        return car_plot

    def plot_line(self, which="speed"): # which est soit "speed", "acc", "steer"

        line_plot = self.plot_car(False, False, True)

        max_x = 0
        if which == "speed":
            for i in range(len(self.line)):
                max_x = max(max_x, abs(self.line[i][2]))
            if len(self.line) > 2000:
                step = math.floor(len(self.line)/500)
            else:
                step = 4

            for i in range(0, len(self.line)-1-step, step):
                line_plot += geom_path(aes(x=[self.line[i][0].x, self.line[i+step][0].x], y=[self.line[i][0].y, self.line[i+step][0].y]), color=color(self.line[i][2], max_x, palette="bi"))


        if which == "acc":
            for i in range(len(self.line)):
                max_x = max(max_x, abs(self.line[i][3]))

            for i in range(0, len(self.line)-5, 4):
                    line_plot += geom_path(aes(x=[self.line[i][0].x, self.line[i+3][0].x], y=[self.line[i][0].y, self.line[i+3][0].y]), color=color(self.line[i][3], max_x, palette="bi"))

        if which == "steer":
            for i in range(len(self.line)):
                max_x = max(max_x, abs(self.line[i][4]))

            for i in range(0, len(self.line)-5, 4):
                    line_plot += geom_path(aes(x=[self.line[i][0].x, self.line[i+3][0].x], y=[self.line[i][0].y, self.line[i+3][0].y]), color=color(self.line[i][4], max_x, palette="bi"))

        return line_plot


