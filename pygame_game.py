import numpy as np
import pygame
import time
from Car import *
from tracks import *
import _pickle as pickle

def meters_to_pixels(car, pt, meters_limitsx=None, meters_limitsy=None, pixels_limitsx=(0, 1023), pixels_limitsy=(0, 700)):

    if meters_limitsx is None:
        meters_limitsx = (car.track.most_left(), car.track.most_right())
    if meters_limitsy is None:
        meters_limitsy = (car.track.most_down(), car.track.most_up())

    assert meters_limitsx[0] != meters_limitsx[1]
    assert meters_limitsy[0] != meters_limitsy[1]
    assert pixels_limitsx[0] != pixels_limitsx[1]
    assert pixels_limitsy[0] != pixels_limitsy[1]

    a_s = ((pixels_limitsx[1] - pixels_limitsx[0]) / (meters_limitsx[1] - meters_limitsx[0]),
           (pixels_limitsy[1] - pixels_limitsy[0]) / (meters_limitsy[1] - meters_limitsy[0]))
    if a_s[0] < a_s[1]:
        a = a_s[0]
        b1 = pixels_limitsx[0] - a * meters_limitsx[0]
        b2 = (pixels_limitsy[1] + pixels_limitsy[0]) / 2 + a * (meters_limitsy[1] + meters_limitsy[0]) / 2
        return (a * pt[0] + b1, -a * pt[1] + b2)
    else:
        a = -a_s[1]
        b2 = pixels_limitsy[0] - a * meters_limitsy[1]
        b1 = (pixels_limitsx[1] + pixels_limitsx[0]) / 2 + a * (meters_limitsx[1] + meters_limitsx[0]) / 2
        return (-a * pt[0] + b1, a * pt[1] + b2)

def display_car_track(car, window, color="red"):
    # pygame.draw.line(surface=window, color="gray", start_pos = meters_to_pixels(car, (car.track.most_left(), car.track.most_down())), end_pos = meters_to_pixels(car, (car.track.most_left(), car.track.most_up())))
    # pygame.draw.line(surface=window, color="gray", start_pos = meters_to_pixels(car, (car.track.most_right(), car.track.most_down())), end_pos = meters_to_pixels(car, (car.track.most_right(), car.track.most_up())))

    pygame.draw.circle(surface=window, color="green", center=meters_to_pixels(car, (car.centre.x, car.centre.y)),
                           radius=2)

    pygame.draw.line(surface=window, color=color,
                         start_pos=meters_to_pixels(car, (car.outer_walls[0].point1.x, car.outer_walls[0].point1.y)),
                         end_pos=meters_to_pixels(car, (car.outer_walls[0].point2.x, car.outer_walls[0].point2.y)),
                         width=2)
    pygame.draw.line(surface=window, color=color,
                         start_pos=meters_to_pixels(car, (car.outer_walls[1].point1.x, car.outer_walls[1].point1.y)),
                         end_pos=meters_to_pixels(car, (car.outer_walls[1].point2.x, car.outer_walls[1].point2.y)),
                         width=2)
    pygame.draw.line(surface=window, color=color,
                         start_pos=meters_to_pixels(car, (car.outer_walls[2].point1.x, car.outer_walls[2].point1.y)),
                         end_pos=meters_to_pixels(car, (car.outer_walls[2].point2.x, car.outer_walls[2].point2.y)),
                         width=2)
    pygame.draw.line(surface=window, color=color,
                         start_pos=meters_to_pixels(car, (car.outer_walls[3].point1.x, car.outer_walls[3].point1.y)),
                         end_pos=meters_to_pixels(car, (car.outer_walls[3].point2.x, car.outer_walls[3].point2.y)),
                         width=2)

    for wall in car.track.walls:
        pygame.draw.line(window, color="white", start_pos=meters_to_pixels(car, (wall.point1.x, wall.point1.y)),
                             end_pos=meters_to_pixels(car, (wall.point2.x, wall.point2.y)), width=2)
    pygame.draw.line(window, color="red",
                         start_pos=meters_to_pixels(car, (car.track.finish.point1.x, car.track.finish.point1.y)),
                         end_pos=meters_to_pixels(car, (car.track.finish.point2.x, car.track.finish.point2.y)), width=2)


def record_with_game(car, buffer):

    window = pygame.display.set_mode((1023, 700))
    pygame.init()

    FPS = 1/car.t_eps

    run = True
    first = 0
    clock = pygame.time.Clock()
    y=0
    myfont = pygame.font.SysFont('arial', 45)

    dims = pygame.display.get_surface().get_size()

    gas_range = 0
    steering_range = 0

    while run:

        clock.tick(FPS)

        pygame.display.update()
        window.fill("black")

        display_car_track(car, window)

        if first == 0:
            for count_down in [5, 4, 3, 2, 1, "GO!"]:
                pygame.time.wait(1000)
                window.fill("black")
                display_car_track(car, window)
                textsurface6 = myfont.render(f'{count_down}', False, "white")
                window.blit(textsurface6, (dims[0] / 2, dims[1] / 2))
                textsurface = myfont.render(f'speed: {round(car.speed, 1)}', False, "white")
                textsurface2 = myfont.render(f'orien: {round(car.orien, 1)}', False, "white")
                window.blit(textsurface2, (dims[0] / 8, dims[1] / 2 + 45))
                window.blit(textsurface, (dims[0] / 8, dims[1] / 2))
                pygame.display.update()

        first += 1

        textsurface = myfont.render(f'speed: {round(car.speed, 1)}', False, "white")
        textsurface2 = myfont.render(f'orien: {round(car.orien, 1)}', False, "white")

        window.blit(textsurface2, (dims[0] / 8, dims[1] / 2 + 45))
        window.blit(textsurface, (dims[0] / 8, dims[1] / 2))


        if car.has_collapsed:
            reward = -1000
            done = 1
        elif car.has_finished:
            reward = 10000
            done = 1
        elif car.speed < 0:
            reward = -100
            done = 1
        else:
            reward = -1
            done = 0
        y = 0.999 * y + reward
        buffer.append((car.radars_dists, car.speed, car.physics_acc, car.physics_steer, reward, None, None, done, reward + 0.999 * (1 - done) * y))

        if done == 1:
            run = False
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        acc = 0
        steer = 0
        keys = pygame.key.get_pressed()

        if keys[pygame.K_RIGHT]:
            steer += -22
            steering_range = min(0, max(steering_range - 1, -5))
        if keys[pygame.K_LEFT]:
            steer += 22
            steering_range = max(0, min(steering_range + 1, 5))
        if keys[pygame.K_DOWN]:
            acc += -11
            gas_range = min(0, max(gas_range - 1, -5))
        if keys[pygame.K_UP]:
            acc += 10
            gas_range = max(0, min(gas_range + 1, 5))

        if not keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
            steering_range -= sign(steering_range)

        if not keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
            gas_range -= sign(gas_range)

        car.move(acc * abs(gas_range) / 5, steer * abs(steering_range) / 5)

    pygame.quit()


def testing_animation(cars):

    FPS = 1 / cars[0].t_eps

    for c in range(len(cars)):
        window = pygame.display.set_mode((1023, 700))
        pygame.init()

        car = Car(x=cars[c].line[0][0].x, y=cars[c].line[0][0].y, orien=cars[c].line[0][1], speed=cars[c].line[0][2])
        car.put_on_track(cars[c].track)
        car.line = cars[c].line
        itera = 0

        run = True
        first = 0
        clock = pygame.time.Clock()

        myfont = pygame.font.SysFont('arial', 45)

        dims = pygame.display.get_surface().get_size()

        while run:

            clock.tick(FPS)

            pygame.display.update()
            window.fill("black")

            display_car_track(car, window)

            first += 1
            textsurface = myfont.render(f'speed: {round(car.speed, 1)}', False, "white")
            textsurface2 = myfont.render(f'orien: {round(car.orien, 1)}', False, "white")

            window.blit(textsurface2, (dims[0] / 8, dims[1] / 2 + 45))
            window.blit(textsurface, (dims[0] / 8, dims[1] / 2))


            if car.has_collapsed:
                done = 1
            elif car.has_finished:
                done = 1
            elif car.speed < 0:
                done = 1
            else:
                done = 0

            if done == 1:
                run = False
                break

            itera += 1
            car = Car(x=cars[c].line[itera][0].x, y=cars[c].line[itera][0].y, orien=cars[c].line[itera][1],
                      speed=cars[c].line[itera][2])
            car.put_on_track(cars[c].track)
            #car.move(car.line[itera][3], car.line[itera][4])

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

        pygame.quit()


def save_all_tracks():
    # tuple (temps record, Track, line record)
    Recorded_Tracks = [(np.Inf, Track(Line(Point(25, 0), Point(35, 0)),
                                      [Point(-5, -2), Point(-5, -2), Point(-5, -2), Point(-5, -2), Point(-5, -2),
                                       Point(-5, 100), Point(35, 100), Point(35, 0)], Line(Point(-5, -2), Point(5, -2)),
                                      [Point(5, -2), Point(5, 80), Point(5 + 10 - cosd(45) * 10, 80 + sind(45) * 10),
                                       Point(15, 90), Point(15 + cosd(45) * 10, 80 + sind(45) * 10), Point(25, 80),
                                       Point(25, 0)]), None),
                       (np.Inf, Track(Line(Point(115, 120), Point(115, 110)),
                                      [Point(-5, -2), Point(-5, -2), Point(-5, -2), Point(-5, 120), Point(115, 120)],
                                      Line(Point(-5, -2), Point(5, -2)),
                                      [Point(5, -2), Point(5, 100), Point(5 + 10 - cosd(45) * 10, 100 + sind(45) * 10),
                                       Point(15, 110), Point(115, 110)]), None),
                       (np.Inf, Track(Line(Point(55, 100), Point(65, 100)),
                                      [Point(-5, -2), Point(-5, 100), Point(35, 100), Point(35, 20), Point(45-cosd(45)*10, 20-sind(45)*10),
                                       Point(45, 10), Point(45+cosd(45)*10, 20-sind(45)*10), Point(55, 20), Point(55, 100)],
                                      Line(Point(-5, -2), Point(5, -2)),
                                      [Point(5, -2), Point(5, 80), Point(15 - cosd(45) * 10, 80 + sind(45) * 10),
                                       Point(15, 90), Point(15 + cosd(45) * 10, 80 + sind(45) * 10), Point(25, 80), Point(25, 0), Point(65, 0), Point(65, 100)]), None),
                       (np.Inf, Track(Line(Point(215, 70), Point(215, 60)),
                                      [Point(-5, -2), Point(-5, -2), Point(-5, -2), Point(-5, 70), Point(215, 70)],
                                      Line(Point(-5, -2), Point(5, -2)),
                                      [Point(5, -2), Point(5, 50), Point(15 - cosd(45) * 10, 50 + sind(45) * 10),
                                       Point(15, 60), Point(215, 60)]), None),
                       (np.Inf, Track(Line(Point(35, 220), Point(35, 210)),
                                      [Point(-5, -2), Point(-5, -2), Point(-5, -2), Point(-5, 220), Point(35, 220)],
                                      Line(Point(-5, -2), Point(5, -2)),
                                      [Point(5, -2), Point(5, 200), Point(15 - cosd(45) * 10, 200 + sind(45) * 10),
                                       Point(15, 210), Point(35, 210)]), None),
                       (np.Inf, Track(Line(Point(-676.58, 97.3), Point(-667, 100.62)),
                                      [Point(-5, -2), Point(-1.95, 7.89), Point(14.94, 11.72), Point(34.53, -1.03),
                                       Point(48, -27.9), Point(50.4, -64.5), Point(39.2, -106), Point(13.88, -147.9),
                                       Point(-24.8, -186), Point(-75.15, -217), Point(-134.9, -238.6),
                                       Point(-201.5, -248.8), Point(-272.17, -246.6), Point(-344.3, -231.5), Point(-415.2, -203.6),
                                       Point(-482.5, -163.4), Point(-544, -111.9), Point(-597.9, -50.2), Point(-642.5, 20.05), Point(-676.58, 97.3)],
                                      Line(Point(-5, -2), Point(5, -2)),
                                      [Point(5, -2), Point(5, 0.03), Point(5.44, 1.15), Point(12.69, 1.97),
                                       Point(26.96, -7.57), Point(38.37, -30.4), Point(40.48, -63.48), Point(30.03, -102),
                                       Point(6.05, -141.7), Point(-30.97, -178), Point(-79.5, -208), Point(-137.4, -228.9),
                                       Point(-202, -238.8), Point(-270.9, -236.6), Point(-341.4, -222), Point(-410.8, -194.6),
                                       Point(-477, -155), Point(-537, -104.7), Point(-633.7, 24.77), Point(-667, 100.62)]), None),
                       (np.Inf, Track(Line(Point(20, 250), Point(80, 250)),
                                      [Point(-10, -2), Point(-10, -2), Point(-10, 40), Point(-15, 80), Point(-40, 110),
                                       Point(-58, 126), Point(-75, 130), Point(-170, 90), Point(-150, 95),
                                       Point(-100, 100), Point(-80, 80), Point(-75, -10),
                                       Point(20, -10), Point(20, 250)],
                                      Line(Point(-10, -2), Point(10, -2)),
                                      [Point(10, -2), Point(10, 40), Point(5, 85), Point(-25, 130),
                                       Point(-80, 150), Point(-200, 120), Point(-200, 50),
                                       Point(-170, 50),
                                       Point(-140, 60), Point(-120, 50), Point(-120, -50),
                                       Point(100, -50),
                                       Point(100, 100), Point(80, 250)]), None)
                       ]

    with open('records_tracks.pkl', 'wb') as outp:
        pickle.dump(Recorded_Tracks, outp)
        return Recorded_Tracks

def add_new_track(track):
    with open('records_tracks.pkl', 'rb') as inpt:
        Recorded_Tracks = pickle.load(inpt)
    Recorded_Tracks.append((np.Inf, track, None))

    with open('records_tracks.pkl', 'wb') as outp:
        pickle.dump(Recorded_Tracks, outp)

def play_tracks(track_number):

    with open('records_tracks.pkl', 'rb') as outp:
        Recorded_Tracks = pickle.load(outp)


    window = pygame.display.set_mode((1023, 700))
    pygame.init()
    car = Car(x = 0, y = 0, orien=90)
    car.put_on_track(Recorded_Tracks[track_number][1])
    record = Recorded_Tracks[track_number][0]
    line = Recorded_Tracks[track_number][2]
    if line is not None:
        car_record = Car(x=0, y=0, orien=90)
        car_record.put_on_track(Recorded_Tracks[track_number][1])

    itera = 0
    FPS = 1 / car.t_eps

    run = True
    first = 0
    clock = pygame.time.Clock()

    myfont = pygame.font.SysFont('arial', 45)

    dims = pygame.display.get_surface().get_size()

    gas_range = 0
    steering_range = 0
    t = 0

    while run:

        clock.tick(FPS)

        pygame.display.update()
        window.fill("black")

        if line is not None:
            display_car_track(car_record, window, "gray")
        display_car_track(car, window)

        if first == 0:
            for count_down in [5, 4, 3, 2, 1, "GO!"]:
                pygame.time.wait(1000)
                window.fill("black")
                display_car_track(car, window)
                textsurface6 = myfont.render(f'{count_down}', False, "white")
                window.blit(textsurface6, (dims[0] / 2, dims[1] / 2))
                pygame.display.update()



        first += 1
        textsurface = myfont.render(f'speed: {round(car.speed, 1)}', False, "white")
        textsurface2 = myfont.render(f'orien: {round(car.orien, 1)}', False, "white")

        window.blit(textsurface2, (dims[0] / 8, dims[1] / 2 + 45))
        window.blit(textsurface, (dims[0] / 8, dims[1] / 2))

        #textsurface3 = myfont.render(f'gas_range: {gas_range}', False, "white")
        #window.blit(textsurface3, (dims[0] / 8, dims[1] / 2 + 90))
        #textsurface4 = myfont.render(f'steering_range: {steering_range}', False, "white")
        #window.blit(textsurface4, (dims[0] / 8, dims[1] / 2 + 135))

        t += car.t_eps

        if car.has_collapsed:
            done = 1
        elif car.has_finished:
            done = 1
            if t < record:
                textsurface5 = myfont.render(f'New record! {math.floor(t*100)/100}', False, "white")
                window.blit(textsurface5, (dims[0] / 2, dims[1] / 2))
                pygame.display.update()
                Recorded_Tracks[track_number] = (t, Recorded_Tracks[track_number][1], car.line)
                with open('records_tracks.pkl', 'wb') as outp:
                    pickle.dump(Recorded_Tracks, outp)
                pygame.time.wait(1000)
        elif car.speed < 0:
            done = 1
        else:
            done = 0

        if done == 1:
            run = False
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        acc = 0
        steer = 0
        keys = pygame.key.get_pressed()

        if keys[pygame.K_RIGHT]:
            steer += -22
            steering_range = min(0, max(steering_range - 1, -5))
        if keys[pygame.K_LEFT]:
            steer += 22
            steering_range = max(0, min(steering_range + 1, 5))
        if keys[pygame.K_DOWN]:
            acc += -11
            gas_range = min(0, max(gas_range - 1, -5))
        if keys[pygame.K_UP]:
            acc += 10
            gas_range = max(0, min(gas_range + 1, 5))

        if not keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
            steering_range -= sign(steering_range)

        if not keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
            gas_range -= sign(gas_range)


        if line is not None:
            itera += 1
            if itera <= len(line)-1:
                car_record = Car(x=line[itera][0].x, y=line[itera][0].y, orien=line[itera][1],
                          speed=line[itera][2])
                car_record.put_on_track(Recorded_Tracks[track_number][1])
                #car_record.move(car_record.line[itera][3], car_record.line[itera][4])

        car.move(acc * abs(gas_range) / 5, steer * abs(steering_range) / 5)

    pygame.quit()