import tabnanny

import plotnine.ggplot
import torch

from Car import *
import _pickle as pickle
from AI import *
from tracks import *
import numpy as np
import random
from copy import deepcopy
from plotnine import *
import time
import matplotlib.pyplot as plt
import keyboard
from pygame_game import *

class Buffer():
    capacity = 1000000
    def __init__(self, capacity=capacity):
        self.capacity = capacity
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def print(self):
        for i in self.buffer:
            print(f'speed:{i[1]}, acc:{i[2]}, steer:{i[3]}, reward:{i[4]}, new_speed:{i[6]}, done:{i[7]}')

    def plot(self, x, y):
        x_list=[]
        y_list=[]
        for buf in self.buffer:
            x_list.append(buf[x])
            y_list.append(buf[y])

        print((ggplot()+geom_point(aes(x=x_list, y=y_list))))

    def plot3d(self, x, y, z, xlabel="z", ylabel="y", zlabel="z"):
        x_list = []
        y_list = []
        z_list = []
        for buf in self.buffer:
            x_list.append(buf[x][12])
            y_list.append(buf[y])
            z_list.append(buf[z])


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        s = ax.scatter(x_list, y_list, z_list, c='r', marker='o', depthshade=False, alpha=0.1)
        s.set_edgecolors = s.set_facecolors = lambda *args: None

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        plt.show()

    def append(self, experience):
        """Add experience to the buffer.
        Args:
            experience: tuple (dists, speed, acc, steer, reward, new_dists, new_speed, done)
        """
        self.buffer.append(deepcopy(experience))
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[1:]

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        dists, speed, acc, steer, rewards, new_dists, new_speed, dones, y = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(dists),
            np.array(speed),
            np.array(acc),
            np.array(steer),
            np.array(rewards),
            np.array(new_dists),
            np.array(new_speed),
            np.array(dones),
            np.array(y)
        ), batch_size

    def split(self, split=0.8):
        batch_size = len(self.buffer)
        n_train = math.floor(batch_size * split)

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        training = indices[:n_train]
        valid = indices[n_train:]

        train_buf = Buffer()
        valid_buf = Buffer()

        for i in training:
            train_buf.append(self.buffer[i])
        for i in valid:
            valid_buf.append(self.buffer[i])

        return train_buf, valid_buf


def linear_loss(output):
    return -torch.mean(output) #negatif car backward, mais on vet monter

def record_pre_train(n=1, reset=False, random_track=True, straight=True):
    if random_track:
        tracks = [Track_generator(straight=straight) for i in range(n)]
    else:
        with open('records_tracks.pkl', 'rb') as outp:
            Recorded_Tracks = pickle.load(outp)
            tracks = [Recorded_Tracks[j][1] for j in range(len(Recorded_Tracks))]


    for iter in range(n):
        if not reset:
            with open('buffer_for_pretrain.pkl', 'rb') as intp:
                buffer = pickle.load(intp)
        else:
            buffer = Buffer()

        reset = False

        track = tracks[iter]

        if straight:
            car = Car(x=random.uniform(track.most_left(), track.most_right()),
                      y=random.uniform(track.most_down(), track.most_up()), speed=np.random.beta(6, 15)*95-5,
                      orien=random.uniform(0, 180))
        else:
            car = Car(x=random.uniform(track.most_left(), track.most_right()),
                      y=random.uniform(track.most_down(), track.most_up()), speed=np.random.beta(6, 15) * 95 - 5,
                      orien=random.uniform(0, 360))

        car.put_on_track(track)
        while (not car.is_inside or car.has_finished or car.has_collapsed):
            if straight:
                car = Car(x=random.uniform(track.most_left(), track.most_right()),
                          y=random.uniform(track.most_down(), track.most_up()), speed=np.random.beta(6, 15)*95-5,
                          orien=random.uniform(0, 180))
            else:
                car = Car(x=random.uniform(track.most_left(), track.most_right()),
                          y=random.uniform(track.most_down(), track.most_up()), speed=np.random.beta(6, 15) * 95 - 5,
                          orien=random.uniform(0, 360))
            car.put_on_track(track)
        record_with_game(car, buffer)

        with open('buffer_for_pretrain.pkl', 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(buffer, outp)
        print(f'saved! {iter+1}/{n}')
    return buffer

def load_pre_train():
    with open('buffer_for_pretrain.pkl', 'rb') as intp:
        buffer = pickle.load(intp)
        print(f'elements in buffer : {len(buffer.buffer)}')
        return buffer

def pre_train(n_epochs, buffer, actor, critic=None):

    actor.train()
    if critic is not None:
        critic.train()

    car = Car()

    optim = torch.optim.Adam(actor.parameters(), lr=0.01)
    if critic is not None:
        optim2 = torch.optim.Adam(critic.parameters(), lr=0.01)


    loss_train = []
    loss_valid = []

    loss_train_critic = []
    loss_valid_critic = []

    training, valid = buffer.split()

    for j in range(n_epochs):
        print(f'Epoch: {j}')
        batch_size = len(training)
        batch, _ = training.sample(batch_size)
        batch_size2 = len(valid)
        batch2, _ = valid.sample(batch_size2)

        optim.zero_grad()
        if critic is not None:
            optim2.zero_grad()

        actions_predites = actor(torch.reshape(torch.FloatTensor(batch[0]), (batch_size, 1, car.radars_count)),
                                 torch.reshape(torch.FloatTensor(batch[1]), (batch_size, 1)))


        actions_predites2 = actor(torch.reshape(torch.FloatTensor(batch2[0]), (batch_size2, 1, car.radars_count)),
                                 torch.reshape(torch.FloatTensor(batch2[1]), (batch_size2, 1)))

        loss = nn.L1Loss()(actions_predites, torch.from_numpy(np.transpose(np.concatenate((batch[2], batch[3]), axis=0).reshape((2, batch_size)))).float())


        loss2 = nn.L1Loss()(actions_predites2, torch.from_numpy(np.transpose(np.concatenate((batch2[2], batch2[3]), axis=0).reshape((2, batch_size2)))).float())

        loss.backward()

        if critic is not None:
            y_predit = critic(torch.reshape(torch.FloatTensor(batch[0]), (batch_size, 1, car.radars_count)),
                                 torch.reshape(torch.FloatTensor(batch[1]), (batch_size, 1)),
                          torch.reshape(torch.FloatTensor(batch[2]), (batch_size, 1)), torch.reshape(torch.FloatTensor(batch[3]), (batch_size, 1)))

            y_predit2 = critic(torch.reshape(torch.FloatTensor(batch2[0]), (batch_size2, 1, car.radars_count)),
                          torch.reshape(torch.FloatTensor(batch2[1]), (batch_size2, 1)),
                          torch.reshape(torch.FloatTensor(batch2[2]), (batch_size2, 1)),
                          torch.reshape(torch.FloatTensor(batch2[3]), (batch_size2, 1)))

            lossC = nn.L1Loss()(y_predit, torch.reshape(torch.FloatTensor(batch[8]), (batch_size, 1)))
            lossC2 = nn.L1Loss()(y_predit2, torch.reshape(torch.FloatTensor(batch2[8]), (batch_size2, 1)))

            lossC.backward()

            lossC2.backward()
            loss_train_critic.append(lossC.item())
            loss_valid_critic.append(lossC2.item())
            optim2.step()

        loss_valid.append(loss2.item())
        loss_train.append(loss.item())

        optim.step()


    print((ggplot()+geom_path(aes(x=seq(1, n_epochs, 1), y=loss_train), color="blue")+geom_path(aes(x=seq(1, n_epochs, 1), y=loss_valid), color="red")))#+
           #geom_path(aes(x=seq(1, n_epochs, 1), y=loss_train_critic), color="green")+geom_path(aes(x=seq(1, n_epochs, 1), y=loss_valid_critic), color="orange")+scale_y_log10()))

    return actor


def train(n_iter, n_epochs, n_cars, n_tracks, actor, critic, batch_size=1):

    tracks = [Track_generator() for i in range(n_tracks)]
    #for track in tracks:
    #    print(track.plot())
    cars = [Car(x=random.uniform(-1, 1), y=random.uniform(-1, 0), orien=90) for i in range(n_cars)]

    reward = [None] * n_cars
    done = [None] * n_cars

    for i in range(n_cars):
        cars[i].put_on_track(tracks[np.random.choice(range(n_tracks), 1, replace=True).item()])

    buffer = Buffer(1000)

    actor_targ = deepcopy(actor)
    critic_targ = deepcopy(critic)
    actor_targ.load_state_dict(actor.state_dict())
    critic_targ.load_state_dict(critic.state_dict())

    names_modules_actor = []
    names_modules_critic = []

    for name, param in actor_targ.named_children():
        #param.requires_grad = False
        names_modules_actor.append(name)
    #print("params theta actor", actor_targ.fc1.bias)

    for name, param in critic_targ.named_children():
        #param.requires_grad = False
        names_modules_critic.append(name)

    #print("params phi critic", critic_targ.fc1.bias)

    actor.train()
    critic.train()
    actor_targ.train()
    critic_targ.train()

    optimC = torch.optim.Adam(critic.parameters(), lr=0.01)
    optimA = torch.optim.Adam(actor.parameters(), lr=0.01)

    loss1 = []
    loss2 = []

    lossC = nn.L1Loss()

    temps1 = 0
    temps2 = 0
    for i in range(1, n_iter+1):
        t1 = time.process_time()
        print(f'Iter {i}')
        states = [(cars[i].radars_dists, cars[i].speed) for i in range(n_cars)]
        actions = actor(torch.FloatTensor([[cars[i].radars_dists] for i in range(n_cars)]), torch.FloatTensor([[cars[i].speed] for i in range(n_cars)])).detach()

        for i in range(n_cars):
            cars[i].move(actions[i][0], actions[i][1])

            if cars[i].has_collapsed:
                reward[i] = -10000
                done[i] = 1
            elif cars[i].has_finished:
                reward[i] = 100000
                done[i] = 1
            elif cars[i].speed < 0:
                reward[i] = -1000
                done[i] = 1
            elif cars[i].check_if_checkpoint_passed():
                reward[i] = 0
                done[i] = 0
            else:
                reward[i] = -1
                done[i] = 0

            buffer.append((states[i][0], states[i][1], cars[i].physics_acc, cars[i].physics_steer, reward[i], cars[i].radars_dists, cars[i].speed, done[i], 0))

        for i in range(n_cars):
            if done[i] == 1:
                cars[i] = Car(x=random.uniform(-1, 1), y=random.uniform(-1, 0), orien=90)
                cars[i].put_on_track(tracks[np.random.choice(range(n_tracks), 1, replace=True).item()])
        t2 = time.process_time()

        for j in range(n_epochs):
            batch, batch_sampled_size = buffer.sample(batch_size)

            mu_theta_targ_s_prime = actor_targ(torch.reshape(torch.FloatTensor(batch[5]), (batch_sampled_size, 1, cars[0].radars_count)), torch.reshape(torch.FloatTensor(batch[6]), (batch_sampled_size, 1)))

            #print("mu_theta prime :", mu_theta_targ_s_prime)
            target_y = torch.reshape(torch.FloatTensor(batch[4]), (batch_sampled_size, 1)) + 0.999 * (1 - torch.reshape(torch.FloatTensor(batch[7]), (batch_sampled_size, 1))) * critic_targ(torch.reshape(torch.FloatTensor(batch[5]), (batch_sampled_size, 1, cars[0].radars_count)), torch.reshape(torch.FloatTensor(batch[6]), (batch_sampled_size, 1)), torch.reshape(mu_theta_targ_s_prime[:, 0], (batch_sampled_size, 1)), torch.reshape(mu_theta_targ_s_prime[:, 1], (batch_sampled_size, 1)))
            #print("target_y", target_y)
            optimC.zero_grad()

            pred_critic = critic(torch.reshape(torch.FloatTensor(batch[0]), (batch_sampled_size, 1, cars[0].radars_count)), torch.reshape(torch.FloatTensor(batch[1]), (batch_sampled_size, 1)), torch.reshape(torch.FloatTensor(batch[2]), (batch_sampled_size, 1)), torch.reshape(torch.FloatTensor(batch[3]), (batch_sampled_size, 1)))
            #print("pred_critic", pred_critic)

            perte = lossC(pred_critic, target_y)
            #print("prediction", pred_critic, "target", target_y, "perte", perte)
            #print(f'Perte : {perte.item()}')
            loss1.append(perte.item())

            perte.backward()
            optimC.step()

            optimA.zero_grad()

            actions_predites = actor(torch.reshape(torch.FloatTensor(batch[0]), (batch_sampled_size, 1, cars[0].radars_count)), torch.reshape(torch.FloatTensor(batch[1]), (batch_sampled_size, 1)))

            reward_loss = linear_loss(critic(torch.reshape(torch.FloatTensor(batch[0]), (batch_sampled_size, 1, cars[0].radars_count)), torch.reshape(torch.FloatTensor(batch[1]), (batch_sampled_size, 1)), torch.reshape(actions_predites[:, 0], (batch_sampled_size, 1)), torch.reshape(actions_predites[:, 1], (batch_sampled_size, 1))))
            #print(f'Reward : {reward_loss}')
            loss2.append(-reward_loss.item())

            reward_loss.backward()

            optimA.step()


            for name in names_modules_actor:
                getattr(actor_targ, name).weight.data = 0.99 * getattr(actor_targ, name).weight.data + 0.01 * getattr(actor, name).weight.data
                getattr(actor_targ, name).bias.data = 0.99 * getattr(actor_targ, name).bias.data + 0.01 * getattr(actor, name).bias.data

            #print("params averaged entre le targ et le modele actor", actor_targ.fc1.bias)
            for name in names_modules_critic:
                getattr(critic_targ, name).weight.data = 0.99 * getattr(critic_targ, name).weight.data + 0.01 * getattr(critic, name).weight.data
                getattr(critic_targ, name).bias.data = 0.99 * getattr(critic_targ, name).bias.data + 0.01 * getattr(critic, name).bias.data


        t3 = time.process_time()
        temps1 += t2-t1
        temps2 += t3-t2
        #print(t2-t1, t3-t2)
    #print(temps1, temps2)
    loss_graphs(loss1, loss2)
    #buffer.print()
    return (actor_targ, critic_targ)


def loss_graphs(loss1, loss2):
    if loss1 is not None:
        print((ggplot() + geom_line(aes(x=range(len(loss1)), y=loss1)) + ggtitle("Perte en précision sur le reward")))
    if loss2 is not None:
        print((ggplot() + geom_line(aes(x=range(len(loss2)), y=loss2)) + ggtitle("Reward prévu")))


def testing(actor, critic, n_cars):
    tracks = [Track_generator() for i in range(n_cars)]
    cars = [Car(x=random.uniform(-1, 1), y=random.uniform(-1, 0), orien=90) for i in range(n_cars)]
    for i in range(n_cars):
        cars[i].put_on_track(tracks[i])

    actor.eval()
    critic.eval()

    actions = actor(torch.reshape(torch.FloatTensor([[cars[i].radars_dists for i in range(n_cars)]]), (n_cars, 1, 24)),
                    torch.reshape(torch.FloatTensor([cars[i].speed for i in range(n_cars)]), (n_cars, 1))).detach()

    for i in range(n_cars):
        while not (cars[i].has_finished or cars[i].has_collapsed or cars[i].speed < 0):
            cars[i].move(actions[i][0].item(), actions[i][1].item())
        #print(cars[i].has_finished , cars[i].has_collapsed , cars[i].speed < 0)

    testing_animation(cars)

def check_models(test_track, actor_reseau, critic_reseau):
    car = Car(x=0, y=0, orien=90)
    actor_reseau.eval()
    critic_reseau.eval()
    car.put_on_track(test_track)

    x = seq(-5, 50, 1)
    y1 = [None] * len(x)
    y2 = [None] * len(x)

    for i in range(len(x)):
        car.speed = x[i]

        actions = actor_reseau(torch.FloatTensor([[car.radars_dists]]),
                               torch.FloatTensor([[car.speed]])).detach()
        # print(actions)
        critic = critic_reseau(torch.FloatTensor([[car.radars_dists]]),
                               torch.FloatTensor([[car.speed]]),
                               torch.FloatTensor([[actions.tolist()[0][0]]]),
                               torch.FloatTensor([[actions.tolist()[0][1]]])).detach()
        #print(critic)
        y1[i] = critic.squeeze().tolist()
        y2[i] = actions.squeeze().tolist()[0]

    print((ggplot() + geom_path(aes(x=x, y=y1)) + geom_path(aes(x=x, y=y2), color="blue") + xlab("speed") + ylab("Q/acc")))