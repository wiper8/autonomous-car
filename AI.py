

# pour controler le char
from Car import *
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(layer, biais = True):
    nn.init.kaiming_uniform_(layer.weight)
    if biais:
      layer.bias.data.fill_(0.05)
    return layer


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        channels1 = 8
        channels2 = 8
        channels3 = 16
        channels4 = 32
        kernel = 3
        self.conv1 = nn.Conv1d(1, channels1, 5, padding="valid") # TODO mettre du weight sharing vu que c'est symétrique
        self.conv2 = nn.Conv1d(channels1, channels2, kernel, padding="valid")
        self.conv3 = nn.Conv1d(channels2, channels3, kernel, padding="valid")
        self.conv4 = nn.Conv1d(channels3, channels4, kernel, padding="valid")
        self.batch1 = nn.BatchNorm1d(channels1)
        self.batch2 = nn.BatchNorm1d(channels2)
        self.batch3 = nn.BatchNorm1d(channels3)
        self.batch4 = nn.BatchNorm1d(channels4)

        self.fc1 = nn.Linear(channels4 + 1, 32) #attention à changer si on ajoute des couches
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 8)
        self.fc7 = nn.Linear(8, 8)
        self.fc8 = nn.Linear(8, 2) #essayer d'ajouter une couche identité pour l'acc et sin ou cos pour l'angle
        #attention de changer la couche des biais par défault dans main

        initialize_weights(self.conv1)
        initialize_weights(self.conv2)
        initialize_weights(self.conv3)
        initialize_weights(self.conv4)
        initialize_weights(self.fc1)
        initialize_weights(self.fc2)
        initialize_weights(self.fc3)
        initialize_weights(self.fc4)
        initialize_weights(self.fc5)
        initialize_weights(self.fc6)
        initialize_weights(self.fc7)
        initialize_weights(self.fc8)


    def forward(self, radars, speed): #radars : tensor (N, 1, 24), speed : tensor (N, 1)

        radars_features = F.avg_pool1d(F.relu(self.batch4(self.conv4(F.relu(self.batch3(self.conv3(F.relu(self.batch2(self.conv2(F.relu(self.batch1(self.conv1(radars)))))))))))), 24 - (5-1) - (3-1) * 3)

        intermediate_inputs = torch.cat((torch.flatten(radars_features, start_dim=1), speed), 1)

        return torch.clamp(self.fc8(F.relu(self.fc7(F.relu(self.fc6(F.relu(self.fc5(F.relu(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(intermediate_inputs))))))))))))))), torch.tensor([-10.868, -21.96]), torch.tensor([8.944, 21.96]))


    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        channels1 = 8
        channels2 = 16
        channels3 = 32
        channels4 = 32
        channels5 = 32
        kernel = 3
        self.conv1 = nn.Conv1d(1, channels1, kernel, padding="valid") # mettre du weight sharing vu que c'est symétrique
        self.conv2 = nn.Conv1d(channels1, channels2, kernel, padding="valid")
        self.conv3 = nn.Conv1d(channels2, channels3, kernel, padding="valid")
        self.conv4 = nn.Conv1d(channels3, channels4, kernel, padding="valid")
        self.conv5 = nn.Conv1d(channels4, channels5, kernel, padding="valid")
        self.batch1 = nn.BatchNorm1d(channels1)
        self.batch2 = nn.BatchNorm1d(channels2)
        self.batch3 = nn.BatchNorm1d(channels3)
        self.batch4 = nn.BatchNorm1d(channels4)
        self.batch5 = nn.BatchNorm1d(channels5)
        hidden = 16
        self.fc1 = nn.Linear(channels5 + 1 + 2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, 1)

    def forward(self, radars, speed, acc, steer):

        radars_features = F.avg_pool1d(F.relu(self.batch5(self.conv5(F.relu(self.batch4(self.conv4(F.relu(self.batch3(self.conv3(F.relu(self.batch2(self.conv2(F.relu(self.batch1(self.conv1(radars))))))))))))))), 24 - (3-1) * 5)

        intermediate_inputs = torch.cat((torch.flatten(radars_features, 1), speed, acc, steer), dim=1)

        return self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(intermediate_inputs)))))))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)