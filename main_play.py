from tests import *

import torch
from train import *
from AI import *
from pygame_game import *
from torchviz import make_dot

if __name__ == "__main__":

    track_number = int(input("Numéro de piste?"))
    play_tracks(track_number)

    #track_generated = Track_generator(straight=False)
    #print(track_generated.plot())

    #record_pre_train(n=30, reset=False, random_track=True, straight=True)








