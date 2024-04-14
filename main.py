from tests import *

import torch
from train import *
from AI import *
from pygame_game import *
from torchviz import make_dot

if __name__ == "__main__":

    # testing
    #test_deg()
    #test_orientation()
    #test_shapes()
    #test_track()
    #test_car()

    #train
    actor_reseau = Actor()
    print(actor_reseau.count_params())
    print(actor_reseau.count_trainable_params())

    critic_reseau = Critic()
    print(critic_reseau.count_params())
    print(critic_reseau.count_trainable_params())

    #save_all_tracks() #attention! ça erase tous les records

    #track_number = int(input("Numéro de piste?"))
    #play_tracks(track_number)

    #track_generated = Track_generator(straight=False)
    #print(track_generated.plot())

    record_pre_train(n=30, reset=False, random_track=True, straight=True)
    pre_trained_buffer = load_pre_train()


    #pre_trained_buffer.plot(1, 4)  #speed vs reward
    pre_trained_buffer.plot3d(0, 1, 2, "top_radar", "speed", "acc")  # top radar vs acc vs speed
    pre_trained_buffer.plot3d(0, 1, 3, "top_radar", "speed", "steer")

    actor_reseau = pre_train(1000, pre_trained_buffer, actor_reseau, None) #critic_reseau

    test_track = Track_generator()

    #actor_reseau, critic_reseau = train(1000, 10, n_cars=8, n_tracks=5, actor=actor_reseau, critic=critic_reseau, batch_size=64)# epochs x batch_size ~<= buffer capacity

    actor_reseau.eval()
    critic_reseau.eval()

    #test
    testing(actor_reseau, critic_reseau, 5)

    #check sens des modèles selon speed
    #check_models(test_track, actor_reseau, critic_reseau)

    with open('actor.pkl', 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(actor_reseau, outp)

    with open('critic.pkl', 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(critic_reseau, outp)







