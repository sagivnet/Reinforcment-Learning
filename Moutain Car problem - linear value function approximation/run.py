##########################################################################################################################
import numpy as np
import gym
import random# setting yperparameters
import statistics
import math
import os
##########################################################################################################################
verbose = False
VECTOR_SIZE = 16
##########################################################################################################################
MAX_STEPS = 50000
EPISODS = 500
EPSILON_UPDATE = 400
EPSILON_DECAY = 1.4
EPSILON_DOWN_THRESHOLD = 0.01
##########################################################################################################################

def run():

    # Learning Configurations
    epsilon = 0.95
    alpha = 0.65
    gamma = 0.6

    env = gym.make("MountainCar-v0").env

    total_steps = 0
    learning_ended = False

    w_left = [0]*VECTOR_SIZE
    w_stay = [0]*VECTOR_SIZE
    w_right = [0]*VECTOR_SIZE


    dones = []
    q_max = 0

    print("Epsilon = " + str(epsilon))
    print("Alpha = " + str(alpha))
    print("Gamma = " + str(gamma))
    print("\nEPSILON_UPDATE = " + str(EPSILON_UPDATE))
    print("EPSILON_DECAY = " + str(EPSILON_DECAY))

    print("\nChoosing:")
    print("           "+str(int((1-epsilon)*100))+"% from experience")
    print("           "+str(int((epsilon)*100))+"% randomly")

    for ep in range(EPISODS):
        q_max = 0
        state = env.reset()
        action_stats = [0, 0, 0]
        
        for step in range (MAX_STEPS):
            # Random action
            if random.uniform(0, 1) < epsilon:      
                action = env.action_space.sample()
                if action == 0:
                    w = w_left
                elif action == 1:
                    w = w_stay
                else:
                    w = w_right

                q_max = calc_q(state, action, w)
            # Choose action
            else:
                (action, q_max) = find_q_max(state, w_left, w_stay, w_right)

                # For Printing
                if action == 0:
                    w = w_left
                elif action == 1:
                    w = w_stay
                else:
                    w = w_right
            print_stuff(action, q_max, w)

            # STATISTICS
            action_stats[action] += 1

            # Take a step
            next_state, reward, done, info = env.step(action)

            # Predict Q Pi
            q_pi = reward + gamma * find_q_max(next_state, w_left, w_stay, w_right)[1]

            # Update w
            if action == 0:
                w_left = add_vectors(w_left, scalar_vector_mul(alpha * (q_pi - q_max), make_features_vec(state[0], state[1])))

            elif action == 1:
                w_stay = add_vectors(w_stay, scalar_vector_mul(alpha * (q_pi - q_max), make_features_vec(state[0], state[1])))

            else:
                w_right = add_vectors(w_right, scalar_vector_mul(alpha * (q_pi - q_max), make_features_vec(state[0], state[1])))

            state = next_state
            env.render()

            # Epsilon Update
            if step % EPSILON_UPDATE == 0 and step != 0:
                os.system('cls')
                
                # STATISTICS
                print("Statistics:\n")
                print("Alpha = " + str(alpha))
                print("Gamma = " + str(gamma)+"\n")
                print("Epsilon = " + str(epsilon))    
                print("\nEPSILON_UPDATE = " + str(EPSILON_UPDATE))
                print("EPSILON_DECAY = " + str(EPSILON_DECAY))

                sum_of_stats = 0
                for stat in action_stats:
                    sum_of_stats += stat
                for i in range(len(action_stats)):
                    output = ""
                    if i == 0:
                        output += "Left: "
                    elif i == 1:
                        output += "Stay: "
                    else:
                        output += "Right: "
                    precentage = (float(action_stats[i])/sum_of_stats) * 100
                    output += str(int(precentage)) + "%"
                    print(output)

                # update
                if(learning_ended == False):
                    epsilon = epsilon**EPSILON_DECAY
                    if epsilon < EPSILON_DOWN_THRESHOLD:
                        epsilon = EPSILON_DOWN_THRESHOLD
                        learning_ended = True
                        print("\n### Epsilon Is At Minimum ###\n")
                    else:
                        print("\nNew Epsilon = " + str(epsilon))
                        print("Choosing:")
                        print("           "+str(int((1-epsilon)*100))+"% from experience")
                        print("           "+str(int((epsilon)*100))+"% randomly")
                else:
                    print("\nEpsilon is at minimum: "+str(epsilon))
                    print("Choosing "+str(int((1-epsilon)*100))+"% from experience")

                print("\nWins:")
                print(dones)

            # Won a game
            if done:
                dones.append(step)
                print("")
                total_steps += step
                if verbose:
                    print ("!! Won the "+ str(ep+1)+ " Game! with Epsilon: "+str(epsilon) + " in #"+str(step) +"# timesteps in timestep: "+ str(total_steps)) 
                    raw_input() # STOP
                break

    # Final Print
    print("The number of steps per trial:")

    
    print(dones)

    score = 0
    for ts in dones:
        score += ts
    print("\nScore: " + str(score))

##########################################################################################################################
#################################################### Functions ###########################################################
##########################################################################################################################

def find_q_max(state, w_left, w_stay, w_right):

    # Action = left
    q_left = calc_q(state, 0, w_left)

    # Action = stay
    q_stay = calc_q(state, 1, w_stay)

    # Action = right
    q_right = calc_q(state, 2, w_right)

    q_array = [q_left, q_stay, q_right]
    
    # Find Q-max
    q_max = max(q_array)
    action = q_array.index(q_max)

    return (action, q_max)

##########################################################################################################################

def calc_q(state, a, w):

    pos = state[0]
    vel = state[1]
    X = make_features_vec(pos, vel)

    return vector_mul(X, w)
    
##########################################################################################################################

def make_features_vec(pos, vel):

    tile_counter = 0
    pos_tilings = make_tilings(-1.3, 0.7)
    vel_tilings = make_tilings(-0.08, 0.08)

    p = []
    v = []

    for tiling in pos_tilings:
        for tile in tiling:
            tile_counter += 1
            if pos >= tile[0] and pos < tile[1]:
                p.append(float(tile_counter)/100)

    tile_counter = 0
    for tiling in vel_tilings:
        for tile in tiling:
            tile_counter += 1
            if vel >= tile[0] and vel < tile[1]:
                v.append(float(tile_counter)/100)

    return p + v

##########################################################################################################################

def make_tilings(minX, maxX):

    tiles_amount = 8
    tilings_amount = 8
    tile_len = (maxX - minX)/tiles_amount
    tilings = []
    offset = 0 # In the beginning no offset

    for i in range(tilings_amount):
        tiles = []
        last_tile_len = minX + offset
        for j in range(tiles_amount):
            this_tile_len = last_tile_len + tile_len
            tiles.append((last_tile_len, this_tile_len))
            last_tile_len = this_tile_len
        tilings.append(tiles)
        offset = tile_len/8

    return tilings

##########################################################################################################################
#################################################### Help Functions ######################################################
##########################################################################################################################

def vector_mul(a, b):

    if len(a) != len(b):
        print("Error vector_mul: length")
        print(a)
        print(b)
        return

    output = 0
    for i in range(len(a)):
        output += a[i] * b[i]

    return output

##########################################################################################################################

def scalar_vector_mul(s, v):

    output = []

    for num in v:
        output.append(s*num)

    return output

##########################################################################################################################

def add_vectors(v1, v2):

    if len(v1) != len(v2):
        print("Error add_vectors: length")
        return

    output = []

    for i in range(len(v1)):
        output.append(v1[i] + v2[i])

    return output

##########################################################################################################################

def sub_vectors(v1, v2):

    if len(v1) != len(v2):
        print("Error sub_vectors: length")
        return

    output = []

    for i in range(len(v1)):
        output.append(v1[i] - v2[i])

    return output

##########################################################################################################################

def print_stuff(action, q_max, w):

    if not verbose:
        return

    output = ""

    if action == 0:
        output += "<<<<, "
    elif action == 1:
        output += "||||, "
    else:
        output += ">>>>, "
    
    output += "q_max: " + str(q_max) + ", "
    output += "w = " + str(w)

    print(output)

##########################################################################################################################
run()
##########################################################################################################################

