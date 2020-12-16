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
TIMES = 1
EPISODS = 1000

GAMMA = 0.15 # Discounted Reward

ALPHA_W = 0.0001 # W Learning Rate
ALPHA_U = 0.0001  # U Learning Rate

##########################################################################################################################

def run():

    env = gym.make("MountainCarContinuous-v0").env

    w = np.random.rand(VECTOR_SIZE)
    u = np.random.rand(VECTOR_SIZE)

    state = None

    for ep in range(EPISODS):

        done = False
        score = 0
        state = env.reset()
        
        while(not done):

            featured_state = make_features_vec (state[0] , state[1])

            # compute statistics
            mu = vector_mul (featured_state,u)
            sig = 0.5

            # choose action
            action = [np.random.normal(mu,sig)]
            
            # step
            next_state,reward,done,_ = env.step(action)
            score += reward

            # render
            # if ep > 10:
            #     env.render()

            # compute TD error
            featured_next_state = make_features_vec (next_state[0] , next_state[1])
            v_state = vector_mul (featured_state,w)
            v_next_state = vector_mul (featured_next_state,w)
            td_target = reward + GAMMA*v_next_state # TD target
            delta = td_target - v_state # TD error

            # update critic
            # w += alpha_w *  delta * X(s)
            w = add_vectors( w , scalar_vector_mul(ALPHA_W  * delta , featured_state))
            
            # update actor
            # u += alpha_u * gradient(Log(s,a))
            gradient_u = scalar_vector_mul ( (action[0] - mu)/sig**2 , featured_state)
            u = add_vectors( u , scalar_vector_mul(ALPHA_U * delta, gradient_u) )

            # update state
            state = next_state
        
        print(str(score))

    env.close()

##########################################################################################################################
#################################################### Functions ###########################################################
##########################################################################################################################
    
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
##########################################################################################################################
for i in range(TIMES):
    print("*"*100)
    run()
##########################################################################################################################

