import numpy as np
import random# setting yperparameters
import gym
import statistics



def learn():
    env = gym.make("Taxi-v3").env
    alpha = 0.25
    gamma = 0.95
    epsilon = 0.25
    rounds = 1
    total_steps = 100000
    list_of_list_of_rewards = []

    for i in range(rounds):
        state = env.reset()
        steps = 0
        done = False
        done_times = 0
        q_table = np.zeros((500, 6))
        

        while steps <= total_steps:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
    
            next_state, reward, done, info = env.step(action) 
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)
            state = next_state
            steps += 1
            if steps % 10000 == 0:
                list_of_rewards = []
                for j in range(0, 9):
                    policy = check_policy(q_table)
                    list_of_rewards.append(policy/100)
                list_of_list_of_rewards.append(list_of_rewards)
            if done:
                state = env.reset()
                done_times += 1
                done = False
        print("Done " + str(done_times) + " runs.")
        print("Episode " + str(i+1) + " done.")
    print_nicely(graph(list_of_list_of_rewards))

def check_policy(q_table):
    env = gym.make("Taxi-v3").env
    gamma = 0.95
    total_steps = 100
    power_of = 0
    total_reward = 0
    current = 0

    while current < total_steps:
        state = env.reset()
        for i in range(current, total_steps):
            action = np.argmax(q_table[state])
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += (gamma ** power_of) * reward
            power_of += 1
            current += 1
            if done:
                break

    return total_reward

def graph(list_of_list_of_rewards):
    list_of_avgs = []
    list_of_sd = []
    for list_of_rewards in list_of_list_of_rewards:
        list_of_avgs.append(np.mean(list_of_rewards))
        list_of_sd.append(statistics.stdev(list_of_rewards))
    return (list_of_avgs, list_of_sd)

def print_nicely(tup_of_items):
    x = 10000
    for i in range(len(tup_of_items[0])):
        print("(x: " + str(x) + ", y: " + str(tup_of_items[0][i]) + ", s: " + str(tup_of_items[1][i]) + ")")
        x += 10000

learn()