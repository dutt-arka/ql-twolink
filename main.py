"""
Two link manipulator using Pyglet and Numpy
Main Program file
please use plot.py to see the graph,if the program is interrupted before 500 steps.
"""
import numpy as np
import os
from env import ArmEnvironment
from rl  import DDPG
import matplotlib.pyplot as plt
#Added matplot lib to plot the reward as a function of time

MAX_EPISODES = 500
MAX_EP_STEPS = 200
ON_TRAIN = True

# setup
env = ArmEnvironment()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound
rl = DDPG(a_dim, s_dim, a_bound)
os.unlink("data.txt")
#steps = []
def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        episode_r = 0.
        for j in range(MAX_EP_STEPS):
            env.render()

            a = rl.choose_action(s)

            s_, r, goal = env.step(a)

            rl.store_transition(s, a, r, s_)

            #increment reward for the episode by reward taken for the specific action
            episode_r += r
            if rl.memory_full:
                rl.learn()

            s = s_
            if goal or j == MAX_EP_STEPS-1:
                
                
                print('Episode: %i | Reward: %.1f' % (i, episode_r))
                # Code for printing to a file
                
                sample = open('data.txt', 'a') 

                print(i, episode_r, file = sample) 
                sample.close() 
                break
    rl.save()


def eval():
    rl.load()
    
    env.render()
    
    #For evaluation need to lock simulation to speed of game
    env.viewer.set_vsync(True)
    while True:
        s = env.reset()
        for _ in range(MAX_EP_STEPS):
            env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            if done:
                break


if ON_TRAIN:
    train()
else:
    eval()

# The graph goes here.
with open ('data.txt','r') as f:
    lines=f.readlines()
    x= [float(line.split()[0]) for line in lines]
    y= [float(line.split()[1]) for line in lines]

plt.title('rewards over episodes')    
plt.xlabel('episodes')
plt.ylabel('reward')

plt.plot(x ,y)
plt.savefig('plot.png')
plt.show()

