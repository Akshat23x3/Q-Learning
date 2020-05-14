import gym
import math
import numpy as np

env = gym.make('Pong-v0')

episodes = 25000
show_every = 1000
learn_rate = 0.01
dicount = 0.95
epsilon = 1
epsilon_start_decay = 1
epsilon_end_decay = episodes//2

env.observation_space.high

discrete_size = [20, 20, 20]
discrete_state_size = (env.observation_space.high - env.observation_space.low) / discrete_size

qtable = np.random.uniform(low = -2, high = 0, size= (discrete_size + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_state_size
    return tuple(discrete_state.astype(np.int))
a = get_discrete_state(env.reset())
a = np.array(a)
a#[10][3]#.shape
for episode in range(episodes):
    
    discrete_state = get_discrete_state(env.reset())
    done = False
    
    if episode % show_every == 0:
        render = True
        env.render()
    else:
        render = False
        
    while not done:
        
        action = np.random.randint(0, env.action_space.n)
        if np.random.random() > epsilon:
            action = np.argmax(qtable[discrete_state])
            































