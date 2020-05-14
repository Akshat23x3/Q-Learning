import gym
import math
import numpy as np
from random import uniform

env = gym.make('MountainCar-v0')

descrete_state_size = [10, 10]
descrete_state_win_size = (env.observation_space.high - env.observation_space.low) / descrete_state_size


learn_rate = 0.1
discount = 0.95
episodes = 25000
epsilon = 1
epsilon_start_decay = 1
epsilon_end_decay = episodes//2
epsilon_decay = epsilon / (epsilon_end_decay - epsilon_start_decay)
show_every = 1000

action_size = env.action_space.n

qtable = np.random.uniform(low = -2, high = 0, size = (descrete_state_size + [action_size]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / descrete_state_win_size
    
    return tuple(discrete_state.astype(np.int))

for episode in range(episodes):
    
    discrete_state = get_discrete_state(env.reset())
    
    done = False
    
    if episode % show_every == 0 :
        print(episode)
        render = True
    else:
        render = False
        
    while not done:
        
        action = np.random.randint(0, action_size)
        if np.random.random() > epsilon:
            action = np.argmax(qtable[discrete_state])
            
        new_state, reward, done, info = env.step(action)
        
        new_discrete_state = get_discrete_state(new_state)
        
        if render:
            
            env.render()
            
        if not done:
            max_future_q = np.max(qtable[new_discrete_state])
            
            current_q = qtable[discrete_state + (action, )]
            
            new_q =(1 - learn_rate) * current_q + learn_rate * (reward + (discount * max_future_q))
            
            qtable[discrete_state + (action, )] = new_q
            
        elif new_state[0] >= env.goal_position:
            print('We made it on episode: ', episode)
            qtable[discrete_state + (action,)] = 0
                
        discrete_state = new_discrete_state
            
    if epsilon_end_decay >= episode >= epsilon_start_decay:
         epsilon -= epsilon_decay
    
env.close()






























