import gym
import numpy as np

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1

DISCOUNT = 0.99
EPISODES = 25000
SHOW_EVERY = 2000

DISCRETE_OS_SIZE = [10, 10]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

print(q_table)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    while not done:

        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)


        new_state, reward, done, _ = env.step(action)

        new_discrete_state = get_discrete_state(new_state)

        if episode % SHOW_EVERY == 0:
            env.render()
        #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # If simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q


        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            #q_table[discrete_state + (action,)] = reward
            print('We made it on episode: ', episode)
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value


env.close()

######################################################################################################

# import gym
# import numpy as np
# import random
# import time

# env = gym.make("FrozenLake-v0")

# action_space_size = env.action_space.n
# state_space_size = env.observation_space.n

# q_table = np.zeros((state_space_size, action_space_size))

# num_episodes = 10000
# max_steps_per_epi = 100
# learning_rate = 0.1
# discount_rate = 0.99

# exploration_rate = 1 #alpha
# max_exploration_rate = 0.99 
# min_exploration_rate = 0.01
# exploration_decay_rate = 0.001 #or 0.01

# rewards_all_list = []

# for episode in range(num_episodes):
#     state = env.reset() #resets the game event in new loop
    
#     done = False
#     rewards_current_episode = 0
    
#     for step in range(max_steps_per_epi):
#         exploration_rate_threshold = random.uniform(0, 1)
        
#         if exploration_rate_threshold > exploration_rate:
#             action = np.argmax(q_table[state, :])
#         else:
#             action = env.action_space.sample()
        
#         new_state, reward, done, info = env.step(action) 
        
#         #Upgrading the q_Table(s,a)
#         q_table[state, action] = q_table[state, action] * (1 - learning_rate) + (learning_rate * (reward + discount_rate * np.max(q_table[new_state, :])))
        
#         state = new_state
#         rewards_current_episode += reward
        
#         if done == True:
#             # print(step)
#             # env.render()
#             break
        
#     exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(
#         -exploration_decay_rate * episode)
#     rewards_all_list.append(rewards_current_episode)

# env.close()
# reward_per_thousand_epi = np.split(np.array(rewards_all_list), num_episodes /1000)
# count = 1000

# print("===Average reward per thousand episode===============\n")

# for r in reward_per_thousand_epi:
#     print(count, ":", str(sum(r / 1000)))
#     count += 1000
    
#--------------------------------------------------------------------------------------------------------
# import os
# import random
# import gym
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow import keras
# from collections import deque
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam

# env = gym.make('MountainCar-v0')

# state_size = env.observation_space.shape[0]

# action_size = env.action_space.n

# batch_size = 32

# episodes_total = 1000

# output_dir = 'model_output/CartPole'

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# class DQNAgent:
    
#     def __init__(self, state_size, action_size):
        
#         self.state_size = state_size
#         self.action_size = action_size
        
#         self.memory = deque(maxlen = 2000)
#         self.gamma = 0.95
#         self.epsilon = 1.0
#         self.epsilon_decay = 0.995
#         self.epsilon_min = 0.01
#         self.learning_rate = 0.001
#         self.model = self._build_model()
        
#     def  _build_model(self):
#         model = Sequential()
#         model.add(Dense(128, input_dim = self.state_size, activation = 'relu'))
#         model.add(Dense(64, activation = 'relu'))
#         model.add(Dense(self.action_size, activation = 'linear'))
#         model.compile(optimizer = Adam(lr = self.learning_rate),
#                       loss = 'mse')
#         return model
        
#     def remeber(self, state, action, reward, new_state, done):
#         self.memory.append((state, action, reward, new_state, done))
    
#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return random.randrange((self.action_size))
#         act_values = self.model.predict(state)
#         return np.argmax(act_values[0])
    
#     def replay(self, batch_size):
        
#         minibatch = random.sample(self.memory, batch_size)
        
#         for state, action, reward, new_state, done in minibatch:
#             target = reward
            
#             if not done:
#                 target = reward + (self.gamma * np.amax(self.model.predict(new_state)[0]))
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
            
#             self.model.fit(state, target_f, epochs = 1, verbose = 0)
            
#             if self.epsilon > self.epsilon_min:
#                 self.epsilon *= self.epsilon_decay
                
                
#     def load(self, name):
#         self.model.load_weights(name)
        
#     def save(self, name):
#         self.model.save_weights(name)
            


# agent = DQNAgent(state_size, action_size)


# done = False
# for episode in range(episodes_total):
#     state = env.reset()
    
#     state = np.reshape(state, [1, state_size])
    
#     for time in range(5000):
#         # env.render()
        
#         action = agent.act(state)
#         new_state, reward, done, _ = env.step(action)
#         reward = reward if not done else -10
#         new_state = np.reshape(new_state, [1, state_size])
        
#         agent.remeber(state, action, reward, new_state, done)
#         state = new_state
        
#         if done == True:
#             env.render()
#             print('Done', done, 'Episode', episode, 'Epsilon', agent.epsilon)
#             break
            
            
#     if len(agent.memory) > batch_size:
#          agent.replay(batch_size)
         
#     if episode % 50 ==0 :
#         agent.save(output_dir + "weights" + '{:04d}'.format(episode) + ".hdf5")


# env.close()

























