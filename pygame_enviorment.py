import pygame
import sys
import numpy as np
import math
from PIL import Image
import pickle

pygame.init()

size = width, height = 600, 600

food_reward = 25
enemyreward = 200
move_reward = 10

episodes = 25000
show = 1000
learn_rate = 0.001
discount = 0.96
epsilon = 1
epsilon_decay = 0.999

start_qtable = None

black = 0, 0, 0

screen = pygame.display.set_mode(size)

#The only observation in this simulation is the distance between the food and player
#Each unit in the game is equal to 20 pixels or 20 units of the window.

if start_qtable == None:
	qtable = {}
	for x1 in range(0, int(height/20 +1)):
		for y1 in range(0, int(height/20 +1)):
			for x2 in range(0, int(height/20 +1)):
				for y2 in range(0, int(height/20 +1)):
					qtable[((x1-x2)*20, (y1 - y2)* 20)] = [np.random.uniform(-2, 0) for i in range(6)]

else:
	with open(start_qtable, "rb") as f: 
		qtable = pickle.load(f)

food_postion = tuple((np.random.randint(0, 20) * 20, np.random.randint(0, 20) * 20))

def food_spawn():
	if food_postion[0] != human_spawn().x & food_postion[1] != human_spawn().y:
		food = pygame.Rect(food_postion[0], food_postion[1],20, 20)
	else:
		food = pygame.Rect(food_postion[0] + 40, food_postion[1] + 40, 20, 20)
	return food

human_player_postion = tuple((200,200))


def human_spawn():
	human_player = pygame.Rect(human_player_postion[0], human_player_postion[1], 20, 20)
	return human_player

kera_player = human_spawn()

def player_action(action, player):
	if(action == 0):
		player.x = human_spawn().x + 20
	if(action == 1):
		player.x = human_spawn().x - 20
	if(action == 2):
		player.y = human_spawn().y + 20
	if(action == 3):
		player.y = human_spawn().x - 20
	if(action == 4):
		player.x, player.y = human_spawn().x + 20,human_spawn().y + 20
	if(action == 5):
		player.x, player.y = human_spawn().x - 20,human_spawn().y - 20
	if human_spawn().left <= 0:
		human_spawn().x = 0
	elif human_spawn().top <= 0:
		human_spawn().y = 0
	elif human_spawn().bottom >= height:
		player = human_spawn().move(0,0)
	elif human_spawn().right >= width:
		player = human_spawn().move(0, 0)
	return player


def display(kera_player):
	screen.fill(black)
	pygame.draw.rect(screen, (0, 0, 255), player_action(action, kera_player))
	pygame.draw.rect(screen, (0, 255, 0), food_spawn())
	pygame.display.flip()


episode_total_rewards = []


for episode in range(episodes):

	for event in pygame.event.get():
		if event.type == pygame.QUIT : sys.exit()

	episode_rewards = 0
	done = False

	kera_player = human_spawn()

	human_player_postion = tuple((200, 200))

	food_postion = tuple((np.random.randint(0, 28) * 20, np.random.randint(0, 28) * 20))
	if food_postion[0] == human_spawn().x & food_postion[1] == human_spawn().y:
		food_spawn().x += 20; food_spawn().y += 20

	if episode % show == 0:
		print('Episode : ', episode, 'epsilon : ', epsilon)
		render = True
	else:
		render = False
		
	current_state = (human_player_postion[0] - food_postion[0], human_player_postion[1] - food_postion[1])
	
	for i in range(200):

		if np.random.random() > epsilon:
			action = np.argmax(qtable[current_state])
		else:
			action = np.random.randint(0, 6)
			action = np.random.randint(0, 6)

		player_action(action, kera_player)


		if human_spawn().colliderect(food_spawn()):
			reward = food_reward
			done = True
		else:
			reward = -move_reward

		new_state = (human_spawn().x - food_spawn().x , human_spawn().y - food_spawn().y)
		max_future_q = np.max(qtable[new_state])
		current_q = qtable[current_state][action]

		new_q = (1 - learn_rate) * current_q + learn_rate * (episode_rewards + (discount * max_future_q))

		if reward == food_reward:
			new_q = food_reward

		qtable[current_state][action] = new_q
		

		if done:
			print('Ate on episode : ', episode)
			break
		if render:
			display(kera_player)
		
		current_state = new_state

	epsilon = epsilon * epsilon_decay
	episode_rewards += reward

	episode_total_rewards.append(episode_rewards)

# print(current_state)
# qtable[current_state][1]
























