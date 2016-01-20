#!/usr/bin/python

#####################################################################
# This script presents how to make use of game variables to implement
# shaping using health_guided.wad scenario
# Health_guided scenario is just like health_gathering 
# (see "../../scenarios/README") but for each collected medkit global
# variable number 1 in acs script (coresponding to USER1) is increased
# by 100.0. It is not considered a part of reward but will possibly
# reduce learning time.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
# 
#####################################################################
from vizia import *
from random import choice
import itertools as it
from time import sleep

import cv2

game = DoomGame()

# Choose scenario config file you wish to watch.
# Don't load two configs cause the second will overrite the first one.
# Multiple config files are ok but combining these ones doesn't make much sense.

game.load_config("config_health_gathering.properties")
game.set_doom_file_path("../../scenarios/health_guided.wad")
#game.set_screen_resolution(ScreenResolution.RES_640X480)

game.init()

# Creates all possible actions.
actions_num = game.get_available_buttons_size()
actions = []
for perm in it.product([False, True], repeat=actions_num):
    actions.append(list(perm))


episodes = 10
sleep_time = 0.05
last_summary_shaping_reward = 0

for i in range(episodes):

	print "Episode #" +str(i+1)
	# Not needed for the first episdoe but the loop is nicer.
	game.new_episode()
	while not game.is_episode_finished():
		

		# Gets the state and possibly to something with it
		s = game.get_state()
		img = s.image_buffer
		misc = s.game_variables

		# Makes a random action and save the reward.
		r = game.make_action(choice(actions))
		
		# Retrieve the shaping reward 
		sr = doom_fixed_to_double(game.get_game_variable(GameVariable.USER1))
		sr = sr - last_summary_shaping_reward
		last_summary_shaping_reward += sr

		print "State #" +str(s.number)
		print "Health:", misc[0]
		print "Last Reward:",r
		print "Last Shaping Reward:", sr
		print "====================="	

		# Sleep some time because processing is too fast to watch.
		if sleep_time>0:
			sleep(sleep_time)

	print "Episode finished!"
	print "Summary reward:", game.get_summary_reward()
	print "************************"


game.close()



for i in range(iters):

	if game.is_episode_finished():
		print "episode finished!"
		print "summary reward:", game.get_summary_reward()
		print "************************"
		sleep(1)
		game.new_episode()

	s = game.get_state()
	r = game.make_action(choice(actions))
	

	print "state #" +str(s.number)
	print "HP:", s.game_variables[0]
	print "reward:",r
	print "summmary shaping reward:", sr
	print "====================="	
	if sleep_time>0:
		sleep(sleep_time)
	


game.close()


    