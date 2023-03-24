#!/usr/bin/env python3

# E. Culurciello, L. Mueller, Z. Boztoprak
# December 2020

import itertools as it
import os
import random
from collections import deque
from time import sleep, time

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.optim as optim
import vizdoom as vzd
import tensorflow as tf
from tqdm import trange


# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
alpha_a = 0.9
alpha_c = 0.9
gamma = 0.9
train_epochs = 5
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 10
huber_del = 1.0

model_savefile = "./model-doom.pth"
save_model = True
load_model = False
skip_learning = False

# AC network outputs: prob dist of choosing action given state and Q(s,a)
dist = torch.empty(1, 1)
values = np.zeros(0)
log_pi = torch.empty(1, 1)

# Configuration file path
config_file_path = os.path.join(vzd.scenarios_path, "simpler_basic.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "rocket_basic.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "basic.cfg")

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")


def preprocess(img):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")

    return game


def test(game, agent):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print(
        "Results: mean: {:.1f} +/- {:.1f},".format(
            test_scores.mean(), test_scores.std()
        ),
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(),
    )


def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=2000):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()

    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        print("\nEpoch #" + str(epoch + 1))

        for _ in trange(steps_per_epoch, leave=False):
            state = preprocess(game.get_state().screen_buffer)
            action = agent.get_action(state, actions)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess(game.get_state().screen_buffer)
            else:
                next_state = np.zeros((1, 30, 45)).astype(np.float32)

            agent.append_memory(state, action, reward, next_state, done)

            if global_step >= agent.batch_size - 1:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        # agent.update_target_net()
        train_scores = np.array(train_scores)

        print(
            "Results: mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

        test(game, agent)
        if save_model:
            print("Saving the network weights to:", model_savefile)
            torch.save(agent.q_net, model_savefile)
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()
    return agent, game

"""
Input of Critic is state output is value of state
"""
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.critic1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.critic2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.critic3 = nn.Sequential(     
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.critic4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.critic5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        value = self.critic1(x)
        # print(value.size())
        value = self.critic2(value)
        # print(value.size())
        value = self.critic3(value)
        # print(value.size())
        value = self.critic4(value)
        print(value.size())
        value = self.critic5(value)
        print(value.size())
        return value.item()
    

"""
Input of Actor is state and output is probability of each action
Actions are generated using Gibbs softmax method: http://incompleteideas.net/book/ebook/node17.html
"""
class Actor(nn.Module):
    def __init__(self, action_size):
        super(Actor, self).__init__()
        self.actor1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.actor2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.actor3 = nn.Sequential(      
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.actor4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.actor5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        probs = self.actor1(x)
        # print(probs.size())
        probs = self.actor2(probs)
        # print(probs.size())
        probs = self.actor3(probs)
        # print(probs.size())
        probs = self.actor4(probs)
        # print(probs.size())
        probs = self.actor5(probs)
        # print(probs.size())
        
        dist = torch.distributions.Categorical(probs)
        return dist


class ACAgent:
    def __init__(
        self,
        action_size,
        memory_size,
        batch_size,
        discount_factor,
        lr,
        load_model,
        epsilon=1,
        epsilon_decay=0.9996,
        epsilon_min=0.1,
    ):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount = discount_factor
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        # self.criterion = nn.MSELoss()

        if load_model:
            print("Loading model from: ", model_savefile)
            self.ac_net = torch.load(model_savefile)
            self.target_net = torch.load(model_savefile)
            self.epsilon = self.epsilon_min

        else:
            print("Initializing new model")
            self.a_net = Actor(len(actions)).to(DEVICE) # TODO: remove len(actions) dep
            self.c_net = Critic().to(DEVICE)

        self.actor_opt = optim.SGD(self.a_net.parameters(), lr=self.lr)
        self.critic_opt = optim.SGD(self.c_net.parameters(), lr=self.lr)


    def get_action(self, state, actions):
            # Signify global dist and value variables are being modified
            global dist
            global values
            global log_pi

            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(DEVICE)

            dist = self.a_net(state)
            # Sample an action and get log prob of sampling action from distribution
            action = dist.sample()
            log_pi = dist.log_prob(action)
            
            return action

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # TODO: batch size is NOT 1 episode, parse through dones matrix
    def get_expected_rewards(self, rewards, are_terminal_samples):
        Gt = np.zeros(0)
        sum = 0.0
        are_terminal_samples = np.flipud(are_terminal_samples)
        terminal_sample_idx = 0
        rewards = np.flipud(rewards)
        # Traverse rewards backwards
        for reward in rewards:
            if(are_terminal_samples[terminal_sample_idx] == 1):
                Gt = np.append(Gt, reward)
                sum = 0
            else:
                sum = reward + gamma * sum
                Gt = np.append(Gt, sum)
            terminal_sample_idx += 1

        Gt = np.flipud(Gt)
        return Gt 
    
    def calc_actor_loss(self, Gt_sa, l_actor, value_sa):
         return l_actor + log_pi * (Gt_sa - value_sa)    

    def calc_critic_loss(self, Gt):
        loss_function = nn.HuberLoss(delta = huber_del) # TODO: tune delta  
        return loss_function(torch.FloatTensor(values), torch.FloatTensor(Gt))

    def train(self):
        global values
        # A batch contains samples from 1 episode
        batch = random.sample(self.memory, self.batch_size)
        states = np.zeros(0)
        actions = np.zeros(0)
        rewards = np.zeros(0)
        next_states = np.zeros(0)
        are_terminal_samples = np.zeros(0)
        

        # Populate states, actions, etc. after global step > batch size
        for sample in batch:
            states = np.append(states, sample[0])
            actions = np.append(actions, (sample[1]).item())
            rewards = np.append(rewards, sample[2])
            next_states = np.append(next_states, sample[3])
            # Array of bools: true if index of sample is terminal sample where episode has ended
            are_terminal_samples = np.append(are_terminal_samples, sample[4])
        
        # Get discounted expected returns
        Gt = self.get_expected_rewards(rewards, are_terminal_samples)

        # Initialize actor and critic loss
        l_actor = 0.0
        l_critic = 0.0

        for row_idx in range(self.batch_size - 1):

            # # Sample action, reward, next state, and next action
            s = states[row_idx]
            values = np.append(values, self.c_net(s))
            # a = actions[row_idx]
            # r = rewards[row_idx]
            # s_1 = next_states[row_idx]
            # a_1 = actions[row_idx + 1]

            # Calculate Actor-Critic loss
            l_actor = self.calc_actor_loss(Gt[row_idx], l_actor, values[row_idx])

        l_critic = self.calc_critic_loss(Gt)
        
        # Loss function where actor loss is negative to reflect a "positive" loss that will be minimized to zero
        print("Actor Loss: ", str(-l_actor))
        print("Critic Loss: ", str(l_critic))
        loss = -l_actor + l_critic
        print("Loss: ", str(loss))

        # Reset gradients, else gradients will be added
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        
        loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()
        """
        
        At every timestep t:     
            Sample reward and next state given sampled action from policy at time t (based on probability of transition at
            time t from state to next state and corresponding reward given sampled action)

            Sample next action from policy given next state
            Update network weights: https://towardsdatascience.com/policy-gradient-methods-104c783251e0
                Adds product of theta learning rates, Q value for state, action, and gradient of log probability that agent 
                selects action from state to current policy (parameterized by theta)
            Critic evaluates new state to determine if things have gone better or worse than expected using TD error (aka advantage)
            Update parameters of Q function
        """  
        # print()


if __name__ == "__main__":
    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    agent = ACAgent(
        len(actions),
        lr=learning_rate,
        batch_size=batch_size,
        memory_size=replay_memory_size,
        discount_factor=discount_factor,
        load_model=load_model,
    )

    # Run the training for the set number of epochs
    if not skip_learning:
        agent, game = run(
            game,
            agent,
            actions,
            num_epochs=train_epochs,
            frame_repeat=frame_repeat,
            steps_per_epoch=learning_steps_per_epoch,
        )

        print("======================================")
        print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
