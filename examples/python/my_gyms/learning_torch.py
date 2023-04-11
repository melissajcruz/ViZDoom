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
from torch.nn.functional import normalize
import torch.optim as optim
import vizdoom as vzd
import matplotlib
import visdom as Visdom
import matplotlib.pyplot as plt 
import warnings
from tqdm import trange


# Q-learning settings
learning_rate = 0.00001
discount_factor = 0.99
train_epochs = 2
gamma = 0.99
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64*3

# Plotting data structures
np_closs = np.zeros(0)
np_aloss = np.zeros(0)
np_episode = np.zeros(0)

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 10
resolution = (30, 45)
episodes_to_watch = 10

# actor_model_savefile = "./a_model-doom.pth"
# critic_model_savefile = "./c_model-doom.pth"
# actor_model_savefile = "./center-line-center-a_model-doom.pth"
# critic_model_savefile = "./center-line-center-c_model-doom.pth"

a_save_to = "./center-line-center-a_model-doom.pth"
c_save_to = "./center-line-center-c_model-doom.pth"

save_model = True
load_model = True
skip_learning = True
show_window = True

# to run GUI event loop for data vis
iteration = 0
# fig = plt.figure()
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)

# Configuration file path
config_file_path = os.path.join(vzd.scenarios_path, "defend_the_center.cfg")

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

"""
Create Visdom class
"""
class VisdomLinePlotter(object):
    def __init__(self, env_name = 'Doom Plots'):
        self.viz = Visdom.Visdom()
        self.env = env_name
        self.plots = {}
    
    def plot(self, var_name, split_name, title_name, x, y,):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=x, Y=y, env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Iterations',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=x, Y=y, env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

# AC network outputs: prob dist of choosing action given state and Q(s,a)
log_pi = torch.zeros(0, 0, requires_grad = False).to(DEVICE)

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
    game.set_window_visible(show_window)
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

def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=1000):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """
    # global values
    global log_pi

    plt.ion()
    plt.title("Loss v. Number of Iterations", fontsize = 20)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    start_time = time()

    game.set_window_visible(show_window)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        print("\nEpoch #" + str(epoch + 1))

        for _ in trange(steps_per_epoch, leave=False):
            state = preprocess(game.get_state().screen_buffer)
            action, log_pi_local = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess(game.get_state().screen_buffer)
            else:
                next_state = np.zeros((1, 30, 45)).astype(np.float32)

            agent.append_memory(state, action, reward, next_state, done, log_pi_local)
        
            if global_step >= agent.batch_size - 1:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            # Reset values array
            # values = torch.empty(1, 1, requires_grad = False)

            global_step += 1

        log_pi = torch.zeros(0, 0, requires_grad = False).to(DEVICE)
        # agent.update_target_net()
        train_scores = np.array(train_scores)

        print(
            "Results: mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

        # test(game, agent)
        if save_model:
            print("Saving the network weights to:", a_save_to, " and ", c_save_to)
            torch.save(agent.a_net, a_save_to)
            torch.save(agent.c_net, c_save_to)
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
        )

        self.critic6 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        value = self.critic1(x)
        value = self.critic2(value)
        # value = self.critic3(value)
        value = self.critic4(value)
        value = self.critic5(value)
        value = normalize(value, p = 1.0, dim = 1)
        value = self.critic6(value)
        # return value.item()
        return value
    

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
        probs = self.actor2(probs)
        # probs = self.actor3(probs)
        probs = self.actor4(probs)
        probs = self.actor5(probs)
        
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
            print("Loading actor model from: ", actor_model_savefile)
            self.a_net = torch.load(actor_model_savefile)

            print("Loading critic model from: ", critic_model_savefile)
            self.c_net = torch.load(critic_model_savefile)

        else:
            print("Initializing new model")
            self.a_net = Actor(len(actions)).to(DEVICE) # TODO: remove len(actions) dep
            self.c_net = Critic().to(DEVICE)

        self.AC_opt = optim.Adam([
            {'params' : self.a_net.parameters()},
            {'params' : self.c_net.parameters(), 'lr' : self.lr}
            ], lr = self.lr)

        # self.actor_opt = optim.Adam(self.a_net.parameters(), lr = self.lr)
        # self.critic_opt = optim.Adam(self.c_net.parameters(), lr = self.lr)

    def get_action(self, state):
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(DEVICE)

            dist = self.a_net(state)
            # Sample an action and get log prob of sampling action from distribution
            action = dist.sample()
            log_pi = dist.log_prob(action)
            return action, log_pi

    def append_memory(self, state, action, reward, next_state, done, log_pi):
        self.memory.append((state, action, reward, next_state, done, log_pi))

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
                sum = reward + self.discount * sum
                Gt = np.append(Gt, sum)
            terminal_sample_idx += 1

        Gt = np.flipud(Gt)
        # Normalization between 0 and 1
        Gt = [((Gt_item - Gt.min()) / (Gt.max() - Gt.min())) for Gt_item in Gt]
        Gt = torch.FloatTensor(Gt.copy()).to(DEVICE).detach()
        return Gt 
    
    def calc_actor_loss(self, advantage, log_pi):
         return -log_pi * advantage
        #  return -log_pi * (Gt_sa) 
        

    """
    Critic loss is positive: transition from S to S' reward is greater than observerd reward
    """
    def calc_critic_loss(self, Gt, next_state, state):
        advantage = Gt + gamma * self.c_net(next_state).float() - self.c_net(state).float()
        return advantage.pow(2).mean(), advantage

    def train(self):
        global iteration
        iteration_arr = np.zeros(0)
        loss_arr = np.zeros(0)

        states = torch.zeros(0, 0, requires_grad = False)
        next_states = torch.zeros(0, 0, requires_grad = False)

        # A batch contains samples from 1 episode
        batch = random.sample(self.memory, self.batch_size)
        rewards = np.zeros(0)
        are_terminal_samples = np.zeros(0)
        log_pi = np.zeros(0)

        bool_first_value = True
        for sample in batch:
            if(bool_first_value):
                log_pi = sample[5]
                bool_first_value = False
            else:
                log_pi = torch.cat((log_pi, sample[5]), 0)
        log_pi.to(DEVICE).detach()

        bool_first_value = True
        # Populate values from critic net given states in tensor form
        for sample in batch:
            state = np.expand_dims(sample[0], axis=0)
            state = torch.from_numpy(state).float().to(DEVICE)

            if(bool_first_value):
            # Create on-policy values: values when acting according to policy pi
                # values = self.c_net(state).to(DEVICE)
                next_states = torch.from_numpy(sample[3]).float().to(DEVICE)
                states = torch.from_numpy(sample[0]).float().to(DEVICE)
                bool_first_value = False
            else:
                # values = torch.cat((values, self.c_net(state)), 0)
                next_states = torch.cat((next_states, torch.from_numpy(sample[3]).float().to(DEVICE)), 0)
                states = torch.cat((states, torch.from_numpy(sample[0]).float().to(DEVICE)), 0)

        # values.to(DEVICE).detach()
        
        # Populate states, actions, etc. after global step > batch size
        for sample in batch:
            rewards = np.append(rewards, sample[2])
            # Array of bools: true if index of sample is terminal sample where episode has ended
            are_terminal_samples = np.append(are_terminal_samples, sample[4])
        
        # Get discounted expected returns
        Gt = self.get_expected_rewards(rewards, are_terminal_samples).detach()
        log_pi_copy = log_pi.clone().detach()

        # Gt_copy = Gt.clone()
        # values_copy = values.clone()

        # Calculate Actor-Critic loss
        for row_idx in range(self.batch_size - 1):

            # Format data structures
            states_item = states[:][:][row_idx]
            states_item = states_item[None]
            states_item = states_item[None]
            next_states_item = next_states[:][:][row_idx]
            next_states_item = next_states_item[None]
            next_states_item = next_states_item[None]

            l_critic, advantage = self.calc_critic_loss(Gt[row_idx], next_states_item, states_item)
            l_actor = self.calc_actor_loss(advantage, log_pi_copy[row_idx])
            loss = l_actor + l_critic

            loss.backward(retain_graph = False)

            iteration += 1

            loss_arr = np.append(loss_arr, loss.item())
            iteration_arr = np.append(iteration_arr, iteration)

        plotter.plot('loss', 'train', 'Class Loss', iteration_arr, loss_arr)
        
        self.AC_opt.zero_grad()
        self.AC_opt.step()

        # print("Loss: ", str(loss))
        
        # Loss function where actor loss is negative to reflect a "positive" loss that will be minimized to zero
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


if __name__ == "__main__":
    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size()
    global plotter
    plotter = VisdomLinePlotter(env_name = 'Doom Plots')
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
            best_action_index = best_action_index[0].item()
            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
