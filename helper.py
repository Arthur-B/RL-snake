# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 19:01:12 2021

@author: Arthur Baucour
"""

import math
import os
import random
from collections import deque

import imageio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =============================================================================
# DQN setup
# =============================================================================


class ReplayMemory(object):
    """ """

    def __init__(self, capacity, Transition):
        self.Transition = Transition
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(
    policy_net, n_actions, state, steps_done, eps_start, eps_end, eps_decay
):
    """
    Select an action to perform following an epsilon greedy policy.
    Careful, the decay is one the steps_done, not the number of iterations
    """

    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(
        -1.0 * steps_done / eps_decay
    )

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)


# =============================================================================
# Neural network
# =============================================================================


class SnakeNet(nn.Module):  # For 3x5x5 input
    def __init__(self):
        super(SnakeNet, self).__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 1),
            nn.ReLU(),
        )

        self.selection_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 3 outputs: left, forward, right
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(-1, self.num_flat_features(x))
        # print(x.shape)
        x = self.selection_fc(x)
        # x = F.softmax(x, dim=1)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# =============================================================================
# Plot
# =============================================================================


def make_plot(x_plot, duration_list, score_list):
    # sns.set()
    # Process the data
    df = pd.DataFrame(
        {"Episode": x_plot, "Duration": duration_list, "Score": score_list}
    )
    df_rolling = df.rolling(5).mean()  # Get the rolling average, using 5 points

    # Build plot
    fig, ax1 = plt.subplots(figsize=(5.4, 4), dpi=300)
    ax2 = ax1.twinx()  # For xyy plot
    axs = [ax1, ax2]

    ax1.scatter(x_plot, duration_list, marker=".", color="C0")
    ax1.plot(df_rolling["Episode"], df_rolling["Duration"], color="C0")
    ax2.scatter(x_plot, score_list, marker=".", color="C1")
    ax2.plot(df_rolling["Episode"], df_rolling["Score"], color="C1")

    # Fluff xyy plot
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Duration", color="C0")
    ax1.tick_params(axis="y", color="C0", labelcolor="C0")

    ax2.set_ylabel("Score", color="C1")
    ax2.tick_params(axis="y", color="C1", labelcolor="C1")
    ax2.spines["right"].set_color("C1")
    ax2.spines["left"].set_color("C0")

    fig.tight_layout()
    fig.savefig("training_process.png")

    return fig, axs


def plot_game_state(game_state, file_name):
    # Build the plot
    fig, ax = plt.subplots(figsize=(3.33, 3.33), dpi=300)
    ax.axis("off")

    # Segment the game state
    matrix_plot = (game_state[0, 0] - game_state[0, 2]).cpu().numpy()
    fruit = np.ma.masked_where(matrix_plot != -1, matrix_plot)
    snake = np.ma.masked_where(matrix_plot < 0, matrix_plot)

    # Add the matrix

    ax.imshow(fruit, cmap=cm.Set1, interpolation="none")
    ax.imshow(
        snake,
        cmap=cm.viridis,
        interpolation="none",
        vmin=0,
    )

    # Save
    fig.tight_layout()
    fig.savefig(file_name)
    plt.close(fig)


def make_gif(filenames: list, name: str):
    with imageio.get_writer(name + ".gif", mode="I") as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in set(filenames):
        os.remove(filename)


def play_game(policy_net, game_env, device, record=False, gif_name="temp"):
    # Initialize and get the current state of the game
    game_env.reset()
    state = game_env.mapState.to(device)

    duration = 0
    if record == True:
        filenames = []
        # record the initial game state
        plot_game_state(state, ".\\images\\temp\\state_0.png")

    while game_env.game_over == 0:
        # Select action following policy net, no randomness
        action = policy_net(state).max(1)[1].view(1, 1)
        # Move the snake
        state, _ = game_env.move_snake(action)
        state = state.to(device)

        # Save the new state
        duration += 1
        if record == True:
            filename = f".\\images\\temp\\state_{duration}.png"
            plot_game_state(state, filename)
            filenames.append(filename)

    # Make the gif
    if record == True:
        gif_path = f".\\images\\{gif_name}"
        make_gif(filenames, gif_path)

    return duration, game_env.score
