# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 15:06:50 2021

To do:
    Move everyone to GPU tensor / scaling up
    Save / load, resume training of agent, etc.
    Move to solid walls?

@author: ArthurBaucour
"""

from collections import namedtuple
from itertools import count

import torch
import torch.nn.functional as F
import torch.optim as optim

# Import the game environment
from game_environment.torch_no_parallel import GameEnvTorch

# General DQN helper
from helper import ReplayMemory, SnakeNet, make_plot, select_action


def optimize_model(
    policy_net,
    target_net,
    optimizer,
    loss_fn,
    gamma,
    memory,
    batch_size,
    Transition,
    device,
):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # Get the state/action/reward batch
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.tensor(batch.action).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch.reward).unsqueeze(1).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either
    # the expected state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1)[0].detach()
    )
    next_state_values = next_state_values.unsqueeze(1)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute the loss
    loss = loss_fn(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)  # Clip the gradient to -1, 1
    optimizer.step()


def main():
    # -------------------------------------------------------------------------
    # Setup

    # loading/saving path

    SAVE_PATH = ".\\models"

    # Hyper-parameters

    num_episodes = 50  # 50, int(1e5)

    batch_size = 128  # 128
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 1000  # 200
    target_update = 10

    n_actions = 3  # Left, forward, right

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on: {device} ({torch.cuda.get_device_name(0)})\n")

    # Game environment
    sizeX, sizeY = 5, 5  # Number of blocks width,height
    gameEnv = GameEnvTorch(sizeX, sizeY)  # Initialize the map

    # Replay memory
    Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
    memory = ReplayMemory(10000, Transition)

    # Networks
    policy_net = SnakeNet().to(device)
    target_net = SnakeNet().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Loss and optimizer
    loss_fn = F.smooth_l1_loss
    # optimizer = optim.RMSprop(policy_net.parameters())
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    # -------------------------------------------------------------------------
    # Training

    steps_done = 0
    mean_score = []
    mean_duration = []

    score_plot = []
    duration_plot = []
    x_plot = []

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        gameEnv.reset()
        # adjust dtype?
        state = gameEnv.mapState
        state = state.to(device)

        for t in count():

            # Select and perform an action
            action = select_action(
                policy_net, n_actions, state, steps_done, eps_start, eps_end, eps_decay
            )
            action = action.squeeze()
            steps_done += 1

            # Get the reward (ditch the map state, stored in gameEnv.mapState)
            _, reward = gameEnv.move_snake(action)
            reward = torch.tensor([reward], device=device)  # Check the dimensions

            # Check if game is over
            done = gameEnv.game_over
            if not done:
                next_state = gameEnv.mapState
                next_state = next_state.to(device)
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(
                policy_net,
                target_net,
                optimizer,
                loss_fn,
                gamma,
                memory,
                batch_size,
                Transition,
                device,
            )
            if done:
                mean_score.append(gameEnv.score)
                mean_duration.append(t + 1)
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

            mean_display = sum(mean_duration) / target_update
            score_display = sum(mean_score) / target_update

            print(
                f"[{i_episode} / {num_episodes}]\t"
                + f"Mean duration: {mean_display:2.1f}\t"
                + f"Mean score: {score_display:3.1f}"
            )

            mean_duration = []
            mean_score = []

            duration_plot.append(mean_display)
            score_plot.append(score_display)
            x_plot.append(i_episode)

    make_plot(x_plot, duration_plot, score_plot)
    print("Complete")


if __name__ == "__main__":
    main()
