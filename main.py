import torch
import os
from DDPG_module.DDPG import DDPG, Actor, Critic, prepare_training_inputs
from DDPG_module.DDPG import OrnsteinUhlenbeckProcess as OUProcess

from DDPG_module.memory import ReplayMemory
from DDPG_module.target_update import soft_update

import matplotlib.pyplot as plt
from scipy.io import savemat
from collections import deque
from param import Hyper_Param

from robotic_env import RoboticEnv
import time
os.environ['KMP_DUPLICATE_LIB_OK'] ='True'

# Hyperparameters
DEVICE = Hyper_Param['DEVICE']
tau = Hyper_Param['tau']
lr_actor = Hyper_Param['lr_actor']
lr_critic = Hyper_Param['lr_critic']
batch_size = Hyper_Param['batch_size']
gamma = Hyper_Param['discount_factor']
memory_size = Hyper_Param['memory_size']
total_eps = Hyper_Param['num_episode']
sampling_only_until = Hyper_Param['train_start']
print_every = Hyper_Param['print_every']

# List storing the results
epi = []
lifting_time =[]
box_z_pos =[]
stable_lifting_time =[]
success_time = []

# Create Environment

env = RoboticEnv()

s_dim = env.state_dim
a_dim = env.action_dim

# initialize target network same as the main network.
actor, actor_target = Actor().to(DEVICE), Actor().to(DEVICE)
critic, critic_target = Critic().to(DEVICE), Critic().to(DEVICE)

agent = DDPG(critic=critic,
             critic_target=critic_target,
             actor=actor,
             actor_target=actor_target,epsilon=Hyper_Param['epsilon'],
             lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma).to(DEVICE)

memory = ReplayMemory(memory_size)


# Episode start
for n_epi in range(total_eps):
    ou_noise = OUProcess(mu=torch.zeros(a_dim))
    s = env.reset()
    epi.append(n_epi)
    episode_return = 0

    while True:
        a = agent.get_action(s, agent.epsilon*ou_noise()).view(-1)

        ns, r, done, info = env.step(a)
        episode_return += r.item()
        experience = (s.view(-1, s_dim),
                      a.view(-1, a_dim),
                      r.view(-1, 1),
                      ns.view(-1, s_dim),
                      torch.tensor(done, device=DEVICE).view(-1, 1))
        memory.push(experience)
        s = ns
        if done:
            break

    # lifting_time.append(env.time_step - 1)
    box_z_pos.append(env.z_pos.item())
    stable_lifting_time.append(env.stable_time)
    success_time.append(env.task_success)

    if len(memory) >= sampling_only_until:
        # train agent
        agent.epsilon = max(agent.epsilon * Hyper_Param['epsilon_decay'], Hyper_Param['epsilon_min'])

        sampled_exps = memory.sample(batch_size)
        sampled_exps = prepare_training_inputs(sampled_exps)
        agent.update(*sampled_exps)

        soft_update(agent.actor, agent.actor_target, tau)
        soft_update(agent.critic, agent.critic_target, tau)

    if n_epi % print_every == 0:
        msg = (n_epi, env.stable_time, env.task_success)
        print("Episode : {:4.0f} | stable lifting time : {:3.0f} | task success time : {:3.0f}:".format(*msg))
    #     plt.xlim(0, total_eps)
    #
    #     plt.plot(epi, epi_returns, color='red')
    #     # plt.plot(epi, score_avg_value, color='red')
    #     # plt.plot(epi, optimal_scoexport LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.sore_avg_value, color='blue')
    #     # plt.plot(epi, cum_rand_score_list, color='blue')
    #     # plt.plot(epi, cum_optimal_score_list, color='green')
    #     plt.xlabel('Episode', labelpad=5)
    #     plt.ylabel('Episode return', labelpad=5)
    #     plt.grid(True)
    #     plt.pause(0.0001)
    #     plt.show()


# Base directory path creation
base_directory = os.path.join(Hyper_Param['today'])

# Subdirectory index calculation
if not os.path.exists(base_directory):
    os.makedirs(base_directory)
    index = 1
else:
    existing_dirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    indices = [int(d) for d in existing_dirs if d.isdigit()]
    index = max(indices) + 1 if indices else 1

# Subdirectory creation
sub_directory = os.path.join(base_directory, str(index))
os.makedirs(sub_directory)


# Store Hyperparameters in txt file
with open(os.path.join(sub_directory, 'Hyper_Param.txt'), 'w') as file:
    for key, value in Hyper_Param.items():
        file.write(f"{key}: {value}\n")

# Store score data (matlab data file)
savemat(os.path.join(sub_directory, 'data.mat'),{'stable_lifting_time': stable_lifting_time, 'box_z_pos': box_z_pos, 'success_time': success_time})
# savemat(os.path.join(sub_directory, 'data.mat'),{'sim_res': cum_score_list,'sim_optimal': optimal_score_avg_value})
# savemat(os.path.join(sub_directory, 'data.mat'), {'sim_res': cum_score_list,'sim_rand_res': cum_rand_score_list,
#                                                   'sim_optimal_res': cum_optimal_score_list})
