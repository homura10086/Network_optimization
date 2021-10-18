import random
from collections import namedtuple
from math import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.random import normal
from torch import nn
from torch import optim
from NT import num_cell
import matplotlib as mpl

'''
弱覆盖、越区覆盖优化算法
'''

torch.manual_seed(1)
torch.cuda.manual_seed(1)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 100
BATCH_SIZE = 32
CAPACITY = 10000
device = 'cpu'
num_samples = 0

num_tes = 10
distances = np.random.normal(300, 30, num_tes)
f = 2.4
n0 = -174
lamuda = 0.7
bw = 1e7


class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3_adv = nn.Linear(n_mid, n_out)
        self.fc3_v = nn.Linear(n_mid, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        adv = self.fc3_adv(h2)
        val = self.fc3_v(h2).expand(-1, adv.size(1))
        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        return output


class Brain:
    def __init__(self, num_states, num_actions, is_train, mode):
        self.mode = mode
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)
        n_in, n_mid, n_out = num_states, 32, num_actions
        if is_train:
            self.main_q_network = Net(n_in, n_mid, n_out).to(device)
            self.target_q_network = Net(n_in, n_mid, n_out).to(device)
        else:
            self.main_q_network = torch.load('main_q_net' + mode + '.pt').to(device)
            self.target_q_network = torch.load('target_q_net' + mode + '.pt').to(device)
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)

    def save_net(self):
        torch.save(self.main_q_network, 'main_q_net' + self.mode + '.pt')
        torch.save(self.target_q_network, 'target_q_net' + self.mode + '.pt')

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = \
            self.make_minibatch()
        self.expected_state_action_values = self.get_expected_state_action_values()
        self.update_main_q_network()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.num_actions)]])
        return action

    def make_minibatch(self):
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        self.main_q_network.eval()
        self.target_q_network.eval()
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, self.batch.next_state)))
        next_state_values = torch.zeros(BATCH_SIZE)
        a_m = torch.zeros(BATCH_SIZE).type(torch.int64)
        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)
        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values
        return expected_state_action_values

    def update_main_q_network(self):
        self.main_q_network.train()
        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())


class Agent:
    def __init__(self, num_states, num_actions, is_train, mode):
        self.brain = Brain(num_states, num_actions, is_train, mode)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()


def get_RSRP(Ptx, RetTilt, distance=300):
    return Ptx - 10 * log10(num_tes * 1e4) - \
           22 * np.log10(np.repeat(distance, len(np.array(RetTilt))) / np.cos(RetTilt)) - 20 * log10(f) - 32.0
    # Ptx:dBm; distance:m; f:GHz;


class Env:
    def __init__(self, is_train, mode):
        if not is_train:  # 从分类数据中读取环境初始参数
            data = pd.read_csv("data_cover_for_dqn.csv", header=0, usecols=['MaxTxPower', 'RetTilt', 'Label'])
            if mode == "弱覆盖":
                data = data[data['Label'] == 1]
            else:
                data = data[data['Label'] == 2]
            global num_samples
            num_samples = data.shape[0]
            self.Rets = np.array(data['RetTilt'] / 10)  # 角度
            self.Ptxs = np.array(data['MaxTxPower'])  # W
        self.reset(is_train, mode, 0)

    def reward(self, Ret, Ptx):
        self.RSRP = get_RSRP(Ptx, Ret)
        snr = np.power(10, (self.RSRP - n0 - 10 * log10(bw / num_tes)) / 10)  # 比值
        R = (bw / num_tes) * np.log2(1 + snr)
        R_avg = R.mean()
        R_edge = np.array(sorted(R)[: int(num_tes * 0.3)]).mean()
        reward = lamuda * R_edge + (1 - lamuda) * R_avg
        return reward, R_avg, snr.mean()

    def step(self, action):
        Ptx_max = 10 * log10(3e4)  # 功率不大于30W
        Ptx_min = 30  # 功率不小于1W
        Ret_max = radians(30)  # 角度不大于30度
        Ret_min = radians(1)  # 角度不小于1度
        if action == 0 and self.Ptx < Ptx_max:
            self.Ptx += 1
        elif action == 1 and self.Ptx > Ptx_min:
            self.Ptx -= 1
        elif action == 2 and self.Ret < Ret_max:
            self.Ret += pi / 180
        elif action == 3 and self.Ret > Ret_min:
            self.Ret -= pi / 180
        reward, R_avg, snr = self.reward(self.Ret, self.Ptx)
        reward = reward / self.base
        RSRP_mean = self.RSRP.mean()
        state_next = np.hstack((self.Ret, self.Ptx, RSRP_mean))
        return state_next, reward, R_avg, snr, RSRP_mean

    def reset(self, is_train, mode, i):
        if is_train:
            if mode == '弱覆盖':  # 弱覆盖
                # 下倾角大，功率小
                MaxRet = random.randint(20, 30)
                MinRet = random.randint(1, MaxRet - 1)
                Ptx = normal(1, 0.02, 1)  # w
            else:  # 越区覆盖
                # 下倾角小，功率大
                MaxRet = random.randint(3, 10)
                MinRet = random.randint(1, MaxRet - 1)
                Ptx = normal(30, 3, 1)  # w
            Ret = random.randint(MinRet, MaxRet)  # 角度
        else:
            Ret = self.Rets[i * num_cell]
            Ptx = self.Ptxs[i * num_cell]
        self.Ret = radians(Ret)  # 弧度
        self.Ptx = 10 * log10(Ptx * 1e3)  # dBm
        self.base, _, _ = self.reward(self.Ret, self.Ptx)
        RSRP_mean = self.RSRP.mean()
        self.state_space = np.hstack((self.Ret, self.Ptx, RSRP_mean))
        self.action_space = np.hstack((self.Ret, self.Ptx))
        return self.state_space


class Environment:
    def __init__(self, is_train: bool, mode: str):
        self.is_train = is_train
        self.mode = mode
        self.env = Env(is_train, mode)
        num_states = self.env.state_space.shape[0]
        num_actions = self.env.action_space.shape[0] * 2
        self.agent = Agent(num_states, num_actions, is_train, mode)

    def run(self):
        reward, R, snr, RSRP = 0, 0, 0, 0
        rewards = []
        RSRPs = []
        Rs = []
        snrs = []
        for episode in range(NUM_EPISODES if self.is_train else num_samples // num_cell):
            state = self.env.reset(self.is_train, self.mode, episode)
            state = torch.from_numpy(state).type(torch.float32)
            state = torch.unsqueeze(state, 0).to(device)
            for step in range(MAX_STEPS):
                action = self.agent.get_action(state, episode).to(device)
                state_next, reward, R, snr, RSRP = self.env.step(action)
                state_next = torch.from_numpy(state_next).type(torch.float32).to(device)
                state_next = torch.unsqueeze(state_next, 0)
                if self.is_train:
                    self.agent.memorize(state, action, state_next, torch.tensor([reward], dtype=torch.float32))
                    self.agent.update_q_function()
                state = state_next
                if step == MAX_STEPS - 1:
                    print('%d Episode | Stopped after %d steps | r = %f' % (episode + 1, step + 1, reward))
            if self.is_train and episode % 2 == 0:
                self.agent.update_target_q_function()
            rewards.append(reward)
            RSRPs.append(RSRP)
            Rs.append(R)
            snrs.append(snr)
        if self.is_train:  # 画出训练趋势
            self.agent.brain.save_net()  # save the model
            plt.figure(1)
            plt.grid()
            plt.ylabel('reward')
            plt.xlabel('epoch')
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.title(self.mode)
            plt.plot(rewards)

            plt.figure(2)
            plt.grid()
            plt.ylabel('RSRP(dBm)')
            plt.xlabel('epoch')
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.title(self.mode)
            plt.plot(RSRPs)

            plt.figure(3)
            plt.grid()
            plt.ylabel('Throughput(bps)')
            plt.xlabel('epoch')
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.title(self.mode)
            plt.plot(Rs)

            plt.figure(4)
            plt.grid()
            plt.ylabel('SNR(ratio)')
            plt.xlabel('epoch')
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.title(self.mode)
            plt.plot(snrs)
            plt.show()
        else:  # 计算平均结果
            # rewards_avg = np.mean(rewards).item()
            # RSRPs_avg = np.mean(RSRPs).item()
            # Rs_avg = np.mean(Rs).item()
            # snrs_avg = np.mean(snrs).item()
            # print('reward: %f, RSRP: %f, R: %f, snr: %f' % (rewards_avg, RSRPs_avg, Rs_avg, snrs_avg))

            # save the data
            data_save = np.array((rewards, RSRPs, Rs, snrs)).transpose()
            col = ('reward', 'RSRP', 'throughput', 'snr')
            pd_data = pd.DataFrame(data_save, columns=col)
            pd_data.to_csv('data_plot_' + self.mode + '.csv', header=True, columns=col, index=False)


def data_plot3D(mode):
    # 绘制三维散点图
    # data = pd.read_csv("data_plot_all1.csv", header=0, usecols=('reward', 'RSRP', 'throughput', 'snr'))
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # X = [i for i in range(data.shape[0])]
    # Y = data['RSRP']
    # Z = data['throughput']
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # mpl.rcParams['legend.fontsize'] = 10
    # plt.title('覆盖总问题')
    # ax.scatter(X, Y, Z, c='r', marker='o')
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('RSRP(dBm)')
    # ax.set_zlabel('Throughput(bps)')
    # plt.savefig('data_plot_all1')

    # 绘制值函数三维图
    num_tiles = 50

    # 弱覆盖/越区覆盖
    Ret = np.linspace(radians(1), radians(30), num=num_tiles).repeat(num_tiles)
    Ptx = np.array([np.linspace(30, 10 * log10(3e4), num=num_tiles)] * num_tiles).reshape(num_tiles ** 2)
    Rsrp = get_RSRP(Ptx, Ret)
    state_cover1 = torch.tensor(np.vstack((Ret, Ptx, Rsrp)).transpose(), dtype=torch.float)
    model_cover1 = torch.load('main_q_net' + mode + '.pt').eval()
    value_cover1 = model_cover1(state_cover1).max(1)[0].detach().numpy()

    # all
    f_dl = np.repeat(399000 * 5e-6, num_cell * num_tiles ** 2)
    f_ul = np.repeat(399000 * 5e-6, num_tiles ** 2)
    Rsrp_main = Rsrp
    Rsrp_sub = get_RSRP(Ptx, Ret, 500)
    state_cover_all = torch.tensor(
        np.hstack((f_dl, f_ul, Ptx, Ret, Rsrp_main, Rsrp_sub)).reshape(9, -1).transpose(), dtype=torch.float)
    print(state_cover_all[1])
    model_cover_all = torch.load('main_q_net_all.pt')
    value_cover_all = model_cover_all(state_cover_all).max(1)[0].detach().numpy()

    fig1 = plt.figure(1)
    ax1 = fig1.gca(projection='3d')
    plt.title('Cover1')
    ax1.scatter(Ret * 180 / pi, Ptx, value_cover1, marker='o', label='cover1')
    ax1.set_xlabel('Ret(°)')
    ax1.set_ylabel('Ptx(dBm)')
    ax1.set_zlabel('Value')
    plt.legend()
    plt.savefig('value_scatter_cover1' + mode)

    fig2 = plt.figure(2)
    ax2 = fig2.gca(projection='3d')
    plt.title('Cover all')
    ax2.scatter(Ret * 180 / pi, Ptx, value_cover_all, marker='o', label='cover all')
    ax2.set_xlabel('Ret(°)')
    ax2.set_ylabel('Ptx(dBm)')
    ax2.set_zlabel('Value')
    plt.legend()
    plt.savefig('value_scatter_cover_all1')

    plt.show()


'''
mode = '弱覆盖' or '越区覆盖' 
'''
if __name__ == '__main__':
    # Environment(is_train=False, mode='弱覆盖').run()
    data_plot3D('弱覆盖')
