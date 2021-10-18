import random
from collections import namedtuple
from math import *
from random import randint, randrange, choices
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
重叠覆盖优化算法
'''

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 100
BATCH_SIZE = 32
CAPACITY = 10000
device = 'cpu'
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)
num_samples = 0

C1 = 1
C2 = 0.1
bw = 1e7  # Hz
N0 = pow(10, -17.4)  # mW
num_cells = 4
num_tes = 10
distance_main = np.random.normal(300, 30, num_tes)
distance_sub1 = np.random.normal(500, 30, num_tes)
distance_sub2 = np.random.normal(500, 30, num_tes)
distance_sub3 = np.random.normal(500, 30, num_tes)
Ret = (np.random.randint(1, 10, 1) * pi) / 180  # 弧度
Ptx = 10 * log10(np.random.normal(30, 3, 1) * 1e3)  # dBm
# f = 2.4  # Ghz
dl = [randrange(422000, 434000, 20), randrange(386000, 398000, 20), randrange(361000, 376000, 20),
      randrange(173800, 178800, 20), randrange(524000, 538000, 20), randrange(185000, 192000, 20),
      randrange(145800, 149200, 20), randrange(151600, 153600, 20), randrange(172000, 175000, 20),
      randrange(158200, 164200, 20), randrange(386000, 399000, 20), randrange(171800, 178800, 20),
      randrange(151600, 160600, 20), randrange(470000, 472000, 20), randrange(402000, 405000, 20),
      randrange(514000, 524000, 20), randrange(376000, 384000, 20), randrange(460000, 480000, 20),
      randrange(499200, 537999, 3), randrange(499200, 537999, 6), randrange(636667, 646666, 1),
      randrange(636668, 646666, 2), randrange(286400, 303400, 20), randrange(285400, 286400, 20),
      randrange(496700, 499000, 20), randrange(422000, 440000, 20), randrange(399000, 404000, 20),
      randrange(123400, 130400, 20), randrange(295000, 303600, 20), randrange(620000, 680000, 1),
      randrange(620000, 680000, 2), randrange(620000, 653333, 1), randrange(620000, 653332, 2),
      randrange(693334, 733333, 1), randrange(693334, 733332, 2), randrange(499200, 538000, 20)]
dl.sort()


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
    def __init__(self, num_states, num_actions, is_train):
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)
        self.num_states = num_states
        n_in, n_mid, n_out = num_states, 32, num_actions
        if is_train:
            self.main_q_network = Net(n_in, n_mid, n_out)
            self.target_q_network = Net(n_in, n_mid, n_out)
        else:
            self.main_q_network = torch.load('main_q_net重叠覆盖.pt').to(device)
            self.target_q_network = torch.load('target_q_net重叠覆盖.pt').to(device)
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)

    def save_net(self):
        torch.save(self.main_q_network, 'main_q_net重叠覆盖.pt')
        torch.save(self.target_q_network, 'target_q_net重叠覆盖.pt')

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
                action = self.main_q_network(state).max(1)[1].view(1, 1).to(device)
        else:
            action = torch.tensor([[random.randrange(self.num_actions)]])
        return action

    def make_minibatch(self):
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        self.main_q_network.eval()
        self.target_q_network.eval()
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, self.batch.next_state)))
        next_state_values = torch.zeros(BATCH_SIZE)
        a_m = torch.zeros(BATCH_SIZE).type(torch.int64)
        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1].to(device)
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1).to(device)
        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze().to(device)
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values.to(device)
        return expected_state_action_values

    def update_main_q_network(self):
        self.main_q_network.train().to(device)
        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())


class Agent:
    def __init__(self, num_states, num_actions, is_train):
        self.brain = Brain(num_states, num_actions, is_train)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()


class Env:
    def __init__(self, is_train):
        if is_train:
            # 重复抽样nrcell_num个，随机分布在dl1中位数两侧，取前num_cells个
            self.Arfcn_DL = np.array(choices(
                dl[len(dl) // 2 - randint(1, 6): len(dl) // 2 + randint(1, 6)], k=12))[:num_cells]
            # print(self.Arfcn_DL)
            self.Afrcn_to_Freq()
            self.get_inter()  # 干扰指数 下行邻频越接近越大
            self.R_base = self.get_R()
            self.state_space = np.hstack((self.f_dl, self.RSRP_sub_sum_avg))
        else:
            data = pd.read_csv("data_cover_for_dqn.csv", header=0, usecols=['ArfcnDL', 'Label'])
            data = data[data['Label'] == 3]
            global num_samples
            num_samples = data.shape[0]
            self.Arfcn_DLs = np.array(data['ArfcnDL'])
            self.reset(is_train, 0)

    def get_RSRP(self, distance, f):
        RSRP_dBm = Ptx - (10 * log10(num_tes * 1e4) + 22 * np.log10(distance / cos(Ret)) + 20 * log10(f) + 32.0)
        return np.power(10, RSRP_dBm / 10)  # Ptx:mW; distance:m; f:GHz;

    def Afrcn_to_Freq(self):
        self.f_dl = 5e-3 * self.Arfcn_DL * 1e-3  # Ghz    arfcn < 599999 / f < 3GHz

    def get_inter(self):
        self.inter = self.f_dl[0] / (self.f_dl[0] + 1e2 * abs(self.f_dl[0] - self.f_dl[1:]))

    def get_R(self):
        self.RSRP_main = self.get_RSRP(distance_main, self.f_dl[0])
        self.RSRP_sub1 = self.get_RSRP(distance_sub1, self.f_dl[1])
        self.RSRP_sub2 = self.get_RSRP(distance_sub2, self.f_dl[2])
        self.RSRP_sub3 = self.get_RSRP(distance_sub3, self.f_dl[3])
        self.RSRP_sub_sum = \
            self.RSRP_sub1 * self.inter[0] + self.RSRP_sub2 * self.inter[1] + self.RSRP_sub3 * self.inter[2]
        self.RSRP_sub_sum_avg = self.RSRP_sub_sum.mean()
        self.snr = self.RSRP_main / (N0 * (bw / num_tes) + self.RSRP_sub_sum)  # 比值
        R = ((bw / num_tes) * np.log2(1 + self.snr)).mean()
        return R

    def step(self, action):
        if action == 0 and self.Arfcn_DL[0] > 362280:
            self.Arfcn_DL[0] -= 20
        elif action == 1 and self.Arfcn_DL[0] < 499302:
            self.Arfcn_DL[0] += 20
        self.Afrcn_to_Freq()
        self.get_inter()
        R = self.get_R()
        reward = R / self.R_base
        state_next = np.hstack((self.f_dl, self.RSRP_sub_sum_avg))
        return state_next, reward, 10 * log10(self.RSRP_main.mean()), self.snr.mean(), R

    def reset(self, is_train, i):
        if is_train:
            self.Arfcn_DL = self.state_space[:-1] / (5e-3 * 1e-3)
            self.Afrcn_to_Freq()
            self.get_inter()
        else:
            self.Arfcn_DL = self.Arfcn_DLs[i * num_cells: (i + 1) * num_cells]
            self.Afrcn_to_Freq()
            self.get_inter()  # 干扰指数 下行邻频越接近越大
            self.R_base = self.get_R()
            self.state_space = np.hstack((self.f_dl, self.RSRP_sub_sum_avg))
        return self.state_space


class Environment:
    def __init__(self, is_train: bool):
        self.is_train = is_train
        self.env = Env(is_train)
        num_states = self.env.state_space.shape[0]
        num_actions = 2
        self.agent = Agent(num_states, num_actions, is_train)

    def run(self):
        reward, R, snr, RSRP = 0, 0, 0, 0
        rewards = []
        RSRPs = []
        Rs = []
        snrs = []
        for episode in range(NUM_EPISODES if self.is_train else num_samples // num_cell):
            state = self.env.reset(self.is_train, episode)
            state = torch.from_numpy(state).type(torch.float32)
            state = torch.unsqueeze(state, 0)
            for step in range(MAX_STEPS):
                action = self.agent.get_action(state, episode)
                state_next, reward, RSRP, snr, R = self.env.step(action)
                state_next = torch.from_numpy(state_next).type(torch.float32)
                state_next = torch.unsqueeze(state_next, 0)
                if self.is_train:
                    self.agent.memorize(state, action, state_next, torch.tensor([reward], dtype=torch.float32))
                    self.agent.update_q_function()
                state = state_next
                if step == MAX_STEPS - 1:
                    print('%d Episode | Stopped after %d steps | r = %f' % (episode + 1, step + 1, reward))
            if self.is_train and episode % 2 == 0:
                self.agent.update_target_q_function()
            Rs.append(R)
            snrs.append(snr)
            rewards.append(reward)
            RSRPs.append(RSRP)
        if self.is_train:  # 画出训练趋势
            self.agent.brain.save_net()  # save the model
            plt.figure(1)
            plt.grid()
            plt.ylabel('reward')
            plt.xlabel('epoch')
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.title('重叠覆盖')
            plt.plot(rewards)

            plt.figure(2)
            plt.grid()
            plt.ylabel('RSRP(dBm)')
            plt.xlabel('epoch')
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.title('重叠覆盖')
            plt.plot(RSRPs)

            plt.figure(3)
            plt.grid()
            plt.ylabel('Throughput(bps)')
            plt.xlabel('epoch')
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.title('重叠覆盖')
            plt.plot(Rs)

            plt.figure(4)
            plt.grid()
            plt.ylabel('SNR(ratio)')
            plt.xlabel('epoch')
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.title('重叠覆盖')
            plt.plot(snrs)
            plt.show()
        else:   # 计算平均结果
            # rewards_avg = np.mean(rewards).item()
            # RSRPs_avg = np.mean(RSRPs).item()
            # Rs_avg = np.mean(Rs).item()
            # snrs_avg = np.mean(snrs).item()
            # print('reward: %f, RSRP: %f, R: %f, snr: %f' % (rewards_avg, RSRPs_avg, Rs_avg, snrs_avg))

            # save the data
            data_save = np.array((rewards, RSRPs, Rs, snrs)).transpose()
            col = ('reward', 'RSRP', 'throughput', 'snr')
            pd_data = pd.DataFrame(data_save, columns=col)
            pd_data.to_csv('data_plot_重叠覆盖.csv', header=True, columns=col, index=False)


def data_plot3D():
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

    # 重叠覆盖
    f_dl = np.linspace(362280 * 5e-6, 499302 * 5e-6, num_tiles).repeat(num_tiles)
    f_dl_c = np.repeat(362280 * 5e-6, num_tiles ** 2)
    f_dl_delt = f_dl - f_dl_c
    f_dl = np.vstack((f_dl, f_dl_c, f_dl_c, f_dl_c))
    Rsrp_sub = np.array([np.linspace(-115, -95, num_tiles)] * num_tiles).reshape(1, num_tiles ** 2)
    state_cover2 = torch.tensor(np.vstack((f_dl, Rsrp_sub)).transpose(), dtype=torch.float)
    model_cover2 = torch.load('main_q_net重叠覆盖.pt').eval()
    value_cover2 = model_cover2(state_cover2).max(1)[0].detach().numpy()

    # all
    Ptx = np.repeat(10 * log10(30 * 1e3), num_tiles ** 2)
    Ret = np.repeat(radians(5), num_tiles ** 2)
    f_ul = np.repeat(399000 * 5e-6, num_tiles ** 2)
    Rsrp_main = Ptx - (10 * log10(num_tes * 1e4) + 22 * np.log10(300 / np.cos(Ret)) + 20 * log10(2.4) + 32.0)
    state_cover_all = torch.tensor(
        np.vstack((f_dl, f_ul, Ptx, Ret, Rsrp_main, Rsrp_sub)).transpose(), dtype=torch.float)
    model_cover_all = torch.load('main_q_net_all.pt')
    value_cover_all = model_cover_all(state_cover_all).max(1)[0].detach().numpy()

    fig1 = plt.figure(1)
    ax1 = fig1.gca(projection='3d')
    plt.title('Cover2')
    ax1.scatter(f_dl_delt, Rsrp_sub, value_cover2, marker='o', label='cover2')
    ax1.set_xlabel('Δf(GHz)')
    ax1.set_ylabel('Rsrp_sub(dBm)')
    ax1.set_zlabel('Value')
    plt.legend()
    plt.savefig('value_scatter_cover2')

    fig2 = plt.figure(2)
    ax2 = fig2.gca(projection='3d')
    plt.title('Cover all')
    ax2.scatter(f_dl_delt, Rsrp_sub, value_cover_all, marker='o', label='cover all')
    ax2.set_xlabel('Δf(GHz)')
    ax2.set_ylabel('Rsrp_sub(dBm)')
    ax2.set_zlabel('Value')
    plt.legend()
    plt.savefig('value_scatter_cover_all2')

    plt.show()


if __name__ == '__main__':
    # Environment(is_train=False).run()
    data_plot3D()
