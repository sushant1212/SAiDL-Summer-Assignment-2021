from numpy.core.fromnumeric import mean
import torch
from optim_env import OptimEnv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

import torch.distributions.normal as n
import torch.distributions.categorical as c
# torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


###############################################
LR = 0.00001
NUM_EPISODES = 10000
GAMMA = 0.99
###############################################


class Reinforce(nn.Module):
    def __init__(self):
        super(Reinforce,self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(8,20),
            nn.ReLU(),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20,4),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.linear_stack(x)*2
        return x

class Agent:
    def __init__(self) -> None:
        self.model = Reinforce().to(device)
        self.env = OptimEnv()
        self.state = self.env.reset()
        self.episode = []
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.G = []
        self.losses = []
        self.total_rewards = 0

    def sample_episode(self):
        self.total_rewards = 0
        self.losses = []
        done = False
        state = self.state
        while done==False:
            output = (self.model((torch.from_numpy(state).unsqueeze(0)).float().to(device)))
            mean_x, std_x, mean_y, std_y = output[0]
            # print(mean_x, mean_y)
            try:
                m1 = n.Normal(mean_x, abs(std_x), validate_args=None)
                m2 = n.Normal(mean_y, abs(std_y), validate_args=None)
            except:
                continue
            action_x = m1.rsample()
            action_y = m2.rsample()
            action = np.array([action_x.item(),action_y.item()])
            # m = c.Categorical(output)
            # action = m.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.episode.append([state, [action_x, action_y], reward])
            self.losses.append(-m1.log_prob(action_x)-m2.log_prob(action_y))
            state = next_state
            self.total_rewards += reward
            # self.env.render()
        self.G = []
        
        for i in range(len(self.episode)):
            self.G.append(0)
            for j in range(i,len(self.episode)):
                _,_,r = self.episode[j]
                self.G[-1] += (GAMMA**(j-i))*r
        
        
    def train_loop(self):
        self.sample_episode()
        loss = 0
        for i in range(len(self.episode)):
            gt = self.G[i]
            li = self.losses[i]
            loss +=  li * gt
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.state = self.env.reset()
        self.episode = []
        return self.total_rewards

if __name__ == "__main__":
    agent = Agent()
    for i in range(NUM_EPISODES):
        total_reward = agent.train_loop()
        writer.add_scalar("reward", total_reward, i)
        writer.flush()
        print(f"Number of Episodes : {i+1}   Total reward = {total_reward}" )
        # print()
        # agent.env.render()
        # print('\n\n')
    writer.close()
    
