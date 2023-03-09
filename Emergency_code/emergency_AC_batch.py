'''
File Name: emergency_AC_batch
Create File Time: 2023/3/6 22:27
File Create By Author: Yang Guanqun
Email: yangguanqun01@corp.netease.com
Corp: Fuxi Tech, Netease
'''

import copy
import platform

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if platform.system() == "Darwin":
    PYTORCH_ENABLE_MPS_FALLBACK = 1
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

value_method = "TD_error"
all_value_method = ["Q_value", "V_value", "TD_error", "Advantage"]
assert value_method in all_value_method, "You choose a wrong value_method!"


class ActorCriticTrainer(nn.Module):
    def __init__(self, env):
        super(ActorCriticTrainer, self).__init__()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.create_training_network()
        self.create_training_method()
        self.GAMMA = 0.9
        self.to(device)

        self.state_batch = []
        self.action_batch = []
        self.reward_batch = []
        self.next_state_batch = []

    def create_training_network(self):
        self.fc = nn.Linear(self.state_dim, 20)
        self.critic = nn.Sequential(self.fc, nn.ReLU(), nn.Linear(20, 1))
        self.actor = nn.Sequential(self.fc, nn.ReLU(), nn.Linear(20, self.action_dim))

    def create_training_method(self):
        self.optim = optim.Adam(self.parameters(), lr=0.001)
        self.value_loss = nn.MSELoss()
        self.actor_loss = nn.LogSoftmax(dim=-1)

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, device=device)
            action_probs = F.softmax(self.actor(state), dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            return action

    def calculate_batch_td_error(self, state_batch: torch.Tensor, reward_batch: torch.Tensor):
        value_batch = self.critic(state_batch)
        values = copy.copy(value_batch).squeeze(-1)
        next_values = torch.cat((copy.copy(value_batch[1:, :]).squeeze(-1), torch.tensor([0.], device=device)))
        td_errors = reward_batch + self.GAMMA * next_values - values
        return td_errors

    def calculate_policy_loss(self, state_batch, action_batch, td_errors):
        action_logits_batch = self.actor(state_batch)
        log_probs = torch.log(F.softmax(action_logits_batch, dim=-1))
        action_log_probs = torch.gather(log_probs, 1, action_batch.unsqueeze(-1)).squeeze(-1)
        policy_loss = action_log_probs * torch.abs(td_errors)
        return policy_loss

    def perceive(self, state, action, reward, next_state):
        self.state_batch.append(state)
        self.action_batch.append(action)
        self.reward_batch.append(reward)
        self.next_state_batch.append(next_state)

    def train_loop(self):
        state_batch = torch.tensor(self.state_batch, device=device)
        action_batch = torch.tensor(self.action_batch, device=device)
        reward_batch = torch.tensor(self.reward_batch, device=device)

        td_errors = self.calculate_batch_td_error(state_batch, reward_batch)
        value_loss = torch.square(td_errors).mean()
        policy_loss = self.calculate_policy_loss(state_batch, action_batch, td_errors.detach()).mean()
        loss = value_loss - policy_loss
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def clear_list(self):
        self.state_batch.clear()
        self.action_batch.clear()
        self.reward_batch.clear()
        self.next_state_batch.clear()


import gym
env_name = "CartPole-v1"
env = gym.make(env_name)
agent = ActorCriticTrainer(env)



import time
def main():
    start_time = time.time()
    for episode in range(3000):
        state, _ = env.reset()
        for step in range(300):
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = -1 if done else 0.01
            agent.perceive(state,action,reward,next_state)
            state = next_state
            if done:
                agent.train_loop()
                agent.clear_list()
                break
        if episode % 100 == 0 and episode != 0:
            total_reward = 0
            for i in range(10):
                state, _ = env.reset()
                for step in range(300):
                    action = agent.choose_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    total_reward += reward
                    state = next_state
                    if done:
                        break
            print(f"episode {episode} total reward is {total_reward/10}")
    end_time = time.time()
    print(f"total time is {end_time - start_time}")


if __name__ == "__main__":
    main()