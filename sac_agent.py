import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.autograd import Variable
from torch.distributions import Normal
from collections import deque, namedtuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor and Critic Networks
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_size=128):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_size=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)


    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class SACAgent:
    def __init__(self, state_size, action_size, seed, buffer_size=int(1e5), batch_size=64, gamma=0.99, tau=1e-3,
                 lr_actor=0.0001, lr_critic=0.0001, alpha=0.2, update_every=1):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.update_every = update_every
        self.t_step = 0

        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        self.critic_local_1 = Critic(state_size, action_size, seed).to(device)
        self.critic_target_1 = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer_1 = optim.Adam(self.critic_local_1.parameters(), lr=lr_critic)

        self.critic_local_2 = Critic(state_size, action_size, seed).to(device)
        self.critic_target_2 = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer_2 = optim.Adam(self.critic_local_2.parameters(), lr=lr_critic)

        self.memory = ReplayBuffer(buffer_size, batch_size, seed)

        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target_1, self.critic_local_1)
        self.hard_update(self.critic_target_2, self.critic_local_2)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state.reshape(1, -1)).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.alpha * np.random.normal(0, 0.1, size=self.action_size)
        return np.clip(action, -1, 1)

    def reset(self):
        pass

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Update critic
        next_actions = self.actor_target(next_states)
        Q_target_next_1 = self.critic_target_1(next_states, next_actions)
        Q_target_next_2 = self.critic_target_2(next_states, next_actions)
        Q_target_next = torch.min(Q_target_next_1, Q_target_next_2)
        Q_targets = rewards + (self.gamma * Q_target_next * (1 - dones))
        Q_expected_1 = self.critic_local_1(states, actions)
        Q_expected_2 = self.critic_local_2(states, actions)
        critic_loss_1 = nn.functional.mse_loss(Q_expected_1, Q_targets.detach())
        critic_loss_2 = nn.functional.mse_loss(Q_expected_2, Q_targets.detach())

        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        # Update actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local_1(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.critic_target_1, self.critic_local_1, self.tau)
        self.soft_update(self.critic_target_2, self.critic_local_2, self.tau)
        self.soft_update(self.actor_target, self.actor_local, self.tau)



