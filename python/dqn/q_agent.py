from collections import deque, namedtuple
import random
import math
import gym
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import count
        
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ExperienceReplay(object):
    def __init__(self, capacity = 1000):
        self.experience = deque([], maxlen=capacity)

    def push(self, *args):
        self.experience.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.experience, batch_size)

    def __len__(self):
        return len(self.experience)

class DQN(nn.Module):
    def __init__(self, oberservation_size, action_size):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(oberservation_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )
        
    def forward(self, x):
        return self.model(x)

class DeepQAgent():
    def __init__(self, gym_env, alpha, epsilon, start_epsilon=0.9, decay_steps=1000, gamma=0.99, tau=0.01):
        if (epsilon > start_epsilon):
            start_epsilon = epsilon
            
        self.env = gym_env
            
        self.alpha: float = alpha
        self.target_epsilon: float = epsilon
        self.start_epsilon: float = start_epsilon
        self.decay_steps: float = decay_steps
        self.gamma: float = gamma
        self.tau = tau
        self.steps: int = 0
        
        n_actions = self.env.action_space.n
        state = self.env.reset()
        n_observations = len(state)
        
        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=alpha, amsgrad=True)

        self.experience = ExperienceReplay()
        
    # Update and return the return
    def update(self, batch_size):
        if len(self.experience) < batch_size:
            return
    
        transitions = self.experience.sample(batch_size)
        
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = torch.gather(self.policy_net(state_batch), 1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = torch.max(self.target_net(non_final_next_states), 1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        

    def choose_action(self, state, epsilon):
        rand = random.random()
        if rand > epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)
        
    def train(self, episodes, render=False):
        episode_durations = []
        
        # to run GUI event loop
        plt.ion()
        
        # here we are creating sub plots
        figure, ax = plt.subplots()
        line1, = ax.plot([], [])
        
        # setting x-axis label and y-axis label
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        
        for i in range(episodes):
            # Initialize the environment and get its state
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            for t in count():
                # Update epsilon
                epsilon = (self.start_epsilon - self.target_epsilon) * math.exp(-self.steps / self.decay_steps) + self.target_epsilon
                self.steps += 1
                
                action = self.choose_action(state, epsilon)
                observation, reward, terminated, truncated = self.env.step(action.item())
                reward = torch.tensor([reward + 5 if truncated else 0])
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float).unsqueeze(0)# Store the transition in memory
        
                self.experience.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.update(128)
                
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)
                
                if render:
                    self.env.render()

                if done:
                    episode_durations.append(t + 1)
                    
                    if render:
                        line1.set_xdata(range(len(episode_durations)))
                        line1.set_ydata(episode_durations)
                        ax.set_xlim([0, len(episode_durations)])
                        ax.set_ylim([0, max(episode_durations)])
                        figure.canvas.draw()
                        figure.canvas.flush_events()
                    
                    break
            
if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    agent = DeepQAgent(env, 0.0001, 0.005)

    agent.train(1000, True)
        


