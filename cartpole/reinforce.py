import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)
    
def select_action(policy_net, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy_net(state)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy = PolicyNet(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

reward_history = []

for episode in range(500):
    state, _ = env.reset()
    log_probs = []
    rewards = []
    done = False
    total_rewards = 0

    while not done:
        action, log_prob = select_action(policy, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state
        total_rewards += reward

    returns = compute_returns(rewards)
    returns =  torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std()+1e-9)

    loss = []
    for log_prob, Gt in zip(log_probs, returns):
        baseline = returns.mean()
        loss.append(-log_prob * (Gt - baseline))

    loss = torch.stack(loss).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(episode +1)%10 ==0:
        print(f"Episode {episode +1}, Total Reward: {total_rewards}")

    reward_history.append(total_rewards)

import matplotlib.pyplot as plt

plt.plot(reward_history)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('REINFORCE CartPole Performance')
plt.grid(True)
plt.show()
