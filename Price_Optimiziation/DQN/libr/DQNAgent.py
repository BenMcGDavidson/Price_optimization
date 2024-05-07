#Import necessary packages
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

#Define DQNAgent
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Softmax(dim=None)  # Only specify the activation function
        self.fc2 = nn.Softmax(dim=None)
        self.fc3 = nn.Softmax(dim=None)
        self.confidence_layer = nn.Softmax(dim=None)

        # Define linear layers separately
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, output_size)
        self.confidence_linear = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc1(self.linear1(x))
        x = self.fc2(self.linear2(x))
        q_values = self.fc3(self.linear3(x))
        confidence = torch.sigmoid(self.confidence_layer(self.confidence_linear(x)))
        return q_values, confidence

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, input_size, action_size, epsilon, output_size=2, capacity=2000, gamma=0.9):
        self.model = DQNNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = ReplayMemory(capacity)
        self.epsilon = epsilon
        self.gamma = gamma
        self.output_size = output_size

    def remember(self, state, action, reward, next_state, done):
        print("State shape: ", state.shape)
        with torch.no_grad():
            q_values, confidence = self.model(state)
        experience = (state, action, reward, next_state, done, confidence)
        self.memory.push(experience)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))  # Convert zip object to list

        state_batch = torch.stack(batch[0])
        action_batch = torch.tensor(batch[1], dtype=torch.long)
        reward_batch = [r if r is not None else 0 for r in batch[2]]
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        next_state_batch = torch.stack(batch[3])
        done_batch = torch.tensor(batch[4], dtype=torch.float32)
        confidence_batch = torch.stack(batch[5])
        
        # Calculate Q-values
        current_q_values, current_confidence = self.model(state_batch)
        current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1))
    
        # Get Q-values for next state
        next_q_values, _ = self.model(next_state_batch)  # Corrected line
        next_q_values = next_q_values.max(1)[0].detach()  # Corrected line
        
        # Ensure done_batch is a tensor
        done_batch = torch.tensor(batch[4], dtype=torch.float32)
        # Convert reward_batch to tensor
        reward_batch = torch.tensor(batch[2], dtype=torch.float32)
    
        target_q_values = reward_batch + (1 - done_batch) * 0.9 * next_q_values

        # Compute loss and perform a gradient step
        loss_q_values = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        loss_confidence = nn.BCELoss()(current_confidence, confidence_batch)
        loss = loss_q_values + loss_confidence
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def act(self, state):
        # Implement the action selection logic with exploration
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action within a certain range
            # Define the range for possible price adjustments
            min_adjustment = -10  # Example: minimum price adjustment
            max_adjustment = 10   # Example: maximum price adjustment
            # Choose a random price adjustment within the defined range
            action = np.random.uniform(min_adjustment, max_adjustment)
            # Convert the chosen action to a PyTorch tensor
            action_tensor = torch.tensor([action], dtype=torch.float32)
        else:
            # Exploit: choose the action with the highest Q-value
            with torch.no_grad():
                q_values, _ = self.model(state)
                action_tensor = torch.argmax(q_values).unsqueeze(0)
        
        return action_tensor

#Custom environment dependent on successive states for reboot rewards
class CustomEnvironment:
    def __init__(self, initial_state, num_rows_before=1, num_rows_after=1):
        if isinstance(initial_state, pd.DataFrame):
            self.initial_state = initial_state
            initial_state = initial_state.iloc[:, 1:].to_numpy()  # Convert DataFrame to NumPy array
        elif isinstance(initial_state, np.ndarray):
            self.initial_state = None
            pass  # No conversion needed
        elif isinstance(initial_state, torch.Tensor):
            self.initial_state = None
            initial_state = initial_state.numpy()  # Convert PyTorch tensor to NumPy array
        else:
            raise ValueError("Unsupported initial_state type")
        
        self.state = initial_state  # Store the initial state directly as 'state'
        self.num_rows_before = 1  # Set default values for these attributes
        self.num_rows_after = 1
        self.current_index = 0
        self.start_index = 0
        self.end_index = min(len(self.state), self.current_index + self.num_rows_after + 1)  # Initialize end_index

    def reset(self):
        # Reset the environment to the initial state
        self.current_index = 0
        self.start_index = 0
        self.end_index = min(len(self.state), self.current_index + self.num_rows_after + 1)
        return self.state[self.start_index:self.end_index]  

    def step(self, action):
        # Increment the index to get the next row of data
        self.current_index += 1
        current_index = self.current_index
        
        if current_index >= len(self.initial_state) or self.end_index >= len(self.initial_state):
            return self.state, 0, True
        else:
            # Retrieve the next row of data
            self.start_index += 1
            self.end_index += 1
            next_state_slice = self.initial_state.iloc[self.start_index:self.end_index, 1:]
            
            # Check if the slice is empty
            if next_state_slice.empty:
                print("Next state slice is empty. Returning current state.")
                return self.state, 0, True
            
            # Convert the slice to a numpy array and then to a PyTorch tensor
            next_state_array = next_state_slice.to_numpy()
            self.state = torch.tensor(next_state_array, dtype=torch.float32)
            
            # Calculate the reward based on changes in health variables
            reward = self.calculate_reward(current_index)
            
            # Return the next state, reward, and done flag
            return self.state, reward, False

    def calculate_reward(self, current_index):
        current_index = int(current_index)
        total_before = self.initial_state.iloc[:current_index, 'total_price']
        total_after = self.initial_state.iloc[current_index:, 'total_price']
        volume_before = self.initial_state.iloc[:current_index, 'volume']
        volume_after = self.initial_state.iloc[current_index, 'volume']
        
        delta_total = total_after.mean() - total_before.mean()
        if len(total_before) < len(total_after):
            total_after = total_after[-1*len(total_before):]
        elif len(total_after) < len(total_before):
            total_before = total_before[-1*len(total_after):]
        delta_volume = volume_after.mean() - volume_before.mean()
        if len(volume_before) < len(volume_after):
            volumeafter = volume_after[-1*len(volume_before):]
        elif len(volume_after) < len(volume_before):
            volume_before = volume_before[-1*len(volume_after):]

        try:
            total_stat, total_p = wilcoxon(total_before, total_after, nan_policy='omit')
        except ValueError:
            total_p = 1
        try:
            vol_stat, vol_p = wilcoxon(volume_before, volume_after, nan_policy='omit')
        except ValueError:
            vol_p = 1
        if (total_p < 0.05 or vol_p < 0.05) and (delta_total > 1 or delta_volume > 1):
            reward = 10  # Good
        elif (total_p < 0.05 or vol_p < 0.05) and (delta_total < 0 or delta_volume < 0):
            reward = -5  # bad change
        else: 
            reward = 0  # no change

        return torch.tensor(reward, dtype=torch.float32)
