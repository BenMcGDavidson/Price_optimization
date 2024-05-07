#Import necessary packages
import concurrent.futures
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
from Price_Optimization.DQN.libr.DQNAgent import DQNAgent, DQNNetwork, CustomEnvironment
from Price_Optimization.DQN.libr.DQN_functions import calculate_optimal_epsilon, train_dqn_agent, process_future_result, evaluate_prices


sales = pd.read_csv('retail_price.csv')
# Initialize DQNAgent
state_size = 20
action_size = 1 #single value for price adjustments
num_episodes = 50000
max_time_steps = 1
agent = DQNAgent(state_size, action_size, epsilon=0.3)
# Initialize a list of epsilon values to experiment with
epsilon_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
num_products = len(sales['product_id'].unique())
batch_size = 100
optimal_epsilon = calculate_optimal_epsilon(sales, epsilon_values, agent, state_size, action_size, batch_size, num_products, num_episodes)
print("optimal epsilon = "+optimal_epsilon)
batch_size = 1000
training_results = train_dqn_agent(sales, optimal_epsilon, agent, state_size, action_size, batch_size, max_time_steps, num_products, num_episodes)
training_results_df = pd.DataFrame(training_results)

def calculate_loss_and_accuracy(agent, df):
    losses = []
    correct_decisions = 0
    total_decisions = 0
    
    # Iterate through the data
    for state, action, reward, next_state, done in df:
        # Predict Q-values for the current state
        q_values = agent.model(torch.tensor(np.reshape(state, [1, state_size]), dtype=torch.float32))
        predicted_q_value = q_values[0, action]

        # Compute target Q-value
        target_q_value = reward
        if not done:
            target_q_value += agent.gamma * torch.max(agent.model(torch.tensor(np.reshape(next_state, [1, state_size]), dtype=torch.float32))).item()

        # Compute loss
        loss = nn.MSELoss()(predicted_q_value, target_q_value)
        losses.append(loss.item())

        # Update accuracy
        if reward == 1:
            total_decisions += 1
            if predicted_q_value == target_q_value:
                correct_decisions += 1

    # Calculate mean loss
    mean_loss = np.mean(losses)

    # Calculate accuracy
    accuracy = correct_decisions / total_decisions if total_decisions > 0 else 0

    return mean_loss, accuracy
#loss, accuracy = calculate_loss_and_accuracy(agent, tdf)
#print(loss, accuracy)
training_results_df.to_csv('training_res_sm.csv')
