import concurrent.futures
import numpy as np
import pandas as pd
import torch
import time
from collections import deque
import random
from DL_RL_comp.lib.DQNAgent import DQNAgent, DQNNetwork, CustomEnvironment

def calculate_optimal_epsilon(df, epsilon_values, agent, state_size, action_size, batch_size, num_products, num_episodes=500000):
    """
    Train the DQN agent.

    Args:
    - epsilon_values: list. List of values to evaluate for epsilon.
    - state_size: int. Size of the state space.
    - action_size: int. Size of the action space.
    - batch_size: int. Batch size for training.
    - max_time_steps: int. Number of time steps per episode.
    - num_products: int. Number of products to evaluate
    - num_episodes: int. Number of episodes for training.

    Returns:
    - training_results: dict. Training results including rewards, actions, and confidence.
    """
    episode_rewards = []
    episode_actions = []
    episode_confidence = []
    episode_losses = []  # List to store loss values for each episode
    
    print('calculating optimal epsilon')

    for episode in range(num_episodes):
        episode_reward_list = []
        episode_action_list = []
        episode_confidence_list = []
        episode_loss_list = []  # List to store loss values for each iteration within an episode

        # Iterate through product IDs
        for product in df['product_id'].unique():
            initial_state = df[df['product_id'] == product].copy()  # Make a copy of the filtered DataFrame
            initial_state = initial_state[[ 'product_id',
                                            'total_price',
                                            'freight_price',
                                            'unit_price',
                                            'product_photos_qty',
                                            'product_weight_g',
                                            'product_score',
                                            'customers',
                                            'weekday',
                                            'weekend',
                                            'holiday',
                                            'month',
                                            'year',
                                            'volume',
                                            'comp_1',
                                            'freight_price_comp1',
                                            'comp_2',
                                            'freight_price_comp2',
                                            'comp_3',
                                            'freight_price_comp3',
                                            'lag_price']]
            env = CustomEnvironment(initial_state)

            state = env.reset()

            for i in range(1):
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action = agent.act(state_tensor)
                next_state, reward, done = env.step(action)

                agent.remember(state, action, reward, next_state, done)

                state = next_state
                episode_reward_list.append(reward)

                agent.replay(batch_size)

                if done:
                    break

            avg_reward = np.mean(episode_reward_list)
            episode_rewards.append(avg_reward)

        results[epsilon] = np.mean(episode_rewards)

    optimal_epsilon = max(results, key=results.get)
    print(f'Optimal Epsilon: {optimal_epsilon}')
    return optimal_epsilon

def train_dqn_agent(df, optimal_epsilon, agent, state_size, action_size, batch_size, max_time_steps, num_products, num_episodes=500000):
    """
    Train the DQN agent.

    Args:
    - optimal_epsilon: float. Optimal epsilon value.
    - state_size: int. Size of the state space.
    - action_size: int. Size of the action space.
    - batch_size: int. Batch size for training.
    - max_time_steps: int. Number of time steps per episode.
    - num_products: int. Number of products to evaluate
    - num_episodes: int. Number of episodes for training.

    Returns:
    - training_results: dict. Training results including rewards, actions, and confidence.
    """
    episode_rewards = []
    episode_actions = []
    episode_confidence = []
    episode_losses = []  # List to store loss values for each episode
    
    print('training agent')

    for episode in range(num_episodes):
        episode_reward_list = []
        episode_action_list = []
        episode_confidence_list = []
        episode_loss_list = []  # List to store loss values for each iteration within an episode

        # Iterate through modem IDs
        for product in df['product_id'].unique():
            initial_state = df[df['product_id'] == product].copy()  # Make a copy of the filtered DataFrame
            initial_state = initial_state[[ 'product_id',
                                            'total_price',
                                            'freight_price',
                                            'unit_price',
                                            'product_photos_qty',
                                            'product_weight_g',
                                            'product_score',
                                            'customers',
                                            'weekday',
                                            'weekend',
                                            'holiday',
                                            'month',
                                            'year',
                                            'volume',
                                            'comp_1',
                                            'freight_price_comp1',
                                            'comp_2',
                                            'freight_price_comp2',
                                            'comp_3',
                                            'freight_price_comp3',
                                            'lag_price']]
            env = CustomEnvironment(initial_state)

            state = env.reset()

            for i in range(max_time_steps):
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action = agent.act(state_tensor)

                next_state, reward, done = env.step(action)

                agent.remember(state, action, reward, next_state, done)

                state = next_state

                _, confidence = agent.model(torch.tensor(np.reshape(state, [1, state_size]), dtype=torch.float32))
                confidence_value = confidence[0, 1].item()

                episode_reward_list.append(reward)
                episode_action_list.append(action.item())
                episode_confidence_list.append(confidence_value)
                loss = agent.replay(batch_size)
                episode_loss_list.append(loss)  # Store the loss value

                if done:
                    break

        episode_rewards.append(episode_reward_list)
        episode_actions.append(episode_action_list)
        episode_confidence.append(episode_confidence_list)
        episode_losses.append(episode_loss_list)  # Append the list of loss values for the episode

    # Aggregate training results into a dictionary
    training_results = {
        'episode_num': list(range(num_episodes)),
        'episode_rewards': episode_rewards,
        'episode_actions': episode_actions,
        'episode_confidence': episode_confidence,
        'episode_losses': episode_losses  # Include loss values in the training results
    }

    return training_results
    
def evaluate_prices(df, agent, state_size, max_threads):
    """
    Evaluate modem data using the DQN agent and environment.

    Args:
    - df: dataframe with price history for each product
    - agent: DQNAgent instance.
    - state_size: int. Size of the state space.

    Returns:
    - modem_df: DataFrame. DataFrame containing evaluation results.
    """

    # Define a helper function for evaluating a single modem
    def evaluate_prices_helper(modem_id, modem_data):
        env = CustomEnvironment(modem_data)
        modem_results = []
        for index in range(len(modem_data)):
            state = env.state
            action = agent.act(state).item()
            next_state, reward, _ = env.step(action)
            with torch.no_grad():
                _, confidence = agent.model(torch.tensor(np.reshape(state, [1, state_size]), dtype=torch.float32))
                confidence_value = confidence[0, 1].item()
            # Extract confidence, append results, etc.
            modem_results.append({'machex': modem_id, 'action': action, 'confidence': confidence_value})
        return modem_results
    
    # Get unique modem IDs
    unique_modem_ids = df['machex'].unique()
    total_modems = len(unique_modem_ids)
    modems_evaluated = 0

    # Iterate over unique modem IDs and submit tasks to the thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        for modem_id in df['machex'].unique():
            modem_data = df[df['machex'] == modem_id][['hsdscore', 'feccorrected', 'fecuncorrected',
                                                        'uspower', 'dspower', 'ussnr', 'dsmer', 'reboot']]
            future = executor.submit(evaluate_prices_helper, modem_id, modem_data)
            future.add_done_callback(lambda future: process_future_result(future, results))
            modems_evaluated += 1
            print(f"Modem {modems_evaluated} out of {total_modems} evaluated")

    # Convert results list to a DataFrame
    modem_df = pd.DataFrame(results)
    
    # Add additional columns
    modem_df['model_version'] = '0.1.3'
    modem_df = modem_df[modem_df['action'] == 1]  # Filter rows where action is 1
    modem_df = modem_df[['machex', 'action', 'confidence', 'model_version']]  # Reorder columns if necessary

    return modem_df

def process_future_result(future, results):
    try:
        modem_results = future.result()
        results.extend(modem_results)
    except Exception as e:
        print(f"Error evaluating modem: {e}")
