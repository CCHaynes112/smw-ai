import numpy as np
import pickle
import time
from smw_api import MarioAPI
from train import calculate_reward, discretize_state, take_action_in_emulator

# Load Q-table
with open("../models/q_table_smw.pkl", "rb") as f:
    q_table = pickle.load(f)


def test(episodes=10):
    # Wait 3 seconds before starting the testing process
    print("Testing will start in 3 seconds...")
    time.sleep(3)

    # Create the Mario API object
    mario_api = MarioAPI()
    total_rewards = []

    for episode in range(episodes):
        # Reset the game environment (e.g., restart ROM)
        mario_api.reset_rom()

        # Get initial state after resetting the game
        mario_api.get_current_state_from_emulator()
        state_discrete = discretize_state(mario_api)
        done = False
        total_reward = 0
        actions = ["Move Left", "Move Right", "Jump", "Stop Run", "Spin Jump", "Crouch", "Look Up"]

        # Print out the Q-values along with the corresponding actions
        for action, q_value in zip(actions, q_table[state_discrete]):
            print(f"{action}: {q_value}")
        print("~~~~~~~~~~~~~~~~~~~~~~~")

        g_rewards_for_x_positions = []

        while not done:
            # Select the best action from the Q-table
            action = np.argmax(q_table[state_discrete])  # Best action from Q-table

            # Apply the action in the emulator (move Mario or perform an action)
            take_action_in_emulator(mario_api, action)

            # Get the next state and reward
            mario_api.get_current_state_from_emulator()
            reward, done, reward_for_x_position, _, _, _, _, _, _, _ = calculate_reward(mario_api)

            g_rewards_for_x_positions.append(reward_for_x_position)

            # Discretize the next state
            state_discrete = discretize_state(mario_api)

            highest_x_position = max(g_rewards_for_x_positions)
            total_reward += highest_x_position

            # Accumulate total reward
            total_reward += reward

            # Slow down the game for visualization purposes
            time.sleep(0.01)

        total_rewards.append(total_reward)
        print(f"Episode: {episode + 1}, Reward: {total_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {episodes} episodes: {avg_reward:.2f}")


if __name__ == "__main__":
    test()
