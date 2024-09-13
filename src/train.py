import os
import time
import numpy as np
import pickle

from smw_api import MarioAPI

# Q-Learning parameters
LEARNING_RATE = 0.05
DISCOUNT_FACTOR = 0.99
EPISODES = 300

# Discrete state buckets
STATE_BUCKETS = [
    50,  # X position (discretized into 50 buckets)
    30,  # Y position (discretized for vertical movement)
    10,  # X velocity (discretized into 10 buckets)
    2,  # Grounded status (binary: grounded or not)
    2,  # Is running status (0 = not running, 1 = running)
    20,  # Time (discretized into 20 buckets)
]

STATE_BOUNDS = [
    [8, 4090],  # X position (from 8 to 4090)
    [0, 352],  # Y position (from 0 to 352, with 352 as the ground level)
    [0, 112],  # X velocity (from 0 to 112)
    [0, 256],  # Grounded status (0 = in the air, 256 = grounded)
    [0, 16],  # Is running status (0 = not running, 16 = running)
    [0, 400],  # Time (from 0 to 400)
]


def discretize_state(mario_api: MarioAPI):
    discrete_state = []

    # Discretize X position
    x_bucket = int(
        (mario_api.mario_x_position - STATE_BOUNDS[0][0])
        / ((STATE_BOUNDS[0][1] - STATE_BOUNDS[0][0]) / STATE_BUCKETS[0])
    )
    x_bucket = min(STATE_BUCKETS[0] - 1, max(0, x_bucket))  # Clamp to valid range
    discrete_state.append(x_bucket)

    # Discretize Y position.
    y_bucket = int(
        (mario_api.mario_y_position - STATE_BOUNDS[1][0])
        / ((STATE_BOUNDS[1][1] - STATE_BOUNDS[1][0]) / STATE_BUCKETS[1])
    )
    y_bucket = min(STATE_BUCKETS[1] - 1, max(0, y_bucket))  # Clamp to valid range
    discrete_state.append(y_bucket)

    # Discretize X velocity
    x_vel_bucket = int(
        (mario_api.mario_x_velocity - STATE_BOUNDS[2][0])
        / ((STATE_BOUNDS[2][1] - STATE_BOUNDS[2][0]) / STATE_BUCKETS[2])
    )
    x_vel_bucket = min(STATE_BUCKETS[2] - 1, max(0, x_vel_bucket))  # Clamp to valid range
    discrete_state.append(x_vel_bucket)

    # Discretize grounded status
    grounded_bucket = 1 if mario_api.mario_is_grounded else 0
    discrete_state.append(grounded_bucket)

    # Discretize running status
    running_bucket = 1 if mario_api.mario_is_running else 0
    discrete_state.append(running_bucket)

    # Discretize time
    time_bucket = int(
        (mario_api.current_timer - STATE_BOUNDS[5][0]) / ((STATE_BOUNDS[5][1] - STATE_BOUNDS[5][0]) / STATE_BUCKETS[5])
    )
    time_bucket = min(STATE_BUCKETS[5] - 1, max(0, time_bucket))  # Clamp to valid range
    discrete_state.append(time_bucket)

    return tuple(discrete_state)


def train():
    # Wait 3 seconds before starting the training process
    print("Training will start in 3 seconds...")
    time.sleep(3)

    # Establish API
    mario_api = MarioAPI()
    # Initialize the Q-table: STATE_BUCKETS for each state variable + action space (5 actions)
    q_table = np.random.uniform(low=-1, high=1, size=(STATE_BUCKETS + [7]))

    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.983  # Decay rate for exploration
    min_epsilon = 0.01
    all_rewards = []

    for episode in range(EPISODES):
        # Reset the game environment (e.g., restart ROM)
        mario_api.reset_rom()

        # Get initial state after resetting the game
        mario_api.get_current_state_from_emulator()
        state_discrete = discretize_state(mario_api)
        done = False
        total_reward = 0

        # print("Move Left | Move Right | Jump | Stop Run | Spin Jump | Crouch | Look Up")
        # print(q_table[state_discrete])
        # print("~~~~~~~~~~~~~~~~~~~~~~~")

        g_rewards_for_x_positions = []
        # g_reward_for_x_position = 0
        g_reward_for_x_velocity = 0
        g_reward_for_grounded_and_moving = 0
        g_small_penalty_for_time = 0
        g_penalty_for_time = 0
        g_penalty_for_not_moving_right = 0
        g_reward_for_level_completion = 0
        g_penalty_for_death = 0

        while not done:
            # Select an action (epsilon-greedy strategy)
            if np.random.random() < epsilon:
                action = np.random.randint(0, 7)
            else:
                action = np.argmax(q_table[state_discrete])  # Best action from Q-table

            # Apply the action in the emulator (move Mario or perform an action)
            take_action_in_emulator(mario_api, action)

            # Get the next state and reward
            mario_api.get_current_state_from_emulator()
            (
                reward,
                done,
                reward_for_x_position,
                reward_for_x_velocity,
                reward_for_grounded_and_moving,
                small_penalty_for_time,
                penalty_for_time,
                penalty_for_not_moving_right,
                reward_for_level_completion,
                penalty_for_death,
            ) = calculate_reward(mario_api, action)

            g_rewards_for_x_positions.append(reward_for_x_position)
            # g_reward_for_x_position += reward_for_x_position
            g_reward_for_x_velocity += reward_for_x_velocity
            g_reward_for_grounded_and_moving += reward_for_grounded_and_moving
            g_small_penalty_for_time += small_penalty_for_time
            g_penalty_for_time += penalty_for_time
            g_penalty_for_not_moving_right += penalty_for_not_moving_right
            g_reward_for_level_completion += reward_for_level_completion
            g_penalty_for_death += penalty_for_death

            next_state_discrete = discretize_state(mario_api)

            # Update Q-value using the Q-learning update rule
            current_q = q_table[state_discrete + (action,)]
            max_future_q = np.max(q_table[next_state_discrete])
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q - current_q)
            q_table[state_discrete + (action,)] = new_q

            # Update state
            state_discrete = next_state_discrete
            total_reward += reward

        highest_x_position = max(g_rewards_for_x_positions)
        total_reward += highest_x_position
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        all_rewards.append(total_reward)

        avg_reward = np.mean(all_rewards[-100:])
        print(f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.4f}")
        print(f"Reward for X Position: {highest_x_position:.2f}")
        print(f"Reward for X Velocity: {g_reward_for_x_velocity:.2f}")
        print(f"Reward for Grounded and Moving: {g_reward_for_grounded_and_moving:.2f}")
        print(f"Small Penalty for Time: {g_small_penalty_for_time:.2f}")
        print(f"Penalty for Time: {g_penalty_for_time:.2f}")
        print(f"Penalty for Not Moving Right: {g_penalty_for_not_moving_right:.2f}")
        print(f"Reward for Level Completion: {g_reward_for_level_completion:.2f}")
        print(f"Penalty for Death: {g_penalty_for_death:.2f}")
        print(f"Total Reward: {total_reward:.2f}")
        print("~~~~~~~~~~~~~~~~~~~~~~~")

    # Save Q-table
    models_dir = "../models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    q_table_path = os.path.join(models_dir, "q_table_smw.pkl")
    with open(q_table_path, "wb") as f:
        pickle.dump(q_table, f)
    print("Training completed and Q-table saved.")


def take_action_in_emulator(mario_api: MarioAPI, action):
    """
    Sends an action command to the emulator to move Mario (left, right, jump, run, or do nothing).
    The action is an integer corresponding to a specific command.
    """
    if action == 0:
        mario_api.move_left()
    elif action == 1:
        mario_api.move_right()
    elif action == 2:
        mario_api.jump()
    elif action == 3:
        mario_api.stop_run()
    elif action == 4:
        mario_api.spin_jump()
    elif action == 5:
        mario_api.crouch()
    elif action == 6:
        mario_api.look_up()


def calculate_reward(mario_api: MarioAPI, action: np.intp = None):
    """
    Calculates the reward based on Mario's progress, velocity, and completion status.
    """

    reward = 0

    # Base reward for moving right (scaled down significantly)
    # reward = mario_api.mario_x_position / 10  # Reduced X position reward
    reward_for_x_position = mario_api.mario_x_position / 10

    # Additional reward for maintaining forward velocity
    reward += mario_api.mario_x_velocity / 2  # Keep a meaningful reward for velocity
    reward_for_x_velocity = mario_api.mario_x_velocity / 2

    # Bonus for being grounded and moving (progress stability)
    reward_for_grounded_and_moving = 0
    if mario_api.mario_is_grounded and mario_api.mario_x_velocity > 0:
        reward += 3  # Increased reward for stable, grounded movement
        reward_for_grounded_and_moving = 3

    # Small penalty for time to encourage quicker level completion
    reward -= 0.02  # Less harsh penalty for slight delays
    small_penalty_for_time = -0.02

    # Time-based penalty to encourage quicker level completion
    reward -= (1000 - mario_api.current_timer) / 2000  # Scaled time penalty (reduced from 1000)
    penalty_for_time = -(1000 - mario_api.current_timer) / 2000

    # Small penalty for actions that aren't moving right
    penalty_for_not_moving_right = 0
    if action is not None and action != 1:  # Assuming action 1 is "Move Right"
        reward -= 0.5  # Less penalty for not moving right
        penalty_for_not_moving_right = -0.5

    # Reward for completing the level
    reward_for_level_completion = 0
    penalty_for_death = 0
    if mario_api.mario_x_position > 4000:  # Assuming 4000+ is level completion
        reward += 1000  # Big reward for reaching the end
        reward_for_level_completion = 1000
        done = True
    elif mario_api.mario_died:  # Heavy penalty for dying
        reward -= 100  # Penalize for dying
        penalty_for_death = -100
        done = True
    else:
        done = False

    return (
        reward,
        done,
        reward_for_x_position,
        reward_for_x_velocity,
        reward_for_grounded_and_moving,
        small_penalty_for_time,
        penalty_for_time,
        penalty_for_not_moving_right,
        reward_for_level_completion,
        penalty_for_death,
    )


if __name__ == "__main__":
    train()
