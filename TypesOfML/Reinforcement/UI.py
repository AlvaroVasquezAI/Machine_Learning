import gym 
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython.display import clear_output
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Create the environment
env = gym.make("Taxi-v3")

# Initialize parameters
alpha = 0.1 # learning rate
gamma = 0.6 # discount factor
epsilon = 0.01 # exploration rate

# Initialize Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Training the agent
def train_agent(episodes):
    for i in range(episodes):
        state = env.reset()
        state = state if isinstance(state, int) else state[0]
        penalties, reward = 0, 0
        done = False
        passenger_picked = False
        passenger_dropped = False

        while not done and not passenger_dropped:
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(Q[state]) # Exploit learned values
            
            step_result = env.step(action)

            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result

            next_state = next_state if isinstance(next_state, int) else next_state[0]
            old_value = Q[state, action]
            next_max = np.max(Q[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            Q[state, action] = new_value
            state = next_state

            if reward == -10:
                penalties += 1

            # Verify if the passenger was picked up
            if action == 4 and reward == -1:
                passenger_picked = True
            
            # Verify if the passenger was dropped off
            if passenger_picked and action == 5 and reward == 20:
                passenger_dropped = True
            
        if i % 100 == 0:
            clear_output(wait=True)
            print(f"Episode: {i}")

    print("Training finished.\n")
    pd.DataFrame(Q).to_csv("Q_table.csv")    

def plot_environment(env, ax, passenger_picked, passenger_dropped):
    grid = np.zeros((5, 5))
    taxi_row, taxi_col, pass_loc, dest_loc = list(env.unwrapped.decode(env.unwrapped.s))

    passenger_locations = {
        0: (0, 0), # Red
        1: (0, 4), # Green
        2: (4, 0), # Yellow
        3: (4, 3), # Blue
        4: (taxi_row, taxi_col) # In the taxi
    }

    destination_locations = {
        0: (0, 0), # Red
        1: (0, 4), # Green
        2: (4, 0), # Yellow
        3: (4, 3) # Blue
    }

    # Taxi: 1, Passenger: 2, Destination: 3, TaxiWithPassenger: 4, Completed: 5
    if passenger_picked:
        if passenger_dropped:
            grid[taxi_row, taxi_col] = 5
        else:
            grid[destination_locations[dest_loc]] = 3
            grid[taxi_row, taxi_col] = 4
    else:
        grid[taxi_row, taxi_col] = 1
        grid[passenger_locations[pass_loc]] = 2
        grid[destination_locations[dest_loc]] = 3
        
    # Colors: white = 0, darkgray = 1, blue = 2, green = 3, gray = 4, lightgreen = 5
    colors = ['white', 'darkgray', 'blue', 'green', 'gray', 'lightgreen']

    # Create a color mapping
    color_map = np.zeros_like(grid, dtype=int)
    unique_values = np.unique(grid).astype(int) 

    for i, val in enumerate(unique_values):
        color_map[grid == val] = i

    cmap = ListedColormap([colors[val] for val in unique_values])

    ax.imshow(color_map, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def evaluate_agent(episodes, max_iterations_per_episode, Q_table):
    print("Running autonomous driving agent...")

    Q = pd.read_csv(Q_table, index_col=0).values

    total_epochs, total_penalties, total_rewards, successful_episodes = 0, 0, 0, 0
    steps_per_episode = []
    penalties_per_episode = []
    rewards_per_episode = []

    root = tk.Tk()
    root.title("Taxi-v3")
    
    fix, ax = plt.subplots(figsize=(6,6))
    canvas = FigureCanvasTkAgg(fix, master=root)
    canvas.get_tk_widget().grid(row=0, column=0, rowspan=10)

    stats_frame = tk.Frame(root)
    stats_frame.grid(row=0, column=1, sticky='nw')

    legend_frame = tk.Frame(root)
    legend_frame.grid(row=1, column=1, sticky='nw')

    # Colors: white = 0, darkgray = 1, blue = 2, green = 3, gray = 4, lightgreen = 5
    colors = ['white', 'darkgray', 'blue', 'green', 'gray', 'lightgreen']
    color_labels = ['Empty', 'Taxi', 'Passenger', 'Destination', 'Taxi with Passenger', 'Completed']

    for i, color in enumerate(colors):
        label = tk.Label(legend_frame, text=f"{color_labels[i]}", background=color)
        label.grid(row=i, column=0, sticky='w')
    
    episode_label = tk.Label(stats_frame, text="Episode: ")
    episode_label.grid(row=0, column=0, sticky='w')
    iteration_label = tk.Label(stats_frame, text="Iteration: ")
    iteration_label.grid(row=1, column=0, sticky='w')
    action_label = tk.Label(stats_frame, text="Action: ")
    action_label.grid(row=2, column=0, sticky='w')
    reward_label = tk.Label(stats_frame, text="Reward: ")
    reward_label.grid(row=3, column=0, sticky='w')
    done_label = tk.Label(stats_frame, text="Done: ")
    done_label.grid(row=4, column=0, sticky='w')
    state_label = tk.Label(stats_frame, text="State: ")
    state_label.grid(row=5, column=0, sticky='w')
    penalty_label = tk.Label(stats_frame, text="Penalties: ")
    penalty_label.grid(row=6, column=0, sticky='w')

    for episode in range(episodes):
        state = env.reset()
        state = state if isinstance(state, int) else state[0] 
        epochs, penalties, reward = 0, 0, 0
        done = False
        episode_rewards = 0
        passenger_picked = False  
        passenger_dropped = False 

        for iteration in range(max_iterations_per_episode):
            action = np.argmax(Q[state]) 
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, terminated, info = step_result

            next_state = next_state if isinstance(next_state, int) else next_state[0]

            if reward == -10:
                penalties += 1

            epochs += 1
            episode_rewards += reward
            state = next_state

            if action == 4 and reward == -1:
                passenger_picked = True

            if passenger_picked and action == 5 and reward == 20:
                passenger_dropped = True
                done = True

            if done and passenger_dropped:
                successful_episodes += 1

                # Update the canvas and statistics
                plot_environment(env, ax, passenger_picked, passenger_dropped)
                canvas.draw()
                time.sleep(0.3)
                episode_label.config(text=f"Episode: {episode + 1}")
                iteration_label.config(text=f"Iteration: {iteration + 1}")
                action_label.config(text=f"Action: {action}")
                reward_label.config(text=f"Reward: {reward}")
                done_label.config(text=f"Done: {done}")
                state_label.config(text=f"State: {state}")
                penalty_label.config(text=f"Penalties: {penalties}")
                root.update()
                break

            # Update the canvas and statistics
            plot_environment(env, ax, passenger_picked, passenger_dropped)
            canvas.draw()
            time.sleep(0.1)
            episode_label.config(text=f"Episode: {episode + 1}")
            iteration_label.config(text=f"Iteration: {iteration + 1}")
            action_label.config(text=f"Action: {action}")
            reward_label.config(text=f"Reward: {reward}")
            done_label.config(text=f"Done: {done}")
            state_label.config(text=f"State: {state}")
            penalty_label.config(text=f"Penalties: {penalties}")
            root.update()

        total_penalties += penalties
        total_epochs += epochs
        total_rewards += episode_rewards

        steps_per_episode.append(epochs)
        penalties_per_episode.append(penalties)
        rewards_per_episode.append(episode_rewards)

    plt.ioff()

    # Calculate metrics
    average_steps_per_episode = total_epochs / episodes
    average_penalties_per_episode = total_penalties / episodes
    average_rewards_per_episode = total_rewards / episodes
    success_rate = successful_episodes / episodes * 100

    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {average_steps_per_episode}")
    print(f"Average penalties per episode: {average_penalties_per_episode}")
    print(f"Average rewards per episode: {average_rewards_per_episode}")
    print(f"Success rate: {success_rate}%")

    # Create a DataFrame with the metrics
    metrics_df = pd.DataFrame({
        'Episode': range(1, episodes + 1),
        'Steps': steps_per_episode,
        'Penalties': penalties_per_episode,
        'Rewards': rewards_per_episode
    })

    # Plot metrics
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(metrics_df['Episode'], metrics_df['Steps'])
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')

    plt.subplot(2, 2, 2)
    plt.plot(metrics_df['Episode'], metrics_df['Penalties'])
    plt.xlabel('Episode')
    plt.ylabel('Penalties')
    plt.title('Penalties per Episode')

    plt.subplot(2, 2, 3)
    plt.plot(metrics_df['Episode'], metrics_df['Rewards'])
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Rewards per Episode')

    plt.subplot(2, 2, 4)
    plt.plot(metrics_df['Episode'], [successful_episodes / (i + 1) * 100 for i in range(episodes)])
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.title('Cumulative Success Rate')

    plt.tight_layout()
    plt.show()

    # Show summary metrics as a table
    summary_metrics_df = pd.DataFrame({
        'Metric': ['Average Steps per Episode', 'Average Penalties per Episode', 'Average Rewards per Episode', 'Success Rate'],
        'Value': [average_steps_per_episode, average_penalties_per_episode, average_rewards_per_episode, success_rate]
    })

    # Show summary metrics as a table
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    plt.table(cellText=summary_metrics_df.values, colLabels=summary_metrics_df.columns, cellLoc='center', loc='center', colColours=['#f2f2f2']*2)
    plt.title('Summary Metrics')
    plt.show()

    root.mainloop()

if __name__ == "__main__":
    #train_agent(10000)
    evaluate_agent(20, 20, "Q_table.csv")