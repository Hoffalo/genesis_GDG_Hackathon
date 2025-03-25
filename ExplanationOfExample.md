# Flow Between Main, Environment, and Bots

This document explains the flow of data and control between the main script, environment, and bots in the GDG_Hackathon project.

## Overview

The project implements a game environment where bots compete against each other. The system uses reinforcement learning to train bots to perform well in the game. The flow of data and control involves three main components:

1. **Main Script** (`main.py`): Initializes the environment and bots, manages the training loop, and tracks metrics.
2. **Environment** (`Environment.py`): Manages the game state, processes bot actions, and calculates rewards.
3. **Bots** (`example_bot.py`, `clean_bot.py`): Receive information from the environment, decide on actions, and learn from experience.

## Data Flow Diagram

```
┌─────────────┐         ┌─────────────────┐         ┌─────────────┐
│             │         │                 │         │             │
│    Main     │ ──────► │   Environment   │ ──────► │    Bots     │
│             │         │                 │         │             │
└─────────────┘         └─────────────────┘         └─────────────┘
      ▲                        │                          │
      │                        │                          │
      │                        ▼                          │
      │                 ┌─────────────────┐               │
      │                 │  Game State     │               │
      │                 │  - Players      │               │
      │                 │  - Obstacles    │               │
      └─────────────── │  - Metrics      │ ◄─────────────┘
                        └─────────────────┘
```

## Key Dictionaries and Data Structures

### 1. From Main to Environment

**Configuration Dictionary (`config`):**
```python
config = {
    "frame_skip": 4,
    "tick_limit": 2400,
    "num_epochs": 50,
    "action_size": 56,
    "hyperparameters": {
        "double_dqn": True,
        "learning_rate": 0.0001,
        "batch_size": 64,
        "gamma": 0.99,
        "epsilon_decay": 0.9999,
    }
}
```
This dictionary configures the training process, including parameters like frame skip, tick limit, number of epochs, and hyperparameters for the reinforcement learning algorithm.

**Curriculum Stages:**
```python
curriculum_stages = [
    {"n_obstacles": 10, "duration": 100},
    {"n_obstacles": 15, "duration": 200},
    {"n_obstacles": 20, "duration": 300}
]
```
This list of dictionaries defines different stages of training, with varying numbers of obstacles and durations.

### 2. From Environment to Bots

**Info Dictionary:**
```python
info = {
    "location": [x, y],
    "rotation": angle,
    "rays": [ray1, ray2, ...],
    "current_ammo": ammo_count,
    "alive": True/False,
    "kills": kill_count,
    "damage_dealt": damage,
    "meters_moved": distance,
    "total_rotation": rotation,
    "health": health_value,
    "closest_opponent": [opponent_x, opponent_y]
}
```
This dictionary provides information about the game state to the bot, including the bot's location, rotation, ray-cast results (for vision), ammo count, and other status information.

**Players Info Dictionary:**
```python
players_info = {
    "player1_username": {
        "location": [x, y],
        "rotation": angle,
        "rays": [ray1, ray2, ...],
        "current_ammo": ammo_count,
        "alive": True/False,
        "kills": kill_count,
        "damage_dealt": damage,
        "meters_moved": distance,
        "total_rotation": rotation,
        "health": health_value
    },
    "player2_username": {
        # Similar structure
    }
}
```
This dictionary contains information about all players in the game, with each player's username as the key.

**Final Info Dictionary:**
```python
final_info = {
    "general_info": {
        "total_players": player_count,
        "alive_players": alive_count
    },
    "players_info": players_info
}
```
This dictionary combines general game information with player-specific information.

### 3. From Bots to Environment

**Actions Dictionary:**
```python
actions = {
    "forward": True/False,
    "right": True/False,
    "down": True/False,
    "left": True/False,
    "rotate": angle,
    "shoot": True/False
}
```
This dictionary specifies the actions that the bot wants to take, including movement directions, rotation angle, and whether to shoot.

### 4. Internal Bot State

**Normalized State Dictionary:**
```python
state = {
    'location': tensor([x/1280.0, y/1280.0]),
    'status': tensor([rotation/360.0, ammo/30.0]),
    'rays': tensor([ray_data]),
    'relative_pos': tensor([rel_x, rel_y]),
    'time_features': tensor([time_since_last_shot/100.0, time_alive/2400.0])
}
```
This dictionary contains normalized state information that the bot uses for decision-making and learning.

## Control Flow

### Initialization Phase

1. **Main Script**:
   - Initializes configuration parameters
   - Creates the environment with specified parameters
   - Creates players and bots
   - Links players and bots to the environment

2. **Environment**:
   - Initializes the game world, including display and surfaces
   - Sets up reward tracking variables
   - Prepares for the game loop

3. **Bots**:
   - Initialize neural networks and learning parameters
   - Prepare memory structures for experience replay

### Training Loop

1. **Main Script**:
   - For each epoch:
     - Determines the current curriculum stage
     - Calls `train_single_episode` to run one episode
     - Saves model checkpoints
     - Tracks and displays metrics

2. **Environment** (during `train_single_episode`):
   - Resets the game state
   - For each step:
     - Gets actions from bots
     - Processes these actions to update the game state
     - Calculates rewards for each bot
     - Provides updated information to bots
     - Checks if the episode is finished

3. **Bots**:
   - For each step:
     - Receive information from the environment
     - Normalize the state information
     - Choose an action (exploration or exploitation)
     - Return the action to the environment
     - Remember the experience (state, action, reward, next state, done)
     - Periodically learn from past experiences

### Learning Phase

1. **Bots**:
   - Sample experiences from memory
   - Calculate target Q-values using the reward and next state
   - Update the neural network to better predict Q-values
   - Gradually reduce exploration (epsilon decay)

## Detailed Flow Example

1. **Main** initializes the environment and bots.
2. **Main** calls `train_single_episode`.
3. **Environment** resets the game state.
4. **Environment** calls `step` to advance the game:
   - For each player, the environment gets the player's information.
   - The environment adds the closest opponent information.
   - The environment calls the bot's `act` method with this information.
5. **Bot** processes the information:
   - Normalizes the state.
   - Chooses an action (random or based on Q-values).
   - Returns an actions dictionary.
6. **Environment** processes the actions:
   - Updates player positions and rotations.
   - Handles shooting and collisions.
   - Updates the game state.
7. **Environment** calculates rewards for each bot.
8. **Environment** provides updated information to bots.
9. **Bot** remembers the experience:
   - Stores the state, action, reward, next state, and done flag.
   - Periodically learns from past experiences.
10. Steps 4-9 repeat until the episode is finished.
11. **Main** tracks metrics and saves model checkpoints.
12. Steps 3-11 repeat for each epoch.

## Conclusion

The flow between main, environment, and bots involves a complex exchange of information through dictionaries. The main script manages the overall training process, the environment manages the game state and calculates rewards, and the bots make decisions and learn from experience. This architecture allows for flexible and modular development of reinforcement learning agents for the game environment.