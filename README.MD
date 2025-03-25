# AI Bot Development Hackathon

## Colab Notebook
You can view the example notebook here:  
[Colab Notebook](https://colab.research.google.com/drive/1HWwft1d0rQaX3zbsRZlCyHAYo5P4Naid)

## Project Overview
This hackathon focuses on developing AI bots that can play a 2D game using reinforcement learning. Participants will create their own bot implementations and reward functions to compete in a dynamic game environment.

## Environment Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
├── bots/                    # Bot implementations
│   ├── simple_example_bot.py # Example bot implementation
│   └── your_bot.py         # Your bot implementation
├── components/             # Game components
├── environment.py          # Game environment
└── main.py                # Main training loop
```

## Hackathon Challenge

### Overview
Your task is to develop an AI bot that can play a 2D game by implementing:
1. The reward function in the environment
2. Your own bot implementation

### Environment Features
The game environment (`environment.py`) provides:
- A 2D game world with obstacles
- Player characters that can move, rotate, and shoot
- State information about the game
- A reward calculation system

### Available Information
Your bot receives the following state information:
- Player position (x, y)
- Player health
- Player angle
- Player velocity (x, y)
- Closest opponent position
- Other game state information

### Available Actions
Your bot can perform these actions:
- Move forward/backward/left/right
- Rotate
- Shoot

## Implementation Guide

### 1. Reward Function
Implement the reward function in `environment.py` to define good and bad behavior. Consider rewarding:
- Staying alive
- Dealing damage to opponents
- Getting kills
- Moving efficiently
- Avoiding obstacles

### 2. Bot Implementation
Create a new bot class in the `bots` directory. Your implementation can include:
- Neural networks
- Experience replay
- Action selection logic
- State processing
- Or any other approach you choose (there are no restrictions on the architecture), it can even not be a neural network

### Example Implementation
A simple example bot (`simple_example_bot.py`) demonstrates:
- Basic neural network structure
- Simple state processing
- Random action exploration
- Experience replay implementation

### Getting Started
1. Create a new file in the `bots` directory for your bot
2. Implement your bot class with the required methods:
   ```python
   class YourBot:
       def __init__(self, action_size):
           # Initialize your bot
           pass

       def act(self, state):
           # Return actions based on state
           pass
   ```
3. Modify the reward function in `environment.py`

## Tips for Success
- Start with a simple reward function and gradually make it more sophisticated
- Use the example bot as a reference but try to improve upon it
- Consider adding more features to your neural network
- Experiment with different hyperparameters
- Test your bot's performance against the example bot

Good luck with the hackathon!
