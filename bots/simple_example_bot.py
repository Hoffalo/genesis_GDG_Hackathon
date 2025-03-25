import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class SimpleBot:
    def __init__(self, action_size):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, self.action_size)
        )
        return model

    def remember(self, reward, next_state, done):
        self.memory.append((reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return self._get_random_action()
        
        state = self._prepare_state(state)
        with torch.no_grad():
            act_values = self.model(state)
        return self._convert_output_to_action(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = torch.zeros((batch_size, 8))
        targets = torch.zeros((batch_size, self.action_size))

        for i, (reward, next_state, done) in enumerate(minibatch):
            state = self._prepare_state(next_state)
            states[i] = state

            with torch.no_grad():
                target = self.model(state)

            if done:
                target[0] = reward
            else:
                with torch.no_grad():
                    next_state = self._prepare_state(next_state)
                    next_target = self.model(next_state)
                    target[0] = reward + self.gamma * torch.max(next_target[0])

            targets[i] = target

        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = nn.MSELoss()(outputs, targets)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _prepare_state(self, state):
        # Extract relevant features from state
        features = [
            state.get('x', 0),
            state.get('y', 0),
            state.get('health', 0),
            state.get('closest_opponent', [0, 0])[0],
            state.get('closest_opponent', [0, 0])[1],
            state.get('angle', 0),
            state.get('velocity_x', 0),
            state.get('velocity_y', 0)
        ]
        return torch.FloatTensor(features)

    def _get_random_action(self):
        # Generate random actions
        return {
            "forward": random.random() > 0.5,
            "right": random.random() > 0.5,
            "down": random.random() > 0.5,
            "left": random.random() > 0.5,
            "rotate": random.uniform(-1, 1),
            "shoot": random.random() > 0.8
        }

    def _convert_output_to_action(self, output):
        # Convert neural network output to game actions
        return {
            "forward": output[0] > 0,
            "right": output[1] > 0,
            "down": output[2] > 0,
            "left": output[3] > 0,
            "rotate": float(output[4]),
            "shoot": output[5] > 0
        }

    def reset_for_new_episode(self):
        self.epsilon = 1.0  # Reset exploration rate
