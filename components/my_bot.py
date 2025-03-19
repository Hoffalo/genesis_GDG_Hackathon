import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


class ImprovedDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImprovedDQN, self).__init__()
        # Enhanced architecture with deeper layers and more neurons
        # Process all inputs together for better learning

        # Deeper network with more capacity
        self.input_net = nn.Sequential(
            nn.Linear(34, 256),  # Increased first layer size
            nn.ReLU(),
            nn.Linear(256, 256),  # Increased hidden layer size
            nn.ReLU(),
            nn.Linear(256, 128),  # Added another layer for more depth
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state_dict):
        # Combine all inputs into a single tensor
        location = state_dict['location']
        status = state_dict['status']
        rays = state_dict['rays']

        # Concatenate all inputs
        combined = torch.cat([location, status, rays], dim=1)

        # Process through the network
        return self.input_net(combined)


class MyBot:
    def __init__(self, action_size=16):  # Increased action space
        self.action_size = action_size
        self.memory = deque(maxlen=100000)  # Increased memory size for better learning
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01  # Lower minimum exploration for better exploitation
        self.epsilon_decay = 0.99  # Faster decay for quicker learning
        self.learning_rate = 0.0005  # Adjusted learning rate for stability with larger network
        self.batch_size = 128  # Increased batch size for more stable learning
        self.min_memory_size = 2000  # Increased minimum memory size for better initial learning
        self.update_target_freq = 200  # More frequent target network updates
        self.train_freq = 2  # Train more frequently for faster learning
        self.steps = 0

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else
                                   "cpu")

        print(f"Using device: {self.device}")

        # Create two networks - one for current Q-values and one for target
        self.model = ImprovedDQN(input_dim=34, output_dim=action_size).to(self.device)
        self.target_model = ImprovedDQN(input_dim=34, output_dim=action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max',
                                                              factor=0.5, patience=5)

        self.last_state = None
        self.last_action = None
        self.training_started = False

    def normalize_state(self, info):
        """Normalize state values to improve learning stability"""
        try:
            state = {
                'location': torch.tensor([
                    info['location'][0] / 1280.0,  # Normalize by world width
                    info['location'][1] / 1280.0  # Normalize by world height
                ], dtype=torch.float32),

                'status': torch.tensor([
                    info['rotation'] / 360.0,  # Normalize rotation
                    info['current_ammo'] / 30.0  # Normalize ammo
                ], dtype=torch.float32),

                'rays': []
            }

            # Process rays
            ray_data = []
            for ray in info.get('rays', []):
                if isinstance(ray, list) and len(ray) == 3:
                    start_pos, end_pos = ray[0]
                    distance = ray[1] if ray[1] is not None else 1500  # Max vision distance
                    hit_type = ray[2]

                    # Normalize positions and distance
                    ray_data.extend([
                        start_pos[0] / 1280.0,
                        start_pos[1] / 1280.0,
                        end_pos[0] / 1280.0,
                        end_pos[1] / 1280.0,
                        distance / 1500.0,  # Normalize by max vision distance
                        1.0 if hit_type == "player" else 0.5 if hit_type == "object" else 0.0
                    ])

            # Pad rays if necessary
            while len(ray_data) < 30:  # 5 rays * 6 features
                ray_data.extend([0.0] * 6)

            state['rays'] = torch.tensor(ray_data[:30], dtype=torch.float32)
            return state

        except Exception as e:
            print(f"Error in normalize_state: {e}")
            print(f"Info received: {info}")
            raise

    def action_to_dict(self, action):
        """Enhanced action space with more granular rotation"""
        movement_directions = ["forward", "right", "down", "left"]
        rotation_angles = [-30, -5, -1, 0, 1, 5, 30]

        # Basic movement commands
        commands = {
            "forward": False,
            "right": False,
            "down": False,
            "left": False,
            "rotate": 0,
            "shoot": False
        }

        # determine block (no-shoot vs shoot)
        if action < 28:
            shoot = False
            local_action = action  # 0..27
        else:
            shoot = True
            local_action = action - 28  # 0..27

        movement_idx = local_action // 7  # 0..3
        angle_idx = local_action % 7  # 0..6

        direction = movement_directions[movement_idx]
        commands[direction] = True
        commands["rotate"] = rotation_angles[angle_idx]
        commands["shoot"] = shoot

        return commands

    def act(self, info):
        try:
            state = self.normalize_state(info)

            # Convert state dict to tensors and add batch dimension
            state_tensors = {
                k: v.unsqueeze(0).to(self.device) for k, v in state.items()
            }

            if random.random() <= self.epsilon:
                action = random.randrange(self.action_size)
            else:
                with torch.no_grad():
                    q_values = self.model(state_tensors)
                    action = torch.argmax(q_values).item()

            self.last_state = state
            self.last_action = action
            return self.action_to_dict(action)

        except Exception as e:
            print(f"Error in act: {e}")
            # Return safe default action
            return {"forward": False, "right": False, "down": False, "left": False, "rotate": 0, "shoot": False}

    def remember(self, reward, next_info, done):
        try:
            next_state = self.normalize_state(next_info)
            self.memory.append((self.last_state, self.last_action, reward, next_state, done))

            # Start training only when we have enough samples
            if len(self.memory) >= self.min_memory_size and not self.training_started:
                print(f"Starting training with {len(self.memory)} samples in memory")
                self.training_started = True

            # Increment step counter
            self.steps += 1

            # Perform learning step if we have enough samples and it's time to train
            if self.training_started and self.steps % self.train_freq == 0:
                self.replay()

                # Print training progress periodically
                if self.steps % 1000 == 0:
                    print(f"Step {self.steps}, epsilon: {self.epsilon:.4f}")

            # Update target network periodically
            if self.steps % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                print(f"Updated target network at step {self.steps}")

        except Exception as e:
            print(f"Error in remember: {e}")

    def replay(self):
        """Safe replay function that checks memory size"""
        if len(self.memory) < self.batch_size:
            return

        try:
            minibatch = random.sample(self.memory, self.batch_size)

            # Prepare batch data
            states = {
                'location': torch.stack([t[0]['location'] for t in minibatch]).to(self.device),
                'status': torch.stack([t[0]['status'] for t in minibatch]).to(self.device),
                'rays': torch.stack([t[0]['rays'] for t in minibatch]).to(self.device)
            }

            next_states = {
                'location': torch.stack([t[3]['location'] for t in minibatch]).to(self.device),
                'status': torch.stack([t[3]['status'] for t in minibatch]).to(self.device),
                'rays': torch.stack([t[3]['rays'] for t in minibatch]).to(self.device)
            }

            actions = torch.tensor([t[1] for t in minibatch], dtype=torch.long).to(self.device)
            rewards = torch.tensor([t[2] for t in minibatch], dtype=torch.float32).to(self.device)
            dones = torch.tensor([t[4] for t in minibatch], dtype=torch.float32).to(self.device)

            # Get current Q values
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

            # Get next Q values from target network
            with torch.no_grad():
                next_q_values = self.target_model(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # Compute loss and optimize
            loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
            self.optimizer.step()

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        except Exception as e:
            print(f"Error in replay: {e}")

    def save(self, filepath):
        """Saves the model weights."""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps
            }, filepath)
            print(f"Model saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load(self, filepath):
        """Loads model weights from the specified file."""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            self.model.to(self.device)
            print(f"Model loaded successfully from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with a fresh model")
            # Initialize a fresh model if loading fails
            self.model = ImprovedDQN(input_dim=34, output_dim=self.action_size).to(self.device)
            self.target_model = ImprovedDQN(input_dim=34, output_dim=self.action_size).to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())
