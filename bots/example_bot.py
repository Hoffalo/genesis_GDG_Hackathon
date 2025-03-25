import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import math


class ImprovedDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImprovedDQN, self).__init__()
        self.input_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state_dict):
        # combine all inputs into a single tensor
        location = state_dict['location']
        status = state_dict['status']
        rays = state_dict['rays']
        relative_pos = state_dict.get('relative_pos', torch.zeros_like(location))
        time_features = state_dict.get('time_features', torch.zeros((location.shape[0], 2), device=location.device))

        # concatenate all inputs
        combined = torch.cat([location, status, rays, relative_pos, time_features], dim=1)

        # process through the network
        return self.input_net(combined)


class MyBot:
    def __init__(self, action_size=16):
        self.action_size = action_size
        self.memory = deque(maxlen=50000)  # Reduced memory size for faster learning
        self.priority_memory = deque(maxlen=50000)  # Use deque for priority memory too
        self.priority_probabilities = deque(maxlen=50000)  # And for priorities
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Increased minimum exploration from 0.01 to allow more exploration
        self.epsilon_decay = 1 #0.999999  # Much slower decay for more exploration throughout training
        self.learning_rate = 0.0001  # Reduced learning rate for stability
        self.batch_size = 64  # Reduced batch size for more stable learning
        self.min_memory_size = 1000  # Reduced minimum memory size for faster start
        self.update_target_freq = 500  # Less frequent target network updates
        self.train_freq = 4  # Train less frequently for stability
        self.steps = 0
        self.use_double_dqn = True  # Enable Double DQN
        self.reset_epsilon = True

        # Prioritized experience replay parameters
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Initial importance sampling weight
        self.beta_increment = 0.001  # Beta increment per sampling
        self.epsilon_pri = 0.01  # Small constant to avoid zero priority
        self.max_priority = 1.0  # Initial max priority

        # Curiosity parameters
        self.exploration_bonus = 0.1
        self.visited_positions = {}  # Track visited positions
        self.position_resolution = 50  # Grid resolution for position tracking

        # Time tracking features
        self.time_since_last_shot = 0
        self.time_alive = 0

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else
                                   "cpu")

        print(f"Using device: {self.device}")

        # Create two networks - one for current Q-values and one for target
        self.model = ImprovedDQN(input_dim=38, output_dim=action_size).to(self.device)  # Updated input dim (34 + 2 for relative position + 2 for time features)
        self.target_model = ImprovedDQN(input_dim=38, output_dim=action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max',
                                                              factor=0.5, patience=10)  # Increased patience

        self.last_state = None
        self.last_action = None
        self.training_started = False

    def normalize_state(self, info):
        """Normalize state values to improve learning stability"""
        try:
            # Regular state components
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

            # Add relative position to opponent if available
            if 'closest_opponent' in info:
                opponent_pos = info['closest_opponent']
                rel_x = (opponent_pos[0] - info['location'][0]) / 1280.0
                rel_y = (opponent_pos[1] - info['location'][1]) / 1280.0
                state['relative_pos'] = torch.tensor([rel_x, rel_y], dtype=torch.float32)
            else:
                state['relative_pos'] = torch.tensor([0.0, 0.0], dtype=torch.float32)

            # Add time-based features
            state['time_features'] = torch.tensor([
                self.time_since_last_shot / 100.0,  # Normalize time since last shot
                self.time_alive / 2400.0            # Normalize time alive by max episode length
            ], dtype=torch.float32)

            # Update time tracking
            self.time_alive += 1
            if info.get('shot_fired', False):
                self.time_since_last_shot = 0
            else:
                self.time_since_last_shot += 1

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

            # Calculate exploration bonus based on position novelty
            pos_x = int(next_state['location'][0].item() * self.position_resolution)
            pos_y = int(next_state['location'][1].item() * self.position_resolution)
            grid_pos = (pos_x, pos_y)

            # Add exploration bonus for less visited areas
            exploration_bonus = 0
            if grid_pos in self.visited_positions:
                self.visited_positions[grid_pos] += 1
                visit_count = self.visited_positions[grid_pos]
                exploration_bonus = self.exploration_bonus / math.sqrt(visit_count)
            else:
                self.visited_positions[grid_pos] = 1
                exploration_bonus = self.exploration_bonus

            # Add exploration bonus to the reward
            reward += exploration_bonus

            # Standard experience memory for backward compatibility
            self.memory.append((self.last_state, self.last_action, reward, next_state, done))

            # Add to prioritized experience replay with max priority for new experiences
            self.priority_memory.append((self.last_state, self.last_action, reward, next_state, done))
            self.priority_probabilities.append(self.max_priority)

            # Start training only when we have enough samples
            if len(self.memory) >= self.min_memory_size and not self.training_started:
                print(f"Starting training with {len(self.memory)} samples in memory")
                self.training_started = True

            # Increment step counter
            self.steps += 1

            # Perform learning step if we have enough samples and it's time to train
            if self.training_started and self.steps % self.train_freq == 0:
                self.prioritized_replay()

                # Print training progress periodically
                if self.steps % 1000 == 0:
                    print(f"Step {self.steps}, epsilon: {self.epsilon:.4f}")

            # Update target network periodically
            if self.steps > 0 and self.steps % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                print(f"Updated target network at step {self.steps}")

            # Reset time alive if episode is done
            if done:
                self.time_alive = 0

        except Exception as e:
            print(f"Error in remember: {e}")

    def prioritized_replay(self):
        """Prioritized experience replay implementation with Double DQN"""
        if len(self.priority_memory) < self.batch_size:
            return

        try:
            # Calculate sampling probabilities
            priorities = np.array(self.priority_probabilities)
            probs = priorities ** self.alpha
            probs /= probs.sum()

            # Sample batch according to priorities
            indices = np.random.choice(len(self.priority_memory), self.batch_size, p=probs)

            # Extract batch
            batch = [self.priority_memory[idx] for idx in indices]

            # Calculate importance sampling weights
            self.beta = min(1.0, self.beta + self.beta_increment)  # Anneal beta
            weights = (len(self.priority_memory) * probs[indices]) ** (-self.beta)
            weights /= weights.max()  # Normalize
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

            # Prepare batch data
            states = {
                'location': torch.stack([t[0]['location'] for t in batch]).to(self.device),
                'status': torch.stack([t[0]['status'] for t in batch]).to(self.device),
                'rays': torch.stack([t[0]['rays'] for t in batch]).to(self.device),
                'relative_pos': torch.stack([t[0].get('relative_pos', torch.zeros(2)) for t in batch]).to(self.device),
                'time_features': torch.stack([t[0].get('time_features', torch.zeros(2)) for t in batch]).to(self.device)
            }

            next_states = {
                'location': torch.stack([t[3]['location'] for t in batch]).to(self.device),
                'status': torch.stack([t[3]['status'] for t in batch]).to(self.device),
                'rays': torch.stack([t[3]['rays'] for t in batch]).to(self.device),
                'relative_pos': torch.stack([t[3].get('relative_pos', torch.zeros(2)) for t in batch]).to(self.device),
                'time_features': torch.stack([t[3].get('time_features', torch.zeros(2)) for t in batch]).to(self.device)
            }

            actions = torch.tensor([t[1] for t in batch], dtype=torch.long).to(self.device)
            rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32).to(self.device)
            dones = torch.tensor([t[4] for t in batch], dtype=torch.float32).to(self.device)

            # Get current Q values
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

            # Get next Q values with Double DQN
            with torch.no_grad():
                if self.use_double_dqn:
                    # Double DQN: select action using policy network
                    next_action_indices = self.model(next_states).max(1)[1].unsqueeze(1)
                    # Evaluate using target network
                    next_q_values = self.target_model(next_states).gather(1, next_action_indices).squeeze()
                else:
                    # Regular DQN: both select and evaluate using target network
                    next_q_values = self.target_model(next_states).max(1)[0]

                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # Compute TD errors for updating priorities
            td_errors = torch.abs(current_q_values.squeeze() - target_q_values).detach().cpu().numpy()

            # Update priorities
            for idx, error in zip(indices, td_errors):
                self.priority_probabilities[idx] = error + self.epsilon_pri
                self.max_priority = max(self.max_priority, error + self.epsilon_pri)

            # Compute weighted loss
            loss = (weights * F.smooth_l1_loss(current_q_values.squeeze(), target_q_values, reduction='none')).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
            self.optimizer.step()

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        except Exception as e:
            print(f"Error in prioritized_replay: {e}")
            # Fall back to regular replay if there's an error
            self.replay()

    def replay(self):
        """Regular replay function with Double DQN support"""
        if len(self.memory) < self.batch_size:
            return

        try:
            minibatch = random.sample(self.memory, self.batch_size)

            # Prepare batch data
            states = {
                'location': torch.stack([t[0]['location'] for t in minibatch]).to(self.device),
                'status': torch.stack([t[0]['status'] for t in minibatch]).to(self.device),
                'rays': torch.stack([t[0]['rays'] for t in minibatch]).to(self.device),
                'relative_pos': torch.stack([t[0].get('relative_pos', torch.zeros(2)) for t in minibatch]).to(self.device),
                'time_features': torch.stack([t[0].get('time_features', torch.zeros(2)) for t in minibatch]).to(self.device)
            }

            next_states = {
                'location': torch.stack([t[3]['location'] for t in minibatch]).to(self.device),
                'status': torch.stack([t[3]['status'] for t in minibatch]).to(self.device),
                'rays': torch.stack([t[3]['rays'] for t in minibatch]).to(self.device),
                'relative_pos': torch.stack([t[3].get('relative_pos', torch.zeros(2)) for t in minibatch]).to(self.device),
                'time_features': torch.stack([t[3].get('time_features', torch.zeros(2)) for t in minibatch]).to(self.device)
            }

            actions = torch.tensor([t[1] for t in minibatch], dtype=torch.long).to(self.device)
            rewards = torch.tensor([t[2] for t in minibatch], dtype=torch.float32).to(self.device)
            dones = torch.tensor([t[4] for t in minibatch], dtype=torch.float32).to(self.device)

            # Get current Q values
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

            # Get next Q values with Double DQN support
            with torch.no_grad():
                if self.use_double_dqn:
                    # Double DQN: select action using policy network
                    next_action_indices = self.model(next_states).max(1)[1].unsqueeze(1)
                    # Evaluate using target network
                    next_q_values = self.target_model(next_states).gather(1, next_action_indices).squeeze()
                else:
                    # Regular DQN: both select and evaluate using target network
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

    def reset_for_new_episode(self):
        """Reset episode-specific variables for a new episode"""
        self.time_alive = 0
        self.time_since_last_shot = 0
        # Reset exploration tracking for curriculum learning
        if self.steps > 1000000:  # Advanced stage - reduce exploration bonus
            self.exploration_bonus = 0.05
        elif self.steps > 500000:  # Intermediate stage
            self.exploration_bonus = 0.08
        # Keep initial exploration bonus for early training

    def get_hyperparameters(self):
        """Return current hyperparameters for logging and tuning"""
        return {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "alpha": self.alpha,
            "beta": self.beta,
            "use_double_dqn": self.use_double_dqn,
            "model_input_dim": 38,
            "action_size": self.action_size,
            "steps": self.steps,
        }

    def save_to_dict(self):
        """Return a checkpoint dictionary of the entire training state."""
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'hyperparameters': self.get_hyperparameters(),
        }

    def load_from_dict(self, checkpoint_dict, map_location=None):
        """Load everything from an in-memory checkpoint dictionary."""
        if map_location is None:
            map_location = self.device

        # First ensure everything is on CPU, then move to final device if needed
        self.model.load_state_dict(checkpoint_dict['model_state_dict'])
        try:
            self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            if map_location != 'cpu':
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(map_location)
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
            print("Continuing with fresh optimizer but keeping model weights")

        if not self.reset_epsilon:
            self.epsilon = checkpoint_dict.get('epsilon', self.epsilon)
        self.steps = checkpoint_dict.get('steps', 0)

        # Move model and target model to final device
        self.device = torch.device(map_location) if isinstance(map_location, str) else map_location
        self.model = self.model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model = self.target_model.to(self.device)
