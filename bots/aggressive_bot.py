import math
import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state_dict):
        location = state_dict['location']
        status = state_dict['status']
        rays = state_dict['rays']
        relative_pos = state_dict.get('relative_pos', torch.zeros_like(location))
        time_features = state_dict.get('time_features', torch.zeros((location.shape[0], 2), device=location.device))

        combined = torch.cat([location, status, rays, relative_pos, time_features], dim=1)
        return self.input_net(combined)


class AggressiveBot:
    def __init__(self, action_size=56):
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.priority_memory = deque(maxlen=50000)
        self.priority_probabilities = deque(maxlen=50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 1
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.min_memory_size = 1000
        self.update_target_freq = 500
        self.train_freq = 4
        self.steps = 0
        self.use_double_dqn = True
        self.reset_epsilon = True
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.epsilon_pri = 0.01
        self.max_priority = 1.0
        self.exploration_bonus = 0.05
        self.visited_positions = {}
        self.position_resolution = 50
        self.time_since_last_shot = 0
        self.time_alive = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ImprovedDQN(input_dim=38, output_dim=action_size).to(self.device)
        self.target_model = ImprovedDQN(input_dim=38, output_dim=action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=10)

        self.last_state = None
        self.last_action = None
        self.training_started = False
        self.last_health = None
        self.last_damage_direction = None
        self.last_known_enemy_pos = None
        self.locked_on_timer = 0

    def normalize_state(self, info):
        try:
            state = {
                'location': torch.tensor([
                    info['location'][0] / 1280.0,
                    info['location'][1] / 1280.0
                ], dtype=torch.float32),
                'status': torch.tensor([
                    info['rotation'] / 360.0,
                    info['current_ammo'] / 30.0
                ], dtype=torch.float32),
                'rays': []
            }

            ray_data = []
            for ray in info.get('rays', []):
                if isinstance(ray, list) and len(ray) == 3:
                    start_pos, end_pos = ray[0]
                    distance = ray[1] if ray[1] is not None else 1500
                    hit_type = ray[2]
                    ray_data.extend([
                        start_pos[0] / 1280.0,
                        start_pos[1] / 1280.0,
                        end_pos[0] / 1280.0,
                        end_pos[1] / 1280.0,
                        distance / 1500.0,
                        1.0 if hit_type == "player" else 0.5 if hit_type == "object" else 0.0
                    ])

            while len(ray_data) < 30:
                ray_data.extend([0.0] * 6)

            state['rays'] = torch.tensor(ray_data[:30], dtype=torch.float32)

            if 'closest_opponent' in info:
                opponent_pos = info['closest_opponent']
                rel_x = (opponent_pos[0] - info['location'][0]) / 1280.0
                rel_y = (opponent_pos[1] - info['location'][1]) / 1280.0
                state['relative_pos'] = torch.tensor([rel_x, rel_y], dtype=torch.float32)
            else:
                state['relative_pos'] = torch.tensor([0.0, 0.0], dtype=torch.float32)

            state['time_features'] = torch.tensor([
                self.time_since_last_shot / 100.0,
                self.time_alive / 2400.0
            ], dtype=torch.float32)

            self.time_alive += 1
            if info.get('shot_fired', False):
                self.time_since_last_shot = 0
            else:
                self.time_since_last_shot += 1

            return state

        except Exception as e:
            print(f"Error in normalize_state: {e}")
            return None

    def _vector_to_angle(self, vector):
        dx, dy = vector
        angle = math.degrees(math.atan2(dy, dx))
        return max(-30, min(30, int(angle / 5) * 5))

    def action_to_dict(self, action):
        movement_directions = ["forward", "right", "down", "left"]
        rotation_angles = [-30, -5, -1, 0, 1, 5, 30]
        commands = {"forward": False, "right": False, "down": False, "left": False, "rotate": 0, "shoot": False}
        if action < 28:
            shoot = False
            local_action = action
        else:
            shoot = True
            local_action = action - 28
        movement_idx = local_action // 7
        angle_idx = local_action % 7
        direction = movement_directions[movement_idx]
        commands[direction] = True
        commands["rotate"] = rotation_angles[angle_idx]
        commands["shoot"] = shoot
        return commands

    def act(self, info):
        try:
            state = self.normalize_state(info)
            if state is None:
                return {"forward": False, "right": False, "down": False, "left": False, "rotate": 0, "shoot": False}

            current_health = info.get("health", 100)
            damage_taken = 0
            if self.last_health is not None:
                damage_taken = self.last_health - current_health
            being_attacked = damage_taken > 0
            self.last_health = current_health

            if being_attacked:
                for ray in info.get("rays", []):
                    if ray[2] == "player":
                        start_pos = ray[0][0]
                        end_pos = ray[0][1]
                        direction_vector = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
                        self.last_damage_direction = direction_vector
                        break

            if "closest_opponent" in info and info["closest_opponent"] is not None:
                self.last_known_enemy_pos = info["closest_opponent"]
                self.locked_on_timer = 10
            elif self.locked_on_timer > 0:
                self.locked_on_timer -= 1
            else:
                self.last_known_enemy_pos = None

            action_dict = {"forward": False, "right": False, "left": False, "down": False, "rotate": 0, "shoot": False}

            if being_attacked and self.last_damage_direction is not None:
                action_dict["rotate"] = self._vector_to_angle(self.last_damage_direction)
                action_dict["shoot"] = True
                return action_dict

            if self.locked_on_timer > 0 and self.last_known_enemy_pos:
                dx = self.last_known_enemy_pos[0] - info["location"][0]
                dy = self.last_known_enemy_pos[1] - info["location"][1]
                dist = (dx ** 2 + dy ** 2) ** 0.5
                near_cover = any(ray[2] == "object" and ray[1] < 150 for ray in info.get("rays", []))
                should_peek = (self.steps % 20) < 8
                if near_cover:
                    action_dict["right"] = should_peek
                    action_dict["left"] = not should_peek
                    action_dict["rotate"] = self._vector_to_angle((dx, dy))
                    action_dict["shoot"] = should_peek and info.get("current_ammo", 0) > 0
                else:
                    action_dict["forward"] = True
                    action_dict["rotate"] = self._vector_to_angle((dx, dy))
                    action_dict["shoot"] = info.get("current_ammo", 0) > 0
                if dist < 0.25:
                    action_dict["shoot"] = True
            else:
                action_dict["forward"] = True
                action_dict["rotate"] = random.choice([-15, 0, 15])

            state_tensors = {k: v.unsqueeze(0).to(self.device) for k, v in state.items()}
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.action_size)
            else:
                with torch.no_grad():
                    q_values = self.model(state_tensors)
                    action_index = torch.argmax(q_values).item()

            self.last_state = state
            self.last_action = action_index
            return action_dict
        


        except Exception as e:
            print(f"AggressiveBot Act Error: {e}")
            return {"forward": False, "right": False, "down": False, "left": False, "rotate": 0, "shoot": False}
        
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
        
    def remember(self, reward, next_info, done):
        try:
            next_state = self.normalize_state(next_info)
            
            if self.last_state is None or next_state is None:
                print("Skipping bad experience (state was None)")
                return

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