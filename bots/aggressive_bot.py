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