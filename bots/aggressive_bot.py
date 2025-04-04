from bots.example_bot import MyBot
import math
import random
import torch
import numpy as np
from collections import deque

class AggressiveBot(MyBot):
    def __init__(self, action_size=56):
        super().__init__(action_size)
        # Customize any hyperparameters for aggressive behavior
        self.epsilon_min = 0.01
        self.exploration_bonus = 0.05  # Less curiosity, more focus
        self.attack_distance_threshold = 0.3  # ~384 pixels in 1280x1280 world

    def act(self, info):
        try:
            state = self.normalize_state(info)

            # Aggression heuristic:
            if 'closest_opponent' in info:
                opp_x, opp_y = info['closest_opponent']
                my_x, my_y = info['location']
                dist_to_opponent = ((opp_x - my_x) ** 2 + (opp_y - my_y) ** 2) ** 0.5 / 1280

                if dist_to_opponent > self.attack_distance_threshold:
                    # Move toward opponent aggressively
                    dx = opp_x - my_x
                    dy = opp_y - my_y
                    forward = abs(dy) > abs(dx)
                    right = dx > 0
                    down = dy > 0
                    left = dx < 0

                    return {
                        "forward": forward,
                        "right": right,
                        "down": down,
                        "left": left,
                        "rotate": self._aim_at_target(my_x, my_y, opp_x, opp_y),
                        "shoot": False  # shoot only when close
                    }

            # If close, act as usual but prioritize shooting
            state_tensors = {k: v.unsqueeze(0).to(self.device) for k, v in state.items()}

            if random.random() <= self.epsilon:
                action = random.randrange(self.action_size)
            else:
                with torch.no_grad():
                    q_values = self.model(state_tensors)
                    action = torch.argmax(q_values).item()

            self.last_state = state
            self.last_action = action
            action_dict = self.action_to_dict(action)

            # Force shoot if enemy close and we have ammo
            if dist_to_opponent < 0.25 and info.get("current_ammo", 0) > 0:
                action_dict["shoot"] = True

            return action_dict

        except Exception as e:
            print(f"AggressiveBot Act Error: {e}")
            return {"forward": False, "right": False, "down": False, "left": False, "rotate": 0, "shoot": False}

    def _aim_at_target(self, my_x, my_y, opp_x, opp_y):
        """Calculate simple directional aiming based on enemy position."""
        dx = opp_x - my_x
        dy = opp_y - my_y
        angle = math.degrees(math.atan2(dy, dx))
        # Normalize and round
        return max(-30, min(30, int(angle / 5) * 5))  # nearest 5 degrees
