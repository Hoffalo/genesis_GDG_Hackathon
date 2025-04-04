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
        self.last_health = None
        self.last_damage_direction = None
        self.last_known_enemy_pos = None
        self.locked_on_timer = 0
        
    def _vector_to_angle(self, vector):
        import math  # ensure math is available
        dx, dy = vector
        angle = math.degrees(math.atan2(dy, dx))
        return max(-30, min(30, int(angle / 5) * 5))  # rounds to nearest 5 and clamps

    
    def act(self, info):
        try:
            # === Normalize input state ===
            state = self.normalize_state(info)

            # === Track damage taken ===
            current_health = info.get("health", 100)
            damage_taken = 0
            if self.last_health is not None:
                damage_taken = self.last_health - current_health
            being_attacked = damage_taken > 0
            self.last_health = current_health

            # === Determine damage direction (if attacked) ===
            if being_attacked:
                for ray in info.get("rays", []):
                    if ray[2] == "player":
                        start_pos = ray[0][0]
                        end_pos = ray[0][1]
                        direction_vector = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
                        self.last_damage_direction = direction_vector
                        break

            # === Enemy lock-on logic ===
            if "closest_opponent" in info and info["closest_opponent"] is not None:
                self.last_known_enemy_pos = info["closest_opponent"]
                self.locked_on_timer = 10  # Stay locked on for 10 frames
            elif self.locked_on_timer > 0:
                self.locked_on_timer -= 1
            else:
                self.last_known_enemy_pos = None

            # === Start with default action dict ===
            action_dict = {
                "forward": False,
                "right": False,
                "left": False,
                "down": False,
                "rotate": 0,
                "shoot": False
            }

            # === Damage override: if hit, turn & shoot ===
            if being_attacked and self.last_damage_direction is not None:
                action_dict["rotate"] = self._vector_to_angle(self.last_damage_direction)
                action_dict["shoot"] = True
                return action_dict

            # === LOCKED ON Behavior ===
            if self.locked_on_timer > 0 and self.last_known_enemy_pos:
                dx = self.last_known_enemy_pos[0] - info["location"][0]
                dy = self.last_known_enemy_pos[1] - info["location"][1]
                dist = (dx ** 2 + dy ** 2) ** 0.5

                near_cover = any(ray[2] == "object" and ray[1] < 150 for ray in info.get("rays", []))
                should_peek = (self.steps % 20) < 8  # peek 8 frames every 20

                if near_cover:
                    # Tactical peek-shoot combo
                    action_dict["right"] = should_peek
                    action_dict["left"] = not should_peek
                    action_dict["rotate"] = self._vector_to_angle((dx, dy))
                    action_dict["shoot"] = should_peek and info.get("current_ammo", 0) > 0
                else:
                    # No cover? Just push toward enemy and fire
                    action_dict["forward"] = True
                    action_dict["rotate"] = self._vector_to_angle((dx, dy))
                    action_dict["shoot"] = info.get("current_ammo", 0) > 0

                if dist < 0.25:
                    action_dict["shoot"] = True

            else:
                # === SEEK MODE ===
                action_dict["forward"] = True
                action_dict["rotate"] = random.choice([-15, 0, 15])  # Look around

            # === Neural network memory update ===
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
            return {
                "forward": False, "right": False, "down": False, "left": False,
                "rotate": 0, "shoot": False
            }



    def _aim_at_target(self, my_x, my_y, opp_x, opp_y):
        """Calculate simple directional aiming based on enemy position."""
        dx = opp_x - my_x
        dy = opp_y - my_y
        angle = math.degrees(math.atan2(dy, dx))
        # Normalize and round
        return max(-30, min(30, int(angle / 5) * 5))  # nearest 5 degrees
