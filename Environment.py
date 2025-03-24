import math
import os
import pygame
from advanced_UI import game_UI
from components.world_gen import spawn_objects


# TODO: add controls for multiple players
# TODO: add dummy bots so that they can train models

class Env:
    def __init__(self, training=False, use_game_ui=True, world_width=1280, world_height=1280, display_width=640,
                 display_height=640, n_of_obstacles=10, frame_skip=4):
        pygame.init()

        self.training_mode = training

        # ONLY FOR DISPLAY
        # create display window with desired display dimensions
        self.display_width = display_width
        self.display_height = display_height
        # only create a window if not in training mode
        if not self.training_mode:
            self.screen = pygame.display.set_mode((display_width, display_height))
        else:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Disable actual video output
            pygame.display.set_mode((1, 1))  # Minimal display

            self.screen = pygame.Surface((display_width, display_height))

        # REAL WORLD DIMENSIONS
        # create an off-screen surface for the game world
        self.world_width = world_width
        self.world_height = world_height
        self.world_surface = pygame.Surface((world_width, world_height))

        self.clock = pygame.time.Clock()
        self.running = True

        self.use_advanced_UI = use_game_ui
        if self.use_advanced_UI:
            self.advanced_UI = game_UI(self.world_surface, self.world_width, self.world_height)

        if not self.training_mode and self.use_advanced_UI:
            self.advanced_UI.display_opening_screen()

        self.n_of_obstacles = n_of_obstacles
        self.min_obstacle_size = (50, 50)
        self.max_obstacle_size = (100, 100)

        # frame skip for training acceleration
        self.frame_skip = frame_skip if training else 1

        # INIT SOME VARIABLES
        self.OG_bots = None
        self.OG_players = None
        self.OG_obstacles = None

        self.bots = None
        self.players = None
        self.obstacles = None

        """REWARD VARIABLES"""
        self.last_positions = {}
        self.last_damage = {}
        self.last_kills = {}
        self.last_health = {}
        self.visited_areas = {}

        self.visited_areas.clear()
        self.last_positions.clear()
        self.last_health.clear()
        self.last_kills.clear()
        self.last_damage.clear()

        self.steps = 0

    def set_players_bots_objects(self, players, bots, obstacles=None):
        self.OG_players = players
        self.OG_bots = bots
        self.OG_obstacles = obstacles

        self.reset()

    def get_world_bounds(self):
        return (0, 0, self.world_width, self.world_height)

    def reset(self, randomize_objects=False, randomize_players=False):
        self.running = True
        if not self.training_mode:
            if not self.use_advanced_UI:
                self.screen.fill("green")
                pygame.display.flip()
                self.clock.tick(1)  # 1 frame per second for 1 second = 1 frame
            else:
                self.advanced_UI.display_reset_screen()

        else:
            self.screen.fill("green")

        self.last_positions = {}
        self.last_damage = {}
        self.last_kills = {}
        self.last_health = {}
        self.visited_areas = {}

        self.steps = 0

        # TODO: add variables for parameters
        if self.use_advanced_UI:
            self.obstacles = self.advanced_UI.obstacles
        else:
            if randomize_objects or self.OG_obstacles is None:
                self.OG_obstacles = spawn_objects(
                    (0, 0, self.world_width, self.world_height),
                    self.max_obstacle_size,
                    self.min_obstacle_size,
                    self.n_of_obstacles
                )
            self.obstacles = self.OG_obstacles

        self.players = self.OG_players.copy()
        self.bots = self.OG_bots
        if randomize_players:
            self.bots = self.bots.shuffle()
            for index in range(len(self.players)):
                self.players[index].related_bot = self.bots[index]  # ensuring bots change location

        else:
            for index in range(len(self.players)):
                self.players[index].related_bot = self.bots[index]

        for player in self.players:
            player.reset()
            temp = self.players.copy()
            temp.remove(player)
            player.players = temp  # Other players
            player.objects = self.obstacles

    def step(self, debugging=False):
        # Only render if not in training mode
        if not self.training_mode:
            if self.use_advanced_UI:
                # Use the background from game_UI
                self.world_surface.blit(self.advanced_UI.background, (0, 0))
            else:
                self.world_surface.fill("purple")

        # Implement frame skipping for training acceleration
        skip_count = self.frame_skip if self.training_mode else 1

        # Track if any frame resulted in game over
        game_over = False
        final_info = None

        # Get actions once and reuse them for all skipped frames
        player_actions = {}
        if self.training_mode:
            for player in self.players:
                if player.alive:
                    # Update player info with closest opponent data before action
                    player_info = player.get_info()
                    player_info['closest_opponent'] = self.find_closest_opponent(player)
                    player_actions[player.username] = player.related_bot.act(player_info)

        # process multiple frames if frame skipping is enabled
        for _ in range(skip_count):
            if game_over:
                break

            self.steps += 1

            players_info = {}
            alive_players = []

            for player in self.players:
                player.update_tick()

                # use stored actions if in training mode with frame skipping
                if self.training_mode and skip_count > 1:
                    actions = player_actions.get(player.username, {})
                else:
                    # update info with closest opponent before getting action
                    player_info = player.get_info()
                    player_info['closest_opponent'] = self.find_closest_opponent(player)
                    actions = player.related_bot.act(player_info)

                if player.alive:
                    alive_players.append(player)
                    player.reload()

                    # skip drawing in training mode for better performance
                    if not self.training_mode:
                        player.draw(self.world_surface)

                    if debugging:
                        print("Bot would like to do:", actions)
                    if actions.get("forward", False):
                        player.move_in_direction("forward")
                    if actions.get("right", False):
                        player.move_in_direction("right")
                    if actions.get("down", False):
                        player.move_in_direction("down")
                    if actions.get("left", False):
                        player.move_in_direction("left")
                    if actions.get("rotate", 0):
                        player.add_rotate(actions["rotate"])
                    if actions.get("shoot", False):
                        player.shoot()

                    if not self.training_mode:
                        # store position for trail
                        if not hasattr(player, 'previous_positions'):
                            player.previous_positions = []
                        player.previous_positions.append(player.rect.center)
                        if len(player.previous_positions) > 10:
                            player.previous_positions.pop(0)

                # add closest opponent info to player info
                player_info = player.get_info()
                player_info["shot_fired"] = actions.get("shoot", False)
                player_info["closest_opponent"] = self.find_closest_opponent(player)
                players_info[player.username] = player_info

            new_dic = {
                "general_info": {
                    "total_players": len(self.players),
                    "alive_players": len(alive_players)
                },
                "players_info": players_info
            }

            # Store the final state
            final_info = new_dic

            # Check if game is over
            if len(alive_players) == 1:
                print("Game Over, winner is:", alive_players[0].username)
                if not self.training_mode:
                    if self.use_advanced_UI:
                        self.advanced_UI.display_winner_screen(alive_players)
                    else:
                        self.screen.fill("green")

                game_over = True
                break

        # Skip all rendering operations in training mode for better performance
        if not self.training_mode:
            if self.use_advanced_UI:
                self.advanced_UI.draw_everything(final_info, self.players, self.obstacles)
            else:
                # Draw obstacles manually if not using advanced UI
                for obstacle in self.obstacles:
                    obstacle.draw(self.world_surface)

            # Scale and display the world surface
            scaled_surface = pygame.transform.scale(self.world_surface, (self.display_width, self.display_height))
            self.screen.blit(scaled_surface, (0, 0))
            pygame.display.flip()

        # In training mode, use a high tick rate but not unreasonably high
        if not self.training_mode:
            self.clock.tick(120)  # Normal gameplay speed
        else:
            # Skip the clock tick entirely in training mode for maximum speed
            pass  # No tick limiting in training mode for maximum speed

        # Return the final state
        if game_over:
            print("Total steps:", self.steps)
            return True, final_info  # Game is over
        else:
            # Return the final state from the last frame
            return False, final_info

    def find_closest_opponent(self, player):
        """Find the position of the closest opponent for a given player"""
        closest_dist = float('inf')
        closest_pos = None

        for other in self.players:
            if other != player and other.alive:
                dist = math.dist(player.rect.center, other.rect.center)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pos = other.rect.center

        # Return default position if no opponents found
        if closest_pos is None:
            return player.rect.center  # Return own position as fallback

        return closest_pos

    """TO MODIFY"""
    def calculate_reward_empty(self, info_dictionary, bot_username):
        """THIS FUNCTION IS USED TO CALCULATE THE REWARD FOR A BOT"""
        """NEEDS TO BE WRITTEN BY YOU TO FINE TUNE YOURS"""

        # retrieve the players' information from the dictionary
        players_info = info_dictionary.get("players_info", {})
        bot_info = players_info.get(bot_username)

        # if the bot is not found, return a default reward of 0
        if bot_info is None:
            print("Bot not found in the dictionary")
            return 0

        # Extract variables from the bot's info
        location = bot_info.get("location", [0, 0])
        rotation = bot_info.get("rotation", 0)
        rays = bot_info.get("rays", [])
        current_ammo = bot_info.get("current_ammo", 0)
        alive = bot_info.get("alive", False)
        kills = bot_info.get("kills", 0)
        damage_dealt = bot_info.get("damage_dealt", 0)
        meters_moved = bot_info.get("meters_moved", 0)
        total_rotation = bot_info.get("total_rotation", 0)
        health = bot_info.get("health", 0)

        # Calculate reward:
        reward = 0
        # Add your reward calculation here

        return reward

    def calculate_reward(self, info_dictionary, bot_username):
        """
        Balanced reward function that encourages:
        1. Survival and health maintenance
        2. Accurate shooting and damage dealing
        3. Strategic movement and positioning
        4. Eliminating opponents
        5. Exploration of new areas
        6. Winning the game (huge reward)
        """
        players_info = info_dictionary.get("players_info", {})
        bot_info = players_info.get(bot_username)
        if bot_info is None:
            print(f"Bot {bot_username} not found in info dictionary.")
            return 0

        # Extract current values
        current_position = bot_info.get("location", [0, 0])
        damage_dealt = bot_info.get("damage_dealt", 0)
        kills = bot_info.get("kills", 0)
        alive = bot_info.get("alive", False)
        health = bot_info.get("health", 100)
        shot_fired = bot_info.get("shot_fired", False)

        # Initialize tracking dictionaries if necessary
        if bot_username not in self.last_positions:
            self.last_positions[bot_username] = current_position
        if bot_username not in self.last_damage:
            self.last_damage[bot_username] = damage_dealt
        if bot_username not in self.last_kills:
            self.last_kills[bot_username] = kills
        if bot_username not in self.last_health:
            self.last_health[bot_username] = health
        if bot_username not in self.visited_areas:
            self.visited_areas[bot_username] = set()

        reward = 0

        # 1. Movement reward - encourage exploration of new areas
        distance_moved = math.dist(current_position, self.last_positions[bot_username])

        # Discretize position to a grid cell (using a grid size of 50x50)
        grid_size = 50
        grid_x = int(current_position[0] / grid_size)
        grid_y = int(current_position[1] / grid_size)
        grid_pos = (grid_x, grid_y)

        # Check if this grid cell has been visited before
        if grid_pos not in self.visited_areas[bot_username]:
            # Reward for discovering a new area
            self.visited_areas[bot_username].add(grid_pos)
            # Scale reward based on number of areas discovered (diminishing returns)
            discovery_reward = min(0.5, 6.0 / len(self.visited_areas[bot_username]))
            reward += discovery_reward
        else:
            # Small reward for movement even if not discovering new areas
            reward += min(distance_moved * 0.0001, 0.1)  # Reduced reward for revisiting

        # 2. Damage reward - encourage accurate shooting
        delta_damage = damage_dealt - self.last_damage[bot_username]
        if delta_damage > 0:
            reward += delta_damage * 5.0

        # 3. Kill reward - significant but not overwhelming
        delta_kills = kills - self.last_kills[bot_username]
        if delta_kills > 0:
            reward += delta_kills * 20.0  # Reduced from 50.0 for more balanced rewards

        # 4. Shot penalty - encourage accuracy without being too punishing
        if shot_fired and delta_damage <= 0:
            reward -= 0.1  # Reduced penalty for missing shots

        # 5. Damage taken penalty - encourage defensive play
        delta_health = self.last_health[bot_username] - health
        if delta_health > 0:
            reward -= delta_health * 0.2  # Reduced penalty for taking damage

        # 6. Survival reward - encourage staying alive
        if alive:
            reward += 0.00001  # Small constant reward for staying alive

        # 7. Health bonus - encourage maintaining high health
        if health > 80:
            reward += 0.000005  # Small bonus for maintaining high health

        # 8. Winning reward - huge reward for winning the game
        game_info = info_dictionary.get("game_info", {})
        alive_players_count = game_info.get("alive_players", 0)
        if alive_players_count == 1 and alive:
            # This bot is the last one standing - it's the winner!
            reward += 100.0  # Huge reward for winning

        # Update tracking values for next step
        self.last_positions[bot_username] = current_position
        self.last_damage[bot_username] = damage_dealt
        self.last_kills[bot_username] = kills
        self.last_health[bot_username] = health

        return reward
