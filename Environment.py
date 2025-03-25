import math
import os
import pygame
from components.advanced_UI import game_UI
from components.world_gen import spawn_objects


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

        # return default position if no opponents found
        if closest_pos is None:
            return player.rect.center

        return closest_pos

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
        # only render if not in training mode
        if not self.training_mode:
            if self.use_advanced_UI:
                # use the background from game_UI
                self.world_surface.blit(self.advanced_UI.background, (0, 0))
            else:
                self.world_surface.fill("purple")

        # frame skipping for training acceleration
        skip_count = self.frame_skip if self.training_mode else 1

        # track if any frame resulted in game over
        game_over = False
        final_info = None

        # get actions once and reuse them for all skipped frames
        player_actions = {}
        if self.training_mode:
            for player in self.players:
                if player.alive:
                    # update player info with closest opponent data before action
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

            # store the final state
            final_info = new_dic

            # check if game is over
            if len(alive_players) == 1:
                print("Game Over, winner is:", alive_players[0].username)
                if not self.training_mode:
                    if self.use_advanced_UI:
                        self.advanced_UI.display_winner_screen(alive_players)
                    else:
                        self.screen.fill("green")

                game_over = True
                break

        # skip all rendering operations in training mode for better performance
        if not self.training_mode:
            if self.use_advanced_UI:
                self.advanced_UI.draw_everything(final_info, self.players, self.obstacles)
            else:
                # draw obstacles manually if not using advanced UI
                for obstacle in self.obstacles:
                    obstacle.draw(self.world_surface)

            # scale and display the world surface
            scaled_surface = pygame.transform.scale(self.world_surface, (self.display_width, self.display_height))
            self.screen.blit(scaled_surface, (0, 0))
            pygame.display.flip()

        # in training mode, use a high tick rate but not unreasonably high
        if not self.training_mode:
            self.clock.tick(120)  # normal gameplay speed
        else:
            # skip the clock tick entirely in training mode for maximum speed
            pass  # no tick limiting in training mode for maximum speed

        # return the final state
        if game_over:
            print("Total steps:", self.steps)
            return True, final_info  # Game is over
        else:
            # return the final state from the last frame
            return False, final_info

    """TO MODIFY"""
    def calculate_reward(self, info_dictionary, bot_username):
        """THIS FUNCTION IS USED TO CALCULATE THE REWARD FOR A BOT"""
        """NEEDS TO BE WRITTEN BY YOU TO FINE TUNE YOURS"""

        # retrieve the players' information from the dictionary
        players_info = info_dictionary.get("players_info", {})
        bot_info = players_info.get(bot_username)

        # if the bot is not found, return a default reward of 0
        if bot_info is None:
            print("Bot not found in the dictionary")
            return 0

        # extract variables from the bot's info
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

        # calculate reward:
        reward = 0
        # add your reward calculation here

        self.last_damage[bot_username] = damage_dealt
        return reward