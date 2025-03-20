import pygame
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
import torch
from multiprocessing import Pool, cpu_count, Manager, Lock
from functools import partial
import copy

from Environment import Env
from components.my_bot import MyBot
from components.character import Character

screen = pygame.display.set_mode((800, 800))

def train_episode(epoch, config, curriculum_stages, world_bounds, display_width, display_height, shared_models, shared_history):
    """Worker function to train a single episode"""
    try:
        # Determine current curriculum stage
        current_stage = 0
        for i, stage in enumerate(curriculum_stages):
            if epoch >= sum(s["duration"] for s in curriculum_stages[:i]):
                current_stage = i
        
        current_obstacles = curriculum_stages[current_stage]["n_obstacles"]
        
        # Create environment for this episode
        env = Env(training=True,
                  use_game_ui=False,
                  world_width=world_bounds[2] - world_bounds[0],
                  world_height=world_bounds[3] - world_bounds[1],
                  display_width=display_width,
                  display_height=display_height,
                  n_of_obstacles=current_obstacles,
                  frame_skip=config["frame_skip"])
        
        # Setup players
        players = [
            Character((world_bounds[2] - 100, world_bounds[3] - 100),
                      env.world_surface, boundaries=world_bounds, username="Ninja"),
            Character((world_bounds[0] + 10, world_bounds[1]+10),
                      env.world_surface, boundaries=world_bounds, username="Faze Jarvis")
        ]
        
        # Create bots with shared models
        bots = []
        for idx in range(2):
            try:
                bot = MyBot(action_size=config["action_size"])
                bot.use_double_dqn = config["hyperparameters"]["double_dqn"]
                bot.learning_rate = config["hyperparameters"]["learning_rate"]
                bot.batch_size = config["hyperparameters"]["batch_size"]
                bot.gamma = config["hyperparameters"]["gamma"]
                bot.epsilon_decay = config["hyperparameters"]["epsilon_decay"]
                bot.optimizer = torch.optim.Adam(bot.model.parameters(), lr=bot.learning_rate)
                
                # Load shared model state with error handling
                if shared_models[idx] is not None:
                    try:
                        bot.model.load_state_dict(shared_models[idx])
                    except Exception as e:
                        print(f"Error loading model state for bot {idx}: {e}")
                        print("Starting with fresh model state")
                bots.append(bot)
            except Exception as e:
                print(f"Error creating bot {idx}: {e}")
                raise
        
        # Link everything together
        env.set_players_bots_objects(players, bots)
        env.reset(randomize_objects=True)
        env.steps = 0
        
        if hasattr(env, 'last_damage_tracker'):
            env.last_damage_tracker = {player.username: 0 for player in players}
        
        for bot in bots:
            bot.reset_for_new_episode()
        
        # Track episode metrics
        episode_metrics = {
            "rewards": {player.username: 0 for player in players},
            "kills": {player.username: 0 for player in players},
            "damage_dealt": {player.username: 0 for player in players},
            "survival_time": {player.username: 0 for player in players},
            "epsilon": {player.username: 0 for player in players},
            "learning_rate": {player.username: 0 for player in players}
        }
        
        while True:
            if env.steps > config["tick_limit"]:
                break
                
            finished, info = env.step(debugging=False)
            
            for player, bot in zip(players, bots):
                try:
                    reward = env.calculate_reward(info, player.username)
                    curriculum_factor = 1.0 - (current_stage * 0.1)
                    reward *= curriculum_factor
                    
                    episode_metrics["rewards"][player.username] += reward
                    player_info = info["players_info"][player.username]
                    episode_metrics["kills"][player.username] = player_info.get("kills", 0)
                    
                    current_damage = player_info.get("damage_dealt", 0)
                    if player.username not in getattr(env, 'last_damage_tracker', {}):
                        if not hasattr(env, 'last_damage_tracker'):
                            env.last_damage_tracker = {}
                        env.last_damage_tracker[player.username] = 0
                        
                    damage_delta = current_damage - env.last_damage_tracker[player.username]
                    if damage_delta > 0:
                        episode_metrics["damage_dealt"][player.username] += damage_delta
                    env.last_damage_tracker[player.username] = current_damage
                    
                    if player_info.get("alive", False):
                        episode_metrics["survival_time"][player.username] += 1
                        
                    next_info = player.get_info()
                    if 'closest_opponent' not in next_info:
                        next_info['closest_opponent'] = env.find_closest_opponent(player)
                    bot.remember(reward, next_info, finished)
                    
                    episode_metrics["epsilon"][player.username] = bot.epsilon
                    episode_metrics["learning_rate"][player.username] = bot.learning_rate
                except Exception as e:
                    print(f"Error processing player {player.username}: {e}")
                    continue
            
            if finished:
                break
        
        # Return model states and metrics
        model_states = [copy.deepcopy(bot.model.state_dict()) for bot in bots]
        return episode_metrics, env.steps, model_states
        
    except Exception as e:
        print(f"Critical error in train_episode: {e}")
        # Return safe default metrics in case of error
        return {
            "rewards": {"Ninja": 0, "Faze Jarvis": 0},
            "kills": {"Ninja": 0, "Faze Jarvis": 0},
            "damage_dealt": {"Ninja": 0, "Faze Jarvis": 0},
            "survival_time": {"Ninja": 0, "Faze Jarvis": 0},
            "epsilon": {"Ninja": 0, "Faze Jarvis": 0},
            "learning_rate": {"Ninja": 0, "Faze Jarvis": 0}
        }, 0, [None, None]

def main():
    # Environment parameters
    world_width = 1280
    world_height = 1280
    display_width = 800
    display_height = 800
    n_of_obstacles = 15
    
    load_back = True
    state_size = 38
    
    # Create training run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"training_runs/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/models", exist_ok=True)
    os.makedirs(f"{run_dir}/plots", exist_ok=True)
    
    # Training configuration
    config = {
        "frame_skip": 4,
        "tick_limit": 2400,
        "num_epochs": 1000,
        "action_size": 56,
        "hyperparameters": {
            "double_dqn": True,
            "learning_rate": 0.0001,
            "batch_size": 64,
            "gamma": 0.99,
            "epsilon_decay": 0.9999,
        }
    }
    
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Create initial environment to get world bounds
    env = Env(training=True,
              use_game_ui=False,
              world_width=world_width,
              world_height=world_height,
              display_width=display_width,
              display_height=display_height,
              n_of_obstacles=n_of_obstacles,
              frame_skip=config["frame_skip"])
    world_bounds = env.get_world_bounds()
    
    # Curriculum learning parameters
    curriculum_stages = [
        {"n_obstacles": 10, "duration": 100},
        {"n_obstacles": 15, "duration": 200},
        {"n_obstacles": 20, "duration": 300},
        {"n_obstacles": 25, "duration": 400}
    ]
    
    # Setup shared memory and synchronization
    manager = Manager()
    
    # Shared model states with proper synchronization
    shared_models = manager.list([None, None])
    
    # Shared training history
    shared_history = manager.dict({
        "total_steps": manager.Value('i', 0),
        "total_episodes": manager.Value('i', 0),
        "best_rewards": manager.dict({
            "Ninja": manager.Value('f', float('-inf')),
            "Faze Jarvis": manager.Value('f', float('-inf'))
        }),
        "last_save": manager.Value('i', 0)
    })
    
    # Shared metrics with proper synchronization
    metrics = manager.dict({
        "episode_rewards": {"Ninja": manager.list(), "Faze Jarvis": manager.list()},
        "avg_rewards": {"Ninja": manager.list(), "Faze Jarvis": manager.list()},
        "episode_steps": manager.list(),
        "kills": {"Ninja": manager.list(), "Faze Jarvis": manager.list()},
        "damage_dealt": {"Ninja": manager.list(), "Faze Jarvis": manager.list()},
        "survival_time": {"Ninja": manager.list(), "Faze Jarvis": manager.list()},
        "epsilon": {"Ninja": manager.list(), "Faze Jarvis": manager.list()},
        "learning_rates": {"Ninja": manager.list(), "Faze Jarvis": manager.list()}
    })
    
    # Initialize models if loading from previous training
    if load_back:
        try:
            for idx in range(2):
                save_path = f"bot_model_{idx}.pth"
                if os.path.exists(save_path):
                    # Create a temporary bot to load the model
                    temp_bot = MyBot(action_size=config["action_size"])
                    temp_bot.load(save_path)
                    # Store the model state in shared memory
                    shared_models[idx] = copy.deepcopy(temp_bot.model.state_dict())
                    print(f"Loaded model {idx} from {save_path}")
                else:
                    print(f"No saved model found for bot {idx}, starting fresh")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Starting with fresh models")
    
    # Setup parallel processing with error handling
    try:
        num_processes = max(1, cpu_count() - 1)  # Leave one CPU free
        print(f"Starting training with {num_processes} processes")
        pool = Pool(processes=num_processes)
        
        # Create partial function with fixed arguments
        train_episode_partial = partial(train_episode,
                                      config=config,
                                      curriculum_stages=curriculum_stages,
                                      world_bounds=world_bounds,
                                      display_width=display_width,
                                      display_height=display_height,
                                      shared_models=shared_models,
                                      shared_history=shared_history)
        
        # Training loop with improved error handling
        for epoch in range(config["num_epochs"]):
            print(f"Starting epoch {epoch + 1}/{config['num_epochs']}")
            
            try:
                # Run multiple episodes in parallel with timeout
                results = pool.map_async(train_episode_partial, [epoch] * num_processes)
                results = results.get(timeout=3600)  # 1-hour timeout per epoch
                
                # Aggregate results with error handling
                for episode_metrics, steps, model_states in results:
                    try:
                        # Update shared history
                        shared_history["total_steps"].value += steps
                        shared_history["total_episodes"].value += 1
                        
                        metrics["episode_steps"].append(steps)
                        for player in ["Ninja", "Faze Jarvis"]:
                            metrics["episode_rewards"][player].append(episode_metrics["rewards"][player])
                            metrics["kills"][player].append(episode_metrics["kills"][player])
                            metrics["damage_dealt"][player].append(episode_metrics["damage_dealt"][player])
                            metrics["survival_time"][player].append(episode_metrics["survival_time"][player])
                            metrics["epsilon"][player].append(episode_metrics["epsilon"][player])
                            metrics["learning_rates"][player].append(episode_metrics["learning_rate"][player])
                            
                            # Update best rewards
                            if episode_metrics["rewards"][player] > shared_history["best_rewards"][player].value:
                                shared_history["best_rewards"][player].value = episode_metrics["rewards"][player]
                            
                            # Calculate average rewards
                            avg_reward = sum(metrics["episode_rewards"][player][-10:]) / min(10, len(metrics["episode_rewards"][player]))
                            metrics["avg_rewards"][player].append(avg_reward)
                            
                            print(f"Episode {epoch + 1} - {player}: "
                                  f"Reward = {episode_metrics['rewards'][player]:.2f}, "
                                  f"Avg(10) = {avg_reward:.2f}, "
                                  f"Kills = {episode_metrics['kills'][player]}, "
                                  f"Damage = {episode_metrics['damage_dealt'][player]:.1f}, "
                                  f"Epsilon = {episode_metrics['epsilon'][player]:.4f}")
                    
                        # Update shared models
                        for idx, model_state in enumerate(model_states):
                            if model_state is not None:
                                shared_models[idx] = model_state
                                
                    except Exception as e:
                        print(f"Error processing episode metrics: {e}")
                        continue
                
                # Save metrics and plots periodically with error handling
                if (epoch + 1) % 10 == 0 or epoch == config["num_epochs"] - 1:
                    try:
                        # Convert shared memory metrics to regular dict for saving
                        save_metrics = {
                            k: (v if not isinstance(v, manager.list) else list(v)) 
                            for k, v in metrics.items()
                        }
                        
                        # Add shared history to saved metrics
                        save_metrics["shared_history"] = {
                            "total_steps": shared_history["total_steps"].value,
                            "total_episodes": shared_history["total_episodes"].value,
                            "best_rewards": {
                                player: shared_history["best_rewards"][player].value 
                                for player in ["Ninja", "Faze Jarvis"]
                            }
                        }
                        
                        with open(f"{run_dir}/metrics.json", "w") as f:
                            json.dump(save_metrics, f, indent=4)
                        create_training_plots(save_metrics, run_dir, epoch+1)
                        
                        # Save model checkpoints
                        for idx, model_state in enumerate(shared_models):
                            if model_state is not None:
                                save_path = f"{run_dir}/models/bot_model_{idx}_epoch_{epoch+1}.pth"
                                torch.save(model_state, save_path)
                                # Also save to standard location for easy loading
                                torch.save(model_state, f"bot_model_{idx}.pth")
                                
                        # Update last save timestamp
                        shared_history["last_save"].value = epoch + 1
                    except Exception as e:
                        print(f"Error saving metrics or models: {e}")
                
            except Exception as e:
                print(f"Error in epoch {epoch + 1}: {e}")
                # Try to recover by saving current state
                try:
                    for idx, model_state in enumerate(shared_models):
                        if model_state is not None:
                            torch.save(model_state, f"bot_model_{idx}_recovery.pth")
                except:
                    pass
                raise
    
    except Exception as e:
        print(f"Critical error during training: {e}")
        raise
    finally:
        # Cleanup with error handling
        try:
            pool.close()
            pool.join()
            pygame.quit()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        print(f"Training complete! Results saved to {run_dir}")
        print(f"Total steps: {shared_history['total_steps'].value}")
        print(f"Total episodes: {shared_history['total_episodes'].value}")
        print("Best rewards:")
        for player in ["Ninja", "Faze Jarvis"]:
            print(f"{player}: {shared_history['best_rewards'][player].value:.2f}")

def create_training_plots(metrics, run_dir, epoch):
    """Create and save training performance plots"""
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    for player, rewards in metrics["episode_rewards"].items():
        plt.plot(rewards, label=f"{player} Rewards")
    for player, avg_rewards in metrics["avg_rewards"].items():
        plt.plot(avg_rewards, label=f"{player} Avg Rewards", linestyle='--')
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    
    # Plot kills - improved clarity
    plt.subplot(2, 2, 2)
    colors = {'Ninja': 'blue', 'Faze Jarvis': 'red'}
    markers = {'Ninja': 'o', 'Faze Jarvis': 's'}
    
    for player, kills in metrics["kills"].items():
        # Use a bar chart for better visibility of discrete kill counts
        episodes = list(range(1, len(kills) + 1))
        plt.bar([e - 0.2 if player == 'Ninja' else e + 0.2 for e in episodes], 
                kills, 
                width=0.4,
                color=colors.get(player, 'green'),
                label=f"{player} Kills")
    
    plt.title("Kills per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Kills")
    plt.legend()
    plt.grid(True)
    
    # Plot damage dealt
    plt.subplot(2, 2, 3)
    for player, damage in metrics["damage_dealt"].items():
        plt.plot(damage, label=f"{player} Damage", 
                color=colors.get(player, 'green'),
                marker=markers.get(player, 'x'),
                markersize=4,
                markevery=max(1, len(damage)//20))  # Show markers every ~20 points
    plt.title("Damage Dealt per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Damage")
    plt.legend()
    plt.grid(True)
    
    # Plot epsilon decay
    plt.subplot(2, 2, 4)
    for player, epsilon in metrics["epsilon"].items():
        plt.plot(epsilon, label=f"{player} Epsilon",
                color=colors.get(player, 'green'))
    plt.title("Exploration Rate (Epsilon) Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{run_dir}/plots/training_progress_epoch_{epoch}.png")
    plt.close()

if __name__ == "__main__":
    main()
