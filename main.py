import pygame
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
import torch

from Environment import Env
from components.my_bot import MyBot
from components.character import Character

screen = pygame.display.set_mode((800, 800))

def main():
    # Environment parameters.
    world_width = 1280
    world_height = 1280
    display_width = 800
    display_height = 800
    n_of_obstacles  = 15

    load_back = True
    state_size = 38  # Updated state size: 34 base + 2 for relative position + 2 for time features

    # Create training run directory for logs and checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"training_runs/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/models", exist_ok=True)
    os.makedirs(f"{run_dir}/plots", exist_ok=True)
    
    # Training configuration - easily adjustable
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
            "epsilon_decay": 0.995,
        }
    }
    
    # Save configuration
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Create the environment with frame skipping for faster training
    env = Env(training=True,
              use_game_ui=False,
              world_width=world_width,
              world_height=world_height,
              display_width=display_width,
              display_height=display_height,
              n_of_obstacles=n_of_obstacles,
              frame_skip=config["frame_skip"])
    screen = env.world_surface
    world_bounds = env.get_world_bounds()

    # Setup two players (characters) with starting positions.
    # The players will be stuck if closer than 5 pixels to the border.
    players = [
        Character((world_bounds[2] - 100, world_bounds[3] - 100),
                  screen, boundaries=world_bounds, username="Ninja"),
        Character((world_bounds[0] + 10, world_bounds[1]+10),
                  screen, boundaries=world_bounds, username="Faze Jarvis")
    ]

    # Create bots with configured hyperparameters
    bots = [
        MyBot(action_size=config["action_size"]),
        MyBot(action_size=config["action_size"])
    ]
    
    # Configure bot hyperparameters
    for bot in bots:
        bot.use_double_dqn = config["hyperparameters"]["double_dqn"]
        bot.learning_rate = config["hyperparameters"]["learning_rate"]
        bot.batch_size = config["hyperparameters"]["batch_size"]
        bot.gamma = config["hyperparameters"]["gamma"]
        bot.epsilon_decay = config["hyperparameters"]["epsilon_decay"]
        # Recreate optimizer with new learning rate
        bot.optimizer = torch.optim.Adam(bot.model.parameters(), lr=bot.learning_rate)

    if load_back:
        for idx, bot in enumerate(bots):
            save_path = f"bot_model_{idx}.pth"
            try:
                bot.load(save_path)
                print(f"Load model for player {players[idx].username} from {save_path}")
            except:
                print(f"Failed to load model for player {players[idx].username} from {save_path}")

    # Link players, bots, and obstacles into the environment.
    env.set_players_bots_objects(players, bots)

    # Training metrics tracking
    metrics = {
        "episode_rewards": {player.username: [] for player in players},
        "avg_rewards": {player.username: [] for player in players},
        "episode_steps": [],
        "kills": {player.username: [] for player in players},
        "damage_dealt": {player.username: [] for player in players},
        "survival_time": {player.username: [] for player in players},
        "epsilon": {player.username: [] for player in players}
    }
    
    # Curriculum learning parameters
    curriculum_stages = [
        {"n_obstacles": 10, "duration": 100},  # Stage 1: Simple environment
        {"n_obstacles": 15, "duration": 200},  # Stage 2: More obstacles
        {"n_obstacles": 20, "duration": 300},  # Stage 3: More challenging
        {"n_obstacles": 25, "duration": 400}   # Stage 4: Full difficulty
    ]

    for epoch in range(config["num_epochs"]):
        print(f"Starting epoch {epoch + 1}/{config['num_epochs']}")
        
        # Implement curriculum learning - adjust difficulty based on training progress
        current_stage = 0
        for i, stage in enumerate(curriculum_stages):
            if epoch >= sum(s["duration"] for s in curriculum_stages[:i]):
                current_stage = i
        
        # Set environment parameters based on current curriculum stage
        current_obstacles = curriculum_stages[current_stage]["n_obstacles"]
        print(f"Curriculum stage {current_stage + 1}: {current_obstacles} obstacles")
        
        # Configure environment for current stage
        env.n_of_obstacles = current_obstacles
        
        # Reset environment with randomized obstacles
        env.reset(randomize_objects=True)
        
        # Reset the step counter at the beginning of each episode
        env.steps = 0
        
        # Reset bots for the new episode
        for bot in bots:
            bot.reset_for_new_episode()

        # Track episode-specific metrics
        episode_metrics = {
            "rewards": {player.username: 0 for player in players},
            "kills": {player.username: 0 for player in players},
            "damage_dealt": {player.username: 0 for player in players},
            "survival_time": {player.username: 0 for player in players}
        }

        while True:
            # If the tick limit for this episode has been reached, break.
            if env.steps > config["tick_limit"]:
                print("Tick limit reached for this episode.")
                break

            # Take a step in the environment.
            finished, info = env.step(debugging=False)

            # For each player, calculate reward, update bot memory, and train the bot.
            for player, bot in zip(players, bots):
                # Calculate the reward for the current step
                reward = env.calculate_reward(info, player.username)
                
                # Apply curriculum-based reward scaling (higher rewards in early stages)
                curriculum_factor = 1.0 - (current_stage * 0.1)  # Gradually decrease reward scaling
                reward *= curriculum_factor
                
                # Update episode metrics
                episode_metrics["rewards"][player.username] += reward
                player_info = info["players_info"][player.username]
                episode_metrics["kills"][player.username] = player_info.get("kills", 0)
                episode_metrics["damage_dealt"][player.username] = player_info.get("damage_dealt", 0)
                if player_info.get("alive", False):
                    episode_metrics["survival_time"][player.username] += 1

                # Retrieve the updated state for the player.
                next_info = player.get_info()
                # Add closest opponent info if it was lost somewhere
                if 'closest_opponent' not in next_info:
                    next_info['closest_opponent'] = env.find_closest_opponent(player)
                # Store the transition (last state, action, reward, next state, done).
                bot.remember(reward, next_info, finished)

            # If the game/episode is over, break out of the loop.
            if finished:
                print(f"Episode {epoch + 1} finished, took {env.steps} ticks.")
                break

        # Record episode metrics
        metrics["episode_steps"].append(env.steps)
        for player in players:
            metrics["episode_rewards"][player.username].append(episode_metrics["rewards"][player.username])
            metrics["kills"][player.username].append(episode_metrics["kills"][player.username])
            metrics["damage_dealt"][player.username].append(episode_metrics["damage_dealt"][player.username])
            metrics["survival_time"][player.username].append(episode_metrics["survival_time"][player.username])
            metrics["epsilon"][player.username].append(player.related_bot.epsilon)
            
            # Calculate average rewards
            avg_reward = sum(metrics["episode_rewards"][player.username][-10:]) / min(10, len(metrics["episode_rewards"][player.username]))
            metrics["avg_rewards"][player.username].append(avg_reward)
            
            # Print episode summary
            print(f"Episode {epoch + 1} - {player.username}: " 
                  f"Reward = {episode_metrics['rewards'][player.username]:.2f}, "
                  f"Avg(10) = {avg_reward:.2f}, "
                  f"Kills = {episode_metrics['kills'][player.username]}, "
                  f"Damage = {episode_metrics['damage_dealt'][player.username]:.1f}, "
                  f"Epsilon = {player.related_bot.epsilon:.4f}")
            
            # Update learning rate using scheduler
            player.related_bot.scheduler.step(avg_reward)

        # Save metrics at regular intervals
        if (epoch + 1) % 10 == 0 or epoch == config["num_epochs"] - 1:
            # Save model checkpoints
            for idx, bot in enumerate(bots):
                save_path = f"{run_dir}/models/bot_model_{idx}_epoch_{epoch+1}.pth"
                bot.save(save_path)
                
            # Save metrics data
            with open(f"{run_dir}/metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
                
            # Create and save plots
            create_training_plots(metrics, run_dir, epoch+1)
                
        # Self-play - save periodic checkpoints for advanced training
        if (epoch + 1) % 100 == 0:
            for idx, bot in enumerate(bots):
                checkpoint_path = f"{run_dir}/models/checkpoint_model_{idx}_epoch_{epoch+1}.pth"
                bot.save(checkpoint_path)
                print(f"Saved checkpoint for player {players[idx].username} to {checkpoint_path}")
                
    # After training is complete, save final models
    for idx, bot in enumerate(bots):
        final_path = f"{run_dir}/models/final_model_{idx}.pth"
        bot.save(final_path)
        print(f"Saved final model for player {players[idx].username} to {final_path}")
        
        # Also save to standard location for easy loading
        bot.save(f"bot_model_{idx}.pth")

    # Final plots and metrics
    create_training_plots(metrics, run_dir, config["num_epochs"])
    print(f"Training complete! Results saved to {run_dir}")

    pygame.quit()
    
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
    
    # Plot kills
    plt.subplot(2, 2, 2)
    for player, kills in metrics["kills"].items():
        plt.plot(kills, label=f"{player} Kills")
    plt.title("Kills per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Kills")
    plt.legend()
    plt.grid(True)
    
    # Plot damage dealt
    plt.subplot(2, 2, 3)
    for player, damage in metrics["damage_dealt"].items():
        plt.plot(damage, label=f"{player} Damage")
    plt.title("Damage Dealt per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Damage")
    plt.legend()
    plt.grid(True)
    
    # Plot epsilon decay
    plt.subplot(2, 2, 4)
    for player, epsilon in metrics["epsilon"].items():
        plt.plot(epsilon, label=f"{player} Epsilon")
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
