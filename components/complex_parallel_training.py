import pygame
import matplotlib.pyplot as plt
import os
import json
import argparse
from datetime import datetime
import torch
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import copy
import time  

from Environment import Env
from bots.my_bot import MyBot
from components.character import Character

# from my_bot import MyBot  # You will add your MyBot class code in a separate file
def create_training_plots(metrics, run_dir, epoch):
    """Create and save training performance plots."""
    plt.figure(figsize=(15, 10))

    total_episodes = len(metrics["episode_rewards"]["Ninja"])
    x_values = list(range(1, total_episodes + 1))

    # 1) Rewards
    plt.subplot(2, 2, 1)
    colors = {'Ninja': 'blue', 'Faze Jarvis': 'red'}

    for player, rewards in metrics["episode_rewards"].items():
        plt.plot(x_values, rewards, label=f"{player} Rewards", color=colors.get(player))

    for player, avg_rewards in metrics["avg_rewards"].items():
        plt.plot(x_values, avg_rewards, label=f"{player} Avg(10) Rewards",
                 linestyle='--', color=colors.get(player), alpha=0.7)

    plt.title("Rewards per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    # 2) Kills
    plt.subplot(2, 2, 2)
    for player, kills in metrics["kills"].items():
        plt.bar(
            [e - 0.2 if player == 'Ninja' else e + 0.2 for e in x_values],
            kills,
            width=0.4,
            color=colors.get(player),
            label=f"{player} Kills"
        )
    plt.title("Kills per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Kills")
    plt.legend()
    plt.grid(True)

    # 3) Damage Dealt
    plt.subplot(2, 2, 3)
    markers = {'Ninja': 'o', 'Faze Jarvis': 's'}

    for player, damage in metrics["damage_dealt"].items():
        plt.plot(x_values, damage, label=f"{player} Damage",
                 color=colors.get(player),
                 marker=markers.get(player, 'x'),
                 markersize=4,
                 markevery=max(1, len(damage)//20))

    plt.title("Damage Dealt per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Damage")
    plt.legend()
    plt.grid(True)

    # 4) Epsilon and LR
    plt.subplot(2, 2, 4)
    ax1 = plt.gca()
    for player, epsilon in metrics["epsilon"].items():
        ax1.plot(x_values, epsilon, label=f"{player} Epsilon",
                 color=colors.get(player), linestyle='-')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Epsilon")

    ax2 = ax1.twinx()
    for player, lr in metrics["learning_rates"].items():
        ax2.plot(x_values, lr, label=f"{player} LR",
                 color=colors.get(player), linestyle=':', alpha=0.7)
    ax2.set_ylabel("Learning Rate")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.title("Exploration and Learning Rates")
    ax1.grid(True)

    if "shared_history" in metrics:
        hist = metrics["shared_history"]
        info_text = (
            f"Total Epochs: {hist.get('current_epoch', total_episodes)}\n"
            f"Total Steps: {hist.get('total_steps', sum(metrics['episode_steps']))}\n"
            f"Best Rewards:\n"
        )
        for player, val in hist.get('best_rewards', {}).items():
            if hasattr(val, 'value'):
                info_text += f"  {player}: {val.value:.2f}\n"
            elif isinstance(val, (int, float)):
                info_text += f"  {player}: {val:.2f}\n"
            else:
                info_text += f"  {player}: {str(val)}\n"

        plt.figtext(0.02, 0.02, info_text, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    try:
        plt.savefig(f"{run_dir}/plots/training_progress_epoch_{epoch}.png")
        print(f"Successfully saved training progress plot for epoch {epoch}")
    except Exception as e:
        print(f"Error saving training progress plot for epoch {epoch}: {e}")
        os.makedirs(f"{run_dir}/plots", exist_ok=True)
        try:
            plt.savefig(f"{run_dir}/plots/training_progress_epoch_{epoch}.png")
            print("Successfully saved training progress plot after creating directory")
        except Exception as e2:
            print(f"Still could not save training progress plot: {e2}")

    # Additional metrics: steps and survival
    plt.figure(figsize=(15, 5))

    # Steps per epoch
    plt.subplot(1, 2, 1)
    plt.plot(x_values, metrics["episode_steps"], 'g-', label="Episode Length")
    plt.title("Steps per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Steps")
    plt.grid(True)
    plt.legend()

    # Survival
    plt.subplot(1, 2, 2)
    for player, survival in metrics["survival_time"].items():
        plt.plot(x_values, survival, label=f"{player} Survival",
                 color=colors.get(player))
    plt.title("Survival Time per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time Steps")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    try:
        plt.savefig(f"{run_dir}/plots/additional_metrics_epoch_{epoch}.png")
        print(f"Successfully saved plots for epoch {epoch}")
    except Exception as e:
        print(f"Error saving additional metrics plot for epoch {epoch}: {e}")
        os.makedirs(f"{run_dir}/plots", exist_ok=True)
        try:
            plt.savefig(f"{run_dir}/plots/additional_metrics_epoch_{epoch}.png")
            print("Successfully saved plots after creating directory")
        except Exception as e2:
            print(f"Still could not save plot: {e2}")
    finally:
        plt.close('all')


def train_episode(epoch, config, curriculum_stages, world_bounds, display_width, display_height, shared_models, shared_history, shared_epsilon, training=True):
    """Worker function to train a single episode"""
    try:
        # Log epoch and process ID to verify parallel environments
        import os
        print(f"Starting epoch {epoch} in process {os.getpid()}")

        # Determine current curriculum stage
        current_stage = 0
        for i, stage in enumerate(curriculum_stages):
            if epoch >= sum(s["duration"] for s in curriculum_stages[:i]):
                current_stage = i
        current_obstacles = curriculum_stages[current_stage]["n_obstacles"]

        # Create environment for this episode
        env = Env(
            training=training,
            use_game_ui=False,
            world_width=world_bounds[2] - world_bounds[0],
            world_height=world_bounds[3] - world_bounds[1],
            display_width=display_width,
            display_height=display_height,
            n_of_obstacles=current_obstacles,
            frame_skip=config["frame_skip"],
        )

        # Setup players
        players = [
            Character((world_bounds[2] - 100, world_bounds[3] - 100),
                      env.world_surface, boundaries=world_bounds, username="Ninja"),
            Character((world_bounds[0] + 10, world_bounds[1] + 10),
                      env.world_surface, boundaries=world_bounds, username="Faze Jarvis"),
        ]

        # Create bots with shared models
        bots = []
        for idx, player in enumerate(players):
            try:
                # Get device from config
                device = config.get("device", "cpu")

                bot = MyBot(action_size=config["action_size"])
                bot.device = torch.device(device)
                bot.model = bot.model.to(bot.device)
                bot.target_model = bot.target_model.to(bot.device)

                bot.use_double_dqn = config["hyperparameters"]["double_dqn"]
                bot.learning_rate = config["hyperparameters"]["learning_rate"]
                bot.batch_size = config["hyperparameters"]["batch_size"]
                bot.gamma = config["hyperparameters"]["gamma"]
                bot.epsilon_decay = config["hyperparameters"]["epsilon_decay"]
                bot.optimizer = torch.optim.Adam(bot.model.parameters(), lr=bot.learning_rate)

                print("Loaded hyperparameters")

                # Load shared model state with error handling
                if shared_models[idx] is not None:
                    try:
                        # We expect a checkpoint dictionary from the parallel manager
                        bot.load_from_dict(shared_models[idx], map_location=device)
                    except Exception as e:
                        print(f"Error loading model state for bot {idx}: {e}")
                        print("Starting with fresh model state")

                # Set epsilon from shared memory
                bot.epsilon = shared_epsilon[player.username].value
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
            "learning_rate": {player.username: 0 for player in players},
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
                    if training:
                        bot.remember(reward, next_info, finished)

                    # Update epsilon and store in shared memory
                    if training:
                        bot.epsilon = max(0.01, bot.epsilon * bot.epsilon_decay)
                        shared_epsilon[player.username].value = bot.epsilon

                    episode_metrics["epsilon"][player.username] = bot.epsilon
                    episode_metrics["learning_rate"][player.username] = bot.learning_rate
                except Exception as e:
                    print(f"Error processing player {player.username}: {e}")
                    continue

            if finished:
                break

        # Return full checkpoint dictionaries (including optimizer, etc.)
        model_checkpoints = []
        for bot in bots:
            # Move bot’s model/optimizer states to CPU so the dictionary is consistent
            bot.model.cpu()
            # Copy the optimizer state to CPU as well
            bot.optimizer.state = {
                k: {
                    subk: v.cpu() for subk, v in val.items() if isinstance(v, torch.Tensor)
                } if isinstance(val, dict) else val
                for k, val in bot.optimizer.state.items()
            }
            checkpoint_dict = bot.save_to_dict()
            model_checkpoints.append(checkpoint_dict)

        return episode_metrics, env.steps, model_checkpoints

    except Exception as e:
        print(f"Critical error in train_episode: {e}")
        # Return safe default metrics in case of error
        return {
            "rewards": {"Ninja": 0, "Faze Jarvis": 0},
            "kills": {"Ninja": 0, "Faze Jarvis": 0},
            "damage_dealt": {"Ninja": 0, "Faze Jarvis": 0},
            "survival_time": {"Ninja": 0, "Faze Jarvis": 0},
            "epsilon": {"Ninja": 0, "Faze Jarvis": 0},
            "learning_rate": {"Ninja": 0, "Faze Jarvis": 0},
        }, 0, [None, None]


def main(num_environments=4, device=None, num_epochs=1000, training=True):
    # Environment parameters
    world_width = 1280
    world_height = 1280
    display_width = 800
    display_height = 800
    n_of_obstacles = 15

    load_back = True
    state_size = 38

    # If not training, force single environment
    if not training:
        num_environments = 1

    # CUDA availability check
    cuda_available = torch.cuda.is_available()
    if device is None:
        # Auto-select best available device
        if cuda_available:
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    # Create training run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"training_runs/{timestamp}"

    # Make sure parent directory exists
    os.makedirs("../training_runs", exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/models", exist_ok=True)
    os.makedirs(f"{run_dir}/plots", exist_ok=True)

    # Training configuration
    config = {
        "frame_skip": 4,
        "tick_limit": 2400,
        "num_epochs": num_epochs,
        "action_size": 56,
        "device": device,
        "hyperparameters": {
            "double_dqn": True,
            "learning_rate": 0.0001,
            "batch_size": 64,
            "gamma": 0.99,
            "epsilon_decay": 0.99996,
        }
    }

    # Create initial environment to get world bounds
    env = Env(
        training=training,
        use_game_ui=False,
        world_width=world_width,
        world_height=world_height,
        display_width=display_width,
        display_height=display_height,
        n_of_obstacles=n_of_obstacles,
        frame_skip=config["frame_skip"],
    )
    world_bounds = env.get_world_bounds()

    # Curriculum learning parameters
    curriculum_stages = [
        {"n_obstacles": 10, "duration": 100},
        {"n_obstacles": 15, "duration": 200},
        {"n_obstacles": 20, "duration": 300},
        {"n_obstacles": 25, "duration": 400},
    ]

    # Setup shared memory and synchronization (only needed for parallel mode)
    manager = Manager()

    # Shared model states with proper synchronization
    shared_models = manager.list([None, None])

    # Shared epsilon values for each bot with proper synchronization
    shared_epsilon = manager.dict({
        "Ninja": manager.Value('f', 1.0),
        "Faze Jarvis": manager.Value('f', 1.0),
    })

    # Shared training history
    shared_history = manager.dict({
        "total_steps": manager.Value('i', 0),
        "total_episodes": manager.Value('i', 0),
        "current_epoch": manager.Value('i', 0),
        "best_rewards": manager.dict({
            "Ninja": manager.Value('f', float('-inf')),
            "Faze Jarvis": manager.Value('f', float('-inf')),
        }),
        "last_save": manager.Value('i', 0),
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
        "learning_rates": {"Ninja": manager.list(), "Faze Jarvis": manager.list()},
    })

    # Models for non-parallel mode
    local_models = [None, None]

    # Initialize models if loading from previous training
    if load_back:
        try:
            for idx in range(2):
                save_path = f"bot_model_{idx}.pth"
                if os.path.exists(save_path):
                    # Create a temporary bot to load the model
                    temp_bot = MyBot(action_size=config["action_size"])
                    try:
                        # Try to load the full checkpoint via MyBot.load
                        temp_bot.load(save_path, map_location=device)

                        # Convert everything to CPU for storing in the manager
                        temp_bot.model.cpu()
                        temp_bot.optimizer.state = {
                            k: {
                                subk: v.cpu() for subk, v in val.items()
                                if isinstance(v, torch.Tensor)
                            } if isinstance(val, dict) else val
                            for k, val in temp_bot.optimizer.state.items()
                        }

                        # Obtain the in-memory dict
                        model_checkpoint = temp_bot.save_to_dict()

                        if num_environments > 1:
                            shared_models[idx] = model_checkpoint
                        local_models[idx] = model_checkpoint

                        print(f"Loaded model {idx} from {save_path} to {device}")
                    except Exception as e:
                        print(f"Error loading model to {device}: {e}")
                        print(f"Trying to load to CPU instead")
                        try:
                            temp_bot.load(save_path, map_location="cpu")
                            temp_bot.model.cpu()
                            temp_bot.optimizer.state = {
                                k: {
                                    subk: v.cpu() for subk, v in val.items()
                                    if isinstance(v, torch.Tensor)
                                } if isinstance(val, dict) else val
                                for k, val in temp_bot.optimizer.state.items()
                            }
                            model_checkpoint = temp_bot.save_to_dict()

                            if num_environments > 1:
                                shared_models[idx] = model_checkpoint
                            local_models[idx] = model_checkpoint
                            print(f"Loaded model {idx} from {save_path} to CPU")
                        except Exception as cpu_e:
                            print(f"Error processing model after loading to CPU: {cpu_e}")
                            # Continue with a fresh model instead
                            print(f"Starting with a fresh model for bot {idx}")
                else:
                    print(f"No saved model found for bot {idx}, starting fresh")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Starting with fresh models")

    # Determine whether to use parallel or sequential training
    use_parallel = num_environments > 1

    if use_parallel:
        print(f"Using parallel training with {num_environments} environments")
        # Setup parallel processing with error handling
        try:
            num_processes = min(max(1, cpu_count() - 1), num_environments)
            print(f"Starting training with {num_processes} processes")
            pool = Pool(processes=num_processes)

            # Create partial function with fixed arguments
            train_episode_partial = partial(
                train_episode,
                config=config,
                curriculum_stages=curriculum_stages,
                world_bounds=world_bounds,
                display_width=display_width,
                display_height=display_height,
                shared_models=shared_models,
                shared_history=shared_history,
                shared_epsilon=shared_epsilon,
                training=training
            )

            # Calculate number of batches needed to reach num_epochs
            num_batches = (num_epochs + num_processes - 1) // num_processes

            # Before the training loop
            start_time = time.time()  # Record the start time

            # Training loop with improved error handling
            for batch in range(num_batches):
                # Calculate epoch range for this batch
                start_epoch = batch * num_processes
                end_epoch = min(start_epoch + num_processes, num_epochs)
                epochs_in_batch = end_epoch - start_epoch

                print(f"Starting batch {batch + 1}/{num_batches} (epochs {start_epoch + 1} to {end_epoch})")

                try:
                    # Generate epoch indices for this batch
                    epoch_indices = list(range(start_epoch, end_epoch))

                    # Run multiple episodes in parallel with timeout (one per epoch)
                    results = pool.map_async(train_episode_partial, epoch_indices)
                    results = results.get(timeout=3600)  # 1-hour timeout per batch

                    # Aggregate results
                    for i, (episode_metrics, steps, model_states) in enumerate(results):
                        try:
                            # Actual epoch
                            epoch = start_epoch + i

                            # Update shared history
                            shared_history["total_steps"].value += steps
                            shared_history["total_episodes"].value += 1
                            shared_history["current_epoch"].value = epoch + 1

                            # Update metrics
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

                                # Average rewards
                                avg_reward = (
                                    sum(metrics["episode_rewards"][player][-10:])
                                    / min(10, len(metrics["episode_rewards"][player]))
                                )
                                metrics["avg_rewards"][player].append(avg_reward)

                                print(f"Epoch {epoch + 1}/{num_epochs} - {player}: "
                                      f"Reward = {episode_metrics['rewards'][player]:.2f}, "
                                      f"Avg(10) = {avg_reward:.2f}, "
                                      f"Kills = {episode_metrics['kills'][player]}, "
                                      f"Damage = {episode_metrics['damage_dealt'][player]:.1f}, "
                                      f"Epsilon = {episode_metrics['epsilon'][player]:.4f}")

                            # Update shared models after each epoch
                            for idx, model_checkpoint in enumerate(model_states):
                                if model_checkpoint is not None:
                                    shared_models[idx] = model_checkpoint
                                    local_models[idx] = model_checkpoint

                        except Exception as e:
                            print(f"Error processing epoch {start_epoch + i} metrics: {e}")
                            continue

                    # Save metrics and plots periodically
                    current_epoch = shared_history["current_epoch"].value
                    if (current_epoch % 1 == 0) or current_epoch >= num_epochs:
                        print("Saving metrics and models...")
                        try:
                            # Convert shared memory metrics to regular dict
                            save_metrics = {}

                            def convert_to_serializable(obj):
                                if isinstance(obj, (list, tuple)):
                                    return [convert_to_serializable(x) for x in obj]
                                elif str(type(obj).__name__) == 'ListProxy':
                                    return [convert_to_serializable(x) for x in obj]
                                elif isinstance(obj, dict):
                                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                                elif hasattr(obj, 'value'):  # handle multiprocessing.Value
                                    return obj.value
                                return obj

                            for k, v in metrics.items():
                                try:
                                    save_metrics[k] = convert_to_serializable(v)
                                except Exception as e:
                                    print(f"Error converting metric {k}: {e}")
                                    save_metrics[k] = []

                            # Add shared history
                            save_metrics["shared_history"] = {
                                "total_steps": shared_history["total_steps"].value,
                                "total_episodes": shared_history["total_episodes"].value,
                                "current_epoch": shared_history["current_epoch"].value,
                                "best_rewards": {
                                    player: shared_history["best_rewards"][player].value
                                    for player in ["Ninja", "Faze Jarvis"]
                                },
                            }

                            # Save metrics
                            metrics_file = f"{run_dir}/metrics.json"
                            temp_file = f"{metrics_file}.tmp"
                            try:
                                with open(temp_file, "w") as f:
                                    json.dump(save_metrics, f, indent=4)
                                os.replace(temp_file, metrics_file)
                                print(f"Successfully saved metrics to {metrics_file}")
                            except Exception as e:
                                print(f"Error saving metrics file: {e}")
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
                                raise

                            # Create and save plots
                            try:
                                create_training_plots(save_metrics, run_dir, current_epoch)
                            except Exception as e:
                                print(f"Error creating plots: {e}")

                            # Save model checkpoints
                            for idx, model_checkpoint in enumerate(shared_models):
                                if model_checkpoint is not None:
                                    try:
                                        temp_bot = MyBot(action_size=config["action_size"])
                                        temp_bot.load_from_dict(model_checkpoint, map_location="cpu")

                                        os.makedirs(f"{run_dir}/models", exist_ok=True)

                                        # Save checkpoint with MyBot.save
                                        save_path = f"{run_dir}/models/bot_model_{idx}_epoch_{current_epoch}.pth"
                                        temp_bot.save(save_path)
                                        print(f"Successfully saved model {idx} checkpoint at epoch {current_epoch}")

                                        # Also save to standard location
                                        standard_path = f"bot_model_{idx}.pth"
                                        temp_bot.save(standard_path)
                                        print(f"Successfully saved model {idx} to standard location")

                                        # Save backup
                                        backup_path = f"{run_dir}/models/bot_model_{idx}_epoch_{current_epoch}_backup.pth"
                                        temp_bot.save(backup_path)
                                        print(f"Successfully saved backup model {idx} at epoch {current_epoch}")

                                    except Exception as model_e:
                                        print(f"Error saving model {idx} at epoch {current_epoch}: {model_e}")
                        except Exception as e:
                            print(f"Error in save operation: {e}")
                            print(f"Error details: {str(e)}")
                            # Try to save at least the metrics
                            try:
                                with open(f"{run_dir}/metrics.json", "w") as f:
                                    json.dump(save_metrics, f, indent=4)
                            except Exception as metrics_e:
                                print(f"Failed to save metrics as fallback: {metrics_e}")

                except Exception as e:
                    print(f"Error in batch {batch + 1}: {e}")
                    # Try to recover by saving current state
                    try:
                        os.makedirs("recovery", exist_ok=True)
                        for idx, model_checkpoint in enumerate(shared_models):
                            if model_checkpoint is not None:
                                recovery_path = f"recovery/bot_model_{idx}_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                                temp_bot = MyBot(action_size=config["action_size"])
                                temp_bot.load_from_dict(model_checkpoint, map_location="cpu")
                                temp_bot.save(recovery_path)
                                print(f"Saved recovery model to {recovery_path}")
                                temp_bot.save(f"bot_model_{idx}_recovery.pth")
                    except Exception as recovery_e:
                        print(f"Error during recovery: {recovery_e}")

        except Exception as e:
            print(f"Critical error during training: {e}")
            raise
        finally:
            try:
                pool.close()
                pool.join()
                pygame.quit()
            except Exception as e:
                print(f"Error during cleanup: {e}")

    else:
        # Non-parallel training path
        print("Using sequential training (non-parallel mode)")

        # Prepare storage for metrics
        local_metrics = {
            "episode_rewards": {"Ninja": [], "Faze Jarvis": []},
            "avg_rewards": {"Ninja": [], "Faze Jarvis": []},
            "episode_steps": [],
            "kills": {"Ninja": [], "Faze Jarvis": []},
            "damage_dealt": {"Ninja": [], "Faze Jarvis": []},
            "survival_time": {"Ninja": [], "Faze Jarvis": []},
            "epsilon": {"Ninja": [], "Faze Jarvis": []},
            "learning_rates": {"Ninja": [], "Faze Jarvis": []},
        }

        total_steps = 0
        total_episodes = 0
        best_rewards = {"Ninja": float('-inf'), "Faze Jarvis": float('-inf')}

        try:
            start_time = time.time()

            # Training loop for non-parallel mode
            for epoch in range(config["num_epochs"]):
                print(f"Starting epoch {epoch + 1}/{config['num_epochs']}")

                try:
                    # Determine current curriculum stage
                    current_stage = 0
                    for i, stage in enumerate(curriculum_stages):
                        if epoch >= sum(s["duration"] for s in curriculum_stages[:i]):
                            current_stage = i
                    current_obstacles = curriculum_stages[current_stage]["n_obstacles"]

                    env = Env(
                        training=training,
                        use_game_ui=False,
                        world_width=world_bounds[2] - world_bounds[0],
                        world_height=world_bounds[3] - world_bounds[1],
                        display_width=display_width,
                        display_height=display_height,
                        n_of_obstacles=current_obstacles,
                        frame_skip=config["frame_skip"],
                    )

                    players = [
                        Character((world_bounds[2] - 100, world_bounds[3] - 100),
                                  env.world_surface, boundaries=world_bounds, username="Ninja"),
                        Character((world_bounds[0] + 10, world_bounds[1] + 10),
                                  env.world_surface, boundaries=world_bounds, username="Faze Jarvis"),
                    ]

                    bots = []
                    for idx in range(2):
                        bot = MyBot(action_size=config["action_size"])
                        bot.use_double_dqn = config["hyperparameters"]["double_dqn"]
                        bot.learning_rate = config["hyperparameters"]["learning_rate"]
                        bot.batch_size = config["hyperparameters"]["batch_size"]
                        bot.gamma = config["hyperparameters"]["gamma"]
                        bot.epsilon_decay = config["hyperparameters"]["epsilon_decay"]

                        bot.device = torch.device(device)
                        bot.model = bot.model.to(bot.device)
                        bot.target_model = bot.target_model.to(bot.device)
                        bot.optimizer = torch.optim.Adam(bot.model.parameters(), lr=bot.learning_rate)

                        # Load local model if present
                        if local_models[idx] is not None:
                            try:
                                bot.load_from_dict(local_models[idx], map_location=device)
                            except Exception as e:
                                print(f"Error loading model state for bot {idx}: {e}")
                                print("Starting with fresh model state")

                        bots.append(bot)

                    env.set_players_bots_objects(players, bots)
                    env.reset(randomize_objects=True)
                    env.steps = 0

                    if hasattr(env, 'last_damage_tracker'):
                        env.last_damage_tracker = {player.username: 0 for player in players}

                    for bot in bots:
                        bot.reset_for_new_episode()

                    episode_metrics = {
                        "rewards": {player.username: 0 for player in players},
                        "kills": {player.username: 0 for player in players},
                        "damage_dealt": {player.username: 0 for player in players},
                        "survival_time": {player.username: 0 for player in players},
                        "epsilon": {player.username: 0 for player in players},
                        "learning_rate": {player.username: 0 for player in players},
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
                                if training:
                                    bot.remember(reward, next_info, finished)

                                episode_metrics["epsilon"][player.username] = bot.epsilon
                                episode_metrics["learning_rate"][player.username] = bot.learning_rate
                            except Exception as e:
                                print(f"Error processing player {player.username}: {e}")
                                continue

                        if finished:
                            break

                    total_steps += env.steps
                    total_episodes += 1

                    local_metrics["episode_steps"].append(env.steps)
                    for player in ["Ninja", "Faze Jarvis"]:
                        local_metrics["episode_rewards"][player].append(episode_metrics["rewards"][player])
                        local_metrics["kills"][player].append(episode_metrics["kills"][player])
                        local_metrics["damage_dealt"][player].append(episode_metrics["damage_dealt"][player])
                        local_metrics["survival_time"][player].append(episode_metrics["survival_time"][player])
                        local_metrics["epsilon"][player].append(episode_metrics["epsilon"][player])
                        local_metrics["learning_rates"][player].append(episode_metrics["learning_rate"][player])

                        if episode_metrics["rewards"][player] > best_rewards[player]:
                            best_rewards[player] = episode_metrics["rewards"][player]

                        avg_reward = sum(local_metrics["episode_rewards"][player][-10:]) / min(10, len(local_metrics["episode_rewards"][player]))
                        local_metrics["avg_rewards"][player].append(avg_reward)

                        print(f"Epoch {epoch + 1}/{config['num_epochs']} - {player}: "
                              f"Reward = {episode_metrics['rewards'][player]:.2f}, "
                              f"Avg(10) = {avg_reward:.2f}, "
                              f"Kills = {episode_metrics['kills'][player]}, "
                              f"Damage = {episode_metrics['damage_dealt'][player]:.1f}, "
                              f"Epsilon = {episode_metrics['epsilon'][player]:.4f}")

                    # Save updated model checkpoints after each epoch
                    for idx, bot in enumerate(bots):
                        # Move to CPU for consistent saving
                        bot.model.cpu()
                        bot.optimizer.state = {
                            k: {
                                subk: v.cpu() for subk, v in val.items()
                                if isinstance(v, torch.Tensor)
                            } if isinstance(val, dict) else val
                            for k, val in bot.optimizer.state.items()
                        }
                        local_models[idx] = bot.save_to_dict()

                    if (epoch + 1) % 10 == 0 or epoch == config["num_epochs"] - 1:
                        try:
                            save_metrics = copy.deepcopy(local_metrics)
                            save_metrics["shared_history"] = {
                                "total_steps": total_steps,
                                "total_episodes": total_episodes,
                                "current_epoch": epoch + 1,
                                "best_rewards": best_rewards,
                            }

                            metrics_file = f"{run_dir}/metrics.json"
                            temp_file = f"{metrics_file}.tmp"
                            try:
                                with open(temp_file, "w") as f:
                                    json.dump(save_metrics, f, indent=4)
                                os.replace(temp_file, metrics_file)
                                print(f"Successfully saved metrics to {metrics_file}")
                            except Exception as e:
                                print(f"Error saving metrics file: {e}")
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
                                # Fallback
                                with open(metrics_file, "w") as f:
                                    json.dump(save_metrics, f, indent=4)
                                print(f"Saved metrics directly as fallback")

                            try:
                                create_training_plots(save_metrics, run_dir, epoch + 1)
                                print(f"Successfully created plots for epoch {epoch+1}")
                            except Exception as e:
                                print(f"Error creating plots: {e}")

                            # Save model checkpoints
                            for idx, model_checkpoint in enumerate(local_models):
                                if model_checkpoint is not None:
                                    try:
                                        temp_bot = MyBot(action_size=config["action_size"])
                                        temp_bot.load_from_dict(model_checkpoint, map_location="cpu")

                                        save_path = f"{run_dir}/models/bot_model_{idx}_epoch_{epoch+1}.pth"
                                        temp_bot.save(save_path)
                                        print(f"Successfully saved model {idx} checkpoint at epoch {epoch+1}")

                                        torch.save(model_checkpoint, f"bot_model_{idx}.pth")
                                        print(f"Successfully saved model {idx} to standard location")
                                    except Exception as model_e:
                                        print(f"Error saving model {idx} at epoch {epoch+1}: {model_e}")
                                        # Attempt directory creation
                                        os.makedirs(f"{run_dir}/models", exist_ok=True)
                                        try:
                                            temp_bot.save(save_path)
                                            torch.save(model_checkpoint, f"bot_model_{idx}.pth")
                                            print(f"Successfully saved model {idx} after creating directory")
                                        except Exception as model_e2:
                                            print(f"Still could not save model {idx}: {model_e2}")
                        except Exception as e:
                            print(f"Error saving metrics or models: {e}")

                    elapsed_time = time.time() - start_time
                    steps_per_second = total_steps / elapsed_time if elapsed_time > 0 else 0

                    print(f"Epoch {epoch + 1}/{config['num_epochs']} - Steps per second: {steps_per_second:.2f}")

                    # Reset total_steps if you only want per-epoch measure
                    total_steps = 0

                except Exception as e:
                    print(f"Error in epoch {epoch + 1}: {e}")
                    # Try to recover
                    try:
                        os.makedirs("recovery", exist_ok=True)
                        for idx, model_checkpoint in enumerate(local_models):
                            if model_checkpoint is not None:
                                recovery_path = f"recovery/bot_model_{idx}_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                                temp_bot = MyBot(action_size=config["action_size"])
                                temp_bot.load_from_dict(model_checkpoint, map_location="cpu")
                                temp_bot.save(recovery_path)
                                print(f"Saved recovery model to {recovery_path}")
                                temp_bot.save(f"bot_model_{idx}_recovery.pth")
                    except Exception as recovery_e:
                        print(f"Error during recovery: {recovery_e}")

        except Exception as e:
            print(f"Critical error during training: {e}")
            raise
        finally:
            try:
                pygame.quit()
            except Exception as e:
                print(f"Error during cleanup: {e}")

    # Print final stats
    if use_parallel:
        print(f"Training complete! Results saved to {run_dir}")
        print(f"Total steps: {shared_history['total_steps'].value}")
        print(f"Total episodes: {shared_history['total_episodes'].value}")
        print(f"Final epoch: {shared_history['current_epoch'].value}")
        print("Best rewards:")
        for player in ["Ninja", "Faze Jarvis"]:
            print(f"{player}: {shared_history['best_rewards'][player].value:.2f}")
    else:
        print(f"Training complete! Results saved to {run_dir}")
        print(f"Total steps: {total_steps}")
        print(f"Total episodes: {total_episodes}")
        print("Best rewards:")
        for player in ["Ninja", "Faze Jarvis"]:
            print(f"{player}: {best_rewards[player]:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train reinforcement learning bots')
    parser.add_argument('--num_environments', type=int, default=4,
                        help='Number of parallel environments to use. Set to 1 to disable parallelization.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, cpu, mps). If not specified, best available device will be used.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs to run')
    parser.add_argument('--training', action='store_true', default=True,
                        help='Enable training mode')
    parser.add_argument('--no-training', dest='training', action='store_false', default=False,
                        help='Disable training mode (only use one environment and do not train)')
    args = parser.parse_args()

    # Set up screen for pygame
    screen = pygame.display.set_mode((800, 800))

    main_kwargs = {
        'num_environments': args.num_environments,
        'device': args.device,
        'num_epochs': args.epochs,
        'training': args.training
    }

    main(**main_kwargs)
