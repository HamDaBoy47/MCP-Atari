from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from atari_policies import MCPAtariPolicy, AtariCNN
from game_configs import get_game_config
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import traceback
import argparse
import datetime
import logging
import torch
import wandb
import json
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

class WandbCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_calls = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            metrics = evaluate_model(self.model, self.eval_env)
            wandb.log(metrics, step=self.n_calls)
        return True

    def reset(self):
        self.n_calls = 0
    
class PacmanSubsetActionWrapper(gym.Wrapper):
    def __init__(self, env, subset_actions):
        super().__init__(env)
        self.subset_actions = subset_actions
        self.action_space = gym.spaces.Discrete(len(subset_actions))

    def step(self, action):
        full_action = self.subset_actions[action]
        return self.env.step(full_action)
    
def plot_action_distributions(pre_training_dist, transfer_dist):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.bar(range(len(pre_training_dist)), pre_training_dist)
    ax1.set_title('Pre-training Action Distribution')
    ax1.set_xlabel('Action')
    ax1.set_ylabel('Probability')
    
    ax2.bar(range(len(transfer_dist)), transfer_dist)
    ax2.set_title('Transfer Learning Action Distribution')
    ax2.set_xlabel('Action')
    ax2.set_ylabel('Probability')
    
    plt.tight_layout()
    return fig
    
def debug_step(env, obs, action, reward, next_obs, done, info):
    logging.info(f"Observation shape: {obs.shape}")
    logging.info(f"Action taken: {action}")
    logging.info(f"Reward received: {reward}")
    logging.info(f"Done flag: {done}")
    logging.info(f"Info: {info}")
    if done:
        logging.info(f"Episode finished. Total reward: {info.get('episode', {}).get('r', 'N/A')}")
    
    # Check for NaN or infinity values
    if np.isnan(obs).any() or np.isinf(obs).any():
        logging.warning("NaN or Inf values detected in observation!")
    if np.isnan(reward) or np.isinf(reward):
        logging.warning("NaN or Inf values detected in reward!")

def create_atari_env(game_name, seed=None):
    """Create Atari environment with full action space"""
    def make_env():
        env = gym.make(f"ALE/{game_name}-v5", render_mode="rgb_array", full_action_space=True)
        env = AtariWrapper(env)
        if seed is not None:
            env.seed(seed)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    return env

def evaluate_model(model, env, num_episodes=100, random_score=0, human_score=10000, success_threshold=0.75, is_transfer=False):
    episode_rewards = []
    episode_lengths = []
    action_counts = np.zeros(env.action_space.n)
    
    # New metrics
    total_frames = 0
    rewards_100ep = deque(maxlen=100)
    best_reward = float('-inf')

    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_actions = []
        episode_rewards = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            action_counts[action[0]] += 1
            total_frames += 1
            
            episode_actions.append(action[0])
            episode_rewards.append(reward[0])
        
        # Detailed logging for each episode
        logging.info(f"Episode {i+1} finished: Total Reward: {episode_reward} Episode Length: {episode_length}")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        rewards_100ep.append(episode_reward)
        best_reward = max(best_reward, episode_reward)

    # Calculate metrics
    normalized_scores = [(score - random_score) / (human_score - random_score) for score in episode_rewards]
    success_rate = sum(score >= success_threshold for score in normalized_scores) / num_episodes
    mean_normalized_score = np.mean(normalized_scores)
    completion_rate = 0  # Since we don't have a clear completion metric for Ms. Pac-Man
    mean_episode_length = np.mean(episode_lengths)

    # Calculate action distribution
    action_distribution = action_counts / np.sum(action_counts)

    # New metrics calculations
    mean_reward = np.mean(episode_rewards)
    median_reward = np.median(episode_rewards)
    mean_reward_100ep = np.mean(rewards_100ep)
    
    # Calculate scores relative to random and human performance
    relative_score = (mean_reward - random_score) / (human_score - random_score) * 100

    # Log overall metrics
    logging.info("Overall Evaluation Metrics:")
    logging.info(f"  Mean Reward: {mean_reward}")
    logging.info(f"  Median Reward: {median_reward}")
    logging.info(f"  Best Reward: {best_reward}")
    logging.info(f"  Mean Normalized Score: {mean_normalized_score}")
    logging.info(f"  Success Rate: {success_rate}")
    logging.info(f"  Mean Episode Length: {mean_episode_length}")
    logging.info(f"  Relative Score: {relative_score}%")
    # logging.info(f"  Action Distribution: {action_distribution}")

    # Log metrics to wandb
    metrics = {
        "success_rate": success_rate,
        "mean_normalized_score": mean_normalized_score,
        "completion_rate": completion_rate,
        "mean_episode_length": mean_episode_length,
        "max_normalized_score": np.max(normalized_scores),
        "min_normalized_score": np.min(normalized_scores),
        "std_normalized_score": np.std(normalized_scores),
        "mean_reward": mean_reward,
        "median_reward": median_reward,
        "best_reward": best_reward,
        "mean_reward_100ep": mean_reward_100ep,
        "relative_score": relative_score,
        "total_frames": total_frames
    }

    # Log action distribution
    for i, prob in enumerate(action_distribution):
        metrics[f"action_{i}_prob"] = prob

    wandb.log(metrics)

    return metrics

def train_mcp_pre_training(env, eval_env, total_timesteps, log_dir, game_config, num_primitives, features_dim, learning_rate):
    """Pre-training phase with full action space"""
    wandb.init(project="mcp_atari_final", name=f"pre-training_{log_dir}", config={
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "num_primitives": num_primitives,
        "features_dim": features_dim
    })
    
    policy_kwargs = dict(
        features_extractor_class=AtariCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
        num_primitives=num_primitives,
    )

    model = PPO(
        MCPAtariPolicy, 
        env, 
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=learning_rate,
        ent_coef=0.01  # Encourage exploration
    )
    
    wandb_callback = WandbCallback(eval_env)
    
    model.learn(total_timesteps=total_timesteps, callback=wandb_callback)
    model.save(f"{log_dir}/final_model")
    
    # Evaluate and log primitive specialization
    results = evaluate_model(
        model, 
        eval_env, 
        random_score=game_config["random_score"],
        human_score=game_config["human_score"],
        success_threshold=game_config["success_threshold"]
    )
    
    wandb.log(results)
    
    return model, results

def transfer_learning_new_task(env, eval_env, model_path, total_timesteps, log_dir, game_config, num_primitives, features_dim, learning_rate):
    """Transfer learning to new task while maintaining primitive structure"""
    wandb.init(project="mcp_atari_final", name=f"transfer_{log_dir}", config={
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate
    })
    
    # Load the pre-trained model
    pre_trained_model = PPO.load(model_path)
    
    # Create new model for transfer task
    policy_kwargs = dict(
        features_extractor_class=AtariCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
        num_primitives=num_primitives,
    )
    
    transferred_model = PPO(
        MCPAtariPolicy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=learning_rate
    )
    
    # Copy pre-trained weights
    transferred_model.policy.features_extractor.load_state_dict(
        pre_trained_model.policy.features_extractor.state_dict()
    )
    transferred_model.policy.mlp_extractor.load_state_dict(
        pre_trained_model.policy.mlp_extractor.state_dict()
    )
    
    # Freeze primitives but allow gate to adapt
    transferred_model.policy.freeze_primitives()
    
    # Recreate optimizer with correct parameters
    transferred_model.policy.optimizer = transferred_model.policy.optimizer_class(
        [p for p in transferred_model.policy.parameters() if p.requires_grad],
        lr=transferred_model.learning_rate,
        **transferred_model.policy.optimizer_kwargs
    )

    wandb_callback = WandbCallback(eval_env)
    
    transferred_model.learn(total_timesteps=total_timesteps, callback=wandb_callback)
    transferred_model.save(f"{log_dir}/final_model")
    
    results = evaluate_model(
        transferred_model, 
        eval_env, 
        random_score=game_config["random_score"],
        human_score=game_config["human_score"],
        success_threshold=game_config["success_threshold"]
    )
    
    wandb.log(results)
    
    return transferred_model, results

def train_baseline_ppo(env, eval_env, total_timesteps, log_dir, game_config):
    wandb.init(project="mcp_atari_final", name=f"baseline_{log_dir}", config={
        "total_timesteps": total_timesteps
    })
        
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)

    # eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
    #                              log_path=log_dir, eval_freq=10000,
    #                              deterministic=True, render=False)
    wandb_callback = WandbCallback(eval_env)

    model.learn(total_timesteps=total_timesteps, callback=wandb_callback)
    model.save(f"{log_dir}/final_model")
    
    results = evaluate_model(
        model,
        eval_env,
        random_score=game_config["random_score"],
        human_score=game_config["human_score"],
        success_threshold=game_config["success_threshold"]
    )
    wandb.log(results)
    
    return model, results

def run_experiment(args):
    results = {}
    game_config = get_game_config(args.game_name)
    
    # Create timestamp and log directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{args.game_name}_primitives{args.num_primitives}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Set seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Create environments for pre-training task
    env = create_atari_env(args.game_name, args.seed)
    eval_env = create_atari_env(args.game_name, args.seed)

    try:
        print(f"Pre-training MCP on {args.game_name}...")
        mcp_model, results["pre_training"] = train_mcp_pre_training(
            env, eval_env, args.pre_training_timesteps, 
            f"{log_dir}/pre_training", game_config, args.num_primitives, 
            args.features_dim, args.pre_training_lr
        )
        print("Pre-training Results:", results["pre_training"])
        wandb.finish()
    except Exception as e:
        print(f"Error in pre-training: {e}")
        print(traceback.format_exc())
        wandb.finish()

    # Create environments for transfer task
    transfer_env = create_atari_env(args.transfer_game_name, args.seed)
    transfer_eval_env = create_atari_env(args.transfer_game_name, args.seed)

    try:
        print(f"Performing transfer learning to {args.transfer_game_name}...")
        transfer_model, results["transfer"] = transfer_learning_new_task(
            transfer_env, transfer_eval_env, 
            f"{log_dir}/pre_training/final_model",
            args.transfer_timesteps, f"{log_dir}/transfer", 
            get_game_config(args.transfer_game_name),
            args.num_primitives, args.features_dim,
            args.transfer_learning_lr
        )
        print("Transfer Results:", results["transfer"])
        wandb.finish()
    except Exception as e:
        print(f"Error in transfer learning: {e}")
        print(traceback.format_exc())
        wandb.finish()

    try:
        print("Training baseline PPO...")
        baseline_model, results["baseline_ppo"] = train_baseline_ppo(
            transfer_env, transfer_eval_env, 
            args.baseline_timesteps, f"{log_dir}/baseline_ppo", 
            get_game_config(args.transfer_game_name)
        )
        print("Baseline PPO Results:", results["baseline_ppo"])
        wandb.finish()
    except Exception as e:
        print(f"Error in baseline PPO training: {e}")
        print(traceback.format_exc())
        wandb.finish()

    # Save results
    with open(f"{log_dir}/experiment_results.json", "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(f"Experiment completed. Results saved to {log_dir}/experiment_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCP experiments on Atari games")
    parser.add_argument("--game_name", type=str, default="MsPacman", help="Name of the Atari game")
    parser.add_argument("--transfer_game_name", type=str, default="MsPacman", help="Name of the Atari game")
    parser.add_argument("--subset_actions", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8], help="Subset of actions for initial training")
    parser.add_argument("--pre_training_timesteps", type=int, default=100000, help="Total timesteps for MCP subset training")
    parser.add_argument("--transfer_timesteps", type=int, default=50000, help="Total timesteps for MCP full action space training")
    parser.add_argument("--baseline_timesteps", type=int, default=100000, help="Total timesteps for baseline PPO training")
    parser.add_argument("--num_primitives", type=int, default=8, help="Number of primitives in the MCP model")
    parser.add_argument("--pre_training_lr", type=float, default=3e-3, help="Learning rate for pre-training")
    parser.add_argument("--transfer_learning_lr", type=float, default=1e-4, help="Learning rate for transfer learning")
    parser.add_argument("--features_dim", type=int, default=512, help="Dimension of the feature extractor output")
    parser.add_argument("--primitive_action_dim", type=int, default=4, help="Dimension of primitive actions")
    parser.add_argument("--eval_freq", type=int, default=10000, help="Frequency of evaluation during training")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    run_experiment(args)