import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from atari_policies import MCPAtariPolicy, AtariCNN
from game_configs import get_game_config
import os
import json
import wandb
import traceback
import argparse
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            metrics = evaluate_model(self.model, self.eval_env)
            wandb.log(metrics, step=self.n_calls)
        return True
    
class PacmanSubsetActionWrapper(gym.Wrapper):
    def __init__(self, env, subset_actions):
        super().__init__(env)
        self.subset_actions = subset_actions
        self.action_space = gym.spaces.Discrete(len(subset_actions))

    def step(self, action):
        full_action = self.subset_actions[action]
        return self.env.step(full_action)

def create_pacman_env(subset_actions=None):
    def make_env():
        env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array", full_action_space=True)
        env = AtariWrapper(env)
        if subset_actions is not None:
            env = PacmanSubsetActionWrapper(env, subset_actions)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env

def evaluate_model(model, env, num_episodes=100, random_score=0, human_score=10000, success_threshold=0.75):
    episode_rewards = []
    episode_lengths = []
    episode_completed = []

    for _ in range(num_episodes):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        episode_reward = 0
        episode_length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            obs, reward, terminated, truncated = step_result[:4]
            info = step_result[4] if len(step_result) > 4 else {}
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_completed.append(info.get('flag_get', False))  # Adjust based on the specific Atari game(model, env, num_episodes=100, random_score=0, human_score=10000, success_threshold=0.75):
    episode_rewards = []
    episode_lengths = []
    episode_completed = []

    for _ in range(num_episodes):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        episode_reward = 0
        episode_length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            obs, reward, terminated, truncated = step_result[:4]
            info = step_result[4] if len(step_result) > 4 else {}
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_completed.append(info.get('flag_get', False))  # Adjust based on the specific Atari game

    # Calculate metrics
    normalized_scores = [(score - random_score) / (human_score - random_score) for score in episode_rewards]
    success_rate = sum(score >= success_threshold for score in normalized_scores) / num_episodes
    mean_normalized_score = np.mean(normalized_scores)
    completion_rate = sum(episode_completed) / num_episodes
    mean_episode_length = np.mean(episode_lengths)

    # Log metrics to wandb
    wandb.log({
        "success_rate": success_rate,
        "mean_normalized_score": mean_normalized_score,
        "completion_rate": completion_rate,
        "mean_episode_length": mean_episode_length,
        "max_normalized_score": np.max(normalized_scores),
        "min_normalized_score": np.min(normalized_scores),
        "std_normalized_score": np.std(normalized_scores)
    })

    return {
        "success_rate": success_rate,
        "mean_normalized_score": mean_normalized_score,
        "completion_rate": completion_rate,
        "mean_episode_length": mean_episode_length,
        "max_normalized_score": np.max(normalized_scores),
        "min_normalized_score": np.min(normalized_scores),
        "std_normalized_score": np.std(normalized_scores)
    }

def train_mcp_pacman_subset(env, eval_env, subset_actions, total_timesteps, log_dir, game_config, num_primitives, features_dim, primitive_action_dim):
    wandb.init(project="mcp_pacman", name=f"pre-training_{log_dir}", config={
        "subset_actions": subset_actions,
        "total_timesteps": total_timesteps
    })
    
    policy_kwargs = dict(
        features_extractor_class=AtariCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
        num_primitives=num_primitives,
        primitive_action_dim=primitive_action_dim
    )

    model = PPO(MCPAtariPolicy, env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_dir)
    
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

def transfer_learning_full_actions(env, eval_env, model_path, total_timesteps, log_dir, game_config, num_primitives, features_dim, primitive_action_dim):
    wandb.init(project="mcp_pacman", name=f"transfer_{log_dir}", config={
        "total_timesteps": total_timesteps
    })
    
    # Load the pre-trained model
    pre_trained_model = PPO.load(model_path)
    
    # Create a new model with the full action space
    policy_kwargs = dict(
        features_extractor_class=AtariCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
        num_primitives=num_primitives,
        primitive_action_dim=primitive_action_dim
    )
    
    transferred_model = PPO(MCPAtariPolicy, env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_dir)
    
    # Copy the pre-trained weights
    transferred_model.policy.features_extractor.load_state_dict(pre_trained_model.policy.features_extractor.state_dict())
    transferred_model.policy.mlp_extractor.load_state_dict(pre_trained_model.policy.mlp_extractor.state_dict())
    
    # Initialize the action_net weights
    torch.nn.init.orthogonal_(transferred_model.policy.action_net.weight, gain=0.01)
    torch.nn.init.constant_(transferred_model.policy.action_net.bias, 0.0)
    
    # Freeze the primitives
    transferred_model.policy.freeze_primitives()
    
    # Ensure the optimizer is recreated with the correct parameters
    transferred_model.policy.optimizer = transferred_model.policy.optimizer_class(
        transferred_model.policy.parameters(),
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
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)

    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=True, render=False)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
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

    # Create a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the log directory with game name, number of primitives, and timestamp
    log_dir = f"logs/{args.game_name}_primitives{args.num_primitives}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    env = create_pacman_env(args.subset_actions)
    eval_env = create_pacman_env(args.subset_actions)

    try:
        print("Training MCP on subset of actions...")
        mcp_subset_model, results["mcp_subset"] = train_mcp_pacman_subset(
            env, eval_env, args.subset_actions, args.mcp_subset_timesteps, 
            f"{log_dir}/mcp_subset", game_config, args.num_primitives, 
            args.features_dim, args.primitive_action_dim
        )
        print("MCP Subset Results:", results["mcp_subset"])
    except Exception as e:
        print(f"Error in MCP subset training: {e}")
        print(traceback.format_exc())  

    env = create_pacman_env()
    eval_env = create_pacman_env()

    try:
        print("Performing transfer learning to full action space...")
        mcp_full_model, results["mcp_full"] = transfer_learning_full_actions(
            env, eval_env, f"{log_dir}/mcp_subset/final_model", 
            args.mcp_full_timesteps, f"{log_dir}/mcp_full", game_config, 
            args.num_primitives, args.features_dim, args.primitive_action_dim
        )
        print("MCP Full Results:", results["mcp_full"])
    except Exception as e:
        print(f"Error in MCP full action space training: {e}")
        print(traceback.format_exc())  

    try:
        print("Training baseline PPO...")
        baseline_model, results["baseline_ppo"] = train_baseline_ppo(
            env, eval_env, args.baseline_ppo_timesteps, 
            f"{log_dir}/baseline_ppo", game_config
        )
        print("Baseline PPO Results:", results["baseline_ppo"])
    except Exception as e:
        print(f"Error in baseline PPO training: {e}")
        print(traceback.format_exc())

    # Save results to a JSON file
    with open(f"{log_dir}/experiment_results.json", "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(f"Experiment completed. Results saved to {log_dir}/experiment_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCP experiments on Atari games")
    parser.add_argument("--game_name", type=str, default="MsPacman", help="Name of the Atari game")
    parser.add_argument("--subset_actions", type=int, nargs="+", default=[3, 2, 1], help="Subset of actions for initial training")
    parser.add_argument("--mcp_subset_timesteps", type=int, default=100000, help="Total timesteps for MCP subset training")
    parser.add_argument("--mcp_full_timesteps", type=int, default=50000, help="Total timesteps for MCP full action space training")
    parser.add_argument("--baseline_ppo_timesteps", type=int, default=100000, help="Total timesteps for baseline PPO training")
    parser.add_argument("--num_primitives", type=int, default=8, help="Number of primitives in the MCP model")
    parser.add_argument("--features_dim", type=int, default=512, help="Dimension of the feature extractor output")
    parser.add_argument("--primitive_action_dim", type=int, default=3, help="Dimension of primitive actions")
    parser.add_argument("--eval_freq", type=int, default=10000, help="Frequency of evaluation during training")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    run_experiment(args)