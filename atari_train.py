import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from atari_policies import MCPAtariPolicy, MCPTransferPolicy, AtariCNN
import os
import json
import wandb
from wandb.integration.sb3 import WandbCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

def calculate_mcp_metrics(rewards, episode_lengths, success_threshold=500):
    num_episodes = len(rewards)
    success_rate = sum(reward >= success_threshold for reward in rewards) / num_episodes
    completion_time = np.mean(episode_lengths)
    mean_reward = np.mean(rewards)
    
    return {
        "success_rate": success_rate,
        "completion_time": completion_time,
        "mean_reward": mean_reward,
    }

def calculate_additional_metrics(rewards, episode_lengths):
    return {
        "median_reward": np.median(rewards),
        "std_reward": np.std(rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "median_episode_length": np.median(episode_lengths),
        "std_episode_length": np.std(episode_lengths),  # Fixed typo here
        "max_reward": np.max(rewards),
        "min_reward": np.min(rewards),
    }

def evaluate_model(model, env, num_episodes=100):
    rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=num_episodes, return_episode_rewards=True)
    mcp_metrics = calculate_mcp_metrics(rewards, episode_lengths)
    additional_metrics = calculate_additional_metrics(rewards, episode_lengths)
    return {**mcp_metrics, **additional_metrics}

def train_mcp_pacman_subset(subset_actions, total_timesteps=100000, log_dir="logs/mcp_subset"):
    wandb.init(project="mcp-pacman", name="mcp_subset", config={
        "algorithm": "MCP-PPO",
        "subset_actions": subset_actions,
        "total_timesteps": total_timesteps,
        "num_primitives": len(subset_actions)
    })
    
    env = create_pacman_env(subset_actions)
    eval_env = create_pacman_env(subset_actions)

    policy_kwargs = dict(
        features_extractor_class=AtariCNN,
        features_extractor_kwargs=dict(features_dim=512),
        num_primitives=len(subset_actions)
    )

    model = PPO(MCPTransferPolicy, env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=log_dir)
    
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=True, render=False)
    
    wandb_callback = WandbCallback(gradient_save_freq=100, model_save_path=f"{wandb.run.dir}/models")

    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, wandb_callback])
    model.save(f"{log_dir}/final_model")
    
    wandb.finish()
    return model

def transfer_learning_full_actions(model_dir="logs/mcp_subset", total_timesteps=50000, log_dir="logs/mcp_full"):
    wandb.init(project="mcp-pacman", name="mcp_full", config={
        "algorithm": "MCP-PPO-Transfer",
        "total_timesteps": total_timesteps,
        "transfer_from": model_dir
    })
    
    full_env = create_pacman_env()
    eval_env = create_pacman_env()

    # Load the model trained on subset actions
    subset_env = create_pacman_env([3, 2, 1])  # LEFT, RIGHT, UP
    subset_model = PPO.load(f"{model_dir}/final_model", env=subset_env)
    
    # Create a new model with the full action space
    model = PPO(MCPTransferPolicy, full_env, verbose=1, tensorboard_log=log_dir)
    
    # Copy the weights that match
    model.policy.features_extractor.load_state_dict(subset_model.policy.features_extractor.state_dict())
    model.policy.mlp_extractor.feature_extractor.load_state_dict(subset_model.policy.mlp_extractor.feature_extractor.state_dict())
    
    # Adapt the model to the new action space
    model.policy.adapt_to_new_action_space(full_env.action_space.n, subset_model.policy.state_dict())
    
    # Freeze the primitives and unfreeze the gate
    model.policy.freeze_primitives()
    model.policy.unfreeze_gate()

    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=True, render=False)
    
    wandb_callback = WandbCallback(gradient_save_freq=100, model_save_path=f"{wandb.run.dir}/models")

    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, wandb_callback])
    model.save(f"{log_dir}/final_model")
    
    wandb.finish()
    return model

def train_baseline_ppo(total_timesteps=150000, log_dir="logs/baseline_ppo"):
    wandb.init(project="mcp-pacman", name="baseline_ppo", config={
        "algorithm": "PPO",
        "total_timesteps": total_timesteps
    })
    
    env = create_pacman_env()
    eval_env = create_pacman_env()

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)

    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=True, render=False)
    
    wandb_callback = WandbCallback(gradient_save_freq=100, model_save_path=f"{wandb.run.dir}/models")

    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, wandb_callback])
    model.save(f"{log_dir}/final_model")
    
    wandb.finish()
    return model

def run_experiment():
    subset_actions = [3, 2, 1]  # LEFT, RIGHT, UP
    results = {}

    try:
        print("Training MCP on subset of actions...")
        mcp_subset_model = train_mcp_pacman_subset(subset_actions)
        subset_env = create_pacman_env(subset_actions)
        results["mcp_subset"] = evaluate_model(mcp_subset_model, subset_env)
        print("MCP Subset Results:", results["mcp_subset"])
    except Exception as e:
        print(f"Error in MCP subset training: {e}")

    try:
        print("Performing transfer learning to full action space...")
        mcp_full_model = transfer_learning_full_actions()
        full_env = create_pacman_env()
        results["mcp_full"] = evaluate_model(mcp_full_model, full_env)
        print("MCP Full Results:", results["mcp_full"])
    except Exception as e:
        print(f"Error in MCP full action space training: {e}")

    try:
        print("Training baseline PPO...")
        baseline_model = train_baseline_ppo()
        baseline_env = create_pacman_env()
        results["baseline_ppo"] = evaluate_model(baseline_model, baseline_env)
        print("Baseline PPO Results:", results["baseline_ppo"])
    except Exception as e:
        print(f"Error in baseline PPO training: {e}")

    # Save results to a JSON file
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Experiment completed. Results saved to experiment_results.json")

if __name__ == "__main__":
    run_experiment()