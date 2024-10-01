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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from stable_baselines3.common.callbacks import BaseCallback

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

def train_mcp_pacman_subset(subset_actions, total_timesteps=1000000, log_dir="logs/mcp_subset", game_config=None):
    wandb.init(project="mcp_pacman", name="subset_training", config={
        "subset_actions": subset_actions,
        "total_timesteps": total_timesteps
    })
    
    env = create_pacman_env(subset_actions)
    eval_env = create_pacman_env(subset_actions)

    policy_kwargs = dict(
        features_extractor_class=AtariCNN,
        features_extractor_kwargs=dict(features_dim=512),
        num_primitives=len(subset_actions)
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

def transfer_learning_full_actions(model_path="logs/mcp_subset/final_model", total_timesteps=50000, log_dir="logs/mcp_full", game_config=None):
    env = create_pacman_env()
    eval_env = create_pacman_env()

    transferred_model = PPO.load(model_path, env=env)
    transferred_model.policy.freeze_primitives()

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

def train_baseline_ppo(total_timesteps=150000, log_dir="logs/baseline_ppo", game_config=None):
    env = create_pacman_env()
    eval_env = create_pacman_env()

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)

    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=True, render=False)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"{log_dir}/final_model")
    
    results = evaluate_model(
        env, 
        eval_env, 
        random_score=game_config["random_score"],
        human_score=game_config["human_score"],
        success_threshold=game_config["success_threshold"]
    )
    wandb.log(results)
    
    return model, results 

def run_experiment():
    subset_actions = [3, 2, 1]  # LEFT, RIGHT, UP
    results = {}
    
    game_name = "MsPacman"
    game_config = get_game_config(game_name)

    try:
        print("Training MCP on subset of actions...")
        mcp_subset_model, results["mcp_subset"] = train_mcp_pacman_subset(subset_actions, game_config=game_config)
        print("MCP Subset Results:", results["mcp_subset"])
    except Exception as e:
        print(f"Error in MCP subset training: {e}")
        print(traceback.format_exc())  

    try:
        print("Performing transfer learning to full action space...")
        mcp_full_model, results["mcp_full"] = transfer_learning_full_actions("logs/mcp_subset/final_model", game_config=game_config)
        print("MCP Full Results:", results["mcp_full"])
    except Exception as e:
        print(f"Error in MCP full action space training: {e}")
        print(traceback.format_exc())  

    try:
        print("Training baseline PPO...")
        baseline_model, results["baseline_ppo"] = train_baseline_ppo(game_config=game_config)
        print("Baseline PPO Results:", results["baseline_ppo"])
    except Exception as e:
        print(f"Error in baseline PPO training: {e}")
        print(traceback.format_exc())  

    # Save results to a JSON file
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Experiment completed. Results saved to experiment_results.json")

if __name__ == "__main__":
    run_experiment()