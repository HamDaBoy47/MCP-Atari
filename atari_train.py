import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.distributions import CategoricalDistribution
from atari_policies import MCPAtariPolicy, AtariCNN
from game_configs import get_game_config
import wandb
import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
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

def train_mcp_pacman_subset(subset_actions, total_timesteps=1000000, log_dir="logs/mcp_subset"):
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
    
    wandb.finish()
    return model

def transfer_learning_full_actions(model_dir="logs/mcp_subset", total_timesteps=500000, log_dir="logs/mcp_full"):
    wandb.init(project="mcp_pacman", name="transfer_learning", config={
        "model_dir": model_dir,
        "total_timesteps": total_timesteps
    })
    
    env = create_pacman_env()  # Full action space environment
    eval_env = create_pacman_env()

    transferred_model = PPO.load(f"{model_dir}/final_model", env=env)
    transferred_model.policy.freeze_primitives()

    # Get the new action space size
    new_action_space_size = env.action_space.n

    # Adjust the action distribution for the new action space
    transferred_model.policy.action_dist = CategoricalDistribution(new_action_space_size)

    # Create a new output layer for each primitive to match the new action space
    for i, primitive in enumerate(transferred_model.policy.mlp_extractor.primitives):
        old_linear = primitive[-1]  # Assume the last layer is the output layer
        in_features = old_linear.in_features
        new_linear = torch.nn.Linear(in_features, new_action_space_size)
        
        # Initialize the new layer with weights from the old layer
        with torch.no_grad():
            new_linear.weight[:old_linear.out_features] = old_linear.weight
            new_linear.bias[:old_linear.out_features] = old_linear.bias
        
        # Replace the old layer with the new one
        primitive[-1] = new_linear

    # Only set requires_grad=True for the gating function parameters
    for param in transferred_model.policy.mlp_extractor.gate.parameters():
        param.requires_grad = True

    # Create a custom learning rate schedule that only updates the gating function
    def custom_lr_schedule(progress_remaining):
        return 3e-4  # You can adjust this value as needed

    # Set the learning rate schedule to only affect the gating function
    transferred_model.policy.optimizer = transferred_model.policy.optimizer_class(
        [param for param in transferred_model.policy.parameters() if param.requires_grad],
        lr=custom_lr_schedule(1),
        **transferred_model.policy.optimizer_kwargs
    )

    wandb_callback = WandbCallback(eval_env)
    
    transferred_model.learn(total_timesteps=total_timesteps, callback=wandb_callback)
    transferred_model.save(f"{log_dir}/final_model")
    
    wandb.finish()
    return transferred_model

def train_baseline_ppo(total_timesteps=150000, log_dir="logs/baseline_ppo"):
    env = create_pacman_env()
    eval_env = create_pacman_env()

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)

    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=True, render=False)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"{log_dir}/final_model")
    return model

def run_experiment():
    subset_actions = [3, 2, 1]  # LEFT, RIGHT, UP
    results = {}
    
    game_name = "MsPacman"  # or whatever game you're currently evaluating
    game_config = get_game_config(game_name)

    try:
        print("Training MCP on subset of actions...")
        mcp_subset_model = train_mcp_pacman_subset(subset_actions)
        subset_env = create_pacman_env(subset_actions)
        results["mcp_subset"] = evaluate_model(
            mcp_subset_model, 
            subset_env, 
            random_score=game_config["random_score"],
            human_score=game_config["human_score"],
            success_threshold=game_config["success_threshold"]
        )
        print("MCP Subset Results:", results["mcp_subset"])
    except Exception as e:
        print(f"Error in MCP subset training: {e}")

    try:
        print("Performing transfer learning to full action space...")
        mcp_full_model = transfer_learning_full_actions(model_dir="logs/mcp_subset")
        full_env = create_pacman_env()
        results["mcp_full"] = evaluate_model(
            mcp_full_model, 
            full_env, 
            random_score=game_config["random_score"],
            human_score=game_config["human_score"],
            success_threshold=game_config["success_threshold"]
        )
        print("MCP Full Results:", results["mcp_full"])
    except Exception as e:
        print(f"Error in MCP full action space training: {e}")

    try:
        print("Training baseline PPO...")
        baseline_model = train_baseline_ppo()
        baseline_env = create_pacman_env()
        results["baseline_ppo"] = evaluate_model(
            baseline_model, 
            baseline_env, 
            random_score=game_config["random_score"],
            human_score=game_config["human_score"],
            success_threshold=game_config["success_threshold"]
        )
        print("Baseline PPO Results:", results["baseline_ppo"])
    except Exception as e:
        print(f"Error in baseline PPO training: {e}")

    # Save results to a JSON file
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Experiment completed. Results saved to experiment_results.json")

if __name__ == "__main__":
    run_experiment()